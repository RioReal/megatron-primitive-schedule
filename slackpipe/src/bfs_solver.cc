#include "slackpipe/bfs_solver.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <sstream>

#include "slackpipe/breadth_first.h"
#include "slackpipe/dag_evaluator.h"

namespace slackpipe {

namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] Tick SafeHorizon(const Instance &instance) {
  const Tick ratio_sum =
      CheckedAdd(instance.backward_ratio_num, instance.backward_ratio_den,
                 "breadth-first horizon ratio sum");
  return CheckedMul(CheckedMul(instance.microbatches, instance.total_layers,
                               "breadth-first horizon"),
                    ratio_sum, "breadth-first horizon");
}

void EnumerateSuffix(const Instance &instance, Index stage, Tick remaining,
                     std::vector<Tick> &split, const SplitVisitor &visitor) {
  if (stage == instance.stages - 1) {
    if (remaining < instance.min_layers) return;
    split[static_cast<std::size_t>(stage)] = remaining;
    visitor(split);
    return;
  }

  const Index stages_left_after = instance.stages - stage - 1;
  const Tick max_here =
      remaining - CheckedMul(stages_left_after, instance.min_layers,
                             "split enumeration remaining minimum");
  for (Tick value = instance.min_layers; value <= max_here; ++value) {
    split[static_cast<std::size_t>(stage)] = value;
    EnumerateSuffix(instance, stage + 1, remaining - value, split, visitor);
  }
}

[[nodiscard]] bool LexicographicallySmaller(const std::vector<Tick> &a,
                                            const std::vector<Tick> &b) {
  return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

[[nodiscard]] std::string JoinTicksForDiagnostic(
    const std::vector<Tick> &values) {
  std::ostringstream out;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ",";
    out << values[i];
  }
  return out.str();
}

[[nodiscard]] std::string WorkerOrderSizesForDiagnostic(
    const MachineOrders &orders) {
  std::ostringstream out;
  for (std::size_t i = 0; i < orders.size(); ++i) {
    if (i != 0) out << ",";
    out << orders[i].size();
  }
  return out.str();
}

[[nodiscard]] std::string BfsCanonicalMismatchDiagnostic(
    const Instance &instance, const BfsSplitOptimizationResult &result,
    Tick raw_objective, Tick deterministic_makespan,
    std::size_t actual_operation_count) {
  std::ostringstream diagnostic;
  diagnostic << "CP-SAT objective does not match deterministic evaluation: "
             << "status=" << result.status
             << " raw_cp_sat_objective=" << raw_objective
             << " deterministic_makespan=" << deterministic_makespan
             << " signed_difference="
             << (deterministic_makespan - raw_objective) << " selected_split=["
             << JoinTicksForDiagnostic(result.split)
             << "] expected_operation_count=" << instance.OperationCount()
             << " actual_operation_count=" << actual_operation_count
             << " worker_order_sizes=["
             << WorkerOrderSizesForDiagnostic(result.machine_orders) << "]";
  return diagnostic.str();
}

}  // namespace

std::uint64_t CountValidSplitsCapped(const Instance &instance,
                                     std::uint64_t cap) {
  instance.Validate();
  if (cap == 0) return 0;

  const Tick slack =
      instance.total_layers -
      CheckedMul(instance.stages, instance.min_layers, "split count slack");
  if (slack < 0) return 0;
  const std::uint64_t n = static_cast<std::uint64_t>(
      CheckedAdd(slack, instance.stages - 1, "split count binomial n"));
  std::uint64_t k = static_cast<std::uint64_t>(instance.stages - 1);
  if (k > n - k) k = n - k;

  __uint128_t value = 1;
  for (std::uint64_t i = 1; i <= k; ++i) {
    value *= (n - k + i);
    value /= i;
    if (value >= cap) return cap;
  }
  return static_cast<std::uint64_t>(value);
}

void SetExactStagePartitionCount(const Instance &instance, SearchStats *stats) {
  if (stats == nullptr) return;
  constexpr std::uint64_t kMaxCount =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  const std::uint64_t count = CountValidSplitsCapped(instance, kMaxCount);
  if (count < kMaxCount) {
    stats->stage_partitions_theoretical_available = true;
    stats->stage_partitions_theoretical = static_cast<std::int64_t>(count);
  } else {
    stats->stage_partitions_theoretical_available = false;
    stats->stage_partitions_theoretical = 0;
  }
}

void EnumerateValidSplits(const Instance &instance,
                          const SplitVisitor &visitor) {
  instance.Validate();
  std::vector<Tick> split(static_cast<std::size_t>(instance.stages), 0);
  EnumerateSuffix(instance, 0, instance.total_layers, split, visitor);
}

void ApplyBfsCanonicalEvaluationPolicy(const Instance &instance,
                                       BfsSplitOptimizationResult &result,
                                       Tick raw_cp_sat_objective,
                                       SearchStats *search_stats) {
  if (search_stats != nullptr) {
    ++search_stats->candidate_schedules_deterministically_evaluated;
  }
  EvaluationResult evaluated =
      EvaluateSchedule(instance, result.split, result.machine_orders);
  if (!evaluated.schedule.ok()) {
    if (search_stats != nullptr) ++search_stats->candidate_schedules_rejected;
    throw Error("deterministic evaluator rejected CP-SAT split");
  }
  if (search_stats != nullptr) ++search_stats->candidate_schedules_accepted;
  const Tick deterministic_makespan = evaluated.schedule.makespan;
  if (deterministic_makespan > raw_cp_sat_objective) {
    throw Error(BfsCanonicalMismatchDiagnostic(
        instance, result, raw_cp_sat_objective, deterministic_makespan,
        evaluated.schedule.operations_by_id.size()));
  }
  if (deterministic_makespan < raw_cp_sat_objective) {
    const std::string diagnostic = BfsCanonicalMismatchDiagnostic(
        instance, result, raw_cp_sat_objective, deterministic_makespan,
        evaluated.schedule.operations_by_id.size());
    if (result.status == "OPTIMAL") {
      throw Error(diagnostic);
    }
    if (result.status == "FEASIBLE") {
      result.diagnostic = diagnostic;
      result.makespan_ticks = deterministic_makespan;
    } else {
      throw Error(diagnostic);
    }
  }
  result.schedule = std::move(evaluated.schedule);
}

BfsSplitOptimizationResult OptimizeBfsSplitEnumerate(
    const Instance &instance, const BfsSplitOptimizerOptions &options) {
  const auto started = Clock::now();
  instance.Validate();
  (void)SafeHorizon(instance);
  SearchStats *stats = options.search_stats;
  SetExactStagePartitionCount(instance, stats);
  if (stats != nullptr) {
    if (stats->algorithm.empty()) stats->algorithm = "optimize-bfs";
    stats->enumerative_search = true;
    stats->stage_partitions_enumerated = true;
    stats->stage_partitions_note =
        "Stage partitions are explicitly enumerated "
        "by OptimizeBfsSplitEnumerate.";
  }

  BfsSplitOptimizationResult result;
  result.method = "enumerate";
  result.status = "OPTIMAL";
  result.solver_status_raw = result.status;
  result.best_bound_ticks = 0;
  result.machine_orders = BreadthFirstOrders(instance);
  result.search_stats_enabled = stats != nullptr;
  result.returned_solution_source = "bfs_enumeration";
  result.solution_source = result.returned_solution_source;

  bool have_best = false;
  EnumerateValidSplits(instance, [&](const std::vector<Tick> &split) {
    ++result.checked_splits;
    if (stats != nullptr) {
      ++stats->stage_partitions_visited;
      ++stats->candidate_schedules_extracted;
      ++stats->candidate_schedules_deterministically_evaluated;
    }
    EvaluationResult evaluated =
        EvaluateSchedule(instance, split, result.machine_orders);
    if (!evaluated.schedule.ok()) {
      if (stats != nullptr) ++stats->candidate_schedules_rejected;
      throw Error("deterministic evaluator rejected enumerated split");
    }
    if (stats != nullptr) ++stats->candidate_schedules_accepted;
    const bool better = !have_best ||
                        evaluated.schedule.makespan < result.makespan_ticks ||
                        (evaluated.schedule.makespan == result.makespan_ticks &&
                         LexicographicallySmaller(split, result.split));
    if (better) {
      if (stats != nullptr) {
        ++stats->stage_partitions_kept;
      }
      have_best = true;
      result.split = split;
      result.makespan_ticks = evaluated.schedule.makespan;
      result.schedule = std::move(evaluated.schedule);
    }
  });

  if (!have_best) throw Error("no valid split found");
  result.best_bound_ticks = result.makespan_ticks;
  result.solver_objective_ticks = result.makespan_ticks;
  result.proven_optimal = true;
  result.first_feasible_objective = result.makespan_ticks;
  result.incumbent_improvement_count = have_best ? 1 : 0;
  result.wall_time_seconds =
      std::chrono::duration<double>(Clock::now() - started).count();
  result.time_to_first_feasible_seconds = result.wall_time_seconds;
  result.time_to_best_incumbent_seconds = result.wall_time_seconds;
  result.timing.incumbent_seconds = result.wall_time_seconds;
  result.timing.total_seconds = result.wall_time_seconds;
  if (stats != nullptr) result.search_stats = *stats;
  return result;
}

#if !SLACKPIPE_HAVE_ORTOOLS
BfsSplitOptimizationResult OptimizeBfsSplitCpSat(
    const Instance &instance, const BfsSplitOptimizerOptions &options) {
  instance.Validate();
  BfsSplitOptimizationResult result;
  result.method = "cpsat";
  result.status = "UNAVAILABLE";
  result.solver_status_raw = result.status;
  result.best_bound_ticks = 0;
  result.proven_optimal = false;
  result.machine_orders = BreadthFirstOrders(instance);
  result.solution_source = "none";
  result.returned_solution_source = "none";
  result.search_stats_enabled = options.search_stats != nullptr;
  if (options.search_stats != nullptr)
    result.search_stats = *options.search_stats;
  return result;
}

bool IsCpSatBfsSplitOptimizerAvailable() { return false; }
#endif

BfsSplitOptimizationResult OptimizeBfsSplitAuto(
    const Instance &instance, const BfsSplitOptimizerOptions &options) {
  const std::uint64_t count =
      CountValidSplitsCapped(instance, options.enumeration_threshold + 1);
  if (count <= options.enumeration_threshold) {
    BfsSplitOptimizationResult result =
        OptimizeBfsSplitEnumerate(instance, options);
    result.method = "auto-enumerate";
    return result;
  }
  BfsSplitOptimizationResult result = OptimizeBfsSplitCpSat(instance, options);
  result.method = "auto-cpsat";
  return result;
}

}  // namespace slackpipe
