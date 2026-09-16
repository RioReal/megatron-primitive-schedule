#include "slackpipe/fixed_order_partition_solver.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "slackpipe/dag_evaluator.h"
#include "slackpipe/deadline.h"
#include "slackpipe/evaluation_method.h"
#include "slackpipe/result_validator.h"
#include "slackpipe/slackpipe_solver.h"
#include "slackpipe/activation_cpsat.h"

#if SLACKPIPE_HAVE_ORTOOLS
#include <mutex>

#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/sat/sat_parameters.pb.h"
#endif

namespace slackpipe {

namespace {

using Clock = std::chrono::steady_clock;

struct PartitionCountEstimate {
  std::uint64_t count = 0;
  bool exact = false;
};

struct ValidatedIncumbent {
  std::vector<Tick> split;
  ScheduleSolution schedule;
  Tick makespan = 0;
  std::string source;
  double runtime_seconds = 0.0;
};

[[nodiscard]] double Since(Clock::time_point started) {
  return std::chrono::duration<double>(Clock::now() - started).count();
}

[[nodiscard]] bool LexicographicallySmaller(const std::vector<Tick> &a,
                                            const std::vector<Tick> &b) {
  return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

[[nodiscard]] std::string NormalizeBackend(std::string backend) {
  if (backend.empty()) return "auto";
  std::transform(
      backend.begin(), backend.end(), backend.begin(),
      [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  if (backend == "auto" || backend == "cpsat" || backend == "enumerate") {
    return backend;
  }
  throw Error("unknown fixed-order partition backend: " + backend);
}

[[nodiscard]] PartitionCountEstimate EstimatePartitions(
    const Instance &instance, std::uint64_t threshold) {
  const std::uint64_t cap =
      threshold == std::numeric_limits<std::uint64_t>::max() ? threshold
                                                             : threshold + 1;
  const std::uint64_t count = CountValidSplitsCapped(instance, cap);
  return PartitionCountEstimate{count, count < cap};
}

void PopulateFixedOrderProvenance(BfsSplitOptimizationResult &result,
                                  const BfsSplitOptimizerOptions &options,
                                  const std::string &requested_backend,
                                  const std::string &effective_backend,
                                  PartitionCountEstimate estimate) {
  result.fixed_order_partition_backend_requested = requested_backend;
  result.fixed_order_partition_backend_effective = effective_backend;
  result.enumeration_safety_threshold = options.enumeration_threshold;
  result.estimated_partition_count = estimate.count;
  result.estimated_partition_count_available = estimate.exact;
}

void PopulateFixedOrderStats(SearchStats *stats,
                             PartitionCountEstimate estimate,
                             bool enumerative_search, const std::string &note) {
  if (stats == nullptr) return;
  if (stats->algorithm.empty()) {
    stats->algorithm = "partition-only-fixed-order";
  }
  stats->enumerative_search = enumerative_search;
  stats->stage_partitions_note = note;
  stats->stage_partitions_enumerated = enumerative_search;
  constexpr std::uint64_t kInt64Max =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  if (estimate.exact && estimate.count <= kInt64Max) {
    stats->stage_partitions_theoretical_available = true;
    stats->stage_partitions_theoretical =
        static_cast<std::int64_t>(estimate.count);
  }
}

[[nodiscard]] ValidatedIncumbent BuildValidatedIncumbent(
    const Instance &instance, const MachineOrders &fixed_orders,
    const std::vector<Tick> &split, const std::string &source,
    const std::string &validation_context) {
  const auto started = Clock::now();
  ValidateSplit(instance, split);
  EvaluationResult evaluated = EvaluateSchedule(instance, split, fixed_orders);
  if (!evaluated.schedule.ok()) {
    throw Error(source + " fixed-order partition is invalid: " +
                evaluated.schedule.validation_errors.front());
  }
  const ResultValidationResult validation = ValidateScheduleSolutionIndependent(
      instance, evaluated.schedule, "FEASIBLE", validation_context);
  if (!validation.passed) {
    throw Error(source +
                " fixed-order partition failed independent validation: " +
                validation.error_code);
  }
  ValidatedIncumbent incumbent;
  incumbent.split = split;
  incumbent.makespan = evaluated.schedule.makespan;
  incumbent.schedule = std::move(evaluated.schedule);
  incumbent.source = source;
  incumbent.runtime_seconds = Since(started);
  return incumbent;
}

void ValidateFixedOrderOrThrow(const Instance &instance,
                               const MachineOrders &fixed_orders) {
  (void)BuildValidatedIncumbent(instance, fixed_orders, UniformSplit(instance),
                                "uniform", "fixed-order-precheck");
}

void ValidateReturnedResultOrThrow(const Instance &instance,
                                   const BfsSplitOptimizationResult &result) {
  if (!IsTerminalFeasibleStatus(result.status)) return;
  const ResultValidationResult validation = ValidateScheduleSolutionIndependent(
      instance, result.schedule, result.status, "partition-only-fixed-order");
  if (!validation.passed) {
    throw Error("fixed-order partition result failed independent validation: " +
                validation.error_code);
  }
}

void MaybeRejectNonOptimal(const BfsSplitOptimizerOptions &options,
                           BfsSplitOptimizationResult &result) {
  if (options.require_optimal && !result.proven_optimal &&
      result.status == "FEASIBLE") {
    result.status = "FEASIBLE_REJECTED_REQUIRE_OPTIMAL";
  }
}

void FinalizeBfsResultTiming(Clock::time_point started,
                             BfsSplitOptimizationResult &result) {
  result.wall_time_seconds = Since(started);
  result.timing.total_seconds = result.wall_time_seconds;
  if (result.solver_status_raw.empty())
    result.solver_status_raw = result.status;
}

BfsSplitOptimizationResult MakeUnavailableCpSatResult(
    const Instance &instance, const MachineOrders &fixed_orders,
    const BfsSplitOptimizerOptions &options, const std::string &method,
    const std::string &requested_backend, const std::string &diagnostic,
    PartitionCountEstimate estimate) {
  const auto started = Clock::now();
  instance.Validate();
  ValidateFixedOrderOrThrow(instance, fixed_orders);
  BfsSplitOptimizationResult result;
  result.method = method;
  result.status = "UNAVAILABLE";
  result.solver_status_raw = result.status;
  result.proven_optimal = false;
  result.machine_orders = fixed_orders;
  result.search_stats_enabled = options.search_stats != nullptr;
  result.diagnostic = diagnostic;
  result.activation_cap_constraints.model_support_level =
      ToString(ActivationCapSolverSupport());
  result.activation_cap_constraints.solver_supported =
      ActivationCapSolverCanEnforce(options.activation_options, true);
  if (options.activation_options.enforce_activation_cap) {
    result.activation_cap_constraints.unsupported_reason =
        ActivationCapSolverUnsupportedReason(options.activation_options, true);
  }
  result.solution_source = "none";
  result.returned_solution_source = "none";
  PopulateFixedOrderProvenance(result, options, requested_backend,
                               "unavailable", estimate);
  if (options.search_stats != nullptr) {
    PopulateFixedOrderStats(
        options.search_stats, estimate, false,
        "Stage partitions would be encoded as CP-SAT variables, but CP-SAT is "
        "not available for this requested backend.");
    result.search_stats = *options.search_stats;
  }
  FinalizeBfsResultTiming(started, result);
  return result;
}

#if SLACKPIPE_HAVE_ORTOOLS
using operations_research::Domain;
using operations_research::sat::CpModelBuilder;
using operations_research::sat::CpSolverResponse;
using operations_research::sat::CpSolverStatus;
using operations_research::sat::IntVar;
using operations_research::sat::LinearExpr;
using operations_research::sat::Model;
using operations_research::sat::NewFeasibleSolutionObserver;
using operations_research::sat::NewSatParameters;
using operations_research::sat::SatParameters;
using operations_research::sat::SolutionIntegerValue;
using operations_research::sat::SolveCpModel;

[[nodiscard]] Tick ConservativeFixedOrderHorizon(const Instance &instance) {
  if (instance.HasCostProfile()) {
    Tick max_slope = 0;
    Tick max_bias = 0;
    for (Index stage = 0; stage < instance.stages; ++stage) {
      const auto index = static_cast<std::size_t>(stage);
      max_slope =
          std::max(max_slope, instance.profile_forward_slope_ticks[index]);
      max_slope =
          std::max(max_slope, instance.profile_backward_slope_ticks[index]);
      max_bias = std::max(max_bias, instance.profile_forward_bias_ticks[index]);
      max_bias =
          std::max(max_bias, instance.profile_backward_bias_ticks[index]);
    }
    const Tick max_duration =
        CheckedAdd(CheckedMul(instance.total_layers, max_slope,
                              "fixed-order profile horizon duration"),
                   max_bias, "fixed-order profile horizon duration");
    const Tick compute = CheckedMul(instance.OperationCount(), max_duration,
                                    "fixed-order profile horizon compute");
    const Tick data_edges = CheckedMul(
        instance.microbatches,
        CheckedAdd(
            CheckedMul(2, instance.stages, "fixed-order data edge count"), -1,
            "fixed-order data edge count"),
        "fixed-order data edge count");
    const Tick communication =
        CheckedMul(data_edges, instance.communication_ticks,
                   "fixed-order horizon communication");
    return CheckedAdd(compute, communication, "fixed-order profile horizon");
  }
  const Tick ratio_sum =
      CheckedAdd(instance.backward_ratio_num, instance.backward_ratio_den,
                 "fixed-order horizon ratio sum");
  const Tick compute =
      CheckedMul(CheckedMul(instance.microbatches, instance.total_layers,
                            "fixed-order horizon compute"),
                 ratio_sum, "fixed-order horizon compute");
  const Tick data_edges = CheckedMul(
      instance.microbatches,
      CheckedAdd(CheckedMul(2, instance.stages, "fixed-order data edge count"),
                 -1, "fixed-order data edge count"),
      "fixed-order data edge count");
  const Tick communication =
      CheckedMul(data_edges, instance.communication_ticks,
                 "fixed-order horizon communication");
  return CheckedAdd(compute, communication, "fixed-order horizon");
}

[[nodiscard]] std::string StatusName(CpSolverStatus status) {
  switch (status) {
    case operations_research::sat::OPTIMAL:
      return "OPTIMAL";
    case operations_research::sat::FEASIBLE:
      return "FEASIBLE";
    case operations_research::sat::INFEASIBLE:
      return "INFEASIBLE";
    case operations_research::sat::MODEL_INVALID:
      return "MODEL_INVALID";
    case operations_research::sat::UNKNOWN:
      return "UNKNOWN";
    default:
      return "UNRECOGNIZED";
  }
}

void EmitLifecycle(const BfsSplitOptimizerOptions &options,
                   Clock::time_point started, const std::string &phase,
                   const std::string &status = "",
                   double effective_limit_seconds = 0.0,
                   const std::string &detail = "") {
  if (!options.lifecycle) return;
  LifecycleEvent event;
  event.phase = phase;
  event.algorithm = "partition-only-fixed-order";
  event.solver_status = status;
  event.elapsed_seconds = Since(started);
  event.configured_solver_limit_seconds = options.time_limit_seconds;
  event.effective_solver_limit_seconds = effective_limit_seconds;
  event.requested_solver_limit_seconds = options.time_limit_seconds;
  event.phase_specific_cap_seconds = options.time_limit_seconds;
  event.solver_threads = options.num_workers;
  event.detail = detail;
  options.lifecycle(event);
}

#endif

}  // namespace

BfsSplitOptimizationResult OptimizePartitionForFixedOrderEnumerate(
    const Instance &instance, const MachineOrders &fixed_orders,
    const BfsSplitOptimizerOptions &options) {
  const auto started = Clock::now();
  instance.Validate();
  ValidateFixedOrderOrThrow(instance, fixed_orders);
  const PartitionCountEstimate estimate =
      EstimatePartitions(instance, options.enumeration_threshold);
  Deadline deadline(options.time_limit_seconds);

  SearchStats *stats = options.search_stats;
  PopulateFixedOrderStats(
      stats, estimate, true,
      "Stage partitions are explicitly enumerated for a supplied fixed worker "
      "order.");

  BfsSplitOptimizationResult result;
  result.method = "fixed-order-enumerate";
  result.status = "UNKNOWN";
  result.solver_status_raw = result.status;
  result.best_bound_ticks = 0;
  result.machine_orders = fixed_orders;
  result.search_stats_enabled = stats != nullptr;
  result.solution_source = "fixed_order_partition_enumeration";
  result.returned_solution_source = result.solution_source;
  result.activation_cap_constraints.model_support_level =
      ToString(ActivationCapSolverSupport());
  result.activation_cap_constraints.solver_supported =
      ActivationCapSolverCanEnforce(options.activation_options, true);
  result.activation_cap_constraints.enforced_by_enumeration =
      options.activation_options.enforce_activation_cap;
  PopulateFixedOrderProvenance(result, options, "enumerate", "enumerate",
                               estimate);

  bool have_best = false;
  bool deadline_hit = false;
  EnumerateValidSplits(instance, [&](const std::vector<Tick> &split) {
    if (deadline.expired()) {
      deadline_hit = true;
      return;
    }
    ++result.checked_splits;
    ++result.enumeration_candidates_total;
    if (stats != nullptr) {
      ++stats->stage_partitions_visited;
      ++stats->candidate_schedules_extracted;
      ++stats->candidate_schedules_deterministically_evaluated;
    }
    EvaluationResult evaluated =
        EvaluateSchedule(instance, split, fixed_orders);
    if (!evaluated.schedule.ok()) {
      if (stats != nullptr) ++stats->candidate_schedules_rejected;
      throw Error(
          "deterministic evaluator rejected enumerated fixed-order split");
    }
    ++result.enumeration_candidates_valid_schedule;
    if (options.activation_options.enforce_activation_cap &&
        !ActivationScheduleSatisfiesCap(instance, evaluated.schedule,
                                        options.activation_options)) {
      ++result.enumeration_candidates_cap_rejected;
      if (stats != nullptr) ++stats->candidate_schedules_rejected;
      return;
    }
    ++result.enumeration_candidates_cap_feasible;
    if (stats != nullptr) ++stats->candidate_schedules_accepted;
    const bool better = !have_best ||
                        evaluated.schedule.makespan < result.makespan_ticks ||
                        (evaluated.schedule.makespan == result.makespan_ticks &&
                         LexicographicallySmaller(split, result.split));
    if (better) {
      if (stats != nullptr) ++stats->stage_partitions_kept;
      have_best = true;
      result.split = split;
      result.makespan_ticks = evaluated.schedule.makespan;
      result.schedule = std::move(evaluated.schedule);
    }
  });

  if (!have_best) {
    result.status = deadline_hit ? "NOT_RUN" : "INFEASIBLE";
    result.enumeration_proved_optimal = !deadline_hit;
    result.optimality_proof_source =
        result.enumeration_proved_optimal ? "exhaustive_enumeration" : "none";
    result.solution_source = "none";
    result.returned_solution_source = "none";
  } else {
    result.status = deadline_hit ? "FEASIBLE" : "OPTIMAL";
    result.proven_optimal = !deadline_hit;
    result.enumeration_proved_optimal = !deadline_hit;
    result.optimality_proof_source =
        result.enumeration_proved_optimal ? "exhaustive_enumeration" : "none";
    result.best_bound_ticks = result.proven_optimal ? result.makespan_ticks : 0;
    result.solver_objective_ticks = static_cast<double>(result.makespan_ticks);
    result.time_to_first_feasible_seconds = Since(started);
    result.time_to_best_incumbent_seconds =
        result.time_to_first_feasible_seconds;
    result.first_feasible_objective = result.makespan_ticks;
    result.incumbent_improvement_count = 1;
    ValidateReturnedResultOrThrow(instance, result);
  }
  result.solver_status_raw = result.status;
  FinalizeBfsResultTiming(started, result);
  result.timing.incumbent_seconds = result.wall_time_seconds;
  result.timing.total_seconds = result.wall_time_seconds;
  if (stats != nullptr) result.search_stats = *stats;
  MaybeRejectNonOptimal(options, result);
  return result;
}

#if !SLACKPIPE_HAVE_ORTOOLS
BfsSplitOptimizationResult OptimizePartitionForFixedOrderCpSat(
    const Instance &instance, const MachineOrders &fixed_orders,
    const BfsSplitOptimizerOptions &options) {
  const PartitionCountEstimate estimate =
      EstimatePartitions(instance, options.enumeration_threshold);
  return MakeUnavailableCpSatResult(
      instance, fixed_orders, options, "fixed-order-cpsat", "cpsat",
      "fixed-order partition CP-SAT is unavailable in this build; rebuild with "
      "SLACKPIPE_ENABLE_ORTOOLS=ON and a discoverable OR-Tools package",
      estimate);
}

bool IsCpSatFixedOrderPartitionOptimizerAvailable() { return false; }
#else
BfsSplitOptimizationResult OptimizePartitionForFixedOrderCpSat(
    const Instance &instance, const MachineOrders &fixed_orders,
    const BfsSplitOptimizerOptions &options) {
  const auto started = Clock::now();
  instance.Validate();
  ValidateFixedOrderOrThrow(instance, fixed_orders);
  const PartitionCountEstimate estimate =
      EstimatePartitions(instance, options.enumeration_threshold);
  SearchStats *stats = options.search_stats;
  PopulateFixedOrderStats(
      stats, estimate, false,
      "Stage partitions are encoded as CP-SAT variables for a supplied fixed "
      "worker order; enumeration is not used by the CP-SAT backend.");

  const ValidatedIncumbent incumbent =
      options.fixed_order_partition_incumbent
          ? BuildValidatedIncumbent(instance, fixed_orders,
                                    *options.fixed_order_partition_incumbent,
                                    "supplied_fixed_order_partition_incumbent",
                                    "partition-only-fixed-order")
          : BuildValidatedIncumbent(
                instance, fixed_orders, UniformSplit(instance),
                "uniform_fixed_order_incumbent", "partition-only-fixed-order");
  ActivationCapConstraintMetadata activation_metadata;
  activation_metadata.model_support_level =
      ToString(ActivationCapSolverSupport());
  activation_metadata.solver_supported =
      ActivationCapSolverCanEnforce(options.activation_options, true);
  if (options.activation_options.enforce_activation_cap &&
      !activation_metadata.solver_supported) {
    activation_metadata.unsupported_reason =
        ActivationCapSolverUnsupportedReason(options.activation_options, true);
    BfsSplitOptimizationResult unavailable = MakeUnavailableCpSatResult(
        instance, fixed_orders, options, "fixed-order-cpsat", "cpsat",
        activation_metadata.unsupported_reason, estimate);
    unavailable.activation_cap_constraints = activation_metadata;
    return unavailable;
  }
  const bool incumbent_cap_feasible =
      !options.activation_options.enforce_activation_cap ||
      ActivationScheduleSatisfiesCap(instance, incumbent.schedule,
                                     options.activation_options);
  activation_metadata.incumbent_rejected_for_activation_cap =
      options.activation_options.enforce_activation_cap &&
      !incumbent_cap_feasible;

  Deadline deadline(options.time_limit_seconds);
  auto return_fallback = [&](const std::string &raw_status,
                             const std::string &reason,
                             double model_build_seconds, double solver_seconds,
                             bool cp_sat_launched, double best_bound,
                             double deterministic_time, Index branches,
                             Index conflicts) {
    BfsSplitOptimizationResult result;
    result.method = "fixed-order-cpsat";
    result.status = incumbent_cap_feasible
                        ? "FEASIBLE"
                        : (cp_sat_launched ? "UNKNOWN" : "NOT_RUN");
    result.solver_status_raw = raw_status;
    result.proven_optimal = false;
    if (incumbent_cap_feasible) {
      result.split = incumbent.split;
      result.makespan_ticks = incumbent.makespan;
    }
    result.best_bound_ticks =
        best_bound > 0.0 ? static_cast<Tick>(best_bound) : 0;
    result.solver_objective_ticks =
        incumbent_cap_feasible ? static_cast<double>(incumbent.makespan) : 0.0;
    if (incumbent_cap_feasible) {
      result.schedule = incumbent.schedule;
    }
    result.machine_orders = fixed_orders;
    result.fallback_available = incumbent_cap_feasible;
    result.fallback_used = incumbent_cap_feasible;
    result.fallback_reason = reason;
    result.fallback_source = incumbent_cap_feasible ? incumbent.source : "none";
    result.solution_source = incumbent_cap_feasible ? incumbent.source : "none";
    result.returned_solution_source = result.solution_source;
    result.cp_sat_launched = cp_sat_launched;
    result.cp_sat_models_solved = cp_sat_launched ? 1 : 0;
    result.deterministic_time = deterministic_time;
    result.branches = branches;
    result.conflicts = conflicts;
    if (incumbent_cap_feasible) {
      result.time_to_first_feasible_seconds = incumbent.runtime_seconds;
      result.time_to_best_incumbent_seconds = incumbent.runtime_seconds;
      result.first_feasible_objective = incumbent.makespan;
      result.incumbent_improvement_count = 1;
    }
    result.timing.incumbent_seconds = incumbent.runtime_seconds;
    result.timing.model_build_seconds = model_build_seconds;
    result.timing.solver_seconds = solver_seconds;
    result.diagnostic =
        incumbent_cap_feasible
            ? reason
            : reason + "; no cap-feasible incumbent is available";
    result.activation_cap_constraints = activation_metadata;
    result.search_stats_enabled = stats != nullptr;
    PopulateFixedOrderProvenance(result, options, "cpsat", "cpsat", estimate);
    if (incumbent_cap_feasible) {
      ValidateReturnedResultOrThrow(instance, result);
    }
    FinalizeBfsResultTiming(started, result);
    if (stats != nullptr) {
      if (cp_sat_launched) {
        stats->cp_sat_available = true;
        stats->cp_sat_status = raw_status;
        stats->cp_sat_objective = 0.0;
        stats->cp_sat_best_bound = result.best_bound_ticks;
        stats->cp_sat_branches = branches;
        stats->cp_sat_conflicts = conflicts;
        stats->cp_sat_wall_time_seconds = 0.0;
        stats->cp_sat_deterministic_time = deterministic_time;
      }
      result.search_stats = *stats;
    }
    MaybeRejectNonOptimal(options, result);
    return result;
  };

  if (deadline.expired()) {
    return return_fallback("NOT_RUN",
                           "global_deadline_expired_before_model_build", 0.0,
                           0.0, false, 0.0, 0.0, 0, 0);
  }

  const auto model_build_started = Clock::now();
  EmitLifecycle(options, started, "MODEL_BUILD_START", "", 0.0,
                "fixed_order_partition_cpsat");
  CpModelBuilder builder;

  const Tick conservative_horizon = ConservativeFixedOrderHorizon(instance);
  const Tick horizon = incumbent_cap_feasible && incumbent.makespan > 0
                           ? incumbent.makespan
                           : conservative_horizon;

  std::vector<IntVar> layers;
  layers.reserve(static_cast<std::size_t>(instance.stages));
  for (Index s = 0; s < instance.stages; ++s) {
    layers.push_back(
        builder.NewIntVar(Domain(instance.min_layers, instance.total_layers))
            .WithName("layers_" + std::to_string(s)));
  }
  builder.AddEquality(LinearExpr::Sum(layers), instance.total_layers);

  const Index op_count = instance.OperationCount();
  std::vector<IntVar> starts;
  std::vector<IntVar> durations;
  std::vector<IntVar> ends;
  starts.reserve(static_cast<std::size_t>(op_count));
  durations.reserve(static_cast<std::size_t>(op_count));
  ends.reserve(static_cast<std::size_t>(op_count));
  for (Index id = 0; id < op_count; ++id) {
    starts.push_back(builder.NewIntVar(Domain(0, horizon))
                         .WithName("start_" + std::to_string(id)));
    durations.push_back(builder.NewIntVar(Domain(0, horizon))
                            .WithName("duration_" + std::to_string(id)));
    ends.push_back(builder.NewIntVar(Domain(0, horizon))
                       .WithName("end_" + std::to_string(id)));
  }

  std::vector<std::optional<OperationId> > worker_predecessors(
      static_cast<std::size_t>(op_count));
  for (const std::vector<OperationId> &worker_order : fixed_orders) {
    for (std::size_t i = 1; i < worker_order.size(); ++i) {
      worker_predecessors[static_cast<std::size_t>(worker_order[i].value)] =
          worker_order[i - 1];
    }
  }

  for (Index id = 0; id < op_count; ++id) {
    const OperationId op{id};
    const OperationView view = DecodeOperation(instance, op);
    const Tick coefficient =
        instance.HasCostProfile()
            ? (view.backward
                   ? instance
                         .profile_backward_slope_ticks[static_cast<std::size_t>(
                             view.stage)]
                   : instance
                         .profile_forward_slope_ticks[static_cast<std::size_t>(
                             view.stage)])
            : (view.backward ? instance.backward_ratio_num
                             : instance.backward_ratio_den);
    const Tick bias =
        instance.HasCostProfile()
            ? (view.backward
                   ? instance
                         .profile_backward_bias_ticks[static_cast<std::size_t>(
                             view.stage)]
                   : instance
                         .profile_forward_bias_ticks[static_cast<std::size_t>(
                             view.stage)])
            : 0;
    builder.AddEquality(
        durations[static_cast<std::size_t>(id)],
        coefficient * layers[static_cast<std::size_t>(view.stage)] + bias);
    builder.AddEquality(ends[static_cast<std::size_t>(id)],
                        starts[static_cast<std::size_t>(id)] +
                            durations[static_cast<std::size_t>(id)]);

    std::vector<LinearExpr> start_predecessors;
    start_predecessors.reserve(4);
    start_predecessors.push_back(0);
    if (const std::optional<OperationId> data = DataPredecessor(instance, op)) {
      const OperationView pred_view = DecodeOperation(instance, *data);
      start_predecessors.push_back(
          ends[static_cast<std::size_t>(data->value)] +
          instance.EdgeDelay(pred_view.worker, view.worker));
    }
    if (const std::optional<OperationId> fifo = FifoPredecessor(instance, op)) {
      start_predecessors.push_back(ends[static_cast<std::size_t>(fifo->value)]);
    }
    if (const std::optional<OperationId> worker =
            worker_predecessors[static_cast<std::size_t>(id)]) {
      start_predecessors.push_back(
          ends[static_cast<std::size_t>(worker->value)]);
    }
    builder.AddMaxEquality(starts[static_cast<std::size_t>(id)],
                           start_predecessors);
  }

  IntVar makespan = builder.NewIntVar(Domain(0, horizon)).WithName("makespan");
  for (const IntVar &end : ends) builder.AddGreaterOrEqual(makespan, end);
  if (incumbent_cap_feasible) {
    builder.AddLessOrEqual(makespan, incumbent.makespan);
  }

  if (!instance.HasCostProfile()) {
    const Tick ratio_sum =
        CheckedAdd(instance.backward_ratio_num, instance.backward_ratio_den,
                   "fixed-order critical-chain lower bound");
    builder.AddGreaterOrEqual(
        makespan, CheckedMul(instance.total_layers, ratio_sum,
                             "fixed-order critical-chain lower bound"));
  }
  builder.Minimize(makespan);

  activation_metadata = AddActivationCapacityConstraints(
      builder, instance, options.activation_options, starts, ends, &layers,
      nullptr, horizon, true, options.activation_cap_model_debug_dump);
  activation_metadata.incumbent_rejected_for_activation_cap =
      options.activation_options.enforce_activation_cap &&
      !incumbent_cap_feasible;
  if (options.activation_options.enforce_activation_cap &&
      !activation_metadata.constraints_added) {
    BfsSplitOptimizationResult unavailable = MakeUnavailableCpSatResult(
        instance, fixed_orders, options, "fixed-order-cpsat", "cpsat",
        activation_metadata.unsupported_reason.empty()
            ? "activation cap constraints were not added"
            : activation_metadata.unsupported_reason,
        estimate);
    unavailable.activation_cap_constraints = activation_metadata;
    return unavailable;
  }

  if (incumbent_cap_feasible) {
    for (Index s = 0; s < instance.stages; ++s) {
      builder.AddHint(layers[static_cast<std::size_t>(s)],
                      incumbent.split[static_cast<std::size_t>(s)]);
    }
    for (const ScheduledOperation &operation :
         incumbent.schedule.operations_by_id) {
      const std::size_t id = static_cast<std::size_t>(operation.id.value);
      builder.AddHint(starts[id], operation.start);
      builder.AddHint(durations[id], operation.duration);
      builder.AddHint(ends[id], operation.end);
    }
    builder.AddHint(makespan, incumbent.makespan);
  }

  const auto cp_model = builder.Build();
  const double model_build_seconds = Since(model_build_started);
  EmitLifecycle(options, started, "MODEL_BUILD_END", "", 0.0,
                "fixed_order_partition_cpsat");

  const double solve_limit =
      options.time_limit_seconds > 0.0 ? deadline.clamp_solver_limit(0.0) : 0.0;
  if (options.time_limit_seconds > 0.0 && solve_limit <= 0.0) {
    return return_fallback("NOT_RUN", "global_deadline_expired_before_cp_sat",
                           model_build_seconds, 0.0, false, 0.0, 0.0, 0, 0);
  }

  Model model;
  SatParameters parameters;
  parameters.set_num_workers(options.num_workers);
  parameters.set_random_seed(options.random_seed);
  parameters.set_log_search_progress(options.log_search_progress);
  if (solve_limit > 0.0) parameters.set_max_time_in_seconds(solve_limit);
  model.Add(NewSatParameters(parameters));

  const auto solve_started = Clock::now();
  std::mutex incumbent_mutex;
  double time_to_first_feasible_seconds =
      incumbent_cap_feasible ? incumbent.runtime_seconds : 0.0;
  double time_to_best_incumbent_seconds =
      incumbent_cap_feasible ? incumbent.runtime_seconds : 0.0;
  Tick first_feasible_objective =
      incumbent_cap_feasible ? incumbent.makespan : 0;
  Tick best_incumbent_objective = incumbent_cap_feasible
                                      ? incumbent.makespan
                                      : std::numeric_limits<Tick>::max();
  Index incumbent_improvement_count = incumbent_cap_feasible ? 1 : 0;
  std::vector<std::pair<double, Tick> > incumbent_trace;
  if (incumbent_cap_feasible) {
    incumbent_trace.push_back({incumbent.runtime_seconds, incumbent.makespan});
  }
  model.Add(NewFeasibleSolutionObserver(
      [&](const CpSolverResponse &callback_response) {
        const Tick objective =
            static_cast<Tick>(callback_response.objective_value());
        std::lock_guard<std::mutex> lock(incumbent_mutex);
        if (incumbent_trace.empty()) {
          time_to_first_feasible_seconds =
              std::chrono::duration<double>(Clock::now() - started).count();
          time_to_best_incumbent_seconds = time_to_first_feasible_seconds;
          first_feasible_objective = objective;
          best_incumbent_objective = objective;
          incumbent_improvement_count = 1;
          incumbent_trace.push_back(
              {time_to_first_feasible_seconds, objective});
          return;
        }
        if (objective < best_incumbent_objective) {
          best_incumbent_objective = objective;
          time_to_best_incumbent_seconds =
              std::chrono::duration<double>(Clock::now() - started).count();
          ++incumbent_improvement_count;
          if (incumbent_trace.size() < 64) {
            incumbent_trace.push_back(
                {time_to_best_incumbent_seconds, objective});
          }
        }
      }));

  EmitLifecycle(options, started, "SOLVE_START", "",
                parameters.max_time_in_seconds(),
                "fixed_order_partition_cpsat");
  const CpSolverResponse response = SolveCpModel(cp_model, &model);
  const double solver_seconds = Since(solve_started);
  const std::string raw_status = StatusName(response.status());
  EmitLifecycle(options, started, "SOLVE_END", raw_status,
                parameters.max_time_in_seconds(),
                "fixed_order_partition_cpsat");

  if (response.status() == operations_research::sat::UNKNOWN) {
    return return_fallback(
        raw_status, "cp_sat_returned_unknown_with_validated_incumbent",
        model_build_seconds, solver_seconds, true,
        response.best_objective_bound(), response.deterministic_time(),
        response.num_branches(), response.num_conflicts());
  }

  BfsSplitOptimizationResult result;
  result.method = "fixed-order-cpsat";
  result.status = raw_status;
  result.solver_status_raw = raw_status;
  result.best_bound_ticks =
      response.best_objective_bound() > 0.0
          ? static_cast<Tick>(response.best_objective_bound())
          : 0;
  result.solver_objective_ticks = response.objective_value();
  result.timing.incumbent_seconds = incumbent.runtime_seconds;
  result.timing.model_build_seconds = model_build_seconds;
  result.timing.solver_seconds = solver_seconds;
  result.timing.ortools_wall_time_seconds = response.wall_time();
  result.branches = response.num_branches();
  result.conflicts = response.num_conflicts();
  result.deterministic_time = response.deterministic_time();
  result.proven_optimal =
      response.status() == operations_research::sat::OPTIMAL;
  result.machine_orders = fixed_orders;
  result.fallback_available = incumbent_cap_feasible;
  result.fallback_source = incumbent_cap_feasible ? incumbent.source : "none";
  result.cp_sat_launched = true;
  result.cp_sat_models_solved = 1;
  result.search_stats_enabled = stats != nullptr;
  result.activation_cap_constraints = activation_metadata;
  PopulateFixedOrderProvenance(result, options, "cpsat", "cpsat", estimate);

  if (response.status() == operations_research::sat::OPTIMAL ||
      response.status() == operations_research::sat::FEASIBLE) {
    const auto extraction_started = Clock::now();
    EmitLifecycle(options, started, "SOLUTION_EXTRACTION_START", raw_status,
                  parameters.max_time_in_seconds(),
                  "fixed_order_partition_cpsat");
    const Tick raw_makespan = SolutionIntegerValue(response, makespan);
    result.split.reserve(static_cast<std::size_t>(instance.stages));
    for (const IntVar &layer : layers) {
      result.split.push_back(SolutionIntegerValue(response, layer));
    }
    result.makespan_ticks = raw_makespan;
    result.solution_source = "fixed_order_partition_cpsat_solution";
    result.returned_solution_source = result.solution_source;
    result.timing.extraction_seconds = Since(extraction_started);
    EmitLifecycle(options, started, "SOLUTION_EXTRACTION_END", raw_status,
                  parameters.max_time_in_seconds(),
                  "fixed_order_partition_cpsat");
    if (stats != nullptr) ++stats->candidate_schedules_extracted;

    const auto canonical_started = Clock::now();
    EvaluationResult evaluated =
        EvaluateSchedule(instance, result.split, fixed_orders);
    if (!evaluated.schedule.ok()) {
      if (stats != nullptr) ++stats->candidate_schedules_rejected;
      throw Error("deterministic evaluator rejected CP-SAT fixed-order split");
    }
    if (stats != nullptr) {
      ++stats->candidate_schedules_deterministically_evaluated;
      ++stats->candidate_schedules_accepted;
    }
    if (evaluated.schedule.makespan > raw_makespan) {
      throw Error(
          "fixed-order CP-SAT objective is smaller than deterministic replay");
    }
    if (evaluated.schedule.makespan < raw_makespan) {
      if (result.proven_optimal) {
        throw Error(
            "fixed-order CP-SAT optimal objective exceeds "
            "deterministic replay");
      }
      result.diagnostic =
          "fixed-order CP-SAT feasible solution replayed to a smaller "
          "deterministic makespan";
      result.makespan_ticks = evaluated.schedule.makespan;
      result.solver_objective_ticks = static_cast<double>(raw_makespan);
    }
    result.schedule = std::move(evaluated.schedule);
    result.timing.canonicalization_seconds = Since(canonical_started);
    if (time_to_first_feasible_seconds <= 0.0) {
      time_to_first_feasible_seconds = Since(started);
    }
    if (first_feasible_objective <= 0) {
      first_feasible_objective = result.makespan_ticks;
    }
    if (time_to_best_incumbent_seconds <= 0.0) {
      time_to_best_incumbent_seconds = time_to_first_feasible_seconds;
    }
    result.time_to_first_feasible_seconds = time_to_first_feasible_seconds;
    result.time_to_best_incumbent_seconds = time_to_best_incumbent_seconds;
    result.first_feasible_objective = first_feasible_objective;
    result.incumbent_improvement_count = incumbent_improvement_count;
    result.incumbent_trace = std::move(incumbent_trace);
    ValidateReturnedResultOrThrow(instance, result);
  } else {
    result.solution_source = "none";
    result.returned_solution_source = "none";
  }

  if (stats != nullptr) {
    stats->cp_sat_available = true;
    stats->cp_sat_status = result.solver_status_raw;
    stats->cp_sat_objective = result.solver_objective_ticks;
    stats->cp_sat_best_bound = result.best_bound_ticks;
    stats->cp_sat_branches = result.branches;
    stats->cp_sat_conflicts = result.conflicts;
    stats->cp_sat_wall_time_seconds = result.timing.ortools_wall_time_seconds;
    stats->cp_sat_deterministic_time = result.deterministic_time;
    result.search_stats = *stats;
  }
  FinalizeBfsResultTiming(started, result);
  MaybeRejectNonOptimal(options, result);
  return result;
}

bool IsCpSatFixedOrderPartitionOptimizerAvailable() { return true; }
#endif

BfsSplitOptimizationResult OptimizePartitionForFixedOrderAuto(
    const Instance &instance, const MachineOrders &fixed_orders,
    const BfsSplitOptimizerOptions &options) {
  const PartitionCountEstimate estimate =
      EstimatePartitions(instance, options.enumeration_threshold);
  if (IsCpSatFixedOrderPartitionOptimizerAvailable()) {
    BfsSplitOptimizationResult result =
        OptimizePartitionForFixedOrderCpSat(instance, fixed_orders, options);
    result.method = "fixed-order-auto-cpsat";
    result.fixed_order_partition_backend_requested = "auto";
    return result;
  }
  if (options.activation_options.enforce_activation_cap) {
    return MakeUnavailableCpSatResult(
        instance, fixed_orders, options, "fixed-order-auto-unavailable", "auto",
        ActivationCapSolverUnsupportedReason(options.activation_options, true),
        estimate);
  }
  if (estimate.exact && estimate.count <= options.enumeration_threshold) {
    BfsSplitOptimizationResult result = OptimizePartitionForFixedOrderEnumerate(
        instance, fixed_orders, options);
    result.method = "fixed-order-auto-enumerate";
    result.fixed_order_partition_backend_requested = "auto";
    return result;
  }
  return MakeUnavailableCpSatResult(
      instance, fixed_orders, options, "fixed-order-auto-unavailable", "auto",
      "fixed-order auto backend cannot safely enumerate this partition space "
      "and CP-SAT is unavailable in this build",
      estimate);
}

BfsSplitOptimizationResult OptimizePartitionForFixedOrder(
    const Instance &instance, const MachineOrders &fixed_orders,
    const BfsSplitOptimizerOptions &options) {
  const std::string backend =
      NormalizeBackend(options.fixed_order_partition_backend);
  if (backend == "enumerate") {
    BfsSplitOptimizationResult result = OptimizePartitionForFixedOrderEnumerate(
        instance, fixed_orders, options);
    result.fixed_order_partition_backend_requested = "enumerate";
    return result;
  }
  if (backend == "cpsat") {
    BfsSplitOptimizationResult result =
        OptimizePartitionForFixedOrderCpSat(instance, fixed_orders, options);
    result.fixed_order_partition_backend_requested = "cpsat";
    return result;
  }
  return OptimizePartitionForFixedOrderAuto(instance, fixed_orders, options);
}

}  // namespace slackpipe
