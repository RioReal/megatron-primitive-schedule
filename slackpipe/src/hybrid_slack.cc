#include "slackpipe/hybrid_slack.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <set>
#include <tuple>

#include "slackpipe/breadth_first.h"
#include "slackpipe/dag_evaluator.h"
#include "slackpipe/slackpipe_solver.h"

namespace slackpipe {

namespace {

using Clock = std::chrono::steady_clock;

constexpr double kSlackEpsilon = 1e-9;

double Since(Clock::time_point started) {
  return std::chrono::duration<double>(Clock::now() - started).count();
}

struct HybridDeadline {
  Clock::time_point started;
  Clock::time_point deadline;
  bool finite = false;

  [[nodiscard]] bool Reached() const {
    return finite && Clock::now() >= deadline;
  }
};

[[nodiscard]] HybridDeadline MakeDeadline(double time_limit_seconds) {
  const Clock::time_point started = Clock::now();
  HybridDeadline deadline;
  deadline.started = started;
  deadline.deadline = started;
  if (time_limit_seconds > 0.0) {
    deadline.finite = true;
    deadline.deadline =
        started + std::chrono::duration_cast<Clock::duration>(
                      std::chrono::duration<double>(time_limit_seconds));
  }
  return deadline;
}

[[nodiscard]] bool OrdersLess(const MachineOrders& a, const MachineOrders& b) {
  return std::lexicographical_compare(
      a.begin(), a.end(), b.begin(), b.end(),
      [](const std::vector<OperationId>& lhs,
         const std::vector<OperationId>& rhs) {
        return std::lexicographical_compare(
            lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
            [](OperationId x, OperationId y) { return x.value < y.value; });
      });
}

struct CandidateState {
  std::vector<Tick> split;
  MachineOrders orders;
  ScheduleSolution schedule;
  HybridSlackScores scores;
  Tick objective = 0;
  std::string source;
};

[[nodiscard]] HybridSlackCandidateSummary SummaryOf(
    const CandidateState& candidate) {
  return HybridSlackCandidateSummary{
      candidate.objective, candidate.scores.partition_min_slack,
      candidate.scores.stage_scores, candidate.split, candidate.orders};
}

[[nodiscard]] bool CandidateBetter(const CandidateState& candidate,
                                   const CandidateState& incumbent) {
  return HybridSlackCandidateBetter(SummaryOf(candidate), SummaryOf(incumbent));
}

[[nodiscard]] bool CandidateBeatsBaseline(const CandidateState& candidate,
                                          const CandidateState& baseline) {
  return candidate.objective < baseline.objective;
}

[[nodiscard]] CandidateState EvaluateCandidate(const Instance& instance,
                                               const std::vector<Tick>& split,
                                               const MachineOrders& orders,
                                               const std::string& source) {
  EvaluationResult evaluated = EvaluateSchedule(instance, split, orders);
  CandidateState candidate;
  candidate.split = split;
  candidate.orders = orders;
  candidate.schedule = std::move(evaluated.schedule);
  if (!candidate.schedule.ok()) return candidate;
  candidate.objective = candidate.schedule.makespan;
  candidate.scores = ComputeHybridSlackScores(instance, split);
  candidate.source = source;
  return candidate;
}

[[nodiscard]] std::vector<Tick> InitialSplit(
    const Instance& instance,
    const std::optional<PartitionRestriction>& restriction) {
  const std::vector<Tick> uniform = UniformSplit(instance);
  if (!restriction ||
      SplitSatisfiesPartitionRestriction(instance, uniform, *restriction)) {
    return uniform;
  }
  return restriction->reference_split;
}

void PopulateHybridFields(
    BfsSplitOptimizationResult& result, const CandidateState& incumbent,
    const CandidateState& baseline, const HybridSlackIncumbentOptions& options,
    double elapsed_seconds, Index iterations, Index candidates_generated,
    Index candidates_simulated, Index partition_moves_accepted,
    Index interleaving_moves_accepted, bool deadline_reached,
    const std::string& termination_reason) {
  result.method = "hybrid-slack";
  result.status = "FEASIBLE";
  result.proven_optimal = false;
  result.split = incumbent.split;
  result.makespan_ticks = incumbent.objective;
  result.best_bound_ticks = 0;
  result.solver_objective_ticks = incumbent.objective;
  result.wall_time_seconds = elapsed_seconds;
  result.timing.incumbent_seconds = elapsed_seconds;
  result.timing.total_seconds = elapsed_seconds;
  result.schedule = incumbent.schedule;
  result.machine_orders = incumbent.orders;
  result.hybrid_min_slack = incumbent.scores.partition_min_slack;
  result.hybrid_stage_scores = incumbent.scores.stage_scores;
  result.hybrid_bottleneck_stages = incumbent.scores.bottleneck_stages;
  result.baseline_primary_objective = baseline.objective;
  result.baseline_hybrid_min_slack = baseline.scores.partition_min_slack;
  result.improved_over_baseline = CandidateBeatsBaseline(incumbent, baseline);
  result.hint_budget_seconds = options.time_limit_seconds;
  result.hint_elapsed_seconds = elapsed_seconds;
  result.hint_iterations = iterations;
  result.hint_candidates_generated = candidates_generated;
  result.hint_candidates_simulated = candidates_simulated;
  result.hint_partition_moves_accepted = partition_moves_accepted;
  result.hint_interleaving_moves_accepted = interleaving_moves_accepted;
  result.hint_deadline_reached = deadline_reached;
  result.hint_termination_reason = termination_reason;
  result.diagnostic =
      "hybrid-slack best feasible incumbent found from baseline_objective=" +
      std::to_string(baseline.objective) + " baseline_hybrid_min_slack=" +
      std::to_string(baseline.scores.partition_min_slack) +
      " final_objective=" + std::to_string(incumbent.objective) +
      " final_hybrid_min_slack=" +
      std::to_string(incumbent.scores.partition_min_slack);
}

struct PartitionMove {
  std::vector<Tick> split;
  double proxy_min_slack = 0.0;
};

struct InterleavingMove {
  Index worker = 0;
  Index index = 0;
  double priority = 0.0;
};

[[nodiscard]] bool StageInSet(Index stage, const std::vector<Index>& stages) {
  return std::find(stages.begin(), stages.end(), stage) != stages.end();
}

}  // namespace

HybridSlackScores ComputeHybridSlackScores(const Instance& instance,
                                           const std::vector<Tick>& split) {
  instance.Validate();
  ValidateSplit(instance, split);
  const Index length = OperationPositionCount(instance);
  std::vector<Tick> costs(static_cast<std::size_t>(length), 0);
  std::vector<Tick> prefix(static_cast<std::size_t>(length + 1), 0);
  for (Index n = 0; n < length; ++n) {
    const Index stage = StageForOperationPosition(instance, n);
    const bool backward = n >= instance.stages;
    costs[static_cast<std::size_t>(n)] =
        instance.Duration(stage, backward, split);
    prefix[static_cast<std::size_t>(n + 1)] = CheckedAdd(
        prefix[static_cast<std::size_t>(n)], costs[static_cast<std::size_t>(n)],
        "hybrid slack prefix sum");
  }

  HybridSlackScores scores;
  scores.operation_scores.assign(static_cast<std::size_t>(length), 0.0);
  for (Index n = 0; n < length; ++n) {
    const Index worker = WorkerForOperationPosition(instance, n);
    std::optional<Index> previous;
    std::optional<Index> next;
    for (Index j = n - 1; j >= 0; --j) {
      if (WorkerForOperationPosition(instance, j) == worker) {
        previous = j;
        break;
      }
    }
    for (Index j = n + 1; j < length; ++j) {
      if (WorkerForOperationPosition(instance, j) == worker) {
        next = j;
        break;
      }
    }
    const Index left = previous ? *previous + 1 : n;
    const Index right = next ? *next - 1 : n;
    const Tick interval_cost = prefix[static_cast<std::size_t>(right + 1)] -
                               prefix[static_cast<std::size_t>(left)];
    scores.operation_scores[static_cast<std::size_t>(n)] =
        static_cast<double>(interval_cost) /
        static_cast<double>(costs[static_cast<std::size_t>(n)]);
  }

  scores.forward_stage_scores.assign(static_cast<std::size_t>(instance.stages),
                                     0.0);
  scores.backward_stage_scores.assign(static_cast<std::size_t>(instance.stages),
                                      0.0);
  scores.stage_scores.assign(static_cast<std::size_t>(instance.stages), 0.0);
  scores.partition_min_slack = std::numeric_limits<double>::infinity();
  for (Index s = 0; s < instance.stages; ++s) {
    const double forward = scores.operation_scores[static_cast<std::size_t>(s)];
    const double backward =
        scores.operation_scores[static_cast<std::size_t>(length - 1 - s)];
    const double stage = std::min(forward, backward);
    scores.forward_stage_scores[static_cast<std::size_t>(s)] = forward;
    scores.backward_stage_scores[static_cast<std::size_t>(s)] = backward;
    scores.stage_scores[static_cast<std::size_t>(s)] = stage;
    scores.partition_min_slack = std::min(scores.partition_min_slack, stage);
  }

  double max_stage_slack = -std::numeric_limits<double>::infinity();
  for (double value : scores.stage_scores) {
    max_stage_slack = std::max(max_stage_slack, value);
  }
  for (Index s = 0; s < instance.stages; ++s) {
    const double value = scores.stage_scores[static_cast<std::size_t>(s)];
    if (std::abs(value - scores.partition_min_slack) <= kSlackEpsilon) {
      scores.bottleneck_stages.push_back(s);
    }
    if (std::abs(value - max_stage_slack) <= kSlackEpsilon) {
      scores.slack_rich_stages.push_back(s);
    }
  }
  return scores;
}

bool HybridSlackCandidateBetter(const HybridSlackCandidateSummary& candidate,
                                const HybridSlackCandidateSummary& incumbent) {
  if (candidate.primary_objective != incumbent.primary_objective) {
    return candidate.primary_objective < incumbent.primary_objective;
  }
  if (candidate.hybrid_min_slack > incumbent.hybrid_min_slack + kSlackEpsilon) {
    return true;
  }
  if (candidate.hybrid_min_slack + kSlackEpsilon < incumbent.hybrid_min_slack) {
    return false;
  }
  std::vector<double> candidate_stage_scores = candidate.hybrid_stage_scores;
  std::vector<double> incumbent_stage_scores = incumbent.hybrid_stage_scores;
  std::sort(candidate_stage_scores.begin(), candidate_stage_scores.end());
  std::sort(incumbent_stage_scores.begin(), incumbent_stage_scores.end());
  for (std::size_t i = 0;
       i < candidate_stage_scores.size() && i < incumbent_stage_scores.size();
       ++i) {
    if (candidate_stage_scores[i] > incumbent_stage_scores[i] + kSlackEpsilon) {
      return true;
    }
    if (candidate_stage_scores[i] + kSlackEpsilon < incumbent_stage_scores[i]) {
      return false;
    }
  }
  if (candidate.split != incumbent.split) {
    return candidate.split < incumbent.split;
  }
  return OrdersLess(candidate.orders, incumbent.orders);
}

std::vector<std::vector<Tick>>
GenerateHybridSlackPartitionMoveCandidatesInternal(
    const Instance& instance, const std::vector<Tick>& split,
    const HybridSlackScores& scores,
    const std::optional<PartitionRestriction>& restriction, Index limit,
    const HybridDeadline* deadline, bool* deadline_reached) {
  instance.Validate();
  ValidateSplit(instance, split);
  if (limit <= 0) return {};

  std::vector<Index> low;
  std::vector<Index> high;
  low.reserve(static_cast<std::size_t>(instance.stages));
  high.reserve(static_cast<std::size_t>(instance.stages));
  for (Index s = 0; s < instance.stages; ++s) {
    low.push_back(s);
    high.push_back(s);
  }
  std::sort(low.begin(), low.end(), [&](Index a, Index b) {
    const double sa = scores.stage_scores[static_cast<std::size_t>(a)];
    const double sb = scores.stage_scores[static_cast<std::size_t>(b)];
    if (sa != sb) return sa < sb;
    return a < b;
  });
  std::sort(high.begin(), high.end(), [&](Index a, Index b) {
    const double sa = scores.stage_scores[static_cast<std::size_t>(a)];
    const double sb = scores.stage_scores[static_cast<std::size_t>(b)];
    if (sa != sb) return sa > sb;
    return a < b;
  });

  std::vector<PartitionMove> moves;
  std::set<std::vector<Tick>> seen;
  for (Index donor : low) {
    if (deadline != nullptr && deadline->Reached()) {
      if (deadline_reached != nullptr) *deadline_reached = true;
      break;
    }
    if (split[static_cast<std::size_t>(donor)] <= instance.min_layers) {
      continue;
    }
    for (Index receiver : high) {
      if (deadline != nullptr && deadline->Reached()) {
        if (deadline_reached != nullptr) *deadline_reached = true;
        break;
      }
      if (donor == receiver) continue;
      std::vector<Tick> candidate = split;
      --candidate[static_cast<std::size_t>(donor)];
      ++candidate[static_cast<std::size_t>(receiver)];
      if (!seen.insert(candidate).second) continue;
      try {
        ValidateSplit(instance, candidate);
        if (restriction && !SplitSatisfiesPartitionRestriction(
                               instance, candidate, *restriction)) {
          continue;
        }
        const HybridSlackScores candidate_scores =
            ComputeHybridSlackScores(instance, candidate);
        moves.push_back(
            PartitionMove{candidate, candidate_scores.partition_min_slack});
      } catch (const Error&) {
        continue;
      }
    }
    if (deadline_reached != nullptr && *deadline_reached) break;
  }

  std::sort(moves.begin(), moves.end(),
            [](const PartitionMove& a, const PartitionMove& b) {
              if (a.proxy_min_slack != b.proxy_min_slack) {
                return a.proxy_min_slack > b.proxy_min_slack;
              }
              return a.split < b.split;
            });
  if (static_cast<Index>(moves.size()) > limit) {
    moves.resize(static_cast<std::size_t>(limit));
  }
  std::vector<std::vector<Tick>> splits;
  splits.reserve(moves.size());
  for (const PartitionMove& move : moves) splits.push_back(move.split);
  return splits;
}

std::vector<std::vector<Tick>> GenerateHybridSlackPartitionMoveCandidates(
    const Instance& instance, const std::vector<Tick>& split,
    const HybridSlackScores& scores,
    const std::optional<PartitionRestriction>& restriction, Index limit) {
  return GenerateHybridSlackPartitionMoveCandidatesInternal(
      instance, split, scores, restriction, limit, nullptr, nullptr);
}

std::vector<MachineOrders>
GenerateHybridSlackInterleavingMoveCandidatesInternal(
    const Instance& instance, const std::vector<Tick>& split,
    const MachineOrders& orders, const HybridSlackScores& scores, Index limit,
    const HybridDeadline* deadline, bool* deadline_reached) {
  instance.Validate();
  ValidateSplit(instance, split);
  if (limit <= 0) return {};

  std::vector<InterleavingMove> moves;
  for (Index w = 0; w < static_cast<Index>(orders.size()); ++w) {
    if (deadline != nullptr && deadline->Reached()) {
      if (deadline_reached != nullptr) *deadline_reached = true;
      break;
    }
    const std::vector<OperationId>& order = orders[static_cast<std::size_t>(w)];
    for (Index i = 1; i < static_cast<Index>(order.size()); ++i) {
      if (deadline != nullptr && deadline->Reached()) {
        if (deadline_reached != nullptr) *deadline_reached = true;
        break;
      }
      const OperationView prev =
          DecodeOperation(instance, order[static_cast<std::size_t>(i - 1)]);
      const OperationView next =
          DecodeOperation(instance, order[static_cast<std::size_t>(i)]);
      if (!StageInSet(prev.stage, scores.bottleneck_stages) &&
          !StageInSet(next.stage, scores.bottleneck_stages)) {
        continue;
      }
      const double priority =
          std::min(scores.stage_scores[static_cast<std::size_t>(prev.stage)],
                   scores.stage_scores[static_cast<std::size_t>(next.stage)]);
      moves.push_back(InterleavingMove{w, i, priority});
    }
    if (deadline_reached != nullptr && *deadline_reached) break;
  }
  std::sort(moves.begin(), moves.end(),
            [](const InterleavingMove& a, const InterleavingMove& b) {
              if (a.priority != b.priority) return a.priority < b.priority;
              if (a.worker != b.worker) return a.worker < b.worker;
              return a.index < b.index;
            });
  if (static_cast<Index>(moves.size()) > limit) {
    moves.resize(static_cast<std::size_t>(limit));
  }

  std::vector<MachineOrders> candidates;
  candidates.reserve(moves.size());
  std::set<std::vector<std::vector<Index>>> seen_keys;
  for (const InterleavingMove& move : moves) {
    if (deadline != nullptr && deadline->Reached()) {
      if (deadline_reached != nullptr) *deadline_reached = true;
      break;
    }
    MachineOrders candidate = orders;
    std::swap(candidate[static_cast<std::size_t>(move.worker)]
                       [static_cast<std::size_t>(move.index - 1)],
              candidate[static_cast<std::size_t>(move.worker)]
                       [static_cast<std::size_t>(move.index)]);
    std::vector<std::vector<Index>> key(candidate.size());
    for (std::size_t w = 0; w < candidate.size(); ++w) {
      for (OperationId id : candidate[w]) key[w].push_back(id.value);
    }
    if (seen_keys.insert(std::move(key)).second) {
      candidates.push_back(std::move(candidate));
    }
  }
  return candidates;
}

std::vector<MachineOrders> GenerateHybridSlackInterleavingMoveCandidates(
    const Instance& instance, const std::vector<Tick>& split,
    const MachineOrders& orders, const HybridSlackScores& scores, Index limit) {
  return GenerateHybridSlackInterleavingMoveCandidatesInternal(
      instance, split, orders, scores, limit, nullptr, nullptr);
}

BfsSplitOptimizationResult BuildHybridSlackIncumbent(
    const Instance& instance, const HybridSlackIncumbentOptions& options) {
  const HybridDeadline deadline = MakeDeadline(options.time_limit_seconds);
  instance.Validate();
  if (options.partition_restriction) {
    ValidatePartitionRestriction(instance, *options.partition_restriction);
  }

  const std::vector<Tick> baseline_split =
      InitialSplit(instance, options.partition_restriction);
  const MachineOrders baseline_orders = BreadthFirstOrders(instance);
  CandidateState baseline =
      EvaluateCandidate(instance, baseline_split, baseline_orders, "baseline");
  if (!baseline.schedule.ok()) {
    throw Error("hybrid-slack baseline incumbent is not feasible");
  }

  CandidateState current = baseline;
  CandidateState best = baseline;
  Index iterations = 0;
  Index candidates_generated = 0;
  Index candidates_simulated = 0;
  Index partition_moves_accepted = 0;
  Index interleaving_moves_accepted = 0;
  bool deadline_reached = false;
  std::string termination_reason = "iteration_limit";
  const Index max_iterations = std::max<Index>(0, options.max_iterations);
  const Index partition_limit =
      std::max<Index>(0, options.partition_candidate_limit);
  const Index interleaving_limit =
      std::max<Index>(0, options.interleaving_candidate_limit);

  for (; iterations < max_iterations; ++iterations) {
    if (deadline.Reached()) {
      deadline_reached = true;
      termination_reason = "deadline";
      break;
    }

    CandidateState best_iteration = current;
    bool accepted_partition = false;
    bool accepted_interleaving = false;

    const std::vector<std::vector<Tick>> partition_candidates =
        GenerateHybridSlackPartitionMoveCandidatesInternal(
            instance, current.split, current.scores,
            options.partition_restriction, partition_limit, &deadline,
            &deadline_reached);
    candidates_generated += static_cast<Index>(partition_candidates.size());
    if (deadline_reached) {
      termination_reason = "deadline";
      break;
    }
    for (const std::vector<Tick>& candidate_split : partition_candidates) {
      if (deadline.Reached()) {
        deadline_reached = true;
        termination_reason = "deadline";
        break;
      }
      ++candidates_simulated;
      CandidateState candidate = EvaluateCandidate(instance, candidate_split,
                                                   current.orders, "partition");
      if (!candidate.schedule.ok()) continue;
      if (CandidateBetter(candidate, best)) best = candidate;
      if (CandidateBetter(candidate, best_iteration)) {
        best_iteration = candidate;
        accepted_partition = true;
        accepted_interleaving = false;
      }
    }
    if (deadline_reached) break;

    const std::vector<MachineOrders> interleaving_candidates =
        GenerateHybridSlackInterleavingMoveCandidatesInternal(
            instance, current.split, current.orders, current.scores,
            interleaving_limit, &deadline, &deadline_reached);
    candidates_generated += static_cast<Index>(interleaving_candidates.size());
    if (deadline_reached) {
      termination_reason = "deadline";
      break;
    }
    for (const MachineOrders& candidate_orders : interleaving_candidates) {
      if (deadline.Reached()) {
        deadline_reached = true;
        termination_reason = "deadline";
        break;
      }
      ++candidates_simulated;
      CandidateState candidate = EvaluateCandidate(
          instance, current.split, candidate_orders, "interleaving");
      if (!candidate.schedule.ok()) continue;
      if (CandidateBetter(candidate, best)) best = candidate;
      if (CandidateBetter(candidate, best_iteration)) {
        best_iteration = candidate;
        accepted_partition = false;
        accepted_interleaving = true;
      }
    }
    if (deadline_reached) break;

    if (!CandidateBetter(best_iteration, current)) {
      termination_reason = "local_convergence";
      ++iterations;
      break;
    }
    current = best_iteration;
    if (accepted_partition) ++partition_moves_accepted;
    if (accepted_interleaving) ++interleaving_moves_accepted;
  }
  if (!deadline_reached && iterations >= max_iterations &&
      termination_reason != "local_convergence") {
    termination_reason = "iteration_limit";
  }

  BfsSplitOptimizationResult result;
  PopulateHybridFields(result, best, baseline, options, Since(deadline.started),
                       iterations, candidates_generated, candidates_simulated,
                       partition_moves_accepted, interleaving_moves_accepted,
                       deadline_reached, termination_reason);
  result.machine_orders = best.orders;
  result.schedule.orders = best.orders;
  result.checked_splits = candidates_simulated;
  if (options.partition_restriction &&
      !SplitSatisfiesPartitionRestriction(instance, result.split,
                                          *options.partition_restriction)) {
    throw Error("hybrid-slack incumbent violates active partition restriction");
  }
  EvaluationResult checked =
      EvaluateSchedule(instance, result.split, result.machine_orders);
  if (!checked.schedule.ok() ||
      checked.schedule.makespan != result.makespan_ticks) {
    throw Error("hybrid-slack incumbent did not replay deterministically");
  }
  result.schedule = std::move(checked.schedule);
  return result;
}

}  // namespace slackpipe
