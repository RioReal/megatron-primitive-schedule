#include "slackpipe/slackpipe_solver.h"

#include <algorithm>
#include <cstdlib>
#include <sstream>

#include "slackpipe/breadth_first.h"

namespace slackpipe {

namespace {

[[nodiscard]] Tick CeilDiv(Tick numerator, Tick denominator) {
  if (denominator <= 0)
    throw Error("non-positive denominator in ceil division");
  return numerator / denominator + (numerator % denominator == 0 ? 0 : 1);
}

}  // namespace

std::string ToString(SlackPipeSplitMode mode) {
  switch (mode) {
    case SlackPipeSplitMode::kFixed:
      return "fixed";
    case SlackPipeSplitMode::kLocal:
      return "local";
    case SlackPipeSplitMode::kGlobal:
      return "global";
    case SlackPipeSplitMode::kWorkerFixed:
      return "worker-fixed";
    case SlackPipeSplitMode::kWorkerLocal:
      return "worker-local";
  }
  return "unknown";
}

SlackPipeSplitMode ParseSlackPipeSplitMode(const std::string& text) {
  if (text == "fixed") return SlackPipeSplitMode::kFixed;
  if (text == "local") return SlackPipeSplitMode::kLocal;
  if (text == "global") return SlackPipeSplitMode::kGlobal;
  if (text == "worker-fixed") return SlackPipeSplitMode::kWorkerFixed;
  if (text == "worker-local") return SlackPipeSplitMode::kWorkerLocal;
  throw Error("unknown SlackPipe split mode: " + text);
}

Tick AnalyticalGlobalLowerBound(const Instance& instance) {
  instance.Validate();
  const Tick ratio_sum =
      CheckedAdd(instance.backward_ratio_num, instance.backward_ratio_den,
                 "SlackPipe analytical lower bound ratio");
  const Tick chain_lb = CheckedMul(instance.total_layers, ratio_sum,
                                   "SlackPipe chain lower bound");
  const Tick total_work =
      CheckedMul(CheckedMul(instance.microbatches, instance.total_layers,
                            "SlackPipe total-work lower bound"),
                 ratio_sum, "SlackPipe total-work lower bound");
  return std::max(chain_lb, CeilDiv(total_work, instance.workers));
}

std::vector<Tick> UniformSplit(const Instance& instance) {
  instance.Validate();
  const Tick minimum_total =
      CheckedMul(instance.stages, instance.min_layers, "uniform split minimum");
  if (instance.total_layers < minimum_total) {
    throw Error("uniform split requires L >= N * min_layers");
  }
  const Tick remaining = instance.total_layers - minimum_total;
  const Tick q = remaining / instance.stages;
  const Tick remainder = remaining % instance.stages;
  std::vector<Tick> split(static_cast<std::size_t>(instance.stages),
                          instance.min_layers + q);
  for (Index s = 0; s < remainder; ++s) {
    ++split[static_cast<std::size_t>(s)];
  }
  ValidateSplit(instance, split);
  return split;
}

std::vector<Tick> LoadBalancedSplit(const Instance& instance) {
  instance.Validate();
  const Tick minimum_total = CheckedMul(instance.stages, instance.min_layers,
                                        "load-balanced split minimum");
  if (instance.total_layers < minimum_total) {
    throw Error("load-balanced split requires L >= N * min_layers");
  }

  std::vector<Tick> split(static_cast<std::size_t>(instance.stages),
                          instance.min_layers);
  std::vector<Tick> worker_loads(static_cast<std::size_t>(instance.workers), 0);
  for (Index s = 0; s < instance.stages; ++s) {
    worker_loads[static_cast<std::size_t>(s % instance.workers)] =
        CheckedAdd(worker_loads[static_cast<std::size_t>(s % instance.workers)],
                   instance.min_layers, "load-balanced worker minimum");
  }

  Tick remaining = instance.total_layers - minimum_total;
  while (remaining > 0) {
    Index selected_worker = 0;
    for (Index w = 1; w < instance.workers; ++w) {
      if (worker_loads[static_cast<std::size_t>(w)] <
          worker_loads[static_cast<std::size_t>(selected_worker)]) {
        selected_worker = w;
      }
    }

    Index selected_stage = -1;
    for (Index s = 0; s < instance.stages; ++s) {
      if (s % instance.workers != selected_worker) continue;
      if (selected_stage < 0 ||
          split[static_cast<std::size_t>(s)] <
              split[static_cast<std::size_t>(selected_stage)]) {
        selected_stage = s;
      }
    }
    if (selected_stage < 0) {
      throw Error("load-balanced split found worker without stages");
    }
    ++split[static_cast<std::size_t>(selected_stage)];
    ++worker_loads[static_cast<std::size_t>(selected_worker)];
    --remaining;
  }
  ValidateSplit(instance, split);
  return split;
}

std::vector<std::vector<Index>> StagesOnWorkers(const Instance& instance) {
  instance.Validate();
  std::vector<std::vector<Index>> stages(
      static_cast<std::size_t>(instance.workers));
  for (Index s = 0; s < instance.stages; ++s) {
    stages[static_cast<std::size_t>(s % instance.workers)].push_back(s);
  }
  return stages;
}

std::vector<Tick> WorkerLayerTotals(const Instance& instance,
                                    const std::vector<Tick>& split) {
  instance.Validate();
  ValidateSplit(instance, split);
  std::vector<Tick> totals(static_cast<std::size_t>(instance.workers), 0);
  for (Index s = 0; s < instance.stages; ++s) {
    totals[static_cast<std::size_t>(s % instance.workers)] +=
        split[static_cast<std::size_t>(s)];
  }
  return totals;
}

std::vector<Tick> WorkerLayerDifferences(
    const std::vector<Tick>& worker_layers,
    const std::vector<Tick>& baseline_worker_layers) {
  if (worker_layers.size() != baseline_worker_layers.size()) {
    throw Error("worker layer vector sizes do not match");
  }
  std::vector<Tick> differences(worker_layers.size(), 0);
  for (std::size_t i = 0; i < worker_layers.size(); ++i) {
    differences[i] = worker_layers[i] - baseline_worker_layers[i];
  }
  return differences;
}

Tick WorkerBalanceL1(const std::vector<Tick>& differences) {
  Tick total = 0;
  for (Tick diff : differences) total += std::llabs(diff);
  return total;
}

Tick WorkerBalanceMaxDeviation(const std::vector<Tick>& differences) {
  Tick max_deviation = 0;
  for (Tick diff : differences) {
    max_deviation =
        std::max(max_deviation, static_cast<Tick>(std::llabs(diff)));
  }
  return max_deviation;
}

void ValidateSlackPipeOptions(const SlackPipeOptions& options) {
  if (options.move_budget < 0) {
    throw Error("SlackPipe move_budget must be non-negative");
  }
  if (options.per_stage_delta && *options.per_stage_delta < 0) {
    throw Error("SlackPipe per_stage_delta must be non-negative");
  }
  if (options.worker_move_budget < 0) {
    throw Error("SlackPipe worker_move_budget must be non-negative");
  }
  if (options.per_worker_delta && *options.per_worker_delta < 0) {
    throw Error("SlackPipe per_worker_delta must be non-negative");
  }

  const bool stage_controls =
      options.move_budget_provided || options.per_stage_delta.has_value();
  const bool worker_controls = options.worker_move_budget_provided ||
                               options.per_worker_delta.has_value();

  switch (options.split_mode) {
    case SlackPipeSplitMode::kFixed:
      if (stage_controls || worker_controls) {
        throw Error(
            "SlackPipe fixed mode does not accept move-budget, "
            "per-stage-delta, worker-move-budget, or per-worker-delta");
      }
      break;
    case SlackPipeSplitMode::kLocal:
      if (worker_controls) {
        throw Error(
            "SlackPipe local mode does not accept worker-move-budget or "
            "per-worker-delta");
      }
      break;
    case SlackPipeSplitMode::kGlobal:
      if (stage_controls || worker_controls) {
        throw Error(
            "SlackPipe global mode does not accept partition-neighborhood "
            "budgets");
      }
      break;
    case SlackPipeSplitMode::kWorkerFixed:
      if (stage_controls || worker_controls) {
        throw Error(
            "SlackPipe worker-fixed mode does not accept "
            "partition-neighborhood "
            "budgets");
      }
      break;
    case SlackPipeSplitMode::kWorkerLocal:
      if (stage_controls) {
        throw Error(
            "SlackPipe worker-local mode does not accept move-budget or "
            "per-stage-delta");
      }
      break;
  }
}

PartitionRestriction MakePartitionRestriction(
    const Instance& instance, const std::vector<Tick>& reference_split,
    const SlackPipeOptions& options) {
  instance.Validate();
  ValidateSplit(instance, reference_split);
  ValidateSlackPipeOptions(options);

  PartitionRestriction restriction;
  restriction.mode = options.split_mode;
  restriction.reference_split = reference_split;
  switch (options.split_mode) {
    case SlackPipeSplitMode::kFixed:
    case SlackPipeSplitMode::kGlobal:
    case SlackPipeSplitMode::kWorkerFixed:
      break;
    case SlackPipeSplitMode::kLocal:
      restriction.move_budget = options.move_budget;
      restriction.per_stage_delta = options.per_stage_delta;
      break;
    case SlackPipeSplitMode::kWorkerLocal:
      restriction.worker_move_budget = options.worker_move_budget;
      restriction.per_worker_delta = options.per_worker_delta;
      break;
  }
  ValidatePartitionRestriction(instance, restriction);
  return restriction;
}

void ValidatePartitionRestriction(const Instance& instance,
                                  const PartitionRestriction& restriction) {
  instance.Validate();
  ValidateSplit(instance, restriction.reference_split);
  if (restriction.move_budget && *restriction.move_budget < 0) {
    throw Error("partition restriction move_budget must be non-negative");
  }
  if (restriction.worker_move_budget && *restriction.worker_move_budget < 0) {
    throw Error(
        "partition restriction worker_move_budget must be non-negative");
  }
  if (restriction.per_stage_delta && *restriction.per_stage_delta < 0) {
    throw Error("partition restriction per_stage_delta must be non-negative");
  }
  if (restriction.per_worker_delta && *restriction.per_worker_delta < 0) {
    throw Error("partition restriction per_worker_delta must be non-negative");
  }

  switch (restriction.mode) {
    case SlackPipeSplitMode::kFixed:
    case SlackPipeSplitMode::kGlobal:
    case SlackPipeSplitMode::kWorkerFixed:
      if (restriction.move_budget || restriction.worker_move_budget ||
          restriction.per_stage_delta || restriction.per_worker_delta) {
        throw Error("partition restriction contains inapplicable budgets");
      }
      break;
    case SlackPipeSplitMode::kLocal:
      if (!restriction.move_budget) {
        throw Error("local partition restriction requires move_budget");
      }
      if (restriction.worker_move_budget || restriction.per_worker_delta) {
        throw Error(
            "local partition restriction contains worker-local budgets");
      }
      break;
    case SlackPipeSplitMode::kWorkerLocal:
      if (!restriction.worker_move_budget) {
        throw Error(
            "worker-local partition restriction requires worker_move_budget");
      }
      if (restriction.move_budget || restriction.per_stage_delta) {
        throw Error(
            "worker-local partition restriction contains stage-local budgets");
      }
      break;
  }
}

bool SplitSatisfiesPartitionRestriction(
    const Instance& instance, const std::vector<Tick>& split,
    const PartitionRestriction& restriction) {
  instance.Validate();
  ValidateSplit(instance, split);
  ValidatePartitionRestriction(instance, restriction);
  if (restriction.mode == SlackPipeSplitMode::kGlobal) return true;
  if (restriction.mode == SlackPipeSplitMode::kFixed) {
    return split == restriction.reference_split;
  }
  if (restriction.mode == SlackPipeSplitMode::kLocal) {
    Tick l1 = 0;
    for (Index s = 0; s < instance.stages; ++s) {
      const Tick diff =
          split[static_cast<std::size_t>(s)] -
          restriction.reference_split[static_cast<std::size_t>(s)];
      const Tick abs_diff = std::llabs(diff);
      if (restriction.per_stage_delta &&
          abs_diff > *restriction.per_stage_delta) {
        return false;
      }
      l1 += abs_diff;
    }
    return l1 <= CheckedMul(2, *restriction.move_budget,
                            "SlackPipe local move budget");
  }

  const std::vector<Tick> reference_workers =
      WorkerLayerTotals(instance, restriction.reference_split);
  const std::vector<Tick> candidate_workers =
      WorkerLayerTotals(instance, split);
  const std::vector<Tick> differences =
      WorkerLayerDifferences(candidate_workers, reference_workers);
  if (restriction.mode == SlackPipeSplitMode::kWorkerFixed) {
    return WorkerBalanceL1(differences) == 0;
  }
  if (restriction.mode == SlackPipeSplitMode::kWorkerLocal) {
    if (restriction.per_worker_delta && WorkerBalanceMaxDeviation(differences) >
                                            *restriction.per_worker_delta) {
      return false;
    }
    return WorkerBalanceL1(differences) <=
           CheckedMul(2, *restriction.worker_move_budget,
                      "SlackPipe worker-local move budget");
  }
  return false;
}

std::string PartitionRestrictionViolationMessage(
    const Instance& instance, const std::vector<Tick>& split,
    const PartitionRestriction& restriction) {
  auto join = [](const std::vector<Tick>& values) {
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (i != 0) out << ",";
      out << values[i];
    }
    out << "]";
    return out.str();
  };

  std::ostringstream msg;
  msg << "PARTITION_RESTRICTION_VIOLATION mode=" << ToString(restriction.mode)
      << " reference_split=" << join(restriction.reference_split)
      << " returned_split=" << join(split);
  if (restriction.move_budget) {
    msg << " move_budget=" << *restriction.move_budget;
  }
  if (restriction.per_stage_delta) {
    msg << " per_stage_delta=" << *restriction.per_stage_delta;
  }
  if (restriction.worker_move_budget) {
    msg << " worker_move_budget=" << *restriction.worker_move_budget;
  }
  if (restriction.per_worker_delta) {
    msg << " per_worker_delta=" << *restriction.per_worker_delta;
  }
  if (!restriction.reference_split.empty() && !split.empty()) {
    const std::vector<Tick> reference_workers =
        WorkerLayerTotals(instance, restriction.reference_split);
    const std::vector<Tick> returned_workers =
        WorkerLayerTotals(instance, split);
    const std::vector<Tick> differences =
        WorkerLayerDifferences(returned_workers, reference_workers);
    msg << " reference_worker_layers=" << join(reference_workers)
        << " returned_worker_layers=" << join(returned_workers)
        << " worker_l1=" << WorkerBalanceL1(differences)
        << " worker_max_delta=" << WorkerBalanceMaxDeviation(differences);
  }
  return msg.str();
}

bool SplitSatisfiesSlackPipeMode(const Instance& instance,
                                 const std::vector<Tick>& split,
                                 const std::vector<Tick>& baseline_split,
                                 const SlackPipeOptions& options) {
  return SplitSatisfiesPartitionRestriction(
      instance, split,
      MakePartitionRestriction(instance, baseline_split, options));
}

#if !SLACKPIPE_HAVE_ORTOOLS
SlackPipeResult SolveCanonicalSlackPipe(const Instance& instance,
                                        const SlackPipeOptions& options) {
  instance.Validate();
  ValidateSlackPipeOptions(options);

  SlackPipeResult result;
  result.algorithm = "slackpipe";
  result.initial_split_method = "uniform";
  result.split_mode = ToString(options.split_mode);
  result.effective_split_mode = ToString(options.split_mode);
  result.move_budget = options.move_budget;
  result.per_stage_delta = options.per_stage_delta;
  result.worker_move_budget = options.worker_move_budget;
  result.per_worker_delta = options.per_worker_delta;
  result.mode_validation_passed = false;
  result.analytical_global_lower_bound = AnalyticalGlobalLowerBound(instance);
  result.initial_uniform_split = UniformSplit(instance);
  result.baseline_worker_layers =
      WorkerLayerTotals(instance, result.initial_uniform_split);
  result.bfs.split = result.initial_uniform_split;
  result.bfs.method = "uniform";
  result.bfs.status = "UNAVAILABLE";
  result.status = "UNAVAILABLE";
  result.joint_status = "UNAVAILABLE";
  result.joint_incumbent_method_requested = options.bfs_method;
  result.joint_incumbent_method_effective = "unavailable";
  result.joint_bfs_incumbent_method_requested = options.bfs_method;
  result.joint_bfs_incumbent_method_effective = "unavailable";
  result.joint_incumbent_source = "none";
  result.joint_incumbent_feasible = false;
  result.joint_incumbent_primary_objective = 0;
  result.joint_incumbent_hybrid_min_slack = 0.0;
  result.joint_incumbent_baseline_primary_objective = 0;
  result.joint_incumbent_baseline_hybrid_min_slack = 0.0;
  result.joint_incumbent_improved_over_baseline = false;
  result.joint_horizon_source = "none";
  result.joint_hint_budget_seconds = 0.0;
  result.joint_hint_elapsed_seconds = 0.0;
  result.joint_hint_termination_reason = "unavailable";
  result.joint_hints_requested = options.use_bfs_hints;
  result.joint_hints_effective = false;
  result.joint_hint_source = "none";
  result.joint_hint_scope = "none";
  result.joint_hint_complete_for_basic_model = false;
  result.joint_hint_complete_for_full_model = false;
  result.joint_fallback_available = false;
  result.joint_fallback_source = "none";
  result.joint_solution_source = "none";
  result.joint_fallback_used = false;
  result.diagnostic = "slackpipe requires a build with OR-Tools CP-SAT support";
  result.activation_cap_constraints.model_support_level =
      ToString(ActivationCapSolverSupport());
  result.activation_cap_constraints.solver_supported =
      ActivationCapSolverCanEnforce(
          options.activation_options,
          !SlackPipeModeFixesFullPartition(options.split_mode));
  if (options.activation_options.enforce_activation_cap) {
    result.activation_cap_constraints.unsupported_reason =
        ActivationCapSolverUnsupportedReason(
            options.activation_options,
            !SlackPipeModeFixesFullPartition(options.split_mode));
  }
  result.worker_balance_constraint = ComputeWorkerBalanceConstraint(
      instance, options.worker_balance_tolerance_percent,
      options.worker_balance_tolerance_layers);
  result.pressure_pruning_stats.enabled = options.pressure_pruning.enabled;
  result.search_stats_enabled = options.search_stats != nullptr;
  if (options.search_stats != nullptr)
    result.search_stats = *options.search_stats;
  return result;
}

bool IsSlackPipeSolverAvailable() { return false; }
#endif

}  // namespace slackpipe
