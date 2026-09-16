#include "slackpipe/alternating_solver.h"

#include <algorithm>
#include <chrono>
#include <optional>

#include "slackpipe/breadth_first.h"
#include "slackpipe/dag_evaluator.h"
#include "slackpipe/deadline.h"
#include "slackpipe/evaluation_method.h"
#include "slackpipe/fixed_order_partition_solver.h"
#include "slackpipe/result_validator.h"
#include "slackpipe/slackpipe_solver.h"

namespace slackpipe {

namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] double Since(Clock::time_point started) {
  return std::chrono::duration<double>(Clock::now() - started).count();
}

[[nodiscard]] bool Feasible(const std::string &status) {
  return status == "OPTIMAL" || status == "FEASIBLE";
}

[[nodiscard]] bool BetterOrTie(const ScheduleSolution &candidate,
                               const std::vector<Tick> &candidate_split,
                               const MachineOrders &candidate_orders,
                               const ScheduleSolution &incumbent,
                               const std::vector<Tick> &incumbent_split,
                               const MachineOrders &incumbent_orders,
                               bool *strict_improvement) {
  *strict_improvement = false;
  if (!candidate.ok()) return false;
  if (!incumbent.ok()) {
    *strict_improvement = true;
    return true;
  }
  if (candidate.makespan < incumbent.makespan) {
    *strict_improvement = true;
    return true;
  }
  if (candidate.makespan > incumbent.makespan) return false;
  if (std::lexicographical_compare(
          candidate_split.begin(), candidate_split.end(),
          incumbent_split.begin(), incumbent_split.end())) {
    return true;
  }
  if (candidate_split != incumbent_split) return false;
  for (std::size_t w = 0;
       w < std::min(candidate_orders.size(), incumbent_orders.size()); ++w) {
    const auto &lhs = candidate_orders[w];
    const auto &rhs = incumbent_orders[w];
    if (std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(),
                                     rhs.end())) {
      return true;
    }
    if (std::lexicographical_compare(rhs.begin(), rhs.end(), lhs.begin(),
                                     lhs.end())) {
      return false;
    }
  }
  return candidate_orders.size() < incumbent_orders.size();
}

[[nodiscard]] ResultValidationResult ValidateCandidate(
    const Instance &instance, const ScheduleSolution &schedule,
    const std::string &status) {
  if (!schedule.ok()) {
    ResultValidationResult validation;
    validation.passed = false;
    validation.error_code = "schedule_invalid";
    validation.message = schedule.validation_errors.empty()
                             ? std::string("invalid schedule")
                             : schedule.validation_errors.front();
    return validation;
  }
  return ValidateScheduleSolutionIndependent(instance, schedule, status,
                                             "alternating-partition-schedule");
}

CanonicalPhaseBudget PhaseBudgetFromTrace(
    const CanonicalAlternatingTraceEntry &trace) {
  CanonicalPhaseBudget budget;
  budget.phase = trace.phase_type;
  budget.requested_limit_seconds = trace.phase_limit_seconds;
  budget.effective_limit_seconds = trace.phase_limit_seconds;
  budget.runtime_seconds = trace.phase_runtime_seconds;
  budget.status = trace.solver_status_raw;
  return budget;
}

void AddTrace(AlternatingOptimizationResult &result,
              CanonicalAlternatingTraceEntry trace) {
  result.phase_budget.phases.push_back(PhaseBudgetFromTrace(trace));
  result.alternating_trace.push_back(std::move(trace));
}

[[nodiscard]] double PhaseLimit(const Deadline &deadline,
                                int remaining_planned_phases) {
  if (!deadline.bounded()) return 0.0;
  return deadline.clamp_solver_limit(AlternatingPhaseLimitSeconds(
      deadline.remaining_seconds(), remaining_planned_phases));
}

[[nodiscard]] AlternatingOptimizationResult MakeInitialResult(
    const Instance &instance, const std::string &method, int max_rounds) {
  const MachineOrders orders = BreadthFirstOrders(instance);
  const std::vector<Tick> split = UniformSplit(instance);
  EvaluationResult evaluated = EvaluateSchedule(instance, split, orders);
  if (!evaluated.schedule.ok()) {
    throw Error("uniform fixed-order baseline is invalid");
  }
  const ResultValidationResult validation = ValidateScheduleSolutionIndependent(
      instance, evaluated.schedule, "FEASIBLE", method);
  if (!validation.passed) {
    throw Error("uniform fixed-order baseline failed validation: " +
                validation.error_code);
  }
  AlternatingOptimizationResult result;
  result.method = method;
  result.split = split;
  result.machine_orders = orders;
  result.schedule = std::move(evaluated.schedule);
  result.makespan_ticks = result.schedule.makespan;
  result.status = "FEASIBLE";
  result.solver_status_raw = "FEASIBLE";
  result.initial_makespan = result.makespan_ticks;
  result.initial_split = split;
  result.alternating_max_rounds = max_rounds;
  result.returned_solution_source = "uniform_breadth_first_incumbent";
  return result;
}

void AcceptCandidate(AlternatingOptimizationResult &result,
                     const std::vector<Tick> &split,
                     const MachineOrders &orders, ScheduleSolution schedule,
                     const std::string &source) {
  result.split = split;
  result.machine_orders = orders;
  result.makespan_ticks = schedule.makespan;
  result.schedule = std::move(schedule);
  result.status = "FEASIBLE";
  result.returned_solution_source = source;
}

JointOptimizerOptions ScheduleOptionsFromAlternating(
    const AlternatingOptimizerOptions &options, double phase_limit,
    int phase_seed) {
  JointOptimizerOptions schedule_options;
  schedule_options.time_limit_seconds = phase_limit;
  schedule_options.num_workers = options.num_workers;
  schedule_options.random_seed = phase_seed;
  schedule_options.require_optimal = false;
  schedule_options.log_search_progress = options.log_search_progress;
  schedule_options.use_bfs_hints = options.use_bfs_hints;
  schedule_options.fifo_ordering = options.fifo_ordering;
  schedule_options.symmetry_break_f0_fifo = options.symmetry_break_f0_fifo;
  schedule_options.activation_options = options.activation_options;
  return schedule_options;
}

BfsSplitOptimizerOptions PartitionOptionsFromAlternating(
    const AlternatingOptimizerOptions &options, double phase_limit,
    int phase_seed) {
  BfsSplitOptimizerOptions partition_options;
  partition_options.time_limit_seconds = phase_limit;
  partition_options.num_workers = options.num_workers;
  partition_options.random_seed = phase_seed;
  partition_options.require_optimal = false;
  partition_options.log_search_progress = options.log_search_progress;
  partition_options.enumeration_threshold = options.enumeration_threshold;
  partition_options.fixed_order_partition_backend =
      options.fixed_order_partition_backend;
  partition_options.activation_options = options.activation_options;
  return partition_options;
}

void ApplyActivationTraceFields(
    CanonicalAlternatingTraceEntry &trace,
    const ActivationAnalysisOptions &options,
    const ActivationCapConstraintMetadata &metadata) {
  trace.activation_cap_requested = options.enforce_activation_cap;
  trace.activation_cap_supported = metadata.solver_supported;
  trace.activation_cap_constraints_added = metadata.constraints_added;
  trace.candidate_rejected_for_activation_cap =
      metadata.incumbent_rejected_for_activation_cap;
}

[[nodiscard]] bool CandidateSatisfiesActivationCap(
    const Instance &instance, const ScheduleSolution &schedule,
    const ActivationAnalysisOptions &options,
    CanonicalAlternatingTraceEntry *trace) {
  if (!options.enforce_activation_cap ||
      options.cap_mode == ActivationCapMode::kNone) {
    if (trace != nullptr) trace->activation_cap_satisfied = std::nullopt;
    return true;
  }
  const bool satisfied =
      ActivationScheduleSatisfiesCap(instance, schedule, options);
  if (trace != nullptr) {
    trace->activation_cap_satisfied = satisfied;
    if (!satisfied) trace->candidate_rejected_for_activation_cap = true;
  }
  return satisfied;
}

void AccumulateActivationMetadata(
    ActivationCapConstraintMetadata &aggregate,
    const ActivationCapConstraintMetadata &phase) {
  aggregate.model_support_level = phase.model_support_level.empty()
                                      ? ToString(ActivationCapSolverSupport())
                                      : phase.model_support_level;
  aggregate.solver_supported =
      aggregate.solver_supported || phase.solver_supported;
  aggregate.retained_interval_count += phase.retained_interval_count;
  aggregate.cumulative_constraint_count += phase.cumulative_constraint_count;
  aggregate.variable_demand_count += phase.variable_demand_count;
  aggregate.fixed_demand_count += phase.fixed_demand_count;
  aggregate.build_runtime_seconds += phase.build_runtime_seconds;
  aggregate.incumbent_rejected_for_activation_cap =
      aggregate.incumbent_rejected_for_activation_cap ||
      phase.incumbent_rejected_for_activation_cap;
  aggregate.constraints_added =
      aggregate.constraints_added || phase.constraints_added;
  aggregate.workers_with_constraints.insert(
      aggregate.workers_with_constraints.end(),
      phase.workers_with_constraints.begin(),
      phase.workers_with_constraints.end());
  if (aggregate.unsupported_reason.empty() &&
      !phase.unsupported_reason.empty()) {
    aggregate.unsupported_reason = phase.unsupported_reason;
  }
}

}  // namespace

AlternatingOptimizationResult OptimizeSequentialPartitionThenSchedule(
    const Instance &instance, const AlternatingOptimizerOptions &options) {
  const auto started = Clock::now();
  instance.Validate();
  if (options.activation_options.enforce_activation_cap &&
      !ActivationCapSolverCanEnforce(options.activation_options, true)) {
    AlternatingOptimizationResult unavailable;
    unavailable.method = "sequential-partition-then-schedule";
    unavailable.status = "UNAVAILABLE";
    unavailable.solver_status_raw = "UNAVAILABLE";
    unavailable.wall_time_seconds = Since(started);
    unavailable.timing.total_seconds = unavailable.wall_time_seconds;
    unavailable.activation_cap_constraints.model_support_level =
        ToString(ActivationCapSolverSupport());
    unavailable.activation_cap_constraints.solver_supported = false;
    unavailable.activation_cap_constraints.unsupported_reason =
        ActivationCapSolverUnsupportedReason(options.activation_options, true);
    unavailable.diagnostic =
        unavailable.activation_cap_constraints.unsupported_reason;
    return unavailable;
  }
  AlternatingOptimizationResult result =
      MakeInitialResult(instance, "sequential-partition-then-schedule", 1);
  bool current_cap_feasible = CandidateSatisfiesActivationCap(
      instance, result.schedule, options.activation_options, nullptr);
  result.activation_cap_constraints.model_support_level =
      ToString(ActivationCapSolverSupport());
  result.activation_cap_constraints.solver_supported = false;
  result.activation_cap_constraints.incumbent_rejected_for_activation_cap =
      options.activation_options.enforce_activation_cap &&
      !current_cap_feasible;
  bool all_activation_phases_constrained = true;
  int activation_phase_count = 0;
  Deadline deadline(options.time_limit_seconds);

  const double partition_limit =
      deadline.bounded()
          ? deadline.clamp_solver_limit(options.time_limit_seconds *
                                        kScheduleOnlyPartitionBudgetFraction)
          : 0.0;
  const auto partition_started = Clock::now();
  BfsSplitOptimizerOptions partition_options = PartitionOptionsFromAlternating(
      options, partition_limit, options.random_seed);
  partition_options.fixed_order_partition_incumbent = result.split;
  const BfsSplitOptimizationResult partition =
      partition_limit == 0.0 && deadline.expired()
          ? BfsSplitOptimizationResult{}
          : OptimizePartitionForFixedOrder(instance, result.machine_orders,
                                           partition_options);
  CanonicalAlternatingTraceEntry partition_trace;
  partition_trace.round_index = 0;
  partition_trace.phase_type = "partition_fixed_order";
  if (deadline.bounded()) partition_trace.phase_limit_seconds = partition_limit;
  partition_trace.phase_runtime_seconds = Since(partition_started);
  partition_trace.input_makespan = result.makespan_ticks;
  partition_trace.solver_status_raw =
      partition.solver_status_raw.empty()
          ? (partition.status.empty() ? std::string("NOT_RUN")
                                      : partition.status)
          : partition.solver_status_raw;
  partition_trace.selected_partition = partition.split;
  partition_trace.seed = options.random_seed;
  partition_trace.solver_threads = options.num_workers;
  ApplyActivationTraceFields(partition_trace, options.activation_options,
                             partition.activation_cap_constraints);
  AccumulateActivationMetadata(result.activation_cap_constraints,
                               partition.activation_cap_constraints);
  if (options.activation_options.enforce_activation_cap) {
    ++activation_phase_count;
    all_activation_phases_constrained =
        all_activation_phases_constrained &&
        partition.activation_cap_constraints.constraints_added;
  }
  result.cp_sat_models_solved += partition.cp_sat_models_solved;
  if (Feasible(partition.status)) {
    partition_trace.candidate_makespan = partition.makespan_ticks;
    const ResultValidationResult validation =
        ValidateCandidate(instance, partition.schedule, partition.status);
    partition_trace.validation_passed = validation.passed;
    const bool cap_ok =
        validation.passed && CandidateSatisfiesActivationCap(
                                 instance, partition.schedule,
                                 options.activation_options, &partition_trace);
    if (validation.passed && cap_ok) {
      result.intermediate_partition_only_makespan = partition.makespan_ticks;
      bool strict = false;
      if (!current_cap_feasible ||
          BetterOrTie(partition.schedule, partition.split,
                      partition.machine_orders, result.schedule, result.split,
                      result.machine_orders, &strict)) {
        AcceptCandidate(result, partition.split, partition.machine_orders,
                        partition.schedule, "partition_fixed_order_phase");
        partition_trace.accepted = true;
        current_cap_feasible = true;
      }
    }
  }
  AddTrace(result, partition_trace);

  const std::vector<Tick> scheduled_split = result.split;
  const double schedule_limit = deadline.clamp_solver_limit(0.0);
  const auto schedule_started = Clock::now();
  JointOptimizationResult schedule =
      deadline.expired()
          ? BuildScheduleOnlyFixedSplitDeadlineFallback(
                instance, scheduled_split,
                ScheduleOptionsFromAlternating(options, schedule_limit,
                                               options.random_seed + 1),
                deadline.elapsed_seconds(),
                "global_deadline_expired_before_cp_sat")
          : OptimizeScheduleForFixedSplitCpSat(
                instance, scheduled_split,
                ScheduleOptionsFromAlternating(options, schedule_limit,
                                               options.random_seed + 1));
  CanonicalAlternatingTraceEntry schedule_trace;
  schedule_trace.round_index = 0;
  schedule_trace.phase_type = "schedule_fixed_split";
  if (deadline.bounded()) schedule_trace.phase_limit_seconds = schedule_limit;
  schedule_trace.phase_runtime_seconds = Since(schedule_started);
  schedule_trace.input_makespan = result.makespan_ticks;
  schedule_trace.solver_status_raw =
      schedule.joint_status.empty() ? schedule.status : schedule.joint_status;
  schedule_trace.fallback_used = schedule.fallback_used;
  schedule_trace.selected_partition = scheduled_split;
  schedule_trace.seed = options.random_seed + 1;
  schedule_trace.solver_threads = options.num_workers;
  ApplyActivationTraceFields(schedule_trace, options.activation_options,
                             schedule.activation_cap_constraints);
  AccumulateActivationMetadata(result.activation_cap_constraints,
                               schedule.activation_cap_constraints);
  if (options.activation_options.enforce_activation_cap) {
    ++activation_phase_count;
    all_activation_phases_constrained =
        all_activation_phases_constrained &&
        schedule.activation_cap_constraints.constraints_added;
  }
  result.cp_sat_models_solved += schedule.cp_sat_models_solved;
  result.best_bound_ticks = schedule.best_bound_ticks;
  if (Feasible(schedule.status)) {
    schedule_trace.candidate_makespan = schedule.makespan_ticks;
    const ResultValidationResult validation =
        ValidateCandidate(instance, schedule.schedule, schedule.status);
    schedule_trace.validation_passed = validation.passed;
    const bool cap_ok =
        validation.passed && CandidateSatisfiesActivationCap(
                                 instance, schedule.schedule,
                                 options.activation_options, &schedule_trace);
    if (validation.passed && cap_ok) {
      bool strict = false;
      if (!current_cap_feasible ||
          BetterOrTie(schedule.schedule, schedule.split,
                      schedule.machine_orders, result.schedule, result.split,
                      result.machine_orders, &strict)) {
        AcceptCandidate(result, schedule.split, schedule.machine_orders,
                        schedule.schedule, schedule.solution_source);
        schedule_trace.accepted = true;
        current_cap_feasible = true;
      }
    }
  }
  AddTrace(result, schedule_trace);
  result.alternating_completed_rounds = 1;
  result.alternating_convergence_reason =
      deadline.expired() ? "deadline_expired" : "sequential_complete";
  result.solver_status_raw =
      schedule.joint_status.empty() ? schedule.status : schedule.joint_status;
  if (options.activation_options.enforce_activation_cap &&
      activation_phase_count > 0) {
    result.activation_cap_constraints.constraints_added =
        all_activation_phases_constrained;
  }
  if (options.activation_options.enforce_activation_cap &&
      !current_cap_feasible) {
    result.schedule = ScheduleSolution{};
    result.split.clear();
    result.machine_orders.clear();
    result.makespan_ticks = 0;
    result.returned_solution_source = "none";
  }
  result.status = result.schedule.ok() ? "FEASIBLE" : "UNKNOWN";
  result.proven_optimal = false;
  result.wall_time_seconds = Since(started);
  result.timing.total_seconds = result.wall_time_seconds;
  result.timing.solver_seconds = result.wall_time_seconds;
  result.phase_budget.reference_phase_limit_seconds =
      deadline.bounded() ? std::optional<double>(partition_limit)
                         : std::nullopt;
  result.phase_budget.schedule_solver_effective_limit_seconds =
      deadline.bounded() ? std::optional<double>(schedule_limit) : std::nullopt;
  if (options.require_optimal && !result.proven_optimal) {
    result.status = "FEASIBLE_REJECTED_REQUIRE_OPTIMAL";
  }
  return result;
}

AlternatingOptimizationResult OptimizeAlternatingPartitionSchedule(
    const Instance &instance, const AlternatingOptimizerOptions &options) {
  const auto started = Clock::now();
  instance.Validate();
  if (options.activation_options.enforce_activation_cap &&
      !ActivationCapSolverCanEnforce(options.activation_options, true)) {
    AlternatingOptimizationResult unavailable;
    unavailable.method = "alternating-partition-schedule";
    unavailable.status = "UNAVAILABLE";
    unavailable.solver_status_raw = "UNAVAILABLE";
    unavailable.wall_time_seconds = Since(started);
    unavailable.timing.total_seconds = unavailable.wall_time_seconds;
    unavailable.activation_cap_constraints.model_support_level =
        ToString(ActivationCapSolverSupport());
    unavailable.activation_cap_constraints.solver_supported = false;
    unavailable.activation_cap_constraints.unsupported_reason =
        ActivationCapSolverUnsupportedReason(options.activation_options, true);
    unavailable.diagnostic =
        unavailable.activation_cap_constraints.unsupported_reason;
    return unavailable;
  }
  const int max_rounds = std::max(0, options.max_rounds);
  AlternatingOptimizationResult result =
      MakeInitialResult(instance, "alternating-partition-schedule", max_rounds);
  bool current_cap_feasible = CandidateSatisfiesActivationCap(
      instance, result.schedule, options.activation_options, nullptr);
  result.activation_cap_constraints.model_support_level =
      ToString(ActivationCapSolverSupport());
  result.activation_cap_constraints.solver_supported = false;
  result.activation_cap_constraints.incumbent_rejected_for_activation_cap =
      options.activation_options.enforce_activation_cap &&
      !current_cap_feasible;
  bool all_activation_phases_constrained = true;
  int activation_phase_count = 0;
  Deadline deadline(options.time_limit_seconds);
  bool stopped = false;

  for (int round = 0; round < max_rounds && !stopped; ++round) {
    bool strict_improvement_this_round = false;

    if (deadline.expired()) {
      result.alternating_convergence_reason = "deadline_expired";
      break;
    }
    const int remaining_partition_phases = 2 * (max_rounds - round);
    const double partition_limit =
        PhaseLimit(deadline, remaining_partition_phases);
    const auto partition_started = Clock::now();
    const int partition_seed = options.random_seed + 2 * round;
    BfsSplitOptimizerOptions partition_options =
        PartitionOptionsFromAlternating(options, partition_limit,
                                        partition_seed);
    partition_options.fixed_order_partition_incumbent = result.split;
    BfsSplitOptimizationResult partition = OptimizePartitionForFixedOrder(
        instance, result.machine_orders, partition_options);
    CanonicalAlternatingTraceEntry partition_trace;
    partition_trace.round_index = round;
    partition_trace.phase_type = "partition_fixed_order";
    if (deadline.bounded())
      partition_trace.phase_limit_seconds = partition_limit;
    partition_trace.phase_runtime_seconds = Since(partition_started);
    partition_trace.input_makespan = result.makespan_ticks;
    partition_trace.candidate_makespan =
        Feasible(partition.status)
            ? std::optional<Tick>(partition.makespan_ticks)
            : std::nullopt;
    partition_trace.solver_status_raw = partition.solver_status_raw.empty()
                                            ? partition.status
                                            : partition.solver_status_raw;
    partition_trace.selected_partition = partition.split;
    partition_trace.seed = partition_seed;
    partition_trace.solver_threads = options.num_workers;
    ApplyActivationTraceFields(partition_trace, options.activation_options,
                               partition.activation_cap_constraints);
    AccumulateActivationMetadata(result.activation_cap_constraints,
                                 partition.activation_cap_constraints);
    if (options.activation_options.enforce_activation_cap) {
      ++activation_phase_count;
      all_activation_phases_constrained =
          all_activation_phases_constrained &&
          partition.activation_cap_constraints.constraints_added;
    }
    result.cp_sat_models_solved += partition.cp_sat_models_solved;
    if (Feasible(partition.status)) {
      const ResultValidationResult validation =
          ValidateCandidate(instance, partition.schedule, partition.status);
      partition_trace.validation_passed = validation.passed;
      const bool cap_ok = validation.passed &&
                          CandidateSatisfiesActivationCap(
                              instance, partition.schedule,
                              options.activation_options, &partition_trace);
      if (validation.passed && cap_ok) {
        bool strict = false;
        if (!current_cap_feasible ||
            BetterOrTie(partition.schedule, partition.split,
                        partition.machine_orders, result.schedule, result.split,
                        result.machine_orders, &strict)) {
          AcceptCandidate(result, partition.split, partition.machine_orders,
                          partition.schedule, "partition_fixed_order_phase");
          partition_trace.accepted = true;
          current_cap_feasible = true;
          strict_improvement_this_round =
              strict_improvement_this_round || strict;
        }
      }
    }
    AddTrace(result, partition_trace);

    if (deadline.expired()) {
      result.alternating_convergence_reason = "deadline_expired";
      break;
    }
    const int remaining_schedule_phases = 2 * (max_rounds - round) - 1;
    const double schedule_limit =
        PhaseLimit(deadline, remaining_schedule_phases);
    const auto schedule_started = Clock::now();
    const int schedule_seed = options.random_seed + 2 * round + 1;
    JointOptimizationResult schedule = OptimizeScheduleForFixedSplitCpSat(
        instance, result.split,
        ScheduleOptionsFromAlternating(options, schedule_limit, schedule_seed));
    CanonicalAlternatingTraceEntry schedule_trace;
    schedule_trace.round_index = round;
    schedule_trace.phase_type = "schedule_fixed_split";
    if (deadline.bounded()) schedule_trace.phase_limit_seconds = schedule_limit;
    schedule_trace.phase_runtime_seconds = Since(schedule_started);
    schedule_trace.input_makespan = result.makespan_ticks;
    schedule_trace.candidate_makespan =
        Feasible(schedule.status) ? std::optional<Tick>(schedule.makespan_ticks)
                                  : std::nullopt;
    schedule_trace.solver_status_raw =
        schedule.joint_status.empty() ? schedule.status : schedule.joint_status;
    schedule_trace.fallback_used = schedule.fallback_used;
    schedule_trace.selected_partition = result.split;
    schedule_trace.seed = schedule_seed;
    schedule_trace.solver_threads = options.num_workers;
    ApplyActivationTraceFields(schedule_trace, options.activation_options,
                               schedule.activation_cap_constraints);
    AccumulateActivationMetadata(result.activation_cap_constraints,
                                 schedule.activation_cap_constraints);
    if (options.activation_options.enforce_activation_cap) {
      ++activation_phase_count;
      all_activation_phases_constrained =
          all_activation_phases_constrained &&
          schedule.activation_cap_constraints.constraints_added;
    }
    result.cp_sat_models_solved += schedule.cp_sat_models_solved;
    result.best_bound_ticks = schedule.best_bound_ticks > 0.0
                                  ? schedule.best_bound_ticks
                                  : result.best_bound_ticks;
    result.solver_status_raw = schedule_trace.solver_status_raw.value_or("");
    if (Feasible(schedule.status)) {
      const ResultValidationResult validation =
          ValidateCandidate(instance, schedule.schedule, schedule.status);
      schedule_trace.validation_passed = validation.passed;
      const bool cap_ok =
          validation.passed && CandidateSatisfiesActivationCap(
                                   instance, schedule.schedule,
                                   options.activation_options, &schedule_trace);
      if (validation.passed && cap_ok) {
        bool strict = false;
        if (!current_cap_feasible ||
            BetterOrTie(schedule.schedule, schedule.split,
                        schedule.machine_orders, result.schedule, result.split,
                        result.machine_orders, &strict)) {
          AcceptCandidate(result, schedule.split, schedule.machine_orders,
                          schedule.schedule, schedule.solution_source);
          schedule_trace.accepted = true;
          current_cap_feasible = true;
          strict_improvement_this_round =
              strict_improvement_this_round || strict;
        }
      }
    }
    AddTrace(result, schedule_trace);
    result.alternating_completed_rounds = round + 1;
    if (!strict_improvement_this_round) {
      result.alternating_convergence_reason = "no_improvement_round";
      stopped = true;
    }
  }

  if (result.alternating_convergence_reason.empty()) {
    result.alternating_convergence_reason =
        deadline.expired() ? "deadline_expired" : "max_rounds";
  }
  if (options.activation_options.enforce_activation_cap &&
      activation_phase_count > 0) {
    result.activation_cap_constraints.constraints_added =
        all_activation_phases_constrained;
  }
  if (options.activation_options.enforce_activation_cap &&
      !current_cap_feasible) {
    result.schedule = ScheduleSolution{};
    result.split.clear();
    result.machine_orders.clear();
    result.makespan_ticks = 0;
    result.returned_solution_source = "none";
  }
  result.status = result.schedule.ok() ? "FEASIBLE" : "UNKNOWN";
  result.proven_optimal = false;
  result.wall_time_seconds = Since(started);
  result.timing.total_seconds = result.wall_time_seconds;
  result.timing.solver_seconds = result.wall_time_seconds;
  result.phase_budget.schedule_solver_effective_limit_seconds =
      deadline.bounded()
          ? std::optional<double>(deadline.remaining_seconds_for_reporting())
          : std::nullopt;
  if (options.require_optimal && !result.proven_optimal) {
    result.status = "FEASIBLE_REJECTED_REQUIRE_OPTIMAL";
  }
  return result;
}

CanonicalOutcome OutcomeFromAlternatingResult(
    const AlternatingOptimizationResult &result) {
  CanonicalOutcome outcome;
  outcome.solver_status_raw = result.solver_status_raw.empty()
                                  ? result.status
                                  : result.solver_status_raw;
  outcome.reported_status = result.status;
  outcome.feasible = Feasible(result.status);
  outcome.optimal = result.proven_optimal;
  outcome.fallback_used = result.fallback_used;
  if (!result.fallback_reason.empty())
    outcome.fallback_reason = result.fallback_reason;
  outcome.returned_solution_source = result.returned_solution_source;
  outcome.makespan = result.makespan_ticks > 0
                         ? std::optional<Tick>(result.makespan_ticks)
                         : std::nullopt;
  outcome.best_objective_bound =
      result.best_bound_ticks > 0.0
          ? std::optional<double>(result.best_bound_ticks)
          : std::nullopt;
  if (outcome.makespan && outcome.best_objective_bound) {
    outcome.relative_optimality_gap =
        RelativeGap(*outcome.makespan, *outcome.best_objective_bound);
  }
  outcome.total_runtime_seconds =
      result.wall_time_seconds > 0.0
          ? std::optional<double>(result.wall_time_seconds)
          : std::nullopt;
  outcome.solver_runtime_seconds = outcome.total_runtime_seconds;
  outcome.phase_budget = result.phase_budget;
  outcome.result_validation_passed = result.schedule.ok();
  if (!result.schedule.validation_errors.empty()) {
    outcome.result_validation_error = result.schedule.validation_errors.front();
  }
  return outcome;
}

void ApplyAlternatingCanonicalFields(
    const AlternatingOptimizationResult &result,
    CanonicalResultMetadata &metadata) {
  metadata.alternating_max_rounds = result.alternating_max_rounds;
  metadata.alternating_completed_rounds = result.alternating_completed_rounds;
  metadata.alternating_convergence_reason =
      result.alternating_convergence_reason;
  metadata.alternating_trace = result.alternating_trace;
  if (result.intermediate_partition_only_makespan > 0) {
    metadata.intermediate_partition_only_makespan =
        result.intermediate_partition_only_makespan;
  }
}

}  // namespace slackpipe
