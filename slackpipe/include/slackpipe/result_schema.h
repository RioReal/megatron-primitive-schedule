#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "slackpipe/activation_analyzer.h"
#include "slackpipe/dag_evaluator.h"
#include "slackpipe/deadline.h"
#include "slackpipe/evaluation_method.h"
#include "slackpipe/instance.h"
#include "slackpipe/partition_restriction.h"
#include "slackpipe/result_validator.h"
#include "slackpipe/worker_balance.h"

namespace slackpipe {

inline constexpr int kEvaluationResultSchemaVersion = 1;
inline constexpr int kBuildInfoSchemaVersion = 1;

struct BfsSplitOptimizationResult;
struct JointOptimizationResult;
struct SlackPipeResult;

struct StageWorkerMappingEntry {
  Index stage = 0;
  Index worker = 0;
};

struct CanonicalWorkerPredecessor {
  Index operation_id = 0;
  std::string operation;
  Index predecessor_id = 0;
  std::string predecessor;
  Index worker = 0;
};

struct CanonicalRequestContext {
  std::optional<std::string> run_id;
  std::optional<std::string> timestamp_utc;
  std::optional<std::string> requested_command;
  std::optional<std::string> requested_method;
  std::optional<std::string> executable_name;
  std::optional<double> requested_time_limit_seconds;
  std::optional<double> effective_time_limit_seconds;
  std::optional<int> random_seed;
  std::optional<int> solver_threads;
};

struct CanonicalSemantics {
  std::string canonical_method;
  std::string actual_solver_path;
  std::string partition_decision = "not_applicable";
  std::string schedule_decision = "not_applicable";
  std::optional<std::string> partition_reference_source;
  std::optional<std::string> schedule_reference_source;
  bool full_partition_fixed = false;
  bool worker_aggregate_loads_fixed = false;
  bool partition_optimized = false;
  bool schedule_optimized = false;
  bool worker_balance_pruning_enabled = false;
  std::optional<bool> worker_balance_pruning_requested;
  std::optional<bool> worker_balance_pruning_effective;
  std::optional<double> worker_balance_tolerance_percent;
  std::optional<Index> worker_balance_tolerance_layers;
  std::optional<Index> worker_balance_lower_bound;
  std::optional<Index> worker_balance_upper_bound;
  std::optional<Index> stage_local_move_budget;
  std::optional<Index> worker_aggregate_move_budget;
  bool predecessor_candidate_restriction_requested = false;
  bool predecessor_candidate_restriction_active = false;
  std::optional<std::string> predecessor_candidate_rule;
  std::optional<bool> fifo_ordering_requested;
  std::optional<bool> fifo_ordering_effective;
  std::optional<Index> fifo_constraint_count;
  std::optional<std::string> incumbent_method_requested;
  std::optional<std::string> incumbent_method_effective;
  std::optional<std::string> fixed_order_partition_backend_requested;
  std::optional<std::string> fixed_order_partition_backend_effective;
  std::optional<std::uint64_t> estimated_partition_count;
  std::optional<std::uint64_t> enumeration_safety_threshold;
  std::optional<bool> cp_sat_launched;
  std::optional<bool> cp_sat_hint_enabled;
  std::optional<bool> incumbent_upper_bound_enabled;
  std::optional<bool> incumbent_bound_requested;
  std::optional<bool> incumbent_bound_effective;
  std::optional<Tick> incumbent_bound_horizon;
  std::optional<bool> incumbent_hints_requested;
  std::optional<bool> incumbent_hints_effective;
  std::optional<Index> incumbent_hint_count;
  std::optional<bool> incumbent_found;
  std::optional<bool> incumbent_valid;
  std::optional<Tick> incumbent_makespan;
  std::optional<Index> cp_sat_models_solved;
};

struct CanonicalPhaseBudget {
  std::string phase;
  std::optional<double> requested_limit_seconds;
  std::optional<double> effective_limit_seconds;
  std::optional<double> remaining_before_seconds;
  std::optional<double> remaining_after_seconds;
  std::optional<double> runtime_seconds;
  bool expired_before_start = false;
  std::optional<std::string> status;
};

struct CanonicalPhaseBudgetSummary {
  std::optional<double> reference_phase_limit_seconds;
  std::optional<double> schedule_solver_effective_limit_seconds;
  std::vector<CanonicalPhaseBudget> phases;
};

struct CanonicalAlternatingTraceEntry {
  Index round_index = 0;
  std::string phase_type;
  std::optional<double> phase_limit_seconds;
  std::optional<double> phase_runtime_seconds;
  std::optional<Tick> input_makespan;
  std::optional<Tick> candidate_makespan;
  bool accepted = false;
  bool validation_passed = false;
  std::vector<Tick> selected_partition;
  std::optional<std::string> solver_status_raw;
  bool fallback_used = false;
  std::optional<int> seed;
  std::optional<int> solver_threads;
  bool activation_cap_requested = false;
  bool activation_cap_supported = false;
  bool activation_cap_constraints_added = false;
  std::optional<bool> activation_cap_satisfied;
  bool candidate_rejected_for_activation_cap = false;
};

struct CanonicalOutcome {
  std::optional<std::string> solver_status_raw;
  std::optional<std::string> reported_status;
  std::optional<bool> feasible;
  std::optional<bool> optimal;
  bool fallback_used = false;
  std::optional<bool> fallback_enabled;
  std::optional<bool> external_incumbent_available;
  std::optional<bool> external_incumbent_used_as_fallback;
  std::optional<bool> solver_solution_available;
  std::optional<bool> final_solution_available;
  std::optional<std::string> final_solution_source;
  std::optional<std::string> no_solution_reason;
  std::optional<std::string> fallback_reason;
  std::optional<std::string> returned_solution_source;
  std::optional<Tick> makespan;
  std::optional<double> best_objective_bound;
  std::optional<double> relative_optimality_gap;
  std::optional<std::string> optimality_proof_source;
  std::optional<bool> enumeration_proved_optimal;
  std::optional<std::uint64_t> enumeration_candidates_total;
  std::optional<std::uint64_t> enumeration_candidates_valid_schedule;
  std::optional<std::uint64_t> enumeration_candidates_cap_feasible;
  std::optional<std::uint64_t> enumeration_candidates_cap_rejected;
  std::optional<double> total_runtime_seconds;
  std::optional<double> incumbent_runtime_seconds;
  std::optional<double> reference_runtime_seconds;
  std::optional<double> model_build_runtime_seconds;
  std::optional<double> solver_runtime_seconds;
  std::optional<double> time_to_first_feasible_seconds;
  std::optional<double> time_to_first_cpsat_feasible_seconds;
  std::optional<Tick> first_cpsat_feasible_objective;
  std::optional<double> time_to_best_solution_seconds;
  std::optional<double> validation_runtime_seconds;
  CanonicalPhaseBudgetSummary phase_budget;
  std::optional<ResultValidationResult> result_validation;
  bool result_validation_passed = false;
  std::optional<std::string> result_validation_error;
};

struct CanonicalResultMetadata {
  int schema_version = kEvaluationResultSchemaVersion;
  int budget_policy_version = kEvaluationBudgetPolicyVersion;
  int evaluation_method_version = kEvaluationMethodVersion;
  std::optional<std::string> run_id;
  std::optional<std::string> timestamp_utc;
  std::optional<std::string> requested_command;
  std::optional<std::string> requested_method;
  std::string canonical_method;
  std::string actual_solver_path;
  std::optional<double> requested_time_limit_seconds;
  std::optional<double> effective_time_limit_seconds;
  std::optional<int> random_seed;
  std::optional<int> solver_threads;

  Index micro_batches = 0;
  Index logical_stages = 0;
  Index physical_workers = 0;
  Index total_layers = 0;
  Index min_layers = 1;
  std::vector<StageWorkerMappingEntry> stage_to_worker_mapping;
  std::string mapping_type = "cyclic_stage_mod_worker";
  Tick forward_cost_ratio_numerator = 1;
  Tick forward_cost_ratio_denominator = 1;
  Tick backward_cost_ratio_numerator = 1;
  Tick backward_cost_ratio_denominator = 1;
  std::string communication_model = "constant_inter_worker_delay";
  Tick communication_ticks = 0;
  std::optional<double> communication_alpha;
  std::optional<double> communication_beta;
  std::optional<std::string> communication_payload;
  std::string fixed_schedule_rule = kFixedScheduleRuleBreadthFirst;
  std::string uniform_partition_rule = kUniformPartitionRule;
  std::optional<Index> alternating_max_rounds;
  std::optional<Index> alternating_completed_rounds;
  std::optional<std::string> alternating_convergence_reason;
  std::vector<CanonicalAlternatingTraceEntry> alternating_trace;
  std::optional<Tick> intermediate_partition_only_makespan;
  std::string method_contract_hash;

  CanonicalSemantics semantics;
  CanonicalOutcome outcome;

  std::optional<std::vector<Tick>> selected_partition;
  std::optional<std::vector<std::vector<std::string>>>
      worker_local_operation_order;
  std::vector<CanonicalWorkerPredecessor> derived_worker_predecessors;

  std::optional<std::string> git_commit;
  std::optional<bool> git_dirty;
  std::optional<std::string> git_dirty_scope;
  std::optional<std::string> build_type;
  std::optional<std::string> executable_name;
  std::optional<ActivationAnalysisResult> activation_analysis;
};

[[nodiscard]] std::string CurrentTimestampUtc();
[[nodiscard]] std::optional<std::string> CurrentGitCommit();
[[nodiscard]] std::optional<bool> CurrentGitDirty(
    std::optional<std::string> *scope = nullptr);
[[nodiscard]] std::optional<std::string> CurrentBuildType();
[[nodiscard]] bool OrToolsCompiledIn();
[[nodiscard]] std::optional<std::string> OrToolsVersion();
[[nodiscard]] bool CumulativeConstraintCompiledIn();
[[nodiscard]] bool VariableCumulativeDemandCompiledIn();
[[nodiscard]] std::string ActivationCapSolverSupportLevelForBuild();
[[nodiscard]] std::string CommandLineFromArgv(int argc, char **argv);
[[nodiscard]] std::string ExecutableNameFromArgv0(const std::string &argv0);
[[nodiscard]] std::string MakeRunId(const std::string &seed);
[[nodiscard]] bool IsTerminalFeasibleStatus(const std::string &status);
[[nodiscard]] std::optional<double> RelativeGap(Tick makespan,
                                                double best_bound);

[[nodiscard]] std::string CanonicalMethodForSlackPipeMode(
    SlackPipeSplitMode mode);
[[nodiscard]] std::string PartitionDecisionForSlackPipeMode(
    SlackPipeSplitMode mode);
[[nodiscard]] bool SlackPipeModeFixesFullPartition(SlackPipeSplitMode mode);
[[nodiscard]] bool SlackPipeModeFixesWorkerAggregateLoads(
    SlackPipeSplitMode mode);
[[nodiscard]] bool SlackPipeModeOptimizesPartition(SlackPipeSplitMode mode);

[[nodiscard]] CanonicalResultMetadata BuildCanonicalResultMetadata(
    const Instance &instance, const CanonicalRequestContext &request,
    const CanonicalSemantics &semantics, const CanonicalOutcome &outcome,
    const std::optional<std::vector<Tick>> &selected_partition,
    const std::optional<MachineOrders> &worker_orders);

[[nodiscard]] ResultValidationInput ValidationInputFromCanonical(
    const Instance &instance, const ScheduleSolution &schedule,
    const CanonicalResultMetadata &metadata);
void ApplyResultValidation(CanonicalOutcome &outcome,
                           const ResultValidationResult &validation);
void ApplyActivationAnalysis(CanonicalResultMetadata &metadata,
                             const ActivationAnalysisResult &analysis);

[[nodiscard]] CanonicalOutcome OutcomeFromSchedule(
    const ScheduleSolution &schedule, const std::string &status,
    std::optional<double> total_runtime_seconds = std::nullopt);
[[nodiscard]] CanonicalOutcome OutcomeFromBfsResult(
    const BfsSplitOptimizationResult &result);
[[nodiscard]] CanonicalOutcome OutcomeFromJointResult(
    const JointOptimizationResult &result);
[[nodiscard]] CanonicalOutcome OutcomeFromSlackPipeResult(
    const SlackPipeResult &result);

[[nodiscard]] CanonicalSemantics SemanticsForUniformFixedOrderBaseline();
[[nodiscard]] CanonicalSemantics SemanticsForUniformInterleavedOneFOneB();
[[nodiscard]] CanonicalSemantics SemanticsForPartitionOnlyFixedOrder(
    const std::string &method);
[[nodiscard]] CanonicalSemantics SemanticsForPartitionOnlyFixedOrder(
    const BfsSplitOptimizationResult &result);
[[nodiscard]] CanonicalSemantics SemanticsForSequentialPartitionThenSchedule(
    Index cp_sat_models_solved);
[[nodiscard]] CanonicalSemantics SemanticsForAlternatingPartitionSchedule(
    Index cp_sat_models_solved);
[[nodiscard]] CanonicalSemantics SemanticsForBfsEvaluate();
[[nodiscard]] CanonicalSemantics SemanticsForPartitionOnlyBfs(
    const std::string &bfs_method);
[[nodiscard]] CanonicalSemantics SemanticsForScheduleOnlyFixedSplit(
    const std::string &partition_reference_source,
    const JointOptimizationResult &result,
    bool predecessor_restriction_requested);
[[nodiscard]] CanonicalSemantics SemanticsForJointUnrestrictedNoOverlap(
    const JointOptimizationResult &result,
    bool predecessor_restriction_requested);
[[nodiscard]] CanonicalSemantics SemanticsForSlackPipe(
    SlackPipeSplitMode split_mode, const SlackPipeResult &result,
    bool predecessor_restriction_requested);

[[nodiscard]] std::string CanonicalResultToJson(
    const CanonicalResultMetadata &metadata, const std::string &indent);
[[nodiscard]] std::string CanonicalResultTopLevelJsonFields(
    const CanonicalResultMetadata &metadata, const std::string &indent,
    bool trailing_comma);

}  // namespace slackpipe
