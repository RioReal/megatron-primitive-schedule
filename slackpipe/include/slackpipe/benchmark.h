#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "slackpipe/activation_analyzer.h"
#include "slackpipe/instance.h"
#include "slackpipe/result_schema.h"
#include "slackpipe/slackpipe_solver.h"

namespace slackpipe::benchmark {

enum class BenchmarkAlgorithm {
  kUniformFixedOrder,
  kOptimizedBfs,
  kJointCpSat,
  kPartitionOnly,
  kScheduleOnly,
  kSequentialPartitionThenSchedule,
  kAlternatingPartitionSchedule,
  kCanonicalSlackPipe,
};

enum class ExperimentMode {
  kLatency,
  kThroughput,
  kAlgorithmComparison,
};

struct InstanceSpec {
  std::string name;
  Instance instance;
};

struct AlgorithmSpec {
  BenchmarkAlgorithm algorithm = BenchmarkAlgorithm::kOptimizedBfs;
  std::string configuration = "default";
  std::string requested_method;
  std::string fixed_partition_source;
  std::string fixed_order_partition_backend = "cpsat";
  std::vector<Tick> fixed_split;
  std::optional<bool> use_bfs_hints;
  SlackPipeSplitMode split_mode = SlackPipeSplitMode::kGlobal;
  Index move_budget = 2;
  bool move_budget_provided = false;
  std::optional<Index> per_stage_delta;
  Index worker_move_budget = 2;
  bool worker_move_budget_provided = false;
  std::optional<Index> per_worker_delta;
  int alternating_max_rounds = kDefaultAlternatingMaxRounds;
};

struct BenchmarkConfig {
  std::vector<InstanceSpec> instances;
  std::vector<AlgorithmSpec> algorithms;
  ExperimentMode mode = ExperimentMode::kAlgorithmComparison;
  std::vector<int> cp_sat_workers{1};
  int repetitions = 1;
  int warmups = 0;
  int parallel_instances = 1;
  int configurable_cpu_budget = 0;
  bool allow_oversubscription = false;
  bool derive_schedule_only_best = true;
  double timeout_seconds = 1.0;
  int random_seed_base = 1;
  bool resume = false;
  std::string requested_command;
  std::string executable_name;
  ActivationAnalysisOptions activation_options;
};

struct GapMetrics {
  bool joint_optimum_when_available = false;
  double absolute_gap_to_joint = 0.0;
  double relative_gap_to_joint = 0.0;
  bool reaches_joint_objective = false;
};

struct SummaryStats {
  int count = 0;
  int successful_count = 0;
  int timeout_count = 0;
  int optimal_count = 0;
  double median = 0.0;
  double mean = 0.0;
  double standard_deviation = 0.0;
  double minimum = 0.0;
  double maximum = 0.0;
  double p25 = 0.0;
  double p75 = 0.0;
  double p90 = 0.0;
  double geometric_mean_positive_runtime_ratio = 0.0;
};

struct BenchmarkRow {
  std::string run_id;
  std::string instance_name;
  BenchmarkAlgorithm algorithm = BenchmarkAlgorithm::kOptimizedBfs;
  std::string configuration;
  int repetition = 0;
  int seed = 0;
  int cp_sat_workers = 1;
  std::string status;
  std::string ablation_mode;
  std::string split_mode;
  std::string fixed_partition_source;
  std::string supplied_split;
  Index worker_move_budget = 0;
  std::optional<Index> per_worker_delta;
  bool completed = false;
  bool proven_optimal = false;
  Tick makespan = 0;
  Tick simulated_iteration_time = 0;
  double pipeline_utilization = 0.0;
  std::string per_worker_busy_time;
  std::string per_worker_idle_time;
  Tick maximum_worker_load = 0;
  Tick pipeline_fill_time = 0;
  Tick pipeline_drain_time = 0;
  std::optional<Tick> communication_blocked_time;
  Tick initial_objective = 0;
  Tick first_feasible_objective = 0;
  Tick final_objective = 0;
  double objective = 0.0;
  double best_bound = 0.0;
  double certified_gap = 0.0;
  std::string split;
  std::string worker_local_operation_order;
  bool canonical_validation_passed = false;
  bool partition_search_enabled = false;
  bool schedule_search_enabled = false;
  bool partition_changed = false;
  bool operation_order_changed = false;
  bool partition_fixed_validation_passed = false;
  bool operation_order_fixed_validation_passed = false;
  bool label_validation_passed = false;
  bool hints_requested = false;
  bool hints_effective = false;
  std::string hint_source;
  std::string hint_scope;
  bool hint_complete_for_basic_model = false;
  bool hint_complete_for_full_model = false;
  Index hinted_layer_variable_count = 0;
  Index hinted_operation_variable_count = 0;
  Index hinted_scalar_variable_count = 0;
  Index hinted_auxiliary_variable_count = 0;
  Index hinted_total_variable_count = 0;
  bool fallback_available = false;
  bool fallback_used = false;
  std::string fallback_source;
  std::string solution_source;
  std::string bfs_incumbent_method_requested;
  std::string bfs_incumbent_method_effective;
  std::string incumbent_method_requested;
  std::string incumbent_method_effective;
  std::string incumbent_source;
  bool incumbent_feasible = false;
  Tick incumbent_primary_objective = 0;
  double incumbent_hybrid_min_slack = 0.0;
  Tick incumbent_baseline_primary_objective = 0;
  double incumbent_baseline_hybrid_min_slack = 0.0;
  bool incumbent_improved_over_baseline = false;
  std::string incumbent_hybrid_stage_scores;
  std::string incumbent_hybrid_bottleneck_stages;
  std::string horizon_source;
  double hint_budget_seconds = 0.0;
  double hint_elapsed_seconds = 0.0;
  Index hint_iterations = 0;
  Index hint_candidates_generated = 0;
  Index hint_candidates_simulated = 0;
  Index hint_partition_moves_accepted = 0;
  Index hint_interleaving_moves_accepted = 0;
  bool hint_deadline_reached = false;
  std::string hint_termination_reason;
  bool utilization_ranking_validation_passed = false;
  bool joint_optimum_when_available = false;
  double absolute_gap_to_joint = 0.0;
  double relative_gap_to_joint = 0.0;
  double absolute_gap_to_global = 0.0;
  double relative_gap_to_global = 0.0;
  bool reaches_joint_objective = false;
  bool proven_global_optimal = false;
  std::string global_certificate;
  double external_total_seconds = 0.0;
  double model_build_seconds = 0.0;
  double external_solver_seconds = 0.0;
  double ortools_wall_time_seconds = 0.0;
  double cp_sat_solve_seconds = 0.0;
  double solver_internal_wall_seconds = 0.0;
  double solver_deterministic_time = 0.0;
  double extraction_verification_seconds = 0.0;
  double extraction_seconds = 0.0;
  double canonicalization_seconds = 0.0;
  double policy_seconds = 0.0;
  double serialization_seconds = 0.0;
  double bfs_seconds = 0.0;
  double uniform_bfs_seconds = 0.0;
  double optimized_bfs_seconds = 0.0;
  double time_to_first_feasible_seconds = 0.0;
  double time_to_best_incumbent_seconds = 0.0;
  Index incumbent_improvement_count = 0;
  std::string baseline_worker_layers;
  std::string final_worker_layers;
  Tick worker_balance_l1 = 0;
  Tick worker_balance_max_deviation = 0;
  Tick stage_split_l1 = 0;
  Tick stage_split_max_deviation = 0;
  double orchestration_seconds = 0.0;
  double total_planning_seconds = 0.0;
  double time_to_first_joint_objective_seconds = 0.0;
  double time_to_global_certificate_seconds = 0.0;
  long process_cpu_micros = 0;
  long peak_rss_kb = 0;
  std::string activation_model;
  Tick activation_units_per_layer = 1;
  std::string explicit_stage_activation_units;
  std::string activation_cap_mode;
  std::string activation_cap_units_per_worker;
  bool activation_cap_enforced = false;
  bool activation_cap_enforced_in_solver = false;
  bool activation_cap_enforcement_requested = false;
  std::string activation_cap_enforcement_mode;
  bool activation_cap_solver_supported = false;
  std::string activation_cap_solver_support_level;
  bool activation_cap_constraints_added = false;
  Index activation_retained_interval_count = 0;
  Index activation_cumulative_constraint_count = 0;
  Index activation_variable_demand_count = 0;
  Index activation_fixed_demand_count = 0;
  double activation_constraint_build_runtime_seconds = 0.0;
  bool incumbent_rejected_for_activation_cap = false;
  std::optional<bool> activation_model_validation_agreement;
  std::string activation_model_disagreement_details;
  int activation_cap_formulation_version = kActivationCapFormulationVersion;
  std::optional<bool> activation_cap_satisfied;
  Tick maximum_worker_peak_activation_units = 0;
  Tick global_simultaneous_peak_activation_units = 0;
  std::optional<double> activation_peak_ratio_to_uniform;
  std::string activation_cap_derivation_hash;
  Index cp_sat_models_solved = 0;
  double model_build_seconds_per_model = 0.0;
  double solver_seconds_per_model = 0.0;
  Index branches = 0;
  Index conflicts = 0;
  std::string failure;
  CanonicalResultMetadata canonical;
};

[[nodiscard]] std::string ToString(BenchmarkAlgorithm algorithm);
[[nodiscard]] BenchmarkAlgorithm ParseBenchmarkAlgorithm(
    const std::string &text);
[[nodiscard]] std::string ToString(ExperimentMode mode);
[[nodiscard]] ExperimentMode ParseExperimentMode(const std::string &text);

[[nodiscard]] BenchmarkConfig ParseBenchmarkConfigFile(const std::string &path);
[[nodiscard]] std::vector<InstanceSpec> GenerateNamedSuite(
    const std::string &suite);
[[nodiscard]] std::string DeterministicRunId(
    const InstanceSpec &instance, const AlgorithmSpec &algorithm,
    int repetition, int seed, int workers, double timeout_seconds,
    const ActivationAnalysisOptions &activation_options = {});
[[nodiscard]] GapMetrics CalculateGap(std::optional<Tick> joint_optimum,
                                      Tick candidate);
void ValidateCpuBudget(const BenchmarkConfig &config, int logical_cpus);
[[nodiscard]] std::map<std::string, bool> LoadCompleteRunIds(
    const std::string &results_jsonl_path);
void RecoverIncrementalOutputs(const std::string &output_dir);
[[nodiscard]] SummaryStats ComputeSummaryStats(
    const std::vector<BenchmarkRow> &rows);
[[nodiscard]] std::map<std::string, double> ComputePairedRuntimeRatios(
    const std::vector<BenchmarkRow> &rows, BenchmarkAlgorithm numerator,
    BenchmarkAlgorithm denominator);
[[nodiscard]] std::string HostMetadataJson();
[[nodiscard]] std::string ResultsCsvHeader();
[[nodiscard]] std::string ToCsvRow(const BenchmarkRow &row);
[[nodiscard]] std::string ToJsonLine(const BenchmarkRow &row);
[[nodiscard]] std::vector<BenchmarkRow> RunBenchmark(
    const BenchmarkConfig &config, const std::string &output_dir);

}  // namespace slackpipe::benchmark
