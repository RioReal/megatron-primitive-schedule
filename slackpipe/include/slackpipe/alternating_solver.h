#pragma once

#include <optional>
#include <string>
#include <vector>

#include "slackpipe/bfs_solver.h"
#include "slackpipe/joint_solver.h"
#include "slackpipe/result_schema.h"

namespace slackpipe {

struct AlternatingOptimizerOptions {
  double time_limit_seconds = 0.0;
  int num_workers = 1;
  int random_seed = 1;
  bool require_optimal = false;
  bool log_search_progress = false;
  std::uint64_t enumeration_threshold = 100000;
  bool use_bfs_hints = true;
  bool fifo_ordering = true;
  bool symmetry_break_f0_fifo = true;
  std::string fixed_order_partition_backend = "cpsat";
  int max_rounds = kDefaultAlternatingMaxRounds;
  ActivationAnalysisOptions activation_options;
};

struct AlternatingOptimizationResult {
  std::string method;
  std::vector<Tick> split;
  MachineOrders machine_orders;
  ScheduleSolution schedule;
  Tick makespan_ticks = 0;
  double best_bound_ticks = 0.0;
  std::string status;
  std::string solver_status_raw;
  bool proven_optimal = false;
  double wall_time_seconds = 0.0;
  SolverPhaseTiming timing;
  Index cp_sat_models_solved = 0;
  Tick initial_makespan = 0;
  std::vector<Tick> initial_split;
  Tick intermediate_partition_only_makespan = 0;
  int alternating_max_rounds = kDefaultAlternatingMaxRounds;
  int alternating_completed_rounds = 0;
  std::string alternating_convergence_reason;
  std::vector<CanonicalAlternatingTraceEntry> alternating_trace;
  CanonicalPhaseBudgetSummary phase_budget;
  bool fallback_used = false;
  std::string fallback_reason;
  std::string returned_solution_source = "none";
  std::string diagnostic;
  ActivationCapConstraintMetadata activation_cap_constraints;
  std::optional<CanonicalResultMetadata> canonical;
};

[[nodiscard]] AlternatingOptimizationResult
OptimizeSequentialPartitionThenSchedule(
    const Instance &instance, const AlternatingOptimizerOptions &options = {});
[[nodiscard]] AlternatingOptimizationResult
OptimizeAlternatingPartitionSchedule(
    const Instance &instance, const AlternatingOptimizerOptions &options = {});
[[nodiscard]] CanonicalOutcome OutcomeFromAlternatingResult(
    const AlternatingOptimizationResult &result);
void ApplyAlternatingCanonicalFields(
    const AlternatingOptimizationResult &result,
    CanonicalResultMetadata &metadata);

}  // namespace slackpipe
