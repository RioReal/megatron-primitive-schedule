#pragma once

#include <optional>
#include <string>
#include <vector>

#include "slackpipe/instance.h"
#include "slackpipe/result_validator.h"

namespace slackpipe {

inline constexpr int kEvaluationMethodVersion = 1;
inline constexpr const char* kUniformBreadthFirstMethod =
    "uniform-breadth-first";
inline constexpr const char* kUniformInterleavedOneFOneBMethod =
    "uniform-interleaved-1f1b";
inline constexpr const char* kFixedScheduleRuleBreadthFirst =
    "breadth_first_wavefront_microbatch_plus_position";
inline constexpr const char* kFixedScheduleRuleInterleavedOneFOneB =
    "interleaved_1f1b_stage_local_streams";
inline constexpr const char* kUniformPartitionRule =
    "as_even_as_possible_remainder_to_lowest_stage";
inline constexpr int kDefaultAlternatingMaxRounds = 4;

struct EvaluationMethodDefinition {
  std::string canonical_name;
  std::vector<std::string> legacy_aliases;
  std::string partition_decision;
  std::string schedule_decision;
  std::vector<std::string> fixed_variables;
  std::vector<std::string> optimized_variables;
  bool partition_optimized = false;
  bool schedule_optimized = false;
  bool requires_ortools = false;
  bool deterministic = false;
  bool validation_required = true;
  std::string phase_policy;
  std::string deadline_policy;
  std::string schedule_semantics;
  std::string partition_source;
  std::string reference;
};

struct MethodCompatibilityRecord {
  std::string method_name;
  Index micro_batches = 0;
  Index logical_stages = 0;
  Index physical_workers = 0;
  Index total_layers = 0;
  Index min_layers = 1;
  std::string mapping_type = "cyclic_stage_mod_worker";
  Tick forward_cost_ratio_numerator = 1;
  Tick forward_cost_ratio_denominator = 1;
  Tick backward_cost_ratio_numerator = 1;
  Tick backward_cost_ratio_denominator = 1;
  std::string communication_model = "constant_inter_worker_delay";
  Tick communication_ticks = 0;
  std::optional<double> requested_time_limit_seconds;
  std::optional<int> solver_threads;
  int budget_policy_version = 1;
  int validation_version = kResultValidationVersion;
  std::optional<int> random_seed;
  bool solver_backed = false;
};

struct MethodCompatibilityCheck {
  bool passed = false;
  std::string message;
};

[[nodiscard]] const std::vector<EvaluationMethodDefinition>&
EvaluationMethodRegistry();
[[nodiscard]] const EvaluationMethodDefinition* FindEvaluationMethod(
    const std::string& name);
[[nodiscard]] std::string CanonicalizeEvaluationMethodName(
    const std::string& name);
[[nodiscard]] std::string EvaluationMethodContractHash(
    const EvaluationMethodDefinition& definition);
[[nodiscard]] std::string DescribeEvaluationMethodJson(const std::string& name);
[[nodiscard]] MethodCompatibilityCheck CheckMethodCompatibility(
    const std::vector<MethodCompatibilityRecord>& records,
    bool seeds_are_repetitions);
[[nodiscard]] double AlternatingPhaseLimitSeconds(double remaining_seconds,
                                                  int remaining_planned_phases);

}  // namespace slackpipe
