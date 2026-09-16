#pragma once

#include <optional>
#include <string>
#include <vector>

#include "slackpipe/instance.h"
#include "slackpipe/schedule.h"

namespace slackpipe {

inline constexpr int kActivationAnalysisVersion = 1;
inline constexpr int kActivationCapFormulationVersion = 1;

enum class ActivationModel {
  kCount,
  kLinearInStageLayers,
  kExplicitStageUnits,
};

enum class ActivationCapMode {
  kNone,
  kExplicit,
  kUniformBaseline,
};

enum class ActivationCapSolverSupportLevel {
  kNone,
  kFixedDemandsOnly,
  kVariableDemands,
};

struct ActivationUniformBaseline {
  std::vector<Tick> partition;
  std::vector<Tick> cap_units_per_worker;
  std::string baseline_run_id;
  std::string cap_derivation_hash;
  std::string method_contract_hash;
  double derivation_runtime_seconds = 0.0;
  Tick maximum_worker_peak_units = 0;
  Tick global_simultaneous_peak_units = 0;
  std::vector<std::string> warnings;
  std::vector<std::string> errors;
};

struct ActivationAnalysisOptions {
  ActivationModel model = ActivationModel::kLinearInStageLayers;
  Tick activation_units_per_layer = 1;
  std::vector<Tick> explicit_stage_activation_units;
  std::optional<double> activation_bytes_per_unit;
  ActivationCapMode cap_mode = ActivationCapMode::kNone;
  std::vector<Tick> activation_cap_units;
  bool enforce_activation_cap = false;
  bool emit_activation_trace = false;
  std::optional<ActivationUniformBaseline> uniform_baseline;
};

struct ActivationCapConstraintMetadata {
  bool constraints_added = false;
  bool enforced_by_enumeration = false;
  Index retained_interval_count = 0;
  Index cumulative_constraint_count = 0;
  Index variable_demand_count = 0;
  Index fixed_demand_count = 0;
  std::vector<Index> workers_with_constraints;
  std::string unsupported_reason;
  std::string model_support_level = "none";
  bool solver_supported = false;
  bool incumbent_rejected_for_activation_cap = false;
  double build_runtime_seconds = 0.0;
};

struct ActivationCapModelTerm {
  Index microbatch = 0;
  Index stage = 0;
  Index worker = 0;
  std::string start_variable_name;
  std::string end_variable_name;
  std::string size_variable_name;
  std::string demand_type;
  std::string demand_source;
  Tick capacity_units = 0;
  bool included_in_cumulative = false;
};

struct ActivationCapModelDebugDump {
  std::vector<ActivationCapModelTerm> terms;
};

struct ActivationIdentity {
  Index microbatch = 0;
  Index stage = 0;
  Index worker = 0;
  OperationId forward_operation_id;
  OperationId backward_operation_id;
};

struct ActivationLifetime {
  ActivationIdentity identity;
  Tick start = 0;
  Tick end = 0;
  Tick demand_units = 0;
  bool zero_length = false;
};

struct ActivationEventTraceEntry {
  Tick timestamp = 0;
  std::string event_type;
  ActivationIdentity identity;
  Tick demand_units = 0;
  Index live_activation_count_after_event = 0;
  Tick live_activation_units_after_event = 0;
};

struct ActivationCapViolation {
  Index worker = 0;
  Tick cap_units = 0;
  Tick observed_peak_units = 0;
  Tick excess_units = 0;
  Tick first_violating_timestamp = 0;
  std::vector<ActivationIdentity> live_activations;
};

struct ActivationWorkerMetrics {
  Index worker = 0;
  Index peak_live_activation_count = 0;
  Tick peak_activation_units = 0;
  std::optional<double> peak_activation_bytes;
  Tick time_of_peak = 0;
  double average_live_activation_count = 0.0;
  double average_activation_units = 0.0;
  Tick activation_count_time_area = 0;
  Tick activation_unit_time_area = 0;
  std::optional<Tick> cap_units;
  std::optional<bool> cap_satisfied;
};

struct ActivationGlobalMetrics {
  Index maximum_worker_peak_activation_count = 0;
  Tick maximum_worker_peak_activation_units = 0;
  std::optional<double> maximum_worker_peak_activation_bytes;
  Index peak_simultaneous_activation_count_across_workers = 0;
  Tick peak_simultaneous_activation_units_across_workers = 0;
  std::optional<double> peak_simultaneous_activation_bytes_across_workers;
  Tick time_of_global_peak = 0;
  Tick total_activation_unit_time_area = 0;
};

struct ActivationPeakRatioToUniform {
  std::vector<std::optional<double>> per_worker_peak_units_ratio;
  std::optional<double> maximum_worker_peak_units_ratio;
  std::optional<double> global_simultaneous_peak_units_ratio;
  std::vector<std::string> warnings;
};

struct ActivationAnalysisResult {
  int activation_analysis_version = kActivationAnalysisVersion;
  ActivationModel model = ActivationModel::kLinearInStageLayers;
  Tick activation_units_per_layer = 1;
  std::vector<Tick> explicit_stage_activation_units;
  std::optional<double> activation_bytes_per_unit;

  Index total_activation_lifetimes = 0;
  Index zero_length_activation_lifetimes = 0;
  std::vector<ActivationLifetime> lifetimes;
  std::vector<ActivationWorkerMetrics> per_worker;
  ActivationGlobalMetrics global;

  ActivationCapMode cap_mode = ActivationCapMode::kNone;
  std::vector<Tick> activation_cap_units_per_worker;
  std::string activation_cap_source = "none";
  std::optional<bool> activation_cap_satisfied;
  bool activation_cap_enforced = false;
  bool activation_cap_enforced_in_solver = false;
  bool activation_cap_enforced_by_enumeration = false;
  bool activation_cap_enforcement_requested = false;
  std::string activation_cap_enforcement_mode = "none";
  bool activation_cap_solver_supported = false;
  std::string activation_cap_solver_support_level = "none";
  bool activation_cap_constraints_added = false;
  Index activation_retained_interval_count = 0;
  Index activation_cumulative_constraint_count = 0;
  Index activation_variable_demand_count = 0;
  Index activation_fixed_demand_count = 0;
  std::vector<Index> activation_workers_with_constraints;
  std::string activation_cap_unsupported_reason;
  double activation_constraint_build_runtime_seconds = 0.0;
  bool incumbent_rejected_for_activation_cap = false;
  std::optional<bool> activation_model_validation_agreement;
  std::string activation_model_disagreement_details;
  int activation_cap_formulation_version = kActivationCapFormulationVersion;
  std::optional<ActivationCapViolation> activation_cap_violation;
  std::string activation_baseline_run_id;
  std::string activation_cap_derivation_hash;
  std::vector<Tick> activation_baseline_partition;
  std::string activation_baseline_method_contract_hash;
  double activation_cap_derivation_runtime_seconds = 0.0;
  ActivationPeakRatioToUniform activation_peak_ratio_to_uniform;

  std::vector<ActivationEventTraceEntry> event_trace;
  std::vector<std::string> warnings;
  std::vector<std::string> errors;
  bool passed = true;
};

[[nodiscard]] std::string ToString(ActivationModel model);
[[nodiscard]] std::string ToCliString(ActivationModel model);
[[nodiscard]] ActivationModel ParseActivationModel(const std::string &text);
[[nodiscard]] std::string ToString(ActivationCapMode mode);
[[nodiscard]] std::string ToCliString(ActivationCapMode mode);
[[nodiscard]] ActivationCapMode ParseActivationCapMode(const std::string &text);
[[nodiscard]] std::string ToString(ActivationCapSolverSupportLevel level);
[[nodiscard]] bool ActivationCapCumulativeConstraintSupported();
[[nodiscard]] bool ActivationCapVariableCumulativeDemandSupported();
[[nodiscard]] ActivationCapSolverSupportLevel ActivationCapSolverSupport();
[[nodiscard]] bool ActivationCapSolverCanEnforce(
    const ActivationAnalysisOptions &options, bool partition_optimized);
[[nodiscard]] std::string ActivationCapSolverUnsupportedReason(
    const ActivationAnalysisOptions &options, bool partition_optimized);

void ValidateActivationOptions(const Instance &instance,
                               const ActivationAnalysisOptions &options);
[[nodiscard]] std::vector<Tick> ResolveExplicitActivationCap(
    const Instance &instance, const std::vector<Tick> &cap_units);
[[nodiscard]] std::vector<Tick> ResolveActivationCapUnits(
    const Instance &instance, const ActivationAnalysisOptions &options);
[[nodiscard]] bool ActivationScheduleSatisfiesCap(
    const Instance &instance, const ScheduleSolution &schedule,
    const ActivationAnalysisOptions &options);
void ApplyActivationCapConstraintMetadata(
    ActivationAnalysisResult &analysis,
    const ActivationCapConstraintMetadata &metadata,
    const std::string &enforcement_mode = "");

[[nodiscard]] ActivationAnalysisResult AnalyzeActivationMemory(
    const Instance &instance, const ScheduleSolution &schedule,
    const ActivationAnalysisOptions &options,
    const std::optional<ActivationUniformBaseline> &uniform_baseline =
        std::nullopt,
    bool activation_cap_enforced_in_solver = false);
[[nodiscard]] ActivationAnalysisResult MakeActivationAnalysisMetadata(
    const Instance &instance, const ActivationAnalysisOptions &options);

[[nodiscard]] ActivationUniformBaseline DeriveUniformActivationBaseline(
    const Instance &instance, const ActivationAnalysisOptions &options);

[[nodiscard]] ActivationAnalysisResult
AnalyzeActivationMemoryWithUniformBaseline(
    const Instance &instance, const ScheduleSolution &schedule,
    const ActivationAnalysisOptions &options,
    bool activation_cap_enforced_in_solver = false);

}  // namespace slackpipe
