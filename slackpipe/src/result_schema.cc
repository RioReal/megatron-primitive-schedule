#include "slackpipe/result_schema.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <optional>
#include <sstream>

#include "slackpipe/bfs_solver.h"
#include "slackpipe/joint_solver.h"
#include "slackpipe/operation.h"
#include "slackpipe/slackpipe_solver.h"

#ifndef SLACKPIPE_BUILD_GIT_COMMIT
#define SLACKPIPE_BUILD_GIT_COMMIT ""
#endif

#ifndef SLACKPIPE_BUILD_GIT_DIRTY
#define SLACKPIPE_BUILD_GIT_DIRTY ""
#endif

#ifndef SLACKPIPE_HAVE_ORTOOLS
#define SLACKPIPE_HAVE_ORTOOLS 0
#endif

#ifndef SLACKPIPE_ORTOOLS_VERSION
#define SLACKPIPE_ORTOOLS_VERSION ""
#endif

#ifndef SLACKPIPE_CUMULATIVE_CONSTRAINT_SUPPORTED
#define SLACKPIPE_CUMULATIVE_CONSTRAINT_SUPPORTED 0
#endif

#ifndef SLACKPIPE_VARIABLE_CUMULATIVE_DEMAND_SUPPORTED
#define SLACKPIPE_VARIABLE_CUMULATIVE_DEMAND_SUPPORTED 0
#endif

#ifndef SLACKPIPE_HAVE_CUMULATIVE
#define SLACKPIPE_HAVE_CUMULATIVE SLACKPIPE_CUMULATIVE_CONSTRAINT_SUPPORTED
#endif

#ifndef SLACKPIPE_HAVE_VARIABLE_CUMULATIVE_DEMAND
#define SLACKPIPE_HAVE_VARIABLE_CUMULATIVE_DEMAND \
  SLACKPIPE_VARIABLE_CUMULATIVE_DEMAND_SUPPORTED
#endif

namespace slackpipe {

namespace {

std::optional<std::string> NonEmpty(std::string value) {
  if (value.empty()) return std::nullopt;
  return value;
}

std::optional<std::string> EnvString(const char *name) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0') return std::nullopt;
  return std::string(value);
}

std::optional<bool> ParseBoolText(const std::string &value) {
  if (value == "true" || value == "1" || value == "dirty") return true;
  if (value == "false" || value == "0" || value == "clean") return false;
  return std::nullopt;
}

std::optional<double> PositiveOrNull(double value) {
  return value > 0.0 ? std::optional<double>(value) : std::nullopt;
}

std::optional<Tick> PositiveOrNull(Tick value) {
  return value > 0 ? std::optional<Tick>(value) : std::nullopt;
}

std::optional<double> BoundOrNull(double value) {
  return value > 0.0 ? std::optional<double>(value) : std::nullopt;
}

std::string JsonEscape(const std::string &text) {
  std::ostringstream out;
  for (char ch : text) {
    switch (ch) {
      case '"':
      case '\\':
        out << '\\' << ch;
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << ch;
        break;
    }
  }
  return out.str();
}

void WriteJsonString(std::ostringstream &out, const std::string &value) {
  out << "\"" << JsonEscape(value) << "\"";
}

template <typename T>
void WriteOptionalNumber(std::ostringstream &out,
                         const std::optional<T> &value) {
  if (value) {
    out << *value;
  } else {
    out << "null";
  }
}

void WriteOptionalBool(std::ostringstream &out,
                       const std::optional<bool> &value) {
  if (value) {
    out << (*value ? "true" : "false");
  } else {
    out << "null";
  }
}

void WriteOptionalString(std::ostringstream &out,
                         const std::optional<std::string> &value) {
  if (value) {
    WriteJsonString(out, *value);
  } else {
    out << "null";
  }
}

void WritePhaseBudget(std::ostringstream &out,
                      const CanonicalPhaseBudgetSummary &budget,
                      const std::string &indent) {
  const std::string nested = indent + "  ";
  out << "{\n";
  out << nested << "\"reference_phase_limit_seconds\": ";
  WriteOptionalNumber(out, budget.reference_phase_limit_seconds);
  out << ",\n";
  out << nested << "\"schedule_solver_effective_limit_seconds\": ";
  WriteOptionalNumber(out, budget.schedule_solver_effective_limit_seconds);
  out << ",\n";
  out << nested << "\"phases\": [";
  for (std::size_t i = 0; i < budget.phases.size(); ++i) {
    if (i != 0) out << ",";
    const CanonicalPhaseBudget &phase = budget.phases[i];
    out << "\n" << nested << "  {\n";
    out << nested << "    \"phase\": ";
    WriteJsonString(out, phase.phase);
    out << ",\n";
    out << nested << "    \"requested_limit_seconds\": ";
    WriteOptionalNumber(out, phase.requested_limit_seconds);
    out << ",\n";
    out << nested << "    \"effective_limit_seconds\": ";
    WriteOptionalNumber(out, phase.effective_limit_seconds);
    out << ",\n";
    out << nested << "    \"remaining_before_seconds\": ";
    WriteOptionalNumber(out, phase.remaining_before_seconds);
    out << ",\n";
    out << nested << "    \"remaining_after_seconds\": ";
    WriteOptionalNumber(out, phase.remaining_after_seconds);
    out << ",\n";
    out << nested << "    \"runtime_seconds\": ";
    WriteOptionalNumber(out, phase.runtime_seconds);
    out << ",\n";
    out << nested << "    \"expired_before_start\": "
        << (phase.expired_before_start ? "true" : "false") << ",\n";
    out << nested << "    \"status\": ";
    WriteOptionalString(out, phase.status);
    out << "\n" << nested << "  }";
  }
  if (!budget.phases.empty()) out << "\n" << nested;
  out << "]\n" << indent << "}";
}

void WriteAlternatingTrace(
    std::ostringstream &out,
    const std::vector<CanonicalAlternatingTraceEntry> &trace,
    const std::string &indent) {
  const std::string nested = indent + "  ";
  out << "[";
  for (std::size_t i = 0; i < trace.size(); ++i) {
    if (i != 0) out << ",";
    const CanonicalAlternatingTraceEntry &entry = trace[i];
    out << "\n" << nested << "{\n";
    out << nested << "  \"round_index\": " << entry.round_index << ",\n";
    out << nested << "  \"phase_type\": ";
    WriteJsonString(out, entry.phase_type);
    out << ",\n";
    out << nested << "  \"phase_limit_seconds\": ";
    WriteOptionalNumber(out, entry.phase_limit_seconds);
    out << ",\n";
    out << nested << "  \"phase_runtime_seconds\": ";
    WriteOptionalNumber(out, entry.phase_runtime_seconds);
    out << ",\n";
    out << nested << "  \"input_makespan\": ";
    WriteOptionalNumber(out, entry.input_makespan);
    out << ",\n";
    out << nested << "  \"candidate_makespan\": ";
    WriteOptionalNumber(out, entry.candidate_makespan);
    out << ",\n";
    out << nested << "  \"accepted\": " << (entry.accepted ? "true" : "false")
        << ",\n";
    out << nested << "  \"validation_passed\": "
        << (entry.validation_passed ? "true" : "false") << ",\n";
    out << nested << "  \"selected_partition\": ";
    out << "[";
    for (std::size_t j = 0; j < entry.selected_partition.size(); ++j) {
      if (j != 0) out << ", ";
      out << entry.selected_partition[j];
    }
    out << "]";
    out << ",\n";
    out << nested << "  \"solver_status_raw\": ";
    WriteOptionalString(out, entry.solver_status_raw);
    out << ",\n";
    out << nested
        << "  \"fallback_used\": " << (entry.fallback_used ? "true" : "false")
        << ",\n";
    out << nested << "  \"seed\": ";
    WriteOptionalNumber(out, entry.seed);
    out << ",\n";
    out << nested << "  \"solver_threads\": ";
    WriteOptionalNumber(out, entry.solver_threads);
    out << ",\n";
    out << nested << "  \"activation_cap_requested\": "
        << (entry.activation_cap_requested ? "true" : "false") << ",\n";
    out << nested << "  \"activation_cap_supported\": "
        << (entry.activation_cap_supported ? "true" : "false") << ",\n";
    out << nested << "  \"activation_cap_constraints_added\": "
        << (entry.activation_cap_constraints_added ? "true" : "false") << ",\n";
    out << nested << "  \"activation_cap_satisfied\": ";
    WriteOptionalBool(out, entry.activation_cap_satisfied);
    out << ",\n";
    out << nested << "  \"candidate_rejected_for_activation_cap\": "
        << (entry.candidate_rejected_for_activation_cap ? "true" : "false");
    out << "\n" << nested << "}";
  }
  if (!trace.empty()) out << "\n" << indent;
  out << "]";
}

CanonicalPhaseBudget MakePhaseBudget(const std::string &phase,
                                     std::optional<double> requested_limit,
                                     std::optional<double> effective_limit,
                                     std::optional<double> runtime,
                                     const std::string &status) {
  CanonicalPhaseBudget budget;
  budget.phase = phase;
  budget.requested_limit_seconds = requested_limit;
  budget.effective_limit_seconds = effective_limit;
  budget.runtime_seconds = runtime;
  budget.status = NonEmpty(status);
  return budget;
}

void WriteTickArray(std::ostringstream &out, const std::vector<Tick> &values) {
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ", ";
    out << values[i];
  }
  out << "]";
}

void WriteStringMatrix(std::ostringstream &out,
                       const std::vector<std::vector<std::string>> &values,
                       const std::string &indent) {
  out << "[\n";
  for (std::size_t w = 0; w < values.size(); ++w) {
    out << indent << "  [";
    for (std::size_t i = 0; i < values[w].size(); ++i) {
      if (i != 0) out << ", ";
      WriteJsonString(out, values[w][i]);
    }
    out << "]";
    if (w + 1 != values.size()) out << ",";
    out << "\n";
  }
  out << indent << "]";
}

void WriteOptionalTickArray(std::ostringstream &out,
                            const std::optional<std::vector<Tick>> &values) {
  if (!values) {
    out << "null";
    return;
  }
  WriteTickArray(out, *values);
}

void WriteOptionalStringMatrix(
    std::ostringstream &out,
    const std::optional<std::vector<std::vector<std::string>>> &values,
    const std::string &indent) {
  if (!values) {
    out << "null";
    return;
  }
  WriteStringMatrix(out, *values, indent);
}

void WriteOptionalDouble(std::ostringstream &out,
                         const std::optional<double> &value) {
  if (value) {
    out << *value;
  } else {
    out << "null";
  }
}

void WriteOptionalDoubleArray(
    std::ostringstream &out, const std::vector<std::optional<double>> &values) {
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ", ";
    WriteOptionalDouble(out, values[i]);
  }
  out << "]";
}

void WriteStringArray(std::ostringstream &out,
                      const std::vector<std::string> &values) {
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ", ";
    WriteJsonString(out, values[i]);
  }
  out << "]";
}

void WriteActivationIdentity(std::ostringstream &out,
                             const ActivationIdentity &identity) {
  out << "{\"microbatch\": " << identity.microbatch
      << ", \"stage\": " << identity.stage
      << ", \"worker\": " << identity.worker
      << ", \"forward_operation_id\": " << identity.forward_operation_id.value
      << ", \"backward_operation_id\": " << identity.backward_operation_id.value
      << "}";
}

void WriteActivationLifetime(std::ostringstream &out,
                             const ActivationLifetime &lifetime,
                             const std::string &indent) {
  out << "{\n";
  out << indent << "  \"identity\": ";
  WriteActivationIdentity(out, lifetime.identity);
  out << ",\n";
  out << indent << "  \"start\": " << lifetime.start << ",\n";
  out << indent << "  \"end\": " << lifetime.end << ",\n";
  out << indent << "  \"demand_units\": " << lifetime.demand_units << ",\n";
  out << indent
      << "  \"zero_length\": " << (lifetime.zero_length ? "true" : "false")
      << "\n";
  out << indent << "}";
}

void WriteActivationCapViolation(
    std::ostringstream &out,
    const std::optional<ActivationCapViolation> &violation,
    const std::string &indent) {
  if (!violation) {
    out << "null";
    return;
  }
  const std::string nested = indent + "  ";
  out << "{\n";
  out << nested << "\"worker\": " << violation->worker << ",\n";
  out << nested << "\"cap_units\": " << violation->cap_units << ",\n";
  out << nested << "\"observed_peak_units\": " << violation->observed_peak_units
      << ",\n";
  out << nested << "\"excess_units\": " << violation->excess_units << ",\n";
  out << nested << "\"first_violating_timestamp\": "
      << violation->first_violating_timestamp << ",\n";
  out << nested << "\"live_activations\": [";
  for (std::size_t i = 0; i < violation->live_activations.size(); ++i) {
    if (i != 0) out << ", ";
    WriteActivationIdentity(out, violation->live_activations[i]);
  }
  out << "]\n";
  out << indent << "}";
}

void WriteActivationMetrics(std::ostringstream &out,
                            const ActivationAnalysisResult &analysis,
                            const std::string &indent) {
  const std::string nested = indent + "  ";
  out << "{\n";
  out << nested << "\"total_activation_lifetimes\": "
      << analysis.total_activation_lifetimes << ",\n";
  out << nested << "\"zero_length_activation_lifetimes\": "
      << analysis.zero_length_activation_lifetimes << ",\n";
  out << nested << "\"per_worker\": [";
  for (std::size_t i = 0; i < analysis.per_worker.size(); ++i) {
    if (i != 0) out << ",";
    const ActivationWorkerMetrics &worker = analysis.per_worker[i];
    out << "\n" << nested << "  {\n";
    out << nested << "    \"worker\": " << worker.worker << ",\n";
    out << nested << "    \"peak_live_activation_count\": "
        << worker.peak_live_activation_count << ",\n";
    out << nested
        << "    \"peak_activation_units\": " << worker.peak_activation_units
        << ",\n";
    out << nested << "    \"peak_activation_bytes\": ";
    WriteOptionalDouble(out, worker.peak_activation_bytes);
    out << ",\n";
    out << nested << "    \"time_of_peak\": " << worker.time_of_peak << ",\n";
    out << nested << "    \"average_live_activation_count\": "
        << worker.average_live_activation_count << ",\n";
    out << nested << "    \"average_activation_units\": "
        << worker.average_activation_units << ",\n";
    out << nested << "    \"activation_count_time_area\": "
        << worker.activation_count_time_area << ",\n";
    out << nested << "    \"activation_unit_time_area\": "
        << worker.activation_unit_time_area << ",\n";
    out << nested << "    \"cap_units\": ";
    WriteOptionalNumber(out, worker.cap_units);
    out << ",\n";
    out << nested << "    \"cap_satisfied\": ";
    WriteOptionalBool(out, worker.cap_satisfied);
    out << "\n" << nested << "  }";
  }
  if (!analysis.per_worker.empty()) out << "\n" << nested;
  out << "],\n";
  out << nested << "\"global\": {\n";
  out << nested << "  \"maximum_worker_peak_activation_count\": "
      << analysis.global.maximum_worker_peak_activation_count << ",\n";
  out << nested << "  \"maximum_worker_peak_activation_units\": "
      << analysis.global.maximum_worker_peak_activation_units << ",\n";
  out << nested << "  \"maximum_worker_peak_activation_bytes\": ";
  WriteOptionalDouble(out,
                      analysis.global.maximum_worker_peak_activation_bytes);
  out << ",\n";
  out << nested << "  \"peak_simultaneous_activation_count_across_workers\": "
      << analysis.global.peak_simultaneous_activation_count_across_workers
      << ",\n";
  out << nested << "  \"peak_simultaneous_activation_units_across_workers\": "
      << analysis.global.peak_simultaneous_activation_units_across_workers
      << ",\n";
  out << nested << "  \"peak_simultaneous_activation_bytes_across_workers\": ";
  WriteOptionalDouble(
      out, analysis.global.peak_simultaneous_activation_bytes_across_workers);
  out << ",\n";
  out << nested
      << "  \"time_of_global_peak\": " << analysis.global.time_of_global_peak
      << ",\n";
  out << nested << "  \"total_activation_unit_time_area\": "
      << analysis.global.total_activation_unit_time_area << "\n";
  out << nested << "},\n";
  out << nested << "\"activation_lifetimes\": [";
  for (std::size_t i = 0; i < analysis.lifetimes.size(); ++i) {
    if (i != 0) out << ",";
    out << "\n" << nested << "  ";
    WriteActivationLifetime(out, analysis.lifetimes[i], nested + "  ");
  }
  if (!analysis.lifetimes.empty()) out << "\n" << nested;
  out << "],\n";
  out << nested << "\"event_trace\": [";
  for (std::size_t i = 0; i < analysis.event_trace.size(); ++i) {
    if (i != 0) out << ",";
    const ActivationEventTraceEntry &entry = analysis.event_trace[i];
    out << "\n" << nested << "  {\n";
    out << nested << "    \"timestamp\": " << entry.timestamp << ",\n";
    out << nested << "    \"event_type\": ";
    WriteJsonString(out, entry.event_type);
    out << ",\n";
    out << nested << "    \"activation_id\": ";
    WriteActivationIdentity(out, entry.identity);
    out << ",\n";
    out << nested << "    \"demand_units\": " << entry.demand_units << ",\n";
    out << nested << "    \"live_activation_count_after_event\": "
        << entry.live_activation_count_after_event << ",\n";
    out << nested << "    \"live_activation_units_after_event\": "
        << entry.live_activation_units_after_event << "\n";
    out << nested << "  }";
  }
  if (!analysis.event_trace.empty()) out << "\n" << nested;
  out << "],\n";
  out << nested << "\"warnings\": ";
  WriteStringArray(out, analysis.warnings);
  out << ",\n";
  out << nested << "\"errors\": ";
  WriteStringArray(out, analysis.errors);
  out << "\n" << indent << "}";
}

void WriteActivationPeakRatio(std::ostringstream &out,
                              const ActivationPeakRatioToUniform &ratio,
                              const std::string &indent) {
  const std::string nested = indent + "  ";
  out << "{\n";
  out << nested << "\"per_worker_peak_units_ratio\": ";
  WriteOptionalDoubleArray(out, ratio.per_worker_peak_units_ratio);
  out << ",\n";
  out << nested << "\"maximum_worker_peak_units_ratio\": ";
  WriteOptionalDouble(out, ratio.maximum_worker_peak_units_ratio);
  out << ",\n";
  out << nested << "\"global_simultaneous_peak_units_ratio\": ";
  WriteOptionalDouble(out, ratio.global_simultaneous_peak_units_ratio);
  out << ",\n";
  out << nested << "\"warnings\": ";
  WriteStringArray(out, ratio.warnings);
  out << "\n" << indent << "}";
}

std::vector<StageWorkerMappingEntry> BuildStageMapping(
    const Instance &instance) {
  std::vector<StageWorkerMappingEntry> mapping;
  mapping.reserve(static_cast<std::size_t>(instance.stages));
  for (Index s = 0; s < instance.stages; ++s) {
    mapping.push_back(StageWorkerMappingEntry{s, s % instance.workers});
  }
  return mapping;
}

std::vector<std::vector<std::string>> OperationOrderNames(
    const Instance &instance, const MachineOrders &orders) {
  std::vector<std::vector<std::string>> names;
  names.reserve(orders.size());
  for (const auto &order : orders) {
    std::vector<std::string> worker;
    worker.reserve(order.size());
    for (OperationId id : order) worker.push_back(OperationName(instance, id));
    names.push_back(std::move(worker));
  }
  return names;
}

std::vector<CanonicalWorkerPredecessor> DerivedPredecessors(
    const Instance &instance, const MachineOrders &orders, bool fifo_ordering) {
  const MachinePredecessors predecessors =
      ExtractMachinePredecessors(instance, orders, fifo_ordering);
  std::vector<CanonicalWorkerPredecessor> derived;
  derived.reserve(predecessors.size());
  for (const auto &[id_value, predecessor] : predecessors) {
    const OperationId id{id_value};
    const OperationView view = DecodeOperation(instance, id);
    derived.push_back(CanonicalWorkerPredecessor{
        id_value, OperationName(instance, id), predecessor.value,
        OperationName(instance, predecessor), view.worker});
  }
  return derived;
}

Index SemanticFifoConstraintCount(const Instance &instance,
                                  bool fifo_ordering) {
  if (!fifo_ordering || instance.microbatches <= 1) return 0;
  return (instance.microbatches - 1) * OperationPositionCount(instance);
}

void ApplyFifoDefaults(const Instance &instance,
                       CanonicalSemantics *semantics) {
  if (!semantics->fifo_ordering_requested) {
    semantics->fifo_ordering_requested = true;
  }
  if (!semantics->fifo_ordering_effective) {
    semantics->fifo_ordering_effective = *semantics->fifo_ordering_requested;
  }
  if (!semantics->fifo_constraint_count) {
    semantics->fifo_constraint_count = SemanticFifoConstraintCount(
        instance, semantics->fifo_ordering_effective.value_or(true));
  }
}

void ApplyFifoFromJointResult(CanonicalSemantics *semantics,
                              const JointOptimizationResult &result) {
  semantics->fifo_ordering_requested = result.fifo_ordering_requested;
  semantics->fifo_ordering_effective = result.fifo_ordering_effective;
  semantics->fifo_constraint_count = result.fifo_constraint_count;
}

void ApplyWorkerBalance(CanonicalSemantics &semantics,
                        const WorkerBalanceConstraintResult &constraint) {
  semantics.worker_balance_pruning_enabled = constraint.enabled;
  semantics.worker_balance_pruning_effective = constraint.enabled;
  if (!constraint.enabled) return;
  if (constraint.tolerance_percent >= 0.0) {
    semantics.worker_balance_tolerance_percent = constraint.tolerance_percent;
  }
  semantics.worker_balance_tolerance_layers = constraint.tolerance_layers;
  semantics.worker_balance_lower_bound = constraint.lower_bound;
  semantics.worker_balance_upper_bound = constraint.upper_bound;
}

std::optional<std::string> GitCommitValue() {
  if (const std::optional<std::string> env =
          EnvString("SLACKPIPE_GIT_COMMIT")) {
    return env;
  }
  return NonEmpty(SLACKPIPE_BUILD_GIT_COMMIT);
}

std::optional<bool> GitDirtyValue(std::optional<std::string> *scope) {
  if (const std::optional<std::string> env = EnvString("SLACKPIPE_GIT_DIRTY")) {
    if (const std::optional<bool> parsed = ParseBoolText(*env)) {
      if (scope != nullptr) *scope = std::string("runtime_environment");
      return parsed;
    }
  }
  const std::optional<std::string> built = NonEmpty(SLACKPIPE_BUILD_GIT_DIRTY);
  if (!built) return std::nullopt;
  if (const std::optional<bool> parsed = ParseBoolText(*built)) {
    if (scope != nullptr) *scope = std::string("build_time");
    return parsed;
  }
  return std::nullopt;
}

std::optional<std::string> BuildTypeValue() {
  if (const std::optional<std::string> env =
          EnvString("SLACKPIPE_BUILD_TYPE")) {
    return env;
  }
#ifdef NDEBUG
  return std::string("Release");
#else
  return std::string("Debug");
#endif
}

std::string SolverPathForBfsMethod(const std::string &method) {
  if (method.find("fixed-order") != std::string::npos) {
    if (method.find("enumerate") != std::string::npos) {
      return "fixed-order-partition-enumeration";
    }
    if (method.find("cpsat") != std::string::npos) {
      return "fixed-order-partition-cpsat";
    }
    return "fixed-order-partition-auto";
  }
  if (method.find("cpsat") != std::string::npos) {
    return "bfs-cpsat-fixed-breadth-first";
  }
  if (method.find("enumerate") != std::string::npos) {
    return "bfs-enumeration-fixed-breadth-first";
  }
  if (method.find("hybrid-slack") != std::string::npos) {
    return "hybrid-slack-incumbent-fixed-breadth-first";
  }
  if (method == "uniform" || method == "uniform-fallback") {
    return "uniform-fixed-breadth-first";
  }
  return "bfs-auto-fixed-breadth-first";
}

}  // namespace

std::string CurrentTimestampUtc() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t time = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
  gmtime_r(&time, &utc);
  std::ostringstream out;
  out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return out.str();
}

std::optional<std::string> CurrentGitCommit() { return GitCommitValue(); }

std::optional<bool> CurrentGitDirty(std::optional<std::string> *scope) {
  return GitDirtyValue(scope);
}

std::optional<std::string> CurrentBuildType() { return BuildTypeValue(); }

bool OrToolsCompiledIn() { return SLACKPIPE_HAVE_ORTOOLS != 0; }

std::optional<std::string> OrToolsVersion() {
  return NonEmpty(SLACKPIPE_ORTOOLS_VERSION);
}

bool CumulativeConstraintCompiledIn() {
  return OrToolsCompiledIn() && SLACKPIPE_HAVE_CUMULATIVE != 0;
}

bool VariableCumulativeDemandCompiledIn() {
  return CumulativeConstraintCompiledIn() &&
         SLACKPIPE_HAVE_VARIABLE_CUMULATIVE_DEMAND != 0;
}

std::string ActivationCapSolverSupportLevelForBuild() {
  if (!CumulativeConstraintCompiledIn()) {
    return "none";
  }
  if (VariableCumulativeDemandCompiledIn()) {
    return "variable_demands";
  }
  return "fixed_demands_only";
}

std::string CommandLineFromArgv(int argc, char **argv) {
  std::ostringstream out;
  for (int i = 0; i < argc; ++i) {
    if (i != 0) out << ' ';
    const std::string arg = argv[i] == nullptr ? "" : argv[i];
    const bool needs_quotes =
        arg.find_first_of(" \t\n\"'\\") != std::string::npos;
    if (!needs_quotes) {
      out << arg;
      continue;
    }
    out << '\'';
    for (char ch : arg) {
      if (ch == '\'') {
        out << "'\\''";
      } else {
        out << ch;
      }
    }
    out << '\'';
  }
  return out.str();
}

std::string ExecutableNameFromArgv0(const std::string &argv0) {
  const std::size_t slash = argv0.find_last_of("/\\");
  return slash == std::string::npos ? argv0 : argv0.substr(slash + 1);
}

std::string MakeRunId(const std::string &seed) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (unsigned char ch : seed) {
    hash ^= static_cast<std::uint64_t>(ch);
    hash *= 1099511628211ULL;
  }
  std::ostringstream out;
  out << "run-" << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}

bool IsTerminalFeasibleStatus(const std::string &status) {
  return status == "OPTIMAL" || status == "FEASIBLE";
}

std::optional<double> RelativeGap(Tick makespan, double best_bound) {
  if (makespan <= 0 || best_bound <= 0.0) return std::nullopt;
  return std::max(0.0, (static_cast<double>(makespan) - best_bound) /
                           static_cast<double>(makespan));
}

std::string CanonicalMethodForSlackPipeMode(SlackPipeSplitMode mode) {
  switch (mode) {
    case SlackPipeSplitMode::kFixed:
      return "canonical-slackpipe-fixed-split";
    case SlackPipeSplitMode::kLocal:
      return "canonical-slackpipe-stage-local";
    case SlackPipeSplitMode::kGlobal:
      return "canonical-slackpipe-global";
    case SlackPipeSplitMode::kWorkerFixed:
      return "canonical-slackpipe-worker-aggregate-fixed";
    case SlackPipeSplitMode::kWorkerLocal:
      return "canonical-slackpipe-worker-aggregate-local";
  }
  return "canonical-slackpipe-global";
}

std::string PartitionDecisionForSlackPipeMode(SlackPipeSplitMode mode) {
  switch (mode) {
    case SlackPipeSplitMode::kFixed:
      return "fixed_supplied";
    case SlackPipeSplitMode::kLocal:
      return "optimized_stage_local";
    case SlackPipeSplitMode::kGlobal:
      return "optimized_global";
    case SlackPipeSplitMode::kWorkerFixed:
      return "optimized_worker_aggregate_fixed";
    case SlackPipeSplitMode::kWorkerLocal:
      return "optimized_worker_aggregate_local";
  }
  return "optimized_global";
}

bool SlackPipeModeFixesFullPartition(SlackPipeSplitMode mode) {
  return mode == SlackPipeSplitMode::kFixed;
}

bool SlackPipeModeFixesWorkerAggregateLoads(SlackPipeSplitMode mode) {
  return mode == SlackPipeSplitMode::kWorkerFixed;
}

bool SlackPipeModeOptimizesPartition(SlackPipeSplitMode mode) {
  return mode != SlackPipeSplitMode::kFixed;
}

CanonicalResultMetadata BuildCanonicalResultMetadata(
    const Instance &instance, const CanonicalRequestContext &request,
    const CanonicalSemantics &semantics, const CanonicalOutcome &outcome,
    const std::optional<std::vector<Tick>> &selected_partition,
    const std::optional<MachineOrders> &worker_orders) {
  CanonicalResultMetadata metadata;
  metadata.timestamp_utc =
      request.timestamp_utc ? request.timestamp_utc : CurrentTimestampUtc();
  metadata.requested_command = request.requested_command
                                   ? request.requested_command
                                   : EnvString("SLACKPIPE_REQUESTED_COMMAND");
  metadata.requested_method = request.requested_method;
  metadata.canonical_method = semantics.canonical_method;
  metadata.actual_solver_path = semantics.actual_solver_path;
  if (const EvaluationMethodDefinition *definition =
          FindEvaluationMethod(metadata.canonical_method)) {
    metadata.method_contract_hash = EvaluationMethodContractHash(*definition);
  }
  metadata.requested_time_limit_seconds = request.requested_time_limit_seconds;
  metadata.effective_time_limit_seconds = request.effective_time_limit_seconds;
  metadata.random_seed = request.random_seed;
  metadata.solver_threads = request.solver_threads;
  metadata.run_id =
      request.run_id ? request.run_id : EnvString("SLACKPIPE_RUN_ID");
  if (!metadata.run_id) {
    metadata.run_id = MakeRunId(metadata.timestamp_utc.value_or("") + "|" +
                                metadata.requested_command.value_or("") + "|" +
                                metadata.canonical_method);
  }

  metadata.micro_batches = instance.microbatches;
  metadata.logical_stages = instance.stages;
  metadata.physical_workers = instance.workers;
  metadata.total_layers = instance.total_layers;
  metadata.min_layers = instance.min_layers;
  metadata.stage_to_worker_mapping = BuildStageMapping(instance);
  metadata.forward_cost_ratio_numerator = instance.backward_ratio_den;
  metadata.forward_cost_ratio_denominator = 1;
  metadata.backward_cost_ratio_numerator = instance.backward_ratio_num;
  metadata.backward_cost_ratio_denominator = 1;
  metadata.communication_ticks = instance.communication_ticks;
  if (semantics.schedule_decision == "fixed_interleaved_1f1b") {
    metadata.fixed_schedule_rule = kFixedScheduleRuleInterleavedOneFOneB;
  }
  metadata.semantics = semantics;
  ApplyFifoDefaults(instance, &metadata.semantics);
  metadata.outcome = outcome;
  if (selected_partition && !selected_partition->empty()) {
    metadata.selected_partition = selected_partition;
  }
  if (worker_orders && !worker_orders->empty()) {
    metadata.worker_local_operation_order =
        OperationOrderNames(instance, *worker_orders);
    metadata.derived_worker_predecessors = DerivedPredecessors(
        instance, *worker_orders,
        metadata.semantics.fifo_ordering_effective.value_or(true));
  }

  metadata.git_commit = CurrentGitCommit();
  metadata.git_dirty = CurrentGitDirty(&metadata.git_dirty_scope);
  metadata.build_type = CurrentBuildType();
  metadata.executable_name = request.executable_name
                                 ? request.executable_name
                                 : EnvString("SLACKPIPE_EXECUTABLE_NAME");
  return metadata;
}

ResultValidationInput ValidationInputFromCanonical(
    const Instance &instance, const ScheduleSolution &schedule,
    const CanonicalResultMetadata &metadata) {
  ResultValidationInput input;
  input.instance = instance;
  input.selected_partition = metadata.selected_partition;
  if (!schedule.orders.empty()) input.worker_orders = schedule.orders;
  input.worker_order_names = metadata.worker_local_operation_order;
  input.require_serialized_worker_predecessors =
      metadata.outcome.feasible.value_or(false);
  input.require_serialized_operations =
      metadata.outcome.feasible.value_or(false);
  if (!metadata.derived_worker_predecessors.empty()) {
    input.derived_worker_predecessors.emplace();
    input.derived_worker_predecessors->reserve(
        metadata.derived_worker_predecessors.size());
    for (const CanonicalWorkerPredecessor &edge :
         metadata.derived_worker_predecessors) {
      SerializedWorkerPredecessorRecord record;
      record.operation_id = OperationId{edge.operation_id};
      record.operation_name = edge.operation;
      record.predecessor_id = OperationId{edge.predecessor_id};
      record.predecessor_name = edge.predecessor;
      record.worker = edge.worker;
      input.derived_worker_predecessors->push_back(std::move(record));
    }
  }
  if (!schedule.operations_by_id.empty()) {
    input.serialized_operations.emplace();
    input.serialized_operations->reserve(schedule.operations_by_id.size());
    for (const ScheduledOperation &operation : schedule.operations_by_id) {
      SerializedOperationRecord record;
      record.id = operation.id;
      const OperationView view = DecodeOperation(instance, operation.id);
      record.name = OperationName(instance, operation.id);
      record.microbatch = view.microbatch;
      record.operation_position = view.chain_index;
      record.stage = view.stage;
      record.phase = view.backward ? "B" : "F";
      record.worker = operation.worker;
      record.start = operation.start;
      record.end = operation.end;
      record.duration = operation.duration;
      input.serialized_operations->push_back(std::move(record));
    }
  }
  input.reported_makespan = metadata.outcome.makespan;
  input.feasible_claimed = metadata.outcome.feasible.value_or(
      IsTerminalFeasibleStatus(metadata.outcome.reported_status.value_or("")));
  input.solver_status_raw = metadata.outcome.solver_status_raw;
  input.reported_status = metadata.outcome.reported_status;
  input.method_name = metadata.canonical_method;
  input.communication_model = metadata.communication_model;
  input.communication_alpha = metadata.communication_alpha;
  input.communication_beta = metadata.communication_beta;
  input.communication_payload = metadata.communication_payload;
  input.fifo_ordering =
      metadata.semantics.fifo_ordering_effective.value_or(true);
  input.fifo_ordering_requested = metadata.semantics.fifo_ordering_requested;
  input.fifo_ordering_effective = metadata.semantics.fifo_ordering_effective;
  input.fifo_constraint_count = metadata.semantics.fifo_constraint_count;
  return input;
}

void ApplyResultValidation(CanonicalOutcome &outcome,
                           const ResultValidationResult &validation) {
  outcome.validation_runtime_seconds = validation.validation_runtime_seconds;
  outcome.result_validation = validation;
  outcome.result_validation_passed = validation.passed;
  if (validation.passed) {
    outcome.result_validation_error = std::nullopt;
  } else {
    outcome.result_validation_error =
        validation.error_code.empty()
            ? validation.message
            : validation.error_code + (validation.message.empty()
                                           ? std::string()
                                           : ": " + validation.message);
    outcome.feasible = false;
    outcome.optimal = false;
    outcome.reported_status = "INVALID_RESULT";
    outcome.makespan = std::nullopt;
    outcome.relative_optimality_gap = std::nullopt;
  }
  if (validation.validation_runtime_seconds > 0.0) {
    if (outcome.total_runtime_seconds) {
      *outcome.total_runtime_seconds += validation.validation_runtime_seconds;
    } else {
      outcome.total_runtime_seconds = validation.validation_runtime_seconds;
    }
  }
  CanonicalPhaseBudget phase;
  phase.phase = "result_validation";
  phase.runtime_seconds = validation.validation_runtime_seconds;
  phase.status = validation.passed ? "PASSED" : "FAILED";
  outcome.phase_budget.phases.push_back(std::move(phase));
}

void ApplyActivationAnalysis(CanonicalResultMetadata &metadata,
                             const ActivationAnalysisResult &analysis) {
  metadata.activation_analysis = analysis;
  const bool cap_violation = analysis.activation_cap_satisfied &&
                             !analysis.activation_cap_satisfied.value_or(true);
  const bool invalid =
      !analysis.passed || ((analysis.activation_cap_enforced ||
                            analysis.activation_cap_enforced_in_solver) &&
                           cap_violation);

  if (analysis.activation_cap_derivation_runtime_seconds > 0.0) {
    if (metadata.outcome.total_runtime_seconds) {
      *metadata.outcome.total_runtime_seconds +=
          analysis.activation_cap_derivation_runtime_seconds;
    } else {
      metadata.outcome.total_runtime_seconds =
          analysis.activation_cap_derivation_runtime_seconds;
    }
    CanonicalPhaseBudget phase;
    phase.phase = "activation_uniform_baseline_analysis";
    phase.runtime_seconds = analysis.activation_cap_derivation_runtime_seconds;
    phase.status = analysis.passed ? "PASSED" : "FAILED";
    metadata.outcome.phase_budget.phases.push_back(std::move(phase));
  }

  if (!invalid) return;

  std::string error_code = "activation_analysis_failed";
  std::string message =
      analysis.errors.empty() ? std::string() : analysis.errors.front();
  if (cap_violation) {
    error_code = "activation_cap_violation";
    if (analysis.activation_cap_violation) {
      const ActivationCapViolation &violation =
          *analysis.activation_cap_violation;
      std::ostringstream out;
      out << "worker " << violation.worker << " peak activation units "
          << violation.observed_peak_units << " exceeds cap "
          << violation.cap_units
          << " at t=" << violation.first_violating_timestamp;
      message = out.str();
    } else {
      message = "activation cap violation";
    }
  }

  if (!metadata.outcome.result_validation) {
    metadata.outcome.result_validation.emplace();
  }
  metadata.outcome.result_validation->passed = false;
  metadata.outcome.result_validation->error_code = error_code;
  metadata.outcome.result_validation->error_category = "activation_memory";
  metadata.outcome.result_validation->message = message;
  metadata.outcome.result_validation->warnings = analysis.warnings;
  metadata.outcome.result_validation_passed = false;
  metadata.outcome.result_validation_error =
      error_code + (message.empty() ? std::string() : ": " + message);
  metadata.outcome.feasible = false;
  metadata.outcome.optimal = false;
  metadata.outcome.reported_status = "INVALID_RESULT";
  metadata.outcome.makespan = std::nullopt;
  metadata.outcome.relative_optimality_gap = std::nullopt;
}

CanonicalOutcome OutcomeFromSchedule(
    const ScheduleSolution &schedule, const std::string &status,
    std::optional<double> total_runtime_seconds) {
  CanonicalOutcome outcome;
  outcome.solver_status_raw = status;
  outcome.reported_status = status;
  outcome.feasible = schedule.ok();
  outcome.optimal = false;
  outcome.returned_solution_source = "deterministic_evaluator";
  outcome.makespan =
      schedule.ok() ? std::optional<Tick>(schedule.makespan) : std::nullopt;
  outcome.total_runtime_seconds = total_runtime_seconds;
  outcome.result_validation_passed = schedule.ok();
  if (!schedule.validation_errors.empty()) {
    outcome.result_validation_error = schedule.validation_errors.front();
  }
  return outcome;
}

CanonicalOutcome OutcomeFromBfsResult(
    const BfsSplitOptimizationResult &result) {
  CanonicalOutcome outcome;
  outcome.solver_status_raw = result.solver_status_raw.empty()
                                  ? result.status
                                  : result.solver_status_raw;
  outcome.reported_status = result.status;
  outcome.feasible = IsTerminalFeasibleStatus(result.status);
  outcome.optimal = result.proven_optimal;
  outcome.fallback_used = result.fallback_used;
  if (!result.fallback_reason.empty()) {
    outcome.fallback_reason = result.fallback_reason;
  }
  outcome.returned_solution_source =
      result.returned_solution_source.empty() ||
              result.returned_solution_source == "none"
          ? std::string("bfs_fixed_order")
          : result.returned_solution_source;
  outcome.makespan = PositiveOrNull(result.makespan_ticks);
  outcome.best_objective_bound = BoundOrNull(result.best_bound_ticks);
  if (outcome.makespan && outcome.best_objective_bound) {
    outcome.relative_optimality_gap =
        RelativeGap(*outcome.makespan, *outcome.best_objective_bound);
  }
  if (!result.optimality_proof_source.empty() &&
      result.optimality_proof_source != "none") {
    outcome.optimality_proof_source = result.optimality_proof_source;
  }
  outcome.enumeration_proved_optimal = result.enumeration_proved_optimal;
  outcome.enumeration_candidates_total = result.enumeration_candidates_total;
  outcome.enumeration_candidates_valid_schedule =
      result.enumeration_candidates_valid_schedule;
  outcome.enumeration_candidates_cap_feasible =
      result.enumeration_candidates_cap_feasible;
  outcome.enumeration_candidates_cap_rejected =
      result.enumeration_candidates_cap_rejected;
  outcome.total_runtime_seconds = PositiveOrNull(result.wall_time_seconds)
                                      .value_or(result.timing.total_seconds);
  if (outcome.total_runtime_seconds && *outcome.total_runtime_seconds <= 0.0) {
    outcome.total_runtime_seconds = std::nullopt;
  }
  outcome.incumbent_runtime_seconds =
      PositiveOrNull(result.timing.incumbent_seconds);
  outcome.model_build_runtime_seconds =
      PositiveOrNull(result.timing.model_build_seconds);
  outcome.solver_runtime_seconds = PositiveOrNull(result.timing.solver_seconds);
  outcome.time_to_first_feasible_seconds =
      PositiveOrNull(result.time_to_first_feasible_seconds);
  outcome.time_to_best_solution_seconds =
      PositiveOrNull(result.time_to_best_incumbent_seconds);
  outcome.phase_budget.phases.push_back(
      MakePhaseBudget(result.method.find("fixed-order") != std::string::npos
                          ? std::string("fixed_order_partition")
                          : std::string("partition_bfs"),
                      std::nullopt, PositiveOrNull(result.timing.total_seconds),
                      PositiveOrNull(result.timing.total_seconds),
                      outcome.solver_status_raw.value_or(result.status)));
  outcome.result_validation_passed =
      outcome.feasible.value_or(false) ? result.schedule.ok() : true;
  if (!result.schedule.validation_errors.empty()) {
    outcome.result_validation_error = result.schedule.validation_errors.front();
  } else if (!result.diagnostic.empty() && !result.schedule.ok() &&
             outcome.feasible.value_or(false)) {
    outcome.result_validation_error = result.diagnostic;
  }
  return outcome;
}

CanonicalOutcome OutcomeFromJointResult(const JointOptimizationResult &result) {
  CanonicalOutcome outcome;
  outcome.solver_status_raw =
      result.joint_status.empty() ? result.status : result.joint_status;
  outcome.reported_status = result.status;
  outcome.feasible = IsTerminalFeasibleStatus(result.status);
  outcome.optimal = result.proven_optimal;
  outcome.fallback_used = result.fallback_used;
  outcome.fallback_enabled = result.fallback_enabled;
  outcome.external_incumbent_available = result.incumbent_valid;
  outcome.external_incumbent_used_as_fallback = result.fallback_used;
  outcome.solver_solution_available = result.solver_solution_available;
  outcome.final_solution_available = result.final_solution_available;
  outcome.final_solution_source = result.final_solution_source;
  if (!result.no_solution_reason.empty()) {
    outcome.no_solution_reason = result.no_solution_reason;
  }
  if (result.fallback_used) outcome.fallback_reason = result.diagnostic;
  outcome.returned_solution_source =
      result.fallback_used ? std::string("incumbent") : result.solution_source;
  outcome.makespan = PositiveOrNull(result.makespan_ticks);
  outcome.best_objective_bound = BoundOrNull(result.best_bound_ticks);
  if (outcome.makespan && outcome.best_objective_bound) {
    outcome.relative_optimality_gap =
        RelativeGap(*outcome.makespan, *outcome.best_objective_bound);
  }
  outcome.total_runtime_seconds = PositiveOrNull(result.wall_time_seconds)
                                      .value_or(result.timing.total_seconds);
  if (outcome.total_runtime_seconds && *outcome.total_runtime_seconds <= 0.0) {
    outcome.total_runtime_seconds = std::nullopt;
  }
  outcome.incumbent_runtime_seconds =
      PositiveOrNull(result.incumbent_solve_seconds)
          .value_or(result.timing.incumbent_seconds);
  if (outcome.incumbent_runtime_seconds &&
      *outcome.incumbent_runtime_seconds <= 0.0) {
    outcome.incumbent_runtime_seconds = std::nullopt;
  }
  outcome.model_build_runtime_seconds =
      PositiveOrNull(result.timing.model_build_seconds);
  outcome.solver_runtime_seconds = PositiveOrNull(result.timing.solver_seconds);
  outcome.time_to_first_feasible_seconds =
      PositiveOrNull(result.time_to_first_feasible_seconds);
  outcome.time_to_first_cpsat_feasible_seconds =
      PositiveOrNull(result.time_to_first_cpsat_feasible_seconds);
  outcome.first_cpsat_feasible_objective =
      PositiveOrNull(result.first_cpsat_feasible_objective);
  outcome.time_to_best_solution_seconds =
      PositiveOrNull(result.time_to_best_incumbent_seconds);
  outcome.phase_budget.schedule_solver_effective_limit_seconds =
      PositiveOrNull(result.joint_budget_seconds);
  outcome.phase_budget.phases.push_back(MakePhaseBudget(
      "incumbent", PositiveOrNull(result.incumbent_budget_seconds),
      PositiveOrNull(result.incumbent_budget_seconds),
      PositiveOrNull(result.timing.incumbent_seconds),
      result.incumbent_status));
  outcome.phase_budget.phases.push_back(
      MakePhaseBudget("joint_model_build", std::nullopt, std::nullopt,
                      PositiveOrNull(result.timing.model_build_seconds),
                      result.joint_status == "NOT_RUN" ? "NOT_RUN" : "BUILT"));
  outcome.phase_budget.phases.push_back(MakePhaseBudget(
      "joint_cp_sat", PositiveOrNull(result.joint_budget_seconds),
      PositiveOrNull(result.joint_budget_seconds),
      PositiveOrNull(result.timing.solver_seconds), result.joint_status));
  outcome.result_validation_passed = result.schedule.ok();
  if (!result.schedule.validation_errors.empty()) {
    outcome.result_validation_error = result.schedule.validation_errors.front();
  } else if (!result.diagnostic.empty() && !result.schedule.ok()) {
    outcome.result_validation_error = result.diagnostic;
  }
  return outcome;
}

CanonicalOutcome OutcomeFromSlackPipeResult(const SlackPipeResult &result) {
  CanonicalOutcome outcome;
  outcome.solver_status_raw =
      result.joint_status.empty() ? result.status : result.joint_status;
  outcome.reported_status = result.status;
  outcome.feasible = IsTerminalFeasibleStatus(result.status);
  outcome.optimal = result.proven_optimal;
  outcome.fallback_used = result.joint_fallback_used;
  if (result.joint_fallback_used) outcome.fallback_reason = result.diagnostic;
  outcome.returned_solution_source = result.joint_fallback_used
                                         ? std::string("incumbent")
                                         : result.joint_solution_source;
  outcome.makespan = PositiveOrNull(result.makespan_ticks);
  outcome.best_objective_bound = BoundOrNull(result.best_bound_ticks);
  if (outcome.makespan && outcome.best_objective_bound) {
    outcome.relative_optimality_gap =
        RelativeGap(*outcome.makespan, *outcome.best_objective_bound);
  }
  outcome.total_runtime_seconds = PositiveOrNull(result.total_wall_time_seconds)
                                      .value_or(result.wall_time_seconds);
  if (outcome.total_runtime_seconds && *outcome.total_runtime_seconds <= 0.0) {
    outcome.total_runtime_seconds = std::nullopt;
  }
  outcome.incumbent_runtime_seconds =
      PositiveOrNull(result.timing.incumbent_seconds);
  outcome.reference_runtime_seconds =
      PositiveOrNull(result.reference_solve_seconds);
  outcome.model_build_runtime_seconds =
      PositiveOrNull(result.timing.model_build_seconds);
  outcome.solver_runtime_seconds = PositiveOrNull(result.timing.solver_seconds);
  outcome.time_to_first_feasible_seconds =
      PositiveOrNull(result.time_to_first_feasible_seconds);
  outcome.time_to_best_solution_seconds =
      PositiveOrNull(result.time_to_best_incumbent_seconds);
  outcome.phase_budget.reference_phase_limit_seconds =
      PositiveOrNull(result.reference_budget_seconds);
  outcome.phase_budget.schedule_solver_effective_limit_seconds =
      PositiveOrNull(result.joint_remaining_budget_seconds);
  outcome.phase_budget.phases.push_back(MakePhaseBudget(
      "slackpipe_reference", PositiveOrNull(result.reference_budget_seconds),
      PositiveOrNull(result.reference_budget_seconds),
      PositiveOrNull(result.reference_solve_seconds), result.bfs.status));
  outcome.phase_budget.phases.push_back(
      MakePhaseBudget("joint_model_build", std::nullopt, std::nullopt,
                      PositiveOrNull(result.timing.model_build_seconds),
                      result.joint_status == "NOT_RUN" ? "NOT_RUN" : "BUILT"));
  outcome.phase_budget.phases.push_back(MakePhaseBudget(
      "joint_cp_sat", PositiveOrNull(result.joint_remaining_budget_seconds),
      PositiveOrNull(result.joint_remaining_budget_seconds),
      PositiveOrNull(result.timing.solver_seconds), result.joint_status));
  outcome.result_validation_passed = result.schedule.ok();
  if (!result.schedule.validation_errors.empty()) {
    outcome.result_validation_error = result.schedule.validation_errors.front();
  } else if (!result.diagnostic.empty() && !result.schedule.ok()) {
    outcome.result_validation_error = result.diagnostic;
  }
  return outcome;
}

CanonicalSemantics SemanticsForUniformFixedOrderBaseline() {
  CanonicalSemantics semantics;
  semantics.canonical_method = kUniformBreadthFirstMethod;
  semantics.actual_solver_path = "deterministic-evaluator";
  semantics.partition_decision = "fixed_uniform";
  semantics.schedule_decision = "fixed_breadth_first";
  semantics.partition_reference_source = "uniform_deterministic";
  semantics.schedule_reference_source = "breadth_first";
  semantics.cp_sat_models_solved = 0;
  return semantics;
}

CanonicalSemantics SemanticsForUniformInterleavedOneFOneB() {
  CanonicalSemantics semantics;
  semantics.canonical_method = kUniformInterleavedOneFOneBMethod;
  semantics.actual_solver_path = "deterministic-evaluator";
  semantics.partition_decision = "fixed_uniform";
  semantics.schedule_decision = "fixed_interleaved_1f1b";
  semantics.partition_reference_source = "uniform_deterministic";
  semantics.schedule_reference_source = "interleaved_1f1b";
  semantics.cp_sat_models_solved = 0;
  return semantics;
}

CanonicalSemantics SemanticsForPartitionOnlyFixedOrder(
    const std::string &method) {
  CanonicalSemantics semantics;
  semantics.canonical_method = "partition-only-fixed-order";
  semantics.actual_solver_path = SolverPathForBfsMethod(method);
  semantics.partition_decision = "optimized_global";
  semantics.schedule_decision = "fixed_breadth_first";
  semantics.schedule_reference_source = "breadth_first";
  semantics.partition_optimized = true;
  semantics.schedule_optimized = false;
  semantics.cp_sat_hint_enabled = false;
  semantics.incumbent_upper_bound_enabled = false;
  semantics.cp_sat_models_solved =
      method.find("cpsat") == std::string::npos ? 0 : 1;
  return semantics;
}

CanonicalSemantics SemanticsForPartitionOnlyFixedOrder(
    const BfsSplitOptimizationResult &result) {
  CanonicalSemantics semantics =
      SemanticsForPartitionOnlyFixedOrder(result.method);
  if (!result.fixed_order_partition_backend_requested.empty()) {
    semantics.fixed_order_partition_backend_requested =
        result.fixed_order_partition_backend_requested;
  }
  if (!result.fixed_order_partition_backend_effective.empty()) {
    semantics.fixed_order_partition_backend_effective =
        result.fixed_order_partition_backend_effective;
  }
  if (result.estimated_partition_count_available) {
    semantics.estimated_partition_count = result.estimated_partition_count;
  }
  if (result.enumeration_safety_threshold > 0) {
    semantics.enumeration_safety_threshold =
        result.enumeration_safety_threshold;
  }
  semantics.cp_sat_launched = result.cp_sat_launched;
  semantics.cp_sat_models_solved = result.cp_sat_models_solved;
  semantics.cp_sat_hint_enabled = result.cp_sat_launched;
  semantics.incumbent_upper_bound_enabled =
      result.cp_sat_launched && result.fallback_available;
  if (!result.fixed_order_partition_backend_effective.empty()) {
    if (result.fixed_order_partition_backend_effective == "enumerate") {
      semantics.actual_solver_path = "fixed-order-partition-enumeration";
    } else if (result.fixed_order_partition_backend_effective == "cpsat") {
      semantics.actual_solver_path = "fixed-order-partition-cpsat";
    } else if (result.fixed_order_partition_backend_effective ==
               "unavailable") {
      semantics.actual_solver_path = "fixed-order-partition-unavailable";
    }
  }
  return semantics;
}

CanonicalSemantics SemanticsForSequentialPartitionThenSchedule(
    Index cp_sat_models_solved) {
  CanonicalSemantics semantics;
  semantics.canonical_method = "sequential-partition-then-schedule";
  semantics.actual_solver_path =
      "fixed-order-partition-then-joint-cpsat-fixed-split-no-overlap";
  semantics.partition_decision = "optimized_global_then_fixed";
  semantics.schedule_decision = "optimized_no_overlap";
  semantics.partition_reference_source = "partition_only_within_global_budget";
  semantics.schedule_reference_source = "no_overlap";
  semantics.full_partition_fixed = false;
  semantics.partition_optimized = true;
  semantics.schedule_optimized = true;
  semantics.predecessor_candidate_restriction_requested = false;
  semantics.predecessor_candidate_restriction_active = false;
  semantics.predecessor_candidate_rule = "unrestricted_no_overlap";
  semantics.cp_sat_hint_enabled = true;
  semantics.incumbent_upper_bound_enabled = true;
  semantics.cp_sat_models_solved = cp_sat_models_solved;
  return semantics;
}

CanonicalSemantics SemanticsForAlternatingPartitionSchedule(
    Index cp_sat_models_solved) {
  CanonicalSemantics semantics;
  semantics.canonical_method = "alternating-partition-schedule";
  semantics.actual_solver_path =
      "alternating-fixed-order-partition-and-fixed-split-no-overlap";
  semantics.partition_decision = "alternating_fixed_order_partition";
  semantics.schedule_decision = "alternating_fixed_split_no_overlap";
  semantics.partition_reference_source = "previous_accepted_schedule_order";
  semantics.schedule_reference_source = "no_overlap";
  semantics.partition_optimized = true;
  semantics.schedule_optimized = true;
  semantics.predecessor_candidate_restriction_requested = false;
  semantics.predecessor_candidate_restriction_active = false;
  semantics.predecessor_candidate_rule = "unrestricted_no_overlap";
  semantics.cp_sat_hint_enabled = true;
  semantics.incumbent_upper_bound_enabled = true;
  semantics.cp_sat_models_solved = cp_sat_models_solved;
  return semantics;
}

CanonicalSemantics SemanticsForBfsEvaluate() {
  return SemanticsForUniformFixedOrderBaseline();
}

CanonicalSemantics SemanticsForPartitionOnlyBfs(const std::string &bfs_method) {
  return SemanticsForPartitionOnlyFixedOrder(bfs_method);
}

CanonicalSemantics SemanticsForScheduleOnlyFixedSplit(
    const std::string &partition_reference_source,
    const JointOptimizationResult &result,
    bool predecessor_restriction_requested) {
  CanonicalSemantics semantics;
  const std::string reference_source = partition_reference_source == "manual"
                                           ? "supplied input"
                                           : partition_reference_source;
  semantics.canonical_method =
      reference_source == "uniform" ||
              reference_source == "uniform_deterministic"
          ? "schedule-only-uniform"
          : "schedule-only-fixed-split";
  semantics.actual_solver_path = "joint-cpsat-fixed-split-no-overlap";
  semantics.partition_decision =
      reference_source == "uniform" ||
              reference_source == "uniform_deterministic"
          ? "fixed_uniform"
          : "fixed_supplied";
  semantics.schedule_decision = "optimized_no_overlap";
  semantics.partition_reference_source = reference_source;
  semantics.schedule_reference_source = "no_overlap";
  semantics.full_partition_fixed = true;
  semantics.partition_optimized = false;
  semantics.schedule_optimized = true;
  semantics.predecessor_candidate_restriction_requested =
      predecessor_restriction_requested;
  semantics.predecessor_candidate_restriction_active = false;
  semantics.predecessor_candidate_rule = "unrestricted_no_overlap";
  semantics.incumbent_method_requested = result.incumbent_method_requested;
  semantics.incumbent_method_effective = result.incumbent_method_effective;
  semantics.cp_sat_hint_enabled = result.hints_effective;
  semantics.incumbent_upper_bound_enabled = result.incumbent_bound_effective;
  semantics.incumbent_bound_requested = result.incumbent_bound_requested;
  semantics.incumbent_bound_effective = result.incumbent_bound_effective;
  if (result.incumbent_bound_horizon > 0) {
    semantics.incumbent_bound_horizon = result.incumbent_bound_horizon;
  }
  semantics.incumbent_hints_requested = result.incumbent_hints_requested;
  semantics.incumbent_hints_effective = result.incumbent_hints_effective;
  semantics.incumbent_hint_count = result.hinted_total_variable_count;
  semantics.incumbent_found = result.incumbent_found;
  semantics.incumbent_valid = result.incumbent_valid;
  ApplyFifoFromJointResult(&semantics, result);
  if (result.incumbent_makespan > 0) {
    semantics.incumbent_makespan = result.incumbent_makespan;
  }
  semantics.cp_sat_models_solved = result.cp_sat_models_solved;
  ApplyWorkerBalance(semantics, result.worker_balance_constraint);
  semantics.worker_balance_pruning_requested =
      result.worker_balance_pruning_requested;
  semantics.worker_balance_pruning_effective =
      result.worker_balance_pruning_effective;
  return semantics;
}

CanonicalSemantics SemanticsForJointUnrestrictedNoOverlap(
    const JointOptimizationResult &result,
    bool predecessor_restriction_requested) {
  CanonicalSemantics semantics;
  semantics.canonical_method = "joint-unrestricted-no-overlap";
  semantics.actual_solver_path = "joint-cpsat-unrestricted-no-overlap";
  semantics.partition_decision = "optimized_global";
  semantics.schedule_decision = "optimized_no_overlap";
  semantics.schedule_reference_source = "no_overlap";
  semantics.partition_optimized = true;
  semantics.schedule_optimized = true;
  semantics.predecessor_candidate_restriction_requested =
      predecessor_restriction_requested;
  semantics.predecessor_candidate_restriction_active = false;
  semantics.predecessor_candidate_rule = "unrestricted_no_overlap";
  semantics.incumbent_method_requested = result.incumbent_method_requested;
  semantics.incumbent_method_effective = result.incumbent_method_effective;
  semantics.cp_sat_hint_enabled = result.hints_effective;
  semantics.incumbent_upper_bound_enabled = result.incumbent_bound_effective;
  semantics.incumbent_bound_requested = result.incumbent_bound_requested;
  semantics.incumbent_bound_effective = result.incumbent_bound_effective;
  if (result.incumbent_bound_horizon > 0) {
    semantics.incumbent_bound_horizon = result.incumbent_bound_horizon;
  }
  semantics.incumbent_hints_requested = result.incumbent_hints_requested;
  semantics.incumbent_hints_effective = result.incumbent_hints_effective;
  semantics.incumbent_hint_count = result.hinted_total_variable_count;
  semantics.incumbent_found = result.incumbent_found;
  semantics.incumbent_valid = result.incumbent_valid;
  ApplyFifoFromJointResult(&semantics, result);
  if (result.incumbent_makespan > 0) {
    semantics.incumbent_makespan = result.incumbent_makespan;
  }
  semantics.cp_sat_models_solved = result.cp_sat_models_solved;
  ApplyWorkerBalance(semantics, result.worker_balance_constraint);
  semantics.worker_balance_pruning_requested =
      result.worker_balance_pruning_requested;
  semantics.worker_balance_pruning_effective =
      result.worker_balance_pruning_effective;
  return semantics;
}

CanonicalSemantics SemanticsForSlackPipe(
    SlackPipeSplitMode split_mode, const SlackPipeResult &result,
    bool predecessor_restriction_requested) {
  CanonicalSemantics semantics;
  semantics.canonical_method = CanonicalMethodForSlackPipeMode(split_mode);
  semantics.actual_solver_path =
      "canonical-slackpipe-reference-plus-joint-cpsat-no-overlap";
  semantics.partition_decision = PartitionDecisionForSlackPipeMode(split_mode);
  semantics.schedule_decision = "optimized_no_overlap";
  semantics.partition_reference_source =
      result.initial_split_method.empty() ? std::string("slackpipe_reference")
                                          : result.initial_split_method;
  semantics.schedule_reference_source = "no_overlap";
  semantics.full_partition_fixed = SlackPipeModeFixesFullPartition(split_mode);
  semantics.worker_aggregate_loads_fixed =
      SlackPipeModeFixesWorkerAggregateLoads(split_mode);
  semantics.partition_optimized = SlackPipeModeOptimizesPartition(split_mode);
  semantics.schedule_optimized = true;
  if (split_mode == SlackPipeSplitMode::kLocal) {
    semantics.stage_local_move_budget = result.move_budget;
  }
  if (split_mode == SlackPipeSplitMode::kWorkerLocal) {
    semantics.worker_aggregate_move_budget = result.worker_move_budget;
  }
  semantics.predecessor_candidate_restriction_requested =
      predecessor_restriction_requested;
  semantics.predecessor_candidate_restriction_active = false;
  semantics.predecessor_candidate_rule = "unrestricted_no_overlap";
  semantics.incumbent_method_requested =
      result.joint_incumbent_method_requested;
  semantics.incumbent_method_effective =
      result.joint_incumbent_method_effective;
  semantics.cp_sat_hint_enabled = result.joint_hints_effective;
  semantics.incumbent_upper_bound_enabled =
      result.joint_horizon_source != "none";
  semantics.fifo_ordering_requested = result.joint_fifo_ordering_requested;
  semantics.fifo_ordering_effective = result.joint_fifo_ordering_effective;
  semantics.fifo_constraint_count = result.joint_fifo_constraint_count;
  semantics.cp_sat_models_solved = result.cp_sat_models_solved;
  ApplyWorkerBalance(semantics, result.worker_balance_constraint);
  return semantics;
}

std::string CanonicalResultToJson(const CanonicalResultMetadata &metadata,
                                  const std::string &indent) {
  const std::string nested = indent + "  ";
  std::ostringstream out;
  out << "{\n";
  out << nested << "\"schema_version\": " << metadata.schema_version << ",\n";
  out << nested
      << "\"budget_policy_version\": " << metadata.budget_policy_version
      << ",\n";
  out << nested
      << "\"evaluation_method_version\": " << metadata.evaluation_method_version
      << ",\n";
  out << nested << "\"run_id\": ";
  WriteOptionalString(out, metadata.run_id);
  out << ",\n";
  out << nested << "\"timestamp_utc\": ";
  WriteOptionalString(out, metadata.timestamp_utc);
  out << ",\n";
  out << nested << "\"requested_command\": ";
  WriteOptionalString(out, metadata.requested_command);
  out << ",\n";
  out << nested << "\"requested_method\": ";
  WriteOptionalString(out, metadata.requested_method);
  out << ",\n";
  out << nested << "\"canonical_method\": ";
  WriteJsonString(out, metadata.canonical_method);
  out << ",\n";
  out << nested << "\"actual_solver_path\": ";
  WriteJsonString(out, metadata.actual_solver_path);
  out << ",\n";

  out << nested << "\"micro_batches\": " << metadata.micro_batches << ",\n";
  out << nested << "\"logical_stages\": " << metadata.logical_stages << ",\n";
  out << nested << "\"physical_workers\": " << metadata.physical_workers
      << ",\n";
  out << nested << "\"total_layers\": " << metadata.total_layers << ",\n";
  out << nested << "\"min_layers\": " << metadata.min_layers << ",\n";
  out << nested << "\"stage_to_worker_mapping\": [";
  for (std::size_t i = 0; i < metadata.stage_to_worker_mapping.size(); ++i) {
    if (i != 0) out << ", ";
    const StageWorkerMappingEntry &entry = metadata.stage_to_worker_mapping[i];
    out << "{\"stage\": " << entry.stage << ", \"worker\": " << entry.worker
        << "}";
  }
  out << "],\n";
  out << nested << "\"mapping_type\": ";
  WriteJsonString(out, metadata.mapping_type);
  out << ",\n";
  out << nested << "\"forward_cost_ratio_numerator\": "
      << metadata.forward_cost_ratio_numerator << ",\n";
  out << nested << "\"forward_cost_ratio_denominator\": "
      << metadata.forward_cost_ratio_denominator << ",\n";
  out << nested << "\"backward_cost_ratio_numerator\": "
      << metadata.backward_cost_ratio_numerator << ",\n";
  out << nested << "\"backward_cost_ratio_denominator\": "
      << metadata.backward_cost_ratio_denominator << ",\n";
  out << nested << "\"communication_model\": ";
  WriteJsonString(out, metadata.communication_model);
  out << ",\n";
  out << nested << "\"communication_ticks\": " << metadata.communication_ticks
      << ",\n";
  out << nested << "\"communication_alpha\": ";
  WriteOptionalNumber(out, metadata.communication_alpha);
  out << ",\n";
  out << nested << "\"communication_beta\": ";
  WriteOptionalNumber(out, metadata.communication_beta);
  out << ",\n";
  out << nested << "\"communication_payload\": ";
  WriteOptionalString(out, metadata.communication_payload);
  out << ",\n";
  out << nested << "\"fixed_schedule_rule\": ";
  WriteJsonString(out, metadata.fixed_schedule_rule);
  out << ",\n";
  out << nested << "\"uniform_partition_rule\": ";
  WriteJsonString(out, metadata.uniform_partition_rule);
  out << ",\n";
  out << nested << "\"alternating_max_rounds\": ";
  WriteOptionalNumber(out, metadata.alternating_max_rounds);
  out << ",\n";
  out << nested << "\"alternating_completed_rounds\": ";
  WriteOptionalNumber(out, metadata.alternating_completed_rounds);
  out << ",\n";
  out << nested << "\"alternating_convergence_reason\": ";
  WriteOptionalString(out, metadata.alternating_convergence_reason);
  out << ",\n";
  out << nested << "\"alternating_trace\": ";
  WriteAlternatingTrace(out, metadata.alternating_trace, nested);
  out << ",\n";
  out << nested << "\"intermediate_partition_only_makespan\": ";
  WriteOptionalNumber(out, metadata.intermediate_partition_only_makespan);
  out << ",\n";
  out << nested << "\"method_contract_hash\": ";
  WriteJsonString(out, metadata.method_contract_hash);
  out << ",\n";

  const CanonicalSemantics &semantics = metadata.semantics;
  out << nested << "\"partition_decision\": ";
  WriteJsonString(out, semantics.partition_decision);
  out << ",\n";
  out << nested << "\"schedule_decision\": ";
  WriteJsonString(out, semantics.schedule_decision);
  out << ",\n";
  out << nested << "\"partition_reference_source\": ";
  WriteOptionalString(out, semantics.partition_reference_source);
  out << ",\n";
  out << nested << "\"schedule_reference_source\": ";
  WriteOptionalString(out, semantics.schedule_reference_source);
  out << ",\n";
  out << nested << "\"full_partition_fixed\": "
      << (semantics.full_partition_fixed ? "true" : "false") << ",\n";
  out << nested << "\"worker_aggregate_loads_fixed\": "
      << (semantics.worker_aggregate_loads_fixed ? "true" : "false") << ",\n";
  out << nested << "\"partition_optimized\": "
      << (semantics.partition_optimized ? "true" : "false") << ",\n";
  out << nested << "\"schedule_optimized\": "
      << (semantics.schedule_optimized ? "true" : "false") << ",\n";
  out << nested << "\"worker_balance_pruning_enabled\": "
      << (semantics.worker_balance_pruning_enabled ? "true" : "false") << ",\n";
  out << nested << "\"worker_balance_pruning_requested\": ";
  WriteOptionalBool(out, semantics.worker_balance_pruning_requested);
  out << ",\n";
  out << nested << "\"worker_balance_pruning_effective\": ";
  WriteOptionalBool(out, semantics.worker_balance_pruning_effective);
  out << ",\n";
  out << nested << "\"worker_balance_tolerance\": {\n";
  out << nested << "  \"percent\": ";
  WriteOptionalNumber(out, semantics.worker_balance_tolerance_percent);
  out << ",\n";
  out << nested << "  \"layers\": ";
  WriteOptionalNumber(out, semantics.worker_balance_tolerance_layers);
  out << ",\n";
  out << nested << "  \"lower_bound\": ";
  WriteOptionalNumber(out, semantics.worker_balance_lower_bound);
  out << ",\n";
  out << nested << "  \"upper_bound\": ";
  WriteOptionalNumber(out, semantics.worker_balance_upper_bound);
  out << "\n" << nested << "},\n";
  out << nested << "\"stage_local_move_budget\": ";
  WriteOptionalNumber(out, semantics.stage_local_move_budget);
  out << ",\n";
  out << nested << "\"worker_aggregate_move_budget\": ";
  WriteOptionalNumber(out, semantics.worker_aggregate_move_budget);
  out << ",\n";
  out << nested << "\"predecessor_candidate_restriction_requested\": "
      << (semantics.predecessor_candidate_restriction_requested ? "true"
                                                                : "false")
      << ",\n";
  out << nested << "\"predecessor_candidate_restriction_active\": "
      << (semantics.predecessor_candidate_restriction_active ? "true" : "false")
      << ",\n";
  out << nested << "\"predecessor_candidate_rule\": ";
  WriteOptionalString(out, semantics.predecessor_candidate_rule);
  out << ",\n";
  out << nested << "\"fifo_ordering_requested\": ";
  WriteOptionalBool(out, semantics.fifo_ordering_requested);
  out << ",\n";
  out << nested << "\"fifo_ordering_effective\": ";
  WriteOptionalBool(out, semantics.fifo_ordering_effective);
  out << ",\n";
  out << nested << "\"fifo_constraint_count\": ";
  WriteOptionalNumber(out, semantics.fifo_constraint_count);
  out << ",\n";
  out << nested << "\"requested_time_limit_seconds\": ";
  WriteOptionalNumber(out, metadata.requested_time_limit_seconds);
  out << ",\n";
  out << nested << "\"effective_time_limit_seconds\": ";
  WriteOptionalNumber(out, metadata.effective_time_limit_seconds);
  out << ",\n";
  out << nested << "\"random_seed\": ";
  WriteOptionalNumber(out, metadata.random_seed);
  out << ",\n";
  out << nested << "\"solver_threads\": ";
  WriteOptionalNumber(out, metadata.solver_threads);
  out << ",\n";

  out << nested << "\"incumbent_method_requested\": ";
  WriteOptionalString(out, semantics.incumbent_method_requested);
  out << ",\n";
  out << nested << "\"incumbent_method_effective\": ";
  WriteOptionalString(out, semantics.incumbent_method_effective);
  out << ",\n";
  out << nested << "\"fixed_order_partition_backend_requested\": ";
  WriteOptionalString(out, semantics.fixed_order_partition_backend_requested);
  out << ",\n";
  out << nested << "\"fixed_order_partition_backend_effective\": ";
  WriteOptionalString(out, semantics.fixed_order_partition_backend_effective);
  out << ",\n";
  out << nested << "\"estimated_partition_count\": ";
  WriteOptionalNumber(out, semantics.estimated_partition_count);
  out << ",\n";
  out << nested << "\"enumeration_safety_threshold\": ";
  WriteOptionalNumber(out, semantics.enumeration_safety_threshold);
  out << ",\n";
  out << nested << "\"cp_sat_launched\": ";
  WriteOptionalBool(out, semantics.cp_sat_launched);
  out << ",\n";
  out << nested << "\"cp_sat_hint_enabled\": ";
  WriteOptionalBool(out, semantics.cp_sat_hint_enabled);
  out << ",\n";
  out << nested << "\"incumbent_upper_bound_enabled\": ";
  WriteOptionalBool(out, semantics.incumbent_upper_bound_enabled);
  out << ",\n";
  out << nested << "\"incumbent_bound_requested\": ";
  WriteOptionalBool(out, semantics.incumbent_bound_requested);
  out << ",\n";
  out << nested << "\"incumbent_bound_effective\": ";
  WriteOptionalBool(out, semantics.incumbent_bound_effective);
  out << ",\n";
  out << nested << "\"incumbent_bound_horizon\": ";
  WriteOptionalNumber(out, semantics.incumbent_bound_horizon);
  out << ",\n";
  out << nested << "\"incumbent_hints_requested\": ";
  WriteOptionalBool(out, semantics.incumbent_hints_requested);
  out << ",\n";
  out << nested << "\"incumbent_hints_effective\": ";
  WriteOptionalBool(out, semantics.incumbent_hints_effective);
  out << ",\n";
  out << nested << "\"incumbent_hint_count\": ";
  WriteOptionalNumber(out, semantics.incumbent_hint_count);
  out << ",\n";
  out << nested << "\"incumbent_found\": ";
  WriteOptionalBool(out, semantics.incumbent_found);
  out << ",\n";
  out << nested << "\"incumbent_valid\": ";
  WriteOptionalBool(out, semantics.incumbent_valid);
  out << ",\n";
  out << nested << "\"incumbent_makespan\": ";
  WriteOptionalNumber(out, semantics.incumbent_makespan);
  out << ",\n";

  const CanonicalOutcome &outcome = metadata.outcome;
  out << nested << "\"solver_status_raw\": ";
  WriteOptionalString(out, outcome.solver_status_raw);
  out << ",\n";
  out << nested << "\"reported_status\": ";
  WriteOptionalString(out, outcome.reported_status);
  out << ",\n";
  out << nested << "\"feasible\": ";
  WriteOptionalBool(out, outcome.feasible);
  out << ",\n";
  out << nested << "\"optimal\": ";
  WriteOptionalBool(out, outcome.optimal);
  out << ",\n";
  out << nested
      << "\"fallback_used\": " << (outcome.fallback_used ? "true" : "false")
      << ",\n";
  out << nested << "\"fallback_enabled\": ";
  WriteOptionalBool(out, outcome.fallback_enabled);
  out << ",\n";
  out << nested << "\"external_incumbent_available\": ";
  WriteOptionalBool(out, outcome.external_incumbent_available);
  out << ",\n";
  out << nested << "\"external_incumbent_used_as_fallback\": ";
  WriteOptionalBool(out, outcome.external_incumbent_used_as_fallback);
  out << ",\n";
  out << nested << "\"solver_solution_available\": ";
  WriteOptionalBool(out, outcome.solver_solution_available);
  out << ",\n";
  out << nested << "\"final_solution_available\": ";
  WriteOptionalBool(out, outcome.final_solution_available);
  out << ",\n";
  out << nested << "\"final_solution_source\": ";
  WriteOptionalString(out, outcome.final_solution_source);
  out << ",\n";
  out << nested << "\"no_solution_reason\": ";
  WriteOptionalString(out, outcome.no_solution_reason);
  out << ",\n";
  out << nested << "\"fallback_reason\": ";
  WriteOptionalString(out, outcome.fallback_reason);
  out << ",\n";
  out << nested << "\"returned_solution_source\": ";
  WriteOptionalString(out, outcome.returned_solution_source);
  out << ",\n";
  out << nested << "\"makespan\": ";
  WriteOptionalNumber(out, outcome.makespan);
  out << ",\n";
  out << nested << "\"best_objective_bound\": ";
  WriteOptionalNumber(out, outcome.best_objective_bound);
  out << ",\n";
  out << nested << "\"relative_optimality_gap\": ";
  WriteOptionalNumber(out, outcome.relative_optimality_gap);
  out << ",\n";
  out << nested << "\"optimality_proof_source\": ";
  WriteOptionalString(out, outcome.optimality_proof_source);
  out << ",\n";
  out << nested << "\"enumeration_proved_optimal\": ";
  WriteOptionalBool(out, outcome.enumeration_proved_optimal);
  out << ",\n";
  out << nested << "\"enumeration_candidates_total\": ";
  WriteOptionalNumber(out, outcome.enumeration_candidates_total);
  out << ",\n";
  out << nested << "\"enumeration_candidates_valid_schedule\": ";
  WriteOptionalNumber(out, outcome.enumeration_candidates_valid_schedule);
  out << ",\n";
  out << nested << "\"enumeration_candidates_cap_feasible\": ";
  WriteOptionalNumber(out, outcome.enumeration_candidates_cap_feasible);
  out << ",\n";
  out << nested << "\"enumeration_candidates_cap_rejected\": ";
  WriteOptionalNumber(out, outcome.enumeration_candidates_cap_rejected);
  out << ",\n";
  out << nested << "\"total_runtime_seconds\": ";
  WriteOptionalNumber(out, outcome.total_runtime_seconds);
  out << ",\n";
  out << nested << "\"incumbent_runtime_seconds\": ";
  WriteOptionalNumber(out, outcome.incumbent_runtime_seconds);
  out << ",\n";
  out << nested << "\"reference_runtime_seconds\": ";
  WriteOptionalNumber(out, outcome.reference_runtime_seconds);
  out << ",\n";
  out << nested << "\"model_build_runtime_seconds\": ";
  WriteOptionalNumber(out, outcome.model_build_runtime_seconds);
  out << ",\n";
  out << nested << "\"solver_runtime_seconds\": ";
  WriteOptionalNumber(out, outcome.solver_runtime_seconds);
  out << ",\n";
  out << nested << "\"time_to_first_feasible_seconds\": ";
  WriteOptionalNumber(out, outcome.time_to_first_feasible_seconds);
  out << ",\n";
  out << nested << "\"time_to_first_cpsat_feasible_seconds\": ";
  WriteOptionalNumber(out, outcome.time_to_first_cpsat_feasible_seconds);
  out << ",\n";
  out << nested << "\"first_cpsat_feasible_objective\": ";
  WriteOptionalNumber(out, outcome.first_cpsat_feasible_objective);
  out << ",\n";
  out << nested << "\"time_to_best_solution_seconds\": ";
  WriteOptionalNumber(out, outcome.time_to_best_solution_seconds);
  out << ",\n";
  out << nested << "\"validation_runtime_seconds\": ";
  WriteOptionalNumber(out, outcome.validation_runtime_seconds);
  out << ",\n";
  out << nested << "\"phase_budget\": ";
  WritePhaseBudget(out, outcome.phase_budget, nested);
  out << ",\n";
  out << nested << "\"cp_sat_models_solved\": ";
  WriteOptionalNumber(out, semantics.cp_sat_models_solved);
  out << ",\n";

  const std::optional<ActivationAnalysisResult> &activation =
      metadata.activation_analysis;
  out << nested << "\"activation_analysis_version\": ";
  if (activation) {
    out << activation->activation_analysis_version;
  } else {
    out << "null";
  }
  out << ",\n";
  out << nested << "\"activation_model\": ";
  if (activation) {
    WriteJsonString(out, ToString(activation->model));
  } else {
    out << "null";
  }
  out << ",\n";
  out << nested << "\"activation_units_per_layer\": ";
  if (activation) {
    out << activation->activation_units_per_layer;
  } else {
    out << "null";
  }
  out << ",\n";
  out << nested << "\"explicit_stage_activation_units\": ";
  if (activation) {
    WriteTickArray(out, activation->explicit_stage_activation_units);
  } else {
    out << "null";
  }
  out << ",\n";
  out << nested << "\"activation_bytes_per_unit\": ";
  if (activation) {
    WriteOptionalDouble(out, activation->activation_bytes_per_unit);
  } else {
    out << "null";
  }
  out << ",\n";
  out << nested << "\"activation_memory_metrics\": ";
  if (activation) {
    WriteActivationMetrics(out, *activation, nested);
  } else {
    out << "null";
  }
  out << ",\n";
  out << nested << "\"activation_cap_mode\": ";
  if (activation) {
    WriteJsonString(out, ToString(activation->cap_mode));
  } else {
    out << "null";
  }
  out << ",\n";
  out << nested << "\"activation_cap_units_per_worker\": ";
  if (activation) {
    WriteTickArray(out, activation->activation_cap_units_per_worker);
  } else {
    out << "null";
  }
  out << ",\n";
  out << nested << "\"activation_cap_source\": ";
  if (activation) {
    WriteJsonString(out, activation->activation_cap_source);
  } else {
    out << "null";
  }
  out << ",\n";
  out << nested << "\"activation_cap_satisfied\": ";
  if (activation) {
    WriteOptionalBool(out, activation->activation_cap_satisfied);
  } else {
    out << "null";
  }
  out << ",\n";
  out << nested << "\"activation_cap_enforced\": ";
  if (activation) {
    out << (activation->activation_cap_enforced ? "true" : "false");
  } else {
    out << "false";
  }
  out << ",\n";
  out << nested << "\"activation_cap_enforced_in_solver\": ";
  if (activation) {
    out << (activation->activation_cap_enforced_in_solver ? "true" : "false");
  } else {
    out << "false";
  }
  out << ",\n";
  out << nested << "\"activation_cap_enforced_by_enumeration\": ";
  if (activation) {
    out << (activation->activation_cap_enforced_by_enumeration ? "true"
                                                               : "false");
  } else {
    out << "false";
  }
  out << ",\n";
  out << nested << "\"activation_cap_enforcement_requested\": ";
  if (activation) {
    out << (activation->activation_cap_enforcement_requested ? "true"
                                                             : "false");
  } else {
    out << "false";
  }
  out << ",\n";
  out << nested << "\"activation_cap_enforcement_mode\": ";
  if (activation) {
    WriteJsonString(out, activation->activation_cap_enforcement_mode);
  } else {
    out << "null";
  }
  out << ",\n";
  out << nested << "\"activation_cap_solver_supported\": ";
  if (activation) {
    out << (activation->activation_cap_solver_supported ? "true" : "false");
  } else {
    out << "false";
  }
  out << ",\n";
  out << nested << "\"activation_cap_solver_support_level\": ";
  if (activation) {
    WriteJsonString(out, activation->activation_cap_solver_support_level);
  } else {
    out << "null";
  }
  out << ",\n";
  out << nested << "\"activation_cap_constraints_added\": ";
  if (activation) {
    out << (activation->activation_cap_constraints_added ? "true" : "false");
  } else {
    out << "false";
  }
  out << ",\n";
  out << nested << "\"activation_retained_interval_count\": ";
  if (activation) {
    out << activation->activation_retained_interval_count;
  } else {
    out << "0";
  }
  out << ",\n";
  out << nested << "\"activation_cumulative_constraint_count\": ";
  if (activation) {
    out << activation->activation_cumulative_constraint_count;
  } else {
    out << "0";
  }
  out << ",\n";
  out << nested << "\"activation_variable_demand_count\": ";
  if (activation) {
    out << activation->activation_variable_demand_count;
  } else {
    out << "0";
  }
  out << ",\n";
  out << nested << "\"activation_fixed_demand_count\": ";
  if (activation) {
    out << activation->activation_fixed_demand_count;
  } else {
    out << "0";
  }
  out << ",\n";
  out << nested << "\"activation_workers_with_constraints\": ";
  if (activation) {
    WriteTickArray(out, activation->activation_workers_with_constraints);
  } else {
    out << "[]";
  }
  out << ",\n";
  out << nested << "\"activation_constraint_build_runtime_seconds\": ";
  if (activation) {
    out << activation->activation_constraint_build_runtime_seconds;
  } else {
    out << "0";
  }
  out << ",\n";
  out << nested << "\"incumbent_rejected_for_activation_cap\": ";
  if (activation) {
    out << (activation->incumbent_rejected_for_activation_cap ? "true"
                                                              : "false");
  } else {
    out << "false";
  }
  out << ",\n";
  out << nested << "\"activation_model_validation_agreement\": ";
  if (activation) {
    WriteOptionalBool(out, activation->activation_model_validation_agreement);
  } else {
    out << "null";
  }
  out << ",\n";
  out << nested << "\"activation_model_disagreement_details\": ";
  if (activation &&
      !activation->activation_model_disagreement_details.empty()) {
    WriteJsonString(out, activation->activation_model_disagreement_details);
  } else {
    out << "null";
  }
  out << ",\n";
  out << nested << "\"activation_cap_unsupported_reason\": ";
  if (activation && !activation->activation_cap_unsupported_reason.empty()) {
    WriteJsonString(out, activation->activation_cap_unsupported_reason);
  } else {
    out << "null";
  }
  out << ",\n";
  out << nested << "\"activation_cap_formulation_version\": ";
  if (activation) {
    out << activation->activation_cap_formulation_version;
  } else {
    out << "null";
  }
  out << ",\n";
  out << nested << "\"activation_cap_violation\": ";
  if (activation) {
    WriteActivationCapViolation(out, activation->activation_cap_violation,
                                nested);
  } else {
    out << "null";
  }
  out << ",\n";
  out << nested << "\"activation_baseline_run_id\": ";
  if (activation && !activation->activation_baseline_run_id.empty()) {
    WriteJsonString(out, activation->activation_baseline_run_id);
  } else {
    out << "null";
  }
  out << ",\n";
  out << nested << "\"activation_cap_derivation_hash\": ";
  if (activation && !activation->activation_cap_derivation_hash.empty()) {
    WriteJsonString(out, activation->activation_cap_derivation_hash);
  } else {
    out << "null";
  }
  out << ",\n";
  out << nested << "\"activation_baseline_partition\": ";
  if (activation && !activation->activation_baseline_partition.empty()) {
    WriteTickArray(out, activation->activation_baseline_partition);
  } else {
    out << "null";
  }
  out << ",\n";
  out << nested << "\"activation_baseline_method_contract_hash\": ";
  if (activation &&
      !activation->activation_baseline_method_contract_hash.empty()) {
    WriteJsonString(out, activation->activation_baseline_method_contract_hash);
  } else {
    out << "null";
  }
  out << ",\n";
  out << nested << "\"activation_cap_derivation_runtime_seconds\": ";
  if (activation) {
    out << activation->activation_cap_derivation_runtime_seconds;
  } else {
    out << "null";
  }
  out << ",\n";
  out << nested << "\"activation_peak_ratio_to_uniform\": ";
  if (activation) {
    WriteActivationPeakRatio(out, activation->activation_peak_ratio_to_uniform,
                             nested);
  } else {
    out << "null";
  }
  out << ",\n";

  out << nested << "\"selected_partition\": ";
  WriteOptionalTickArray(out, metadata.selected_partition);
  out << ",\n";
  out << nested << "\"worker_local_operation_order\": ";
  WriteOptionalStringMatrix(out, metadata.worker_local_operation_order, nested);
  out << ",\n";
  out << nested << "\"derived_worker_predecessors\": [";
  for (std::size_t i = 0; i < metadata.derived_worker_predecessors.size();
       ++i) {
    if (i != 0) out << ", ";
    const CanonicalWorkerPredecessor &edge =
        metadata.derived_worker_predecessors[i];
    out << "{\"operation_id\": " << edge.operation_id << ", \"operation\": ";
    WriteJsonString(out, edge.operation);
    out << ", \"predecessor_id\": " << edge.predecessor_id
        << ", \"predecessor\": ";
    WriteJsonString(out, edge.predecessor);
    out << ", \"worker\": " << edge.worker << "}";
  }
  out << "],\n";
  out << nested << "\"derived_worker_predecessor_rule\": ";
  WriteJsonString(out,
                  "adjacent operations in each worker_local_operation_order "
                  "become worker predecessor edges when they have different "
                  "operation positions and are not already data predecessors");
  out << ",\n";

  out << nested << "\"result_validation\": ";
  if (outcome.result_validation) {
    out << ResultValidationToJson(*outcome.result_validation, nested);
  } else {
    out << "null";
  }
  out << ",\n";
  out << nested << "\"result_validation_passed\": "
      << (outcome.result_validation_passed ? "true" : "false") << ",\n";
  out << nested << "\"result_validation_error\": ";
  WriteOptionalString(out, outcome.result_validation_error);
  out << ",\n";
  out << nested << "\"git_commit\": ";
  WriteOptionalString(out, metadata.git_commit);
  out << ",\n";
  out << nested << "\"git_dirty\": ";
  WriteOptionalBool(out, metadata.git_dirty);
  out << ",\n";
  out << nested << "\"git_dirty_scope\": ";
  WriteOptionalString(out, metadata.git_dirty_scope);
  out << ",\n";
  out << nested << "\"build_type\": ";
  WriteOptionalString(out, metadata.build_type);
  out << ",\n";
  out << nested << "\"executable_name\": ";
  WriteOptionalString(out, metadata.executable_name);
  out << "\n" << indent << "}";
  return out.str();
}

std::string CanonicalResultTopLevelJsonFields(
    const CanonicalResultMetadata &metadata, const std::string &indent,
    bool trailing_comma) {
  std::ostringstream out;
  out << indent << "\"schema_version\": " << metadata.schema_version << ",\n";
  out << indent
      << "\"budget_policy_version\": " << metadata.budget_policy_version
      << ",\n";
  out << indent
      << "\"evaluation_method_version\": " << metadata.evaluation_method_version
      << ",\n";
  out << indent << "\"canonical_method\": ";
  WriteJsonString(out, metadata.canonical_method);
  out << ",\n";
  out << indent << "\"actual_solver_path\": ";
  WriteJsonString(out, metadata.actual_solver_path);
  out << ",\n";
  out << indent
      << "\"canonical_result\": " << CanonicalResultToJson(metadata, indent);
  if (trailing_comma) out << ",";
  out << "\n";
  return out.str();
}

}  // namespace slackpipe
