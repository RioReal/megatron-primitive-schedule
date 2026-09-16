#include "slackpipe/evaluation_method.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>

#include "slackpipe/deadline.h"

namespace slackpipe {

namespace {

std::string JsonEscape(const std::string& text) {
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
      default:
        out << ch;
        break;
    }
  }
  return out.str();
}

void WriteStringArray(std::ostringstream& out,
                      const std::vector<std::string>& values) {
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ", ";
    out << "\"" << JsonEscape(values[i]) << "\"";
  }
  out << "]";
}

std::uint64_t Fnv1a(const std::string& text) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (unsigned char ch : text) {
    hash ^= static_cast<std::uint64_t>(ch);
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::string HashHex(const std::string& text) {
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << Fnv1a(text);
  return out.str();
}

std::string ContractText(const EvaluationMethodDefinition& definition) {
  std::ostringstream out;
  out << "evaluation_method_version=" << kEvaluationMethodVersion << "\n";
  out << "canonical_name=" << definition.canonical_name << "\n";
  out << "partition_decision=" << definition.partition_decision << "\n";
  out << "schedule_decision=" << definition.schedule_decision << "\n";
  out << "partition_optimized=" << definition.partition_optimized << "\n";
  out << "schedule_optimized=" << definition.schedule_optimized << "\n";
  out << "requires_ortools=" << definition.requires_ortools << "\n";
  out << "deterministic=" << definition.deterministic << "\n";
  out << "phase_policy=" << definition.phase_policy << "\n";
  out << "deadline_policy=" << definition.deadline_policy << "\n";
  out << "schedule_semantics=" << definition.schedule_semantics << "\n";
  out << "partition_source=" << definition.partition_source << "\n";
  for (const std::string& value : definition.fixed_variables) {
    out << "fixed=" << value << "\n";
  }
  for (const std::string& value : definition.optimized_variables) {
    out << "optimized=" << value << "\n";
  }
  return out.str();
}

bool MatchesName(const EvaluationMethodDefinition& definition,
                 const std::string& name) {
  if (definition.canonical_name == name) return true;
  return std::find(definition.legacy_aliases.begin(),
                   definition.legacy_aliases.end(),
                   name) != definition.legacy_aliases.end();
}

template <typename T>
bool OptionalEqual(const std::optional<T>& a, const std::optional<T>& b) {
  if (!a && !b) return true;
  if (a && b) return *a == *b;
  return false;
}

std::string CompatibilityMismatch(const MethodCompatibilityRecord& expected,
                                  const MethodCompatibilityRecord& actual) {
  auto mismatch = [](const std::string& field, const auto& a,
                     const auto& b) -> std::optional<std::string> {
    if (a == b) return std::nullopt;
    std::ostringstream out;
    out << field << " mismatch";
    return out.str();
  };
  if (auto msg = mismatch("B", expected.micro_batches, actual.micro_batches)) {
    return *msg;
  }
  if (auto msg =
          mismatch("N", expected.logical_stages, actual.logical_stages)) {
    return *msg;
  }
  if (auto msg =
          mismatch("W", expected.physical_workers, actual.physical_workers)) {
    return *msg;
  }
  if (auto msg = mismatch("L", expected.total_layers, actual.total_layers)) {
    return *msg;
  }
  if (auto msg =
          mismatch("min_layers", expected.min_layers, actual.min_layers)) {
    return *msg;
  }
  if (auto msg = mismatch("mapping_type", expected.mapping_type,
                          actual.mapping_type)) {
    return *msg;
  }
  if (auto msg = mismatch("forward_cost_ratio_numerator",
                          expected.forward_cost_ratio_numerator,
                          actual.forward_cost_ratio_numerator)) {
    return *msg;
  }
  if (auto msg = mismatch("forward_cost_ratio_denominator",
                          expected.forward_cost_ratio_denominator,
                          actual.forward_cost_ratio_denominator)) {
    return *msg;
  }
  if (auto msg = mismatch("backward_cost_ratio_numerator",
                          expected.backward_cost_ratio_numerator,
                          actual.backward_cost_ratio_numerator)) {
    return *msg;
  }
  if (auto msg = mismatch("backward_cost_ratio_denominator",
                          expected.backward_cost_ratio_denominator,
                          actual.backward_cost_ratio_denominator)) {
    return *msg;
  }
  if (auto msg = mismatch("communication_model", expected.communication_model,
                          actual.communication_model)) {
    return *msg;
  }
  if (auto msg = mismatch("communication_ticks", expected.communication_ticks,
                          actual.communication_ticks)) {
    return *msg;
  }
  if (!OptionalEqual(expected.requested_time_limit_seconds,
                     actual.requested_time_limit_seconds)) {
    return "requested_time_limit_seconds mismatch";
  }
  if (!OptionalEqual(expected.solver_threads, actual.solver_threads)) {
    return "solver_threads mismatch";
  }
  if (auto msg =
          mismatch("budget_policy_version", expected.budget_policy_version,
                   actual.budget_policy_version)) {
    return *msg;
  }
  if (auto msg = mismatch("validation_version", expected.validation_version,
                          actual.validation_version)) {
    return *msg;
  }
  return "";
}

std::string FixedScheduleRuleForMethod(
    const EvaluationMethodDefinition& definition) {
  if (definition.canonical_name == kUniformInterleavedOneFOneBMethod ||
      definition.schedule_decision == "fixed_interleaved_1f1b") {
    return kFixedScheduleRuleInterleavedOneFOneB;
  }
  return kFixedScheduleRuleBreadthFirst;
}

}  // namespace

const std::vector<EvaluationMethodDefinition>& EvaluationMethodRegistry() {
  static const std::vector<EvaluationMethodDefinition> methods = {
      EvaluationMethodDefinition{
          kUniformBreadthFirstMethod,
          {"eval-bfs"},
          "fixed_uniform",
          "fixed_breadth_first",
          {"B", "N", "W", "L", "cyclic_stage_to_worker_mapping",
           "uniform_partition", "breadth_first_worker_order",
           "data_dependencies", "fifo_dependencies", "communication_model"},
          {},
          false,
          false,
          false,
          true,
          true,
          "deterministic construction and independent validation",
          "no solver deadline; construction and validation runtime recorded",
          "worker-local breadth-first wavefront order sorted by "
          "(microbatch + operation_position, -operation_position, "
          "microbatch)",
          "uniform deterministic partition",
          "canonical fixed-order baseline"},
      EvaluationMethodDefinition{
          kUniformInterleavedOneFOneBMethod,
          {"eval-1f1b", "uniform-1f1b", "interleaved-1f1b"},
          "fixed_uniform",
          "fixed_interleaved_1f1b",
          {"B", "N", "W", "L", "cyclic_stage_to_worker_mapping",
           "uniform_partition", "interleaved_1f1b_worker_order",
           "data_dependencies", "fifo_dependencies", "communication_model"},
          {},
          false,
          false,
          false,
          true,
          true,
          "deterministic construction and independent validation",
          "no solver deadline; construction and validation runtime recorded",
          "stage-local 1F1B streams with warmup, steady-state forward/backward "
          "alternation, and drain; cyclic worker-local orders are a "
          "deterministic topological merge of those streams",
          "uniform deterministic partition",
          "canonical interleaved 1F1B fixed-order baseline"},
      EvaluationMethodDefinition{
          "partition-only-fixed-order",
          {"optimize-bfs", "partition-only"},
          "optimized_global",
          "fixed_breadth_first",
          {"B", "N", "W", "L", "cyclic_stage_to_worker_mapping",
           "breadth_first_worker_order", "data_dependencies",
           "fifo_dependencies", "communication_model"},
          {"stage_layer_partition"},
          true,
          false,
          false,
          false,
          true,
          "one fixed-order partition phase",
          "phase consumes at most the parent deadline",
          "same breadth-first worker-local order as uniform-breadth-first",
          "optimized partition under fixed worker order",
          "canonical partition-only baseline"},
      EvaluationMethodDefinition{
          "schedule-only-uniform",
          {"schedule-only"},
          "fixed_uniform",
          "optimized_no_overlap",
          {"B", "N", "W", "L", "cyclic_stage_to_worker_mapping",
           "uniform_partition", "data_dependencies", "fifo_dependencies",
           "communication_model"},
          {"worker_local_operation_order"},
          false,
          true,
          true,
          false,
          true,
          "one fixed-split unrestricted NoOverlap CP-SAT phase",
          "phase consumes at most the parent deadline",
          "unrestricted CP-SAT NoOverlap order per worker; no predecessor "
          "candidate restriction",
          "uniform deterministic partition",
          "canonical schedule-only baseline"},
      EvaluationMethodDefinition{
          "sequential-partition-then-schedule",
          {"schedule-only-partition-only"},
          "optimized_global_then_fixed",
          "optimized_no_overlap",
          {"B", "N", "W", "L", "cyclic_stage_to_worker_mapping",
           "data_dependencies", "fifo_dependencies", "communication_model"},
          {"stage_layer_partition", "worker_local_operation_order"},
          true,
          true,
          true,
          false,
          true,
          "partition fixed-order phase capped at 50%, then fixed-split "
          "unrestricted NoOverlap schedule phase",
          "one global deadline; unused partition time carries to schedule",
          "partition phase uses breadth-first order; schedule phase uses "
          "unrestricted CP-SAT NoOverlap",
          "partition-only result within global budget",
          "canonical sequential baseline"},
      EvaluationMethodDefinition{
          "alternating-partition-schedule",
          {},
          "alternating_fixed_order_partition",
          "alternating_fixed_split_no_overlap",
          {"B", "N", "W", "L", "cyclic_stage_to_worker_mapping",
           "data_dependencies", "fifo_dependencies", "communication_model"},
          {"stage_layer_partition", "worker_local_operation_order"},
          true,
          true,
          true,
          false,
          true,
          "alternate fixed-order partition and fixed-split schedule phases",
          "one global deadline; each phase cap is remaining time divided by "
          "remaining planned phases",
          "partition phases preserve the current worker-local order exactly; "
          "schedule phases use unrestricted CP-SAT NoOverlap",
          "starts from uniform-breadth-first",
          "canonical alternating baseline"},
      EvaluationMethodDefinition{
          "joint-unrestricted-no-overlap",
          {"joint", "joint-cpsat", "direct-joint", "optimize-joint"},
          "optimized_global",
          "optimized_no_overlap",
          {"B", "N", "W", "L", "cyclic_stage_to_worker_mapping",
           "data_dependencies", "fifo_dependencies", "communication_model"},
          {"stage_layer_partition", "worker_local_operation_order"},
          true,
          true,
          true,
          false,
          true,
          "hybrid/uniform incumbent followed by one direct joint CP-SAT model",
          "one global deadline with unchanged 10% incumbent policy",
          "unrestricted CP-SAT NoOverlap order per worker; no predecessor "
          "candidate restriction",
          "joint partition variables",
          "canonical direct joint baseline"}};
  return methods;
}

const EvaluationMethodDefinition* FindEvaluationMethod(
    const std::string& name) {
  for (const EvaluationMethodDefinition& definition :
       EvaluationMethodRegistry()) {
    if (MatchesName(definition, name)) return &definition;
  }
  return nullptr;
}

std::string CanonicalizeEvaluationMethodName(const std::string& name) {
  const EvaluationMethodDefinition* definition = FindEvaluationMethod(name);
  return definition == nullptr ? name : definition->canonical_name;
}

std::string EvaluationMethodContractHash(
    const EvaluationMethodDefinition& definition) {
  return HashHex(ContractText(definition));
}

std::string DescribeEvaluationMethodJson(const std::string& name) {
  const EvaluationMethodDefinition* definition = FindEvaluationMethod(name);
  if (definition == nullptr) {
    throw Error("unknown evaluation method: " + name);
  }
  std::ostringstream out;
  out << "{\n";
  out << "  \"evaluation_method_version\": " << kEvaluationMethodVersion
      << ",\n";
  out << "  \"canonical_name\": \"" << JsonEscape(definition->canonical_name)
      << "\",\n";
  out << "  \"legacy_aliases\": ";
  WriteStringArray(out, definition->legacy_aliases);
  out << ",\n";
  out << "  \"partition_decision\": \""
      << JsonEscape(definition->partition_decision) << "\",\n";
  out << "  \"schedule_decision\": \""
      << JsonEscape(definition->schedule_decision) << "\",\n";
  out << "  \"partition_optimized\": "
      << (definition->partition_optimized ? "true" : "false") << ",\n";
  out << "  \"schedule_optimized\": "
      << (definition->schedule_optimized ? "true" : "false") << ",\n";
  out << "  \"fixed_variables\": ";
  WriteStringArray(out, definition->fixed_variables);
  out << ",\n";
  out << "  \"optimized_variables\": ";
  WriteStringArray(out, definition->optimized_variables);
  out << ",\n";
  out << "  \"requires_ortools\": "
      << (definition->requires_ortools ? "true" : "false") << ",\n";
  out << "  \"deterministic\": "
      << (definition->deterministic ? "true" : "false") << ",\n";
  out << "  \"validation_required\": "
      << (definition->validation_required ? "true" : "false") << ",\n";
  out << "  \"phase_policy\": \"" << JsonEscape(definition->phase_policy)
      << "\",\n";
  out << "  \"deadline_policy\": \"" << JsonEscape(definition->deadline_policy)
      << "\",\n";
  out << "  \"schedule_semantics\": \""
      << JsonEscape(definition->schedule_semantics) << "\",\n";
  out << "  \"partition_source\": \""
      << JsonEscape(definition->partition_source) << "\",\n";
  out << "  \"fixed_schedule_rule\": \""
      << JsonEscape(FixedScheduleRuleForMethod(*definition)) << "\",\n";
  out << "  \"uniform_partition_rule\": \"" << kUniformPartitionRule << "\",\n";
  out << "  \"method_contract_hash\": \""
      << EvaluationMethodContractHash(*definition) << "\"\n";
  out << "}\n";
  return out.str();
}

MethodCompatibilityCheck CheckMethodCompatibility(
    const std::vector<MethodCompatibilityRecord>& records,
    bool seeds_are_repetitions) {
  if (records.empty()) return MethodCompatibilityCheck{true, ""};
  const MethodCompatibilityRecord& expected = records.front();
  for (std::size_t i = 1; i < records.size(); ++i) {
    const std::string mismatch = CompatibilityMismatch(expected, records[i]);
    if (!mismatch.empty()) {
      return MethodCompatibilityCheck{
          false, mismatch + " between " + expected.method_name + " and " +
                     records[i].method_name};
    }
    if (!seeds_are_repetitions && expected.solver_backed &&
        records[i].solver_backed &&
        !OptionalEqual(expected.random_seed, records[i].random_seed)) {
      return MethodCompatibilityCheck{
          false, "random_seed mismatch between solver-backed methods"};
    }
  }
  return MethodCompatibilityCheck{true, ""};
}

double AlternatingPhaseLimitSeconds(double remaining_seconds,
                                    int remaining_planned_phases) {
  if (remaining_planned_phases <= 0) return 0.0;
  if (remaining_seconds <= 0.0) return 0.0;
  if (!std::isfinite(remaining_seconds)) return 0.0;
  return remaining_seconds / static_cast<double>(remaining_planned_phases);
}

}  // namespace slackpipe
