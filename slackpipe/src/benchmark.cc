#include "slackpipe/benchmark.h"

#include <sys/resource.h>
#include <sys/utsname.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <thread>

#include "slackpipe/activation_analyzer.h"
#include "slackpipe/alternating_solver.h"
#include "slackpipe/bfs_solver.h"
#include "slackpipe/breadth_first.h"
#include "slackpipe/dag_evaluator.h"
#include "slackpipe/deadline.h"
#include "slackpipe/evaluation_method.h"
#include "slackpipe/fixed_order_partition_solver.h"
#include "slackpipe/io.h"
#include "slackpipe/joint_solver.h"
#include "slackpipe/operation.h"
#include "slackpipe/result_schema.h"
#include "slackpipe/slackpipe_solver.h"

namespace slackpipe::benchmark {
namespace {

using Clock = std::chrono::steady_clock;

std::string Trim(const std::string &text) {
  const std::size_t first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return "";
  const std::size_t last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1);
}

std::vector<std::string> Split(const std::string &text, char delimiter) {
  std::vector<std::string> out;
  std::stringstream ss(text);
  std::string token;
  while (std::getline(ss, token, delimiter)) out.push_back(Trim(token));
  return out;
}

std::string JsonEscape(const std::string &text) {
  std::ostringstream out;
  for (char ch : text) {
    if (ch == '"' || ch == '\\') out << '\\';
    if (ch == '\n') {
      out << "\\n";
    } else {
      out << ch;
    }
  }
  return out.str();
}

void WriteJsonOptionalString(std::ostringstream &out,
                             const std::optional<std::string> &value) {
  if (value) {
    out << "\"" << JsonEscape(*value) << "\"";
  } else {
    out << "null";
  }
}

void WriteJsonOptionalBool(std::ostringstream &out,
                           const std::optional<bool> &value) {
  if (value) {
    out << (*value ? "true" : "false");
  } else {
    out << "null";
  }
}

std::string CsvEscape(const std::string &text) {
  if (text.find_first_of(",\"\n") == std::string::npos) return text;
  std::string out = "\"";
  for (char ch : text) {
    if (ch == '"') out += '"';
    out += ch;
  }
  out += '"';
  return out;
}

std::string CompactJsonForJsonLine(std::string text) {
  text.erase(std::remove(text.begin(), text.end(), '\n'), text.end());
  return text;
}

std::string OptionalStringForCsv(const std::optional<std::string> &value) {
  return value.value_or("");
}

std::string OptionalBoolForCsv(const std::optional<bool> &value) {
  if (!value) return "";
  return *value ? "true" : "false";
}

template <typename T>
std::string OptionalNumberForCsv(const std::optional<T> &value) {
  if (!value) return "";
  std::ostringstream out;
  out << *value;
  return out.str();
}

std::string JoinTicks(const std::vector<Tick> &values) {
  std::ostringstream out;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ';';
    out << values[i];
  }
  return out.str();
}

std::string JoinDoubles(const std::vector<double> &values) {
  std::ostringstream out;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ';';
    out << values[i];
  }
  return out.str();
}

std::vector<Tick> ParseTickList(const std::string &text, char delimiter) {
  std::vector<Tick> values;
  for (const std::string &token : Split(text, delimiter)) {
    if (!token.empty()) values.push_back(std::stoll(token));
  }
  return values;
}

std::string JoinOrders(const Instance &instance, const MachineOrders &orders) {
  std::ostringstream out;
  for (std::size_t w = 0; w < orders.size(); ++w) {
    if (w != 0) out << '|';
    out << 'w' << w << ':';
    for (std::size_t i = 0; i < orders[w].size(); ++i) {
      if (i != 0) out << ' ';
      out << OperationName(instance, orders[w][i]);
    }
  }
  return out.str();
}

std::string RequestedMethod(const AlgorithmSpec &algorithm) {
  if (!algorithm.requested_method.empty()) return algorithm.requested_method;
  if (!algorithm.configuration.empty()) return algorithm.configuration;
  return ToString(algorithm.algorithm);
}

std::string FixedPartitionReferenceSource(const AlgorithmSpec &algorithm) {
  if (!algorithm.fixed_split.empty()) return "supplied input";
  if (algorithm.fixed_partition_source.empty() ||
      algorithm.fixed_partition_source == "uniform") {
    return "uniform_deterministic";
  }
  if (algorithm.fixed_partition_source == "load-balanced") {
    return "load-balanced construction";
  }
  if (algorithm.fixed_partition_source == "partition-only") {
    return "partition_only_within_global_budget";
  }
  return algorithm.fixed_partition_source;
}

CanonicalRequestContext BenchmarkRequestContext(
    const BenchmarkRow &row, const AlgorithmSpec &algorithm, double timeout,
    double effective_time_limit_seconds, const std::string &requested_command,
    const std::string &executable_name) {
  CanonicalRequestContext context;
  context.run_id = row.run_id;
  context.timestamp_utc = CurrentTimestampUtc();
  context.requested_command = requested_command.empty()
                                  ? std::string("slackpipe_benchmark")
                                  : requested_command;
  context.requested_method = RequestedMethod(algorithm);
  context.executable_name = executable_name.empty()
                                ? std::string("slackpipe_benchmark")
                                : executable_name;
  context.requested_time_limit_seconds = timeout;
  context.effective_time_limit_seconds = effective_time_limit_seconds;
  context.random_seed = row.seed;
  context.solver_threads = row.cp_sat_workers;
  return context;
}

CanonicalSemantics PlannedSemanticsForAlgorithm(
    const AlgorithmSpec &algorithm) {
  switch (algorithm.algorithm) {
    case BenchmarkAlgorithm::kUniformFixedOrder:
      return SemanticsForUniformFixedOrderBaseline();
    case BenchmarkAlgorithm::kOptimizedBfs:
    case BenchmarkAlgorithm::kPartitionOnly:
      return SemanticsForPartitionOnlyFixedOrder("auto");
    case BenchmarkAlgorithm::kJointCpSat: {
      CanonicalSemantics semantics;
      semantics.canonical_method = "joint-unrestricted-no-overlap";
      semantics.actual_solver_path = "joint-cpsat-unrestricted-no-overlap";
      semantics.partition_decision = "optimized_global";
      semantics.schedule_decision = "optimized_no_overlap";
      semantics.partition_optimized = true;
      semantics.schedule_optimized = true;
      semantics.predecessor_candidate_rule = "unrestricted_no_overlap";
      return semantics;
    }
    case BenchmarkAlgorithm::kScheduleOnly: {
      CanonicalSemantics semantics;
      semantics.canonical_method = "schedule-only-fixed-split";
      semantics.actual_solver_path = "joint-cpsat-fixed-split-no-overlap";
      semantics.partition_decision =
          FixedPartitionReferenceSource(algorithm) == "uniform_deterministic"
              ? "fixed_uniform"
              : "fixed_supplied";
      semantics.schedule_decision = "optimized_no_overlap";
      semantics.partition_reference_source =
          FixedPartitionReferenceSource(algorithm);
      semantics.full_partition_fixed = true;
      semantics.schedule_optimized = true;
      semantics.predecessor_candidate_rule = "unrestricted_no_overlap";
      return semantics;
    }
    case BenchmarkAlgorithm::kSequentialPartitionThenSchedule:
      return SemanticsForSequentialPartitionThenSchedule(0);
    case BenchmarkAlgorithm::kAlternatingPartitionSchedule:
      return SemanticsForAlternatingPartitionSchedule(0);
    case BenchmarkAlgorithm::kCanonicalSlackPipe: {
      CanonicalSemantics semantics;
      semantics.canonical_method =
          CanonicalMethodForSlackPipeMode(algorithm.split_mode);
      semantics.actual_solver_path =
          "canonical-slackpipe-reference-plus-joint-cpsat-no-overlap";
      semantics.partition_decision =
          PartitionDecisionForSlackPipeMode(algorithm.split_mode);
      semantics.schedule_decision = "optimized_no_overlap";
      semantics.full_partition_fixed =
          SlackPipeModeFixesFullPartition(algorithm.split_mode);
      semantics.worker_aggregate_loads_fixed =
          SlackPipeModeFixesWorkerAggregateLoads(algorithm.split_mode);
      semantics.partition_optimized =
          SlackPipeModeOptimizesPartition(algorithm.split_mode);
      semantics.schedule_optimized = true;
      semantics.predecessor_candidate_rule = "unrestricted_no_overlap";
      return semantics;
    }
  }
  return CanonicalSemantics{};
}

bool OrdersEqual(const MachineOrders &lhs, const MachineOrders &rhs) {
  return lhs == rhs;
}

long ProcessCpuMicros() {
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
  return usage.ru_utime.tv_sec * 1000000L + usage.ru_utime.tv_usec +
         usage.ru_stime.tv_sec * 1000000L + usage.ru_stime.tv_usec;
}

long PeakRssKb() {
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
  return usage.ru_maxrss;
}

double Since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string HashHex(const std::string &text) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (unsigned char ch : text) {
    hash ^= ch;
    hash *= 1099511628211ULL;
  }
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}

bool TimeoutLike(const std::string &status) {
  return status == "FEASIBLE" || status == "UNKNOWN" ||
         status.find("TIME") != std::string::npos;
}

std::string ReadFile(const std::string &path) {
  std::ifstream in(path);
  if (!in) throw Error("unable to open config file: " + path);
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

std::string Needle(const std::string &key) { return "\"" + key + "\""; }

bool HasKey(const std::string &text, const std::string &key) {
  return text.find(Needle(key)) != std::string::npos;
}

std::string ExtractString(const std::string &text, const std::string &key,
                          const std::string &fallback) {
  const std::size_t pos = text.find(Needle(key));
  if (pos == std::string::npos) return fallback;
  const std::size_t colon = text.find(':', pos);
  const std::size_t begin = text.find('"', colon + 1);
  if (colon == std::string::npos || begin == std::string::npos) return fallback;
  const std::size_t end = text.find('"', begin + 1);
  if (end == std::string::npos) return fallback;
  return text.substr(begin + 1, end - begin - 1);
}

long ExtractLong(const std::string &text, const std::string &key,
                 long fallback) {
  const std::size_t pos = text.find(Needle(key));
  if (pos == std::string::npos) return fallback;
  const std::size_t colon = text.find(':', pos);
  if (colon == std::string::npos) return fallback;
  const std::size_t begin = text.find_first_of("-0123456789", colon + 1);
  if (begin == std::string::npos) return fallback;
  std::size_t end = begin + 1;
  while (end < text.size() &&
         std::isdigit(static_cast<unsigned char>(text[end]))) {
    ++end;
  }
  return std::stol(text.substr(begin, end - begin));
}

double ExtractDouble(const std::string &text, const std::string &key,
                     double fallback) {
  const std::size_t pos = text.find(Needle(key));
  if (pos == std::string::npos) return fallback;
  const std::size_t colon = text.find(':', pos);
  if (colon == std::string::npos) return fallback;
  const std::size_t begin = text.find_first_of("-0123456789.", colon + 1);
  if (begin == std::string::npos) return fallback;
  std::size_t end = begin + 1;
  while (end < text.size() &&
         (std::isdigit(static_cast<unsigned char>(text[end])) ||
          text[end] == '.')) {
    ++end;
  }
  return std::stod(text.substr(begin, end - begin));
}

bool ExtractBool(const std::string &text, const std::string &key,
                 bool fallback) {
  const std::size_t pos = text.find(Needle(key));
  if (pos == std::string::npos) return fallback;
  const std::size_t colon = text.find(':', pos);
  if (colon == std::string::npos) return fallback;
  const std::string rest = text.substr(colon + 1, 8);
  if (rest.find("true") != std::string::npos) return true;
  if (rest.find("false") != std::string::npos) return false;
  return fallback;
}

std::vector<int> ExtractIntArray(const std::string &text,
                                 const std::string &key,
                                 const std::vector<int> &fallback) {
  const std::size_t pos = text.find(Needle(key));
  if (pos == std::string::npos) return fallback;
  const std::size_t open = text.find('[', pos);
  const std::size_t close = text.find(']', open);
  if (open == std::string::npos || close == std::string::npos) return fallback;
  std::vector<int> out;
  for (const std::string &token :
       Split(text.substr(open + 1, close - open - 1), ',')) {
    if (!token.empty()) out.push_back(std::stoi(token));
  }
  return out.empty() ? fallback : out;
}

std::vector<Tick> ExtractTickArray(const std::string &text,
                                   const std::string &key,
                                   const std::vector<Tick> &fallback) {
  const std::size_t pos = text.find(Needle(key));
  if (pos == std::string::npos) return fallback;
  const std::size_t open = text.find('[', pos);
  const std::size_t close = text.find(']', open);
  if (open == std::string::npos || close == std::string::npos) return fallback;
  std::vector<Tick> out;
  for (const std::string &token :
       Split(text.substr(open + 1, close - open - 1), ',')) {
    if (!token.empty()) out.push_back(std::stoll(token));
  }
  return out.empty() ? fallback : out;
}

std::vector<std::string> ExtractStringArray(const std::string &text,
                                            const std::string &key) {
  const std::size_t pos = text.find(Needle(key));
  if (pos == std::string::npos) return {};
  const std::size_t open = text.find('[', pos);
  const std::size_t close = text.find(']', open);
  if (open == std::string::npos || close == std::string::npos) return {};
  const std::string body = text.substr(open + 1, close - open - 1);
  std::vector<std::string> out;
  std::size_t cursor = 0;
  while (true) {
    const std::size_t first = body.find('"', cursor);
    if (first == std::string::npos) break;
    const std::size_t second = body.find('"', first + 1);
    if (second == std::string::npos) break;
    out.push_back(body.substr(first + 1, second - first - 1));
    cursor = second + 1;
  }
  return out;
}

std::vector<std::string> ExtractObjects(const std::string &text,
                                        const std::string &key) {
  const std::size_t pos = text.find(Needle(key));
  if (pos == std::string::npos) return {};
  const std::size_t open = text.find('[', pos);
  if (open == std::string::npos) return {};
  std::vector<std::string> objects;
  int brace_level = 0;
  std::size_t start = std::string::npos;
  for (std::size_t i = open + 1; i < text.size(); ++i) {
    if (text[i] == '{') {
      if (brace_level == 0) start = i;
      ++brace_level;
    } else if (text[i] == '}') {
      --brace_level;
      if (brace_level == 0 && start != std::string::npos) {
        objects.push_back(text.substr(start, i - start + 1));
      }
    } else if (text[i] == ']' && brace_level == 0) {
      break;
    }
  }
  return objects;
}

InstanceSpec InstanceFromFields(const std::map<std::string, std::string> &row) {
  auto get = [&](const std::string &key) -> std::string {
    const auto it = row.find(key);
    if (it == row.end() || it->second.empty()) {
      throw Error("missing instance field: " + key);
    }
    return it->second;
  };
  InstanceSpec spec;
  spec.name = row.count("name") != 0 ? row.at("name") : "instance";
  spec.instance.microbatches = std::stoll(get("B"));
  spec.instance.stages = std::stoll(get("N"));
  spec.instance.workers = std::stoll(get("J"));
  spec.instance.total_layers = std::stoll(get("L"));
  spec.instance.backward_ratio_num = std::stoll(get("ratio_num"));
  spec.instance.backward_ratio_den = std::stoll(get("ratio_den"));
  spec.instance.min_layers = std::stoll(get("min_layers"));
  if (row.count("communication_ticks") != 0) {
    spec.instance.communication_ticks =
        std::stoll(row.at("communication_ticks"));
  }
  spec.instance.Validate();
  return spec;
}

AlgorithmSpec AlgorithmFromText(const std::string &text) {
  AlgorithmSpec spec;
  spec.algorithm = ParseBenchmarkAlgorithm(text);
  spec.configuration = text;
  spec.requested_method = text;
  spec.split_mode = SlackPipeSplitMode::kGlobal;
  const std::string canonical = CanonicalizeEvaluationMethodName(text);
  if (canonical == kUniformBreadthFirstMethod) {
    spec.configuration = kUniformBreadthFirstMethod;
  } else if (canonical == "joint-unrestricted-no-overlap") {
    spec.configuration = "joint-unrestricted-no-overlap";
  } else if (canonical == "partition-only-fixed-order") {
    spec.configuration = "partition-only-fixed-order";
  } else if (text == "schedule-only-uniform") {
    spec.configuration = "schedule-only-uniform";
    spec.fixed_partition_source = "uniform";
  } else if (canonical == "schedule-only-uniform") {
    spec.configuration = "schedule-only-uniform";
    spec.fixed_partition_source = "uniform";
  } else if (text == "schedule-only-load-balanced" ||
             text == "schedule-only-balanced") {
    spec.configuration = "schedule-only-load-balanced";
    spec.fixed_partition_source = "load-balanced";
  } else if (canonical == "sequential-partition-then-schedule") {
    spec.configuration = "sequential-partition-then-schedule";
  } else if (canonical == "alternating-partition-schedule") {
    spec.configuration = "alternating-partition-schedule";
  }
  return spec;
}

AlgorithmSpec AlgorithmFromObject(const std::string &text) {
  const std::string requested_algorithm =
      ExtractString(text, "algorithm", "joint-cpsat");
  AlgorithmSpec spec = AlgorithmFromText(requested_algorithm);
  spec.requested_method = requested_algorithm;
  spec.configuration = ExtractString(text, "configuration", spec.configuration);
  spec.fixed_partition_source = ExtractString(text, "fixed_partition_source",
                                              spec.fixed_partition_source);
  spec.fixed_order_partition_backend =
      ExtractString(text, "fixed_order_partition_backend",
                    spec.fixed_order_partition_backend);
  const std::string fixed_split = ExtractString(text, "fixed_split", "");
  if (!fixed_split.empty()) {
    spec.fixed_split = ParseTickList(fixed_split, ';');
    if (spec.fixed_partition_source.empty()) {
      spec.fixed_partition_source = "manual";
    }
  }
  if (HasKey(text, "use_bfs_hints")) {
    spec.use_bfs_hints = ExtractBool(text, "use_bfs_hints", true);
  }
  spec.split_mode = ParseSlackPipeSplitMode(
      ExtractString(text, "split_mode", ToString(spec.split_mode)));
  spec.move_budget = ExtractLong(text, "move_budget", spec.move_budget);
  spec.move_budget_provided = HasKey(text, "move_budget");
  const long delta =
      ExtractLong(text, "per_stage_delta", std::numeric_limits<long>::min());
  if (delta != std::numeric_limits<long>::min()) spec.per_stage_delta = delta;
  spec.worker_move_budget =
      ExtractLong(text, "worker_move_budget", spec.worker_move_budget);
  spec.worker_move_budget_provided = HasKey(text, "worker_move_budget");
  const long worker_delta =
      ExtractLong(text, "per_worker_delta", std::numeric_limits<long>::min());
  if (worker_delta != std::numeric_limits<long>::min()) {
    spec.per_worker_delta = worker_delta;
  }
  spec.alternating_max_rounds = static_cast<int>(
      ExtractLong(text, "alternating_max_rounds", spec.alternating_max_rounds));
  return spec;
}

Tick SplitL1(const std::vector<Tick> &split,
             const std::vector<Tick> &baseline) {
  if (split.size() != baseline.size()) return 0;
  Tick total = 0;
  for (std::size_t i = 0; i < split.size(); ++i) {
    total += std::llabs(split[i] - baseline[i]);
  }
  return total;
}

Tick SplitMaxDeviation(const std::vector<Tick> &split,
                       const std::vector<Tick> &baseline) {
  if (split.size() != baseline.size()) return 0;
  Tick maximum = 0;
  for (std::size_t i = 0; i < split.size(); ++i) {
    maximum = std::max(maximum,
                       static_cast<Tick>(std::llabs(split[i] - baseline[i])));
  }
  return maximum;
}

void ValidateSchedule(
    const Instance &instance, BenchmarkRow &row,
    const ScheduleSolution &schedule,
    const ActivationAnalysisOptions &activation_options,
    const std::optional<std::vector<Tick>> &fixed_partition_reference =
        std::nullopt,
    const ActivationCapConstraintMetadata *activation_constraints = nullptr,
    const std::string &activation_enforcement_mode = "") {
  const auto start = Clock::now();
  row.canonical_validation_passed = false;
  try {
    const ResultValidationResult validation = [&]() {
      ResultValidationInput input =
          ValidationInputFromCanonical(instance, schedule, row.canonical);
      input.fixed_partition_reference = fixed_partition_reference;
      return ValidateResult(input);
    }();
    ApplyResultValidation(row.canonical.outcome, validation);
    row.canonical_validation_passed = validation.passed;
    if (validation.passed && schedule.ok()) {
      ActivationAnalysisResult activation =
          AnalyzeActivationMemoryWithUniformBaseline(
              instance, schedule, activation_options,
              activation_constraints != nullptr &&
                  activation_constraints->constraints_added);
      if (activation_constraints != nullptr) {
        ApplyActivationCapConstraintMetadata(
            activation, *activation_constraints, activation_enforcement_mode);
      } else if (!activation_enforcement_mode.empty()) {
        ActivationCapConstraintMetadata metadata;
        metadata.model_support_level = ToString(ActivationCapSolverSupport());
        metadata.solver_supported =
            ActivationCapSolverCanEnforce(activation_options, false);
        ApplyActivationCapConstraintMetadata(activation, metadata,
                                             activation_enforcement_mode);
      }
      ApplyActivationAnalysis(row.canonical, activation);
      row.activation_model = ToString(activation.model);
      row.activation_units_per_layer = activation.activation_units_per_layer;
      row.explicit_stage_activation_units =
          JoinTicks(activation.explicit_stage_activation_units);
      row.activation_cap_mode = ToString(activation.cap_mode);
      row.activation_cap_units_per_worker =
          JoinTicks(activation.activation_cap_units_per_worker);
      row.activation_cap_enforced = activation.activation_cap_enforced;
      row.activation_cap_enforced_in_solver =
          activation.activation_cap_enforced_in_solver;
      row.activation_cap_enforcement_requested =
          activation.activation_cap_enforcement_requested;
      row.activation_cap_enforcement_mode =
          activation.activation_cap_enforcement_mode;
      row.activation_cap_solver_supported =
          activation.activation_cap_solver_supported;
      row.activation_cap_solver_support_level =
          activation.activation_cap_solver_support_level;
      row.activation_cap_constraints_added =
          activation.activation_cap_constraints_added;
      row.activation_retained_interval_count =
          activation.activation_retained_interval_count;
      row.activation_cumulative_constraint_count =
          activation.activation_cumulative_constraint_count;
      row.activation_variable_demand_count =
          activation.activation_variable_demand_count;
      row.activation_fixed_demand_count =
          activation.activation_fixed_demand_count;
      row.activation_constraint_build_runtime_seconds =
          activation.activation_constraint_build_runtime_seconds;
      row.incumbent_rejected_for_activation_cap =
          activation.incumbent_rejected_for_activation_cap;
      row.activation_model_validation_agreement =
          activation.activation_model_validation_agreement;
      row.activation_model_disagreement_details =
          activation.activation_model_disagreement_details;
      row.activation_cap_formulation_version =
          activation.activation_cap_formulation_version;
      row.activation_cap_satisfied = activation.activation_cap_satisfied;
      row.maximum_worker_peak_activation_units =
          activation.global.maximum_worker_peak_activation_units;
      row.global_simultaneous_peak_activation_units =
          activation.global.peak_simultaneous_activation_units_across_workers;
      row.activation_peak_ratio_to_uniform =
          activation.activation_peak_ratio_to_uniform
              .maximum_worker_peak_units_ratio;
      row.activation_cap_derivation_hash =
          activation.activation_cap_derivation_hash;
      const bool cap_violation =
          activation.activation_cap_satisfied &&
          !activation.activation_cap_satisfied.value_or(true);
      if (!activation.passed ||
          (activation.activation_cap_enforced && cap_violation)) {
        row.canonical_validation_passed = false;
        if (IsTerminalFeasibleStatus(row.status)) row.status = "INVALID_RESULT";
        if (row.failure.empty()) {
          row.failure = row.canonical.outcome.result_validation_error.value_or(
              "activation analysis failed");
        }
      }
    }
    if (!row.canonical_validation_passed) {
      if (IsTerminalFeasibleStatus(row.status)) row.status = "INVALID_RESULT";
      if (row.failure.empty()) {
        row.failure = validation.error_code;
        if (!validation.message.empty()) {
          row.failure += ": " + validation.message;
        }
      }
    }
  } catch (const std::exception &) {
    row.canonical_validation_passed = false;
  }
  row.extraction_verification_seconds += Since(start);
}

BenchmarkRow BaseRow(const InstanceSpec &instance,
                     const AlgorithmSpec &algorithm, int repetition, int seed,
                     int workers, double timeout,
                     const std::string &requested_command,
                     const std::string &executable_name,
                     const ActivationAnalysisOptions &activation_options) {
  BenchmarkRow row;
  row.instance_name = instance.name;
  row.algorithm = algorithm.algorithm;
  row.configuration = algorithm.configuration;
  row.fixed_partition_source = algorithm.fixed_partition_source;
  row.supplied_split = JoinTicks(algorithm.fixed_split);
  row.hints_requested = algorithm.use_bfs_hints.value_or(
      algorithm.algorithm == BenchmarkAlgorithm::kJointCpSat ||
      algorithm.algorithm == BenchmarkAlgorithm::kScheduleOnly ||
      algorithm.algorithm == BenchmarkAlgorithm::kCanonicalSlackPipe);
  row.split_mode = ToString(algorithm.split_mode);
  row.worker_move_budget = algorithm.worker_move_budget;
  row.per_worker_delta = algorithm.per_worker_delta;
  row.repetition = repetition;
  row.seed = seed;
  row.cp_sat_workers = workers;
  row.run_id = DeterministicRunId(instance, algorithm, repetition, seed,
                                  workers, timeout, activation_options);
  row.activation_model = ToString(activation_options.model);
  row.activation_units_per_layer =
      activation_options.activation_units_per_layer;
  row.explicit_stage_activation_units =
      JoinTicks(activation_options.explicit_stage_activation_units);
  row.activation_cap_mode = ToString(activation_options.cap_mode);
  row.activation_cap_enforced = activation_options.enforce_activation_cap;
  row.activation_cap_enforcement_requested =
      activation_options.enforce_activation_cap;
  row.activation_cap_enforcement_mode =
      activation_options.cap_mode == ActivationCapMode::kNone ? "none"
                                                              : "posthoc_only";
  row.activation_cap_solver_supported =
      ActivationCapSolverSupport() != ActivationCapSolverSupportLevel::kNone;
  row.activation_cap_solver_support_level =
      ToString(ActivationCapSolverSupport());
  row.activation_cap_formulation_version = kActivationCapFormulationVersion;
  row.canonical = BuildCanonicalResultMetadata(
      instance.instance,
      BenchmarkRequestContext(row, algorithm, timeout, timeout,
                              requested_command, executable_name),
      PlannedSemanticsForAlgorithm(algorithm), CanonicalOutcome{}, std::nullopt,
      std::nullopt);
  switch (algorithm.algorithm) {
    case BenchmarkAlgorithm::kUniformFixedOrder:
      row.ablation_mode = "uniform-fixed-order";
      row.partition_search_enabled = false;
      row.schedule_search_enabled = false;
      row.partition_fixed_validation_passed = true;
      row.operation_order_fixed_validation_passed = true;
      break;
    case BenchmarkAlgorithm::kJointCpSat:
      row.ablation_mode = "joint";
      row.partition_search_enabled = true;
      row.schedule_search_enabled = true;
      break;
    case BenchmarkAlgorithm::kPartitionOnly:
    case BenchmarkAlgorithm::kOptimizedBfs:
      row.ablation_mode =
          algorithm.algorithm == BenchmarkAlgorithm::kPartitionOnly
              ? "partition-only"
              : "";
      row.partition_search_enabled = true;
      row.schedule_search_enabled = false;
      row.operation_order_fixed_validation_passed = true;
      break;
    case BenchmarkAlgorithm::kScheduleOnly:
      row.ablation_mode = "schedule-only";
      row.partition_search_enabled = false;
      row.schedule_search_enabled = true;
      row.partition_fixed_validation_passed = true;
      break;
    case BenchmarkAlgorithm::kSequentialPartitionThenSchedule:
    case BenchmarkAlgorithm::kAlternatingPartitionSchedule:
      row.ablation_mode =
          algorithm.algorithm ==
                  BenchmarkAlgorithm::kSequentialPartitionThenSchedule
              ? "sequential-partition-then-schedule"
              : "alternating-partition-schedule";
      row.partition_search_enabled = true;
      row.schedule_search_enabled = true;
      break;
    case BenchmarkAlgorithm::kCanonicalSlackPipe:
      break;
  }
  return row;
}

void ApplyGap(BenchmarkRow &row, std::optional<Tick> joint_optimum) {
  const GapMetrics gap = CalculateGap(joint_optimum, row.makespan);
  row.joint_optimum_when_available = gap.joint_optimum_when_available;
  row.absolute_gap_to_joint = gap.absolute_gap_to_joint;
  row.relative_gap_to_joint = gap.relative_gap_to_joint;
  row.reaches_joint_objective = gap.reaches_joint_objective;
}

void ApplyTiming(BenchmarkRow &row, const SolverPhaseTiming &timing) {
  row.model_build_seconds = timing.model_build_seconds;
  row.external_solver_seconds = timing.solver_seconds;
  row.ortools_wall_time_seconds = timing.ortools_wall_time_seconds;
  row.cp_sat_solve_seconds = timing.solver_seconds;
  row.solver_internal_wall_seconds = timing.ortools_wall_time_seconds;
  row.extraction_seconds = timing.extraction_seconds;
  row.canonicalization_seconds = timing.canonicalization_seconds;
  row.extraction_verification_seconds =
      timing.extraction_seconds + timing.canonicalization_seconds;
  row.policy_seconds = timing.policy_seconds;
  row.bfs_seconds = timing.incumbent_seconds;
  row.orchestration_seconds = timing.OrchestrationSeconds();
  if (row.cp_sat_models_solved > 0) {
    row.model_build_seconds_per_model =
        row.model_build_seconds / static_cast<double>(row.cp_sat_models_solved);
    row.solver_seconds_per_model =
        row.external_solver_seconds /
        static_cast<double>(row.cp_sat_models_solved);
  }
}

void ApplyScheduleMetrics(BenchmarkRow &row, const Instance &instance,
                          const ScheduleSolution &schedule) {
  const ScheduleMetrics metrics = ComputeScheduleMetrics(instance, schedule);
  row.simulated_iteration_time = metrics.simulated_iteration_time;
  row.pipeline_utilization = metrics.pipeline_utilization;
  row.per_worker_busy_time = JoinTicks(metrics.per_worker_busy_time);
  row.per_worker_idle_time = JoinTicks(metrics.per_worker_idle_time);
  row.maximum_worker_load = metrics.max_worker_load;
  row.pipeline_fill_time = metrics.pipeline_fill_time;
  row.pipeline_drain_time = metrics.pipeline_drain_time;
  row.communication_blocked_time = metrics.communication_blocked_time;
}

void ApplyJointProvenance(BenchmarkRow &row,
                          const JointOptimizationResult &result) {
  row.hints_requested = result.hints_requested;
  row.hints_effective = result.hints_effective;
  row.hint_source = result.hint_source;
  row.hint_scope = result.hint_scope;
  row.hint_complete_for_basic_model = result.hint_complete_for_basic_model;
  row.hint_complete_for_full_model = result.hint_complete_for_full_model;
  row.hinted_layer_variable_count = result.hinted_layer_variable_count;
  row.hinted_operation_variable_count = result.hinted_operation_variable_count;
  row.hinted_scalar_variable_count = result.hinted_scalar_variable_count;
  row.hinted_auxiliary_variable_count = result.hinted_auxiliary_variable_count;
  row.hinted_total_variable_count = result.hinted_total_variable_count;
  row.fallback_available = result.fallback_available;
  row.fallback_used = result.fallback_used;
  row.fallback_source = result.fallback_source;
  row.solution_source = result.solution_source;
  row.bfs_incumbent_method_requested = result.bfs_incumbent_method_requested;
  row.bfs_incumbent_method_effective = result.bfs_incumbent_method_effective;
  row.incumbent_method_requested = result.incumbent_method_requested;
  row.incumbent_method_effective = result.incumbent_method_effective;
  row.incumbent_source = result.incumbent_source;
  row.incumbent_feasible = result.incumbent_feasible;
  row.incumbent_primary_objective = result.incumbent_primary_objective;
  row.incumbent_hybrid_min_slack = result.incumbent_hybrid_min_slack;
  row.incumbent_baseline_primary_objective =
      result.incumbent_baseline_primary_objective;
  row.incumbent_baseline_hybrid_min_slack =
      result.incumbent_baseline_hybrid_min_slack;
  row.incumbent_improved_over_baseline =
      result.incumbent_improved_over_baseline;
  row.incumbent_hybrid_stage_scores =
      JoinDoubles(result.incumbent_hybrid_stage_scores);
  row.incumbent_hybrid_bottleneck_stages =
      JoinTicks(result.incumbent_hybrid_bottleneck_stages);
  row.horizon_source = result.horizon_source;
  row.hint_budget_seconds = result.hint_budget_seconds;
  row.hint_elapsed_seconds = result.hint_elapsed_seconds;
  row.hint_iterations = result.hint_iterations;
  row.hint_candidates_generated = result.hint_candidates_generated;
  row.hint_candidates_simulated = result.hint_candidates_simulated;
  row.hint_partition_moves_accepted = result.hint_partition_moves_accepted;
  row.hint_interleaving_moves_accepted =
      result.hint_interleaving_moves_accepted;
  row.hint_deadline_reached = result.hint_deadline_reached;
  row.hint_termination_reason = result.hint_termination_reason;
}

void ApplySlackPipeJointProvenance(BenchmarkRow &row,
                                   const SlackPipeResult &result) {
  row.hints_requested = result.joint_hints_requested;
  row.hints_effective = result.joint_hints_effective;
  row.hint_source = result.joint_hint_source;
  row.hint_scope = result.joint_hint_scope;
  row.hint_complete_for_basic_model =
      result.joint_hint_complete_for_basic_model;
  row.hint_complete_for_full_model = result.joint_hint_complete_for_full_model;
  row.hinted_layer_variable_count = result.joint_hinted_layer_variable_count;
  row.hinted_operation_variable_count =
      result.joint_hinted_operation_variable_count;
  row.hinted_scalar_variable_count = result.joint_hinted_scalar_variable_count;
  row.hinted_auxiliary_variable_count =
      result.joint_hinted_auxiliary_variable_count;
  row.hinted_total_variable_count = result.joint_hinted_total_variable_count;
  row.fallback_available = result.joint_fallback_available;
  row.fallback_used = result.joint_fallback_used;
  row.fallback_source = result.joint_fallback_source;
  row.solution_source = result.joint_solution_source;
  row.bfs_incumbent_method_requested =
      result.joint_bfs_incumbent_method_requested;
  row.bfs_incumbent_method_effective =
      result.joint_bfs_incumbent_method_effective;
  row.incumbent_method_requested = result.joint_incumbent_method_requested;
  row.incumbent_method_effective = result.joint_incumbent_method_effective;
  row.incumbent_source = result.joint_incumbent_source;
  row.incumbent_feasible = result.joint_incumbent_feasible;
  row.incumbent_primary_objective = result.joint_incumbent_primary_objective;
  row.incumbent_hybrid_min_slack = result.joint_incumbent_hybrid_min_slack;
  row.incumbent_baseline_primary_objective =
      result.joint_incumbent_baseline_primary_objective;
  row.incumbent_baseline_hybrid_min_slack =
      result.joint_incumbent_baseline_hybrid_min_slack;
  row.incumbent_improved_over_baseline =
      result.joint_incumbent_improved_over_baseline;
  row.incumbent_hybrid_stage_scores =
      JoinDoubles(result.joint_incumbent_hybrid_stage_scores);
  row.incumbent_hybrid_bottleneck_stages =
      JoinTicks(result.joint_incumbent_hybrid_bottleneck_stages);
  row.horizon_source = result.joint_horizon_source;
  row.hint_budget_seconds = result.joint_hint_budget_seconds;
  row.hint_elapsed_seconds = result.joint_hint_elapsed_seconds;
  row.hint_iterations = result.joint_hint_iterations;
  row.hint_candidates_generated = result.joint_hint_candidates_generated;
  row.hint_candidates_simulated = result.joint_hint_candidates_simulated;
  row.hint_partition_moves_accepted =
      result.joint_hint_partition_moves_accepted;
  row.hint_interleaving_moves_accepted =
      result.joint_hint_interleaving_moves_accepted;
  row.hint_deadline_reached = result.joint_hint_deadline_reached;
  row.hint_termination_reason = result.joint_hint_termination_reason;
}

void ValidateAblationLabel(BenchmarkRow &row) {
  if (row.ablation_mode.empty()) {
    row.label_validation_passed = true;
  } else {
    const bool label_matches =
        (row.ablation_mode == "joint" && row.partition_search_enabled &&
         row.schedule_search_enabled) ||
        (row.ablation_mode == "partition-only" &&
         row.partition_search_enabled && !row.schedule_search_enabled) ||
        (row.ablation_mode == "schedule-only" &&
         !row.partition_search_enabled && row.schedule_search_enabled) ||
        (row.ablation_mode == "uniform-fixed-order" &&
         !row.partition_search_enabled && !row.schedule_search_enabled) ||
        ((row.ablation_mode == "sequential-partition-then-schedule" ||
          row.ablation_mode == "alternating-partition-schedule") &&
         row.partition_search_enabled && row.schedule_search_enabled);
    if (!label_matches) {
      throw Error("ABLATION_LABEL_RESTRICTION_MISMATCH configuration=" +
                  row.configuration);
    }
    row.label_validation_passed = true;
  }

  if (row.configuration.find("no-hint") != std::string::npos &&
      row.hints_effective) {
    throw Error("HINT_LABEL_RESTRICTION_MISMATCH configuration=" +
                row.configuration);
  }
  if ((row.configuration.find("with-hints") != std::string::npos ||
       row.configuration.find("hinted") != std::string::npos) &&
      !row.hints_effective) {
    throw Error("HINT_LABEL_RESTRICTION_MISMATCH configuration=" +
                row.configuration);
  }
  row.label_validation_passed = true;
}

struct FixedPartitionResolution {
  std::vector<Tick> split;
  std::string reference_source;
  double phase_limit_seconds = 0.0;
  double runtime_seconds = 0.0;
  Index cp_sat_models_solved = 0;
  std::string status;
};

FixedPartitionResolution ResolveFixedPartition(
    const Instance &instance, const AlgorithmSpec &algorithm, int workers,
    int seed, double timeout, const Deadline &deadline,
    const ActivationAnalysisOptions &activation_options) {
  const auto started = Clock::now();
  FixedPartitionResolution resolution;
  if (!algorithm.fixed_split.empty()) {
    ValidateSplit(instance, algorithm.fixed_split);
    resolution.split = algorithm.fixed_split;
    resolution.reference_source = "supplied input";
    resolution.status = "SUPPLIED";
    resolution.runtime_seconds = Since(started);
    return resolution;
  }
  if (algorithm.fixed_partition_source == "uniform" ||
      algorithm.fixed_partition_source.empty()) {
    resolution.split = UniformSplit(instance);
    resolution.reference_source = "uniform_deterministic";
    resolution.status = "DETERMINISTIC";
    resolution.runtime_seconds = Since(started);
    return resolution;
  }
  if (algorithm.fixed_partition_source == "load-balanced") {
    resolution.split = LoadBalancedSplit(instance);
    resolution.reference_source = "load-balanced construction";
    resolution.status = "DETERMINISTIC";
    resolution.runtime_seconds = Since(started);
    return resolution;
  }
  if (algorithm.fixed_partition_source == "partition-only") {
    BfsSplitOptimizerOptions options;
    options.num_workers = workers;
    options.random_seed = seed;
    options.enumeration_threshold =
        timeout > 0.0 ? 0 : options.enumeration_threshold;
    resolution.phase_limit_seconds =
        timeout > 0.0 ? deadline.clamp_solver_limit(
                            timeout * kScheduleOnlyPartitionBudgetFraction)
                      : 0.0;
    if (timeout > 0.0 && resolution.phase_limit_seconds <= 0.0) {
      resolution.split = UniformSplit(instance);
      resolution.reference_source = "partition_only_within_global_budget";
      resolution.status = "NOT_RUN";
      resolution.runtime_seconds = Since(started);
      return resolution;
    }
    options.time_limit_seconds = resolution.phase_limit_seconds;
    options.require_optimal = false;
    options.activation_options = activation_options;
    const BfsSplitOptimizationResult result =
        OptimizeBfsSplitAuto(instance, options);
    if (!(result.status == "OPTIMAL" || result.status == "FEASIBLE")) {
      throw Error("partition-only fixed partition source did not solve");
    }
    if (result.machine_orders != BreadthFirstOrders(instance)) {
      throw Error("PARTITION_ONLY_ORDER_MUTATION canonical order changed");
    }
    resolution.split = result.split;
    resolution.reference_source = "partition_only_within_global_budget";
    resolution.status = result.status;
    resolution.cp_sat_models_solved =
        result.method.find("cpsat") == std::string::npos ? 0 : 1;
    resolution.runtime_seconds = Since(started);
    return resolution;
  }
  throw Error("unknown fixed partition source: " +
              algorithm.fixed_partition_source);
}

BenchmarkRow RunOne(const InstanceSpec &instance,
                    const AlgorithmSpec &algorithm, int repetition, int seed,
                    int workers, double timeout,
                    std::optional<Tick> joint_optimum,
                    const std::string &requested_command,
                    const std::string &executable_name,
                    const ActivationAnalysisOptions &activation_options_input) {
  ActivationAnalysisOptions activation_options = activation_options_input;
  if (activation_options.cap_mode == ActivationCapMode::kUniformBaseline &&
      !activation_options.uniform_baseline) {
    activation_options.uniform_baseline =
        DeriveUniformActivationBaseline(instance.instance, activation_options);
  }
  BenchmarkRow row =
      BaseRow(instance, algorithm, repetition, seed, workers, timeout,
              requested_command, executable_name, activation_options);
  const long cpu_start = ProcessCpuMicros();
  const auto total_start = Clock::now();
  const Deadline row_deadline(timeout);
  bool reached_joint_objective = false;
  double first_joint_time_seconds = 0.0;
  try {
    if (algorithm.algorithm == BenchmarkAlgorithm::kUniformFixedOrder) {
      const MachineOrders orders = BreadthFirstOrders(instance.instance);
      const std::vector<Tick> uniform_split = UniformSplit(instance.instance);
      const EvaluationResult evaluated =
          EvaluateSchedule(instance.instance, uniform_split, orders);
      row.status = evaluated.schedule.ok() ? "FEASIBLE" : "INVALID";
      row.proven_optimal = false;
      row.makespan = evaluated.schedule.makespan;
      row.simulated_iteration_time = evaluated.schedule.makespan;
      row.initial_objective = evaluated.schedule.makespan;
      row.first_feasible_objective = evaluated.schedule.makespan;
      row.final_objective = evaluated.schedule.makespan;
      row.objective = static_cast<double>(evaluated.schedule.makespan);
      row.best_bound = 0.0;
      row.split = JoinTicks(uniform_split);
      row.worker_local_operation_order = JoinOrders(instance.instance, orders);
      row.supplied_split = JoinTicks(uniform_split);
      row.canonical = BuildCanonicalResultMetadata(
          instance.instance,
          BenchmarkRequestContext(row, algorithm, timeout, timeout,
                                  requested_command, executable_name),
          SemanticsForUniformFixedOrderBaseline(),
          OutcomeFromSchedule(evaluated.schedule, row.status), uniform_split,
          orders);
      ApplyScheduleMetrics(row, instance.instance, evaluated.schedule);
      row.partition_changed = false;
      row.operation_order_changed = false;
      row.partition_fixed_validation_passed = true;
      row.operation_order_fixed_validation_passed = true;
      ValidateAblationLabel(row);
      ValidateSchedule(instance.instance, row, evaluated.schedule,
                       activation_options, std::nullopt, nullptr,
                       activation_options.enforce_activation_cap
                           ? "deterministic_postconstruction_check"
                           : "");
    } else if (algorithm.algorithm == BenchmarkAlgorithm::kOptimizedBfs ||
               algorithm.algorithm == BenchmarkAlgorithm::kPartitionOnly) {
      BfsSplitOptimizerOptions options;
      options.num_workers = workers;
      options.random_seed = seed;
      options.time_limit_seconds = timeout;
      options.require_optimal = false;
      options.fixed_order_partition_backend =
          algorithm.fixed_order_partition_backend;
      options.activation_options = activation_options;
      const MachineOrders fixed_orders = BreadthFirstOrders(instance.instance);
      const BfsSplitOptimizationResult result = OptimizePartitionForFixedOrder(
          instance.instance, fixed_orders, options);
      row.status = result.status;
      row.proven_optimal = result.proven_optimal;
      row.makespan = result.makespan_ticks;
      row.simulated_iteration_time = result.makespan_ticks;
      row.final_objective = result.makespan_ticks;
      row.objective = static_cast<double>(result.makespan_ticks);
      row.best_bound = static_cast<double>(result.best_bound_ticks);
      row.split = JoinTicks(result.split);
      row.worker_local_operation_order =
          JoinOrders(instance.instance, result.machine_orders);
      const MachineOrders canonical_orders =
          BreadthFirstOrders(instance.instance);
      const std::vector<Tick> uniform_split = UniformSplit(instance.instance);
      const EvaluationResult uniform_canonical =
          EvaluateSchedule(instance.instance, uniform_split, canonical_orders);
      if (uniform_canonical.schedule.ok()) {
        row.initial_objective = uniform_canonical.schedule.makespan;
      }
      row.first_feasible_objective = result.makespan_ticks;
      row.cp_sat_models_solved = result.cp_sat_models_solved;
      row.branches = result.branches;
      row.conflicts = result.conflicts;
      ApplyTiming(row, result.timing);
      row.canonical = BuildCanonicalResultMetadata(
          instance.instance,
          BenchmarkRequestContext(row, algorithm, timeout, timeout,
                                  requested_command, executable_name),
          SemanticsForPartitionOnlyFixedOrder(result),
          OutcomeFromBfsResult(result), result.split, result.machine_orders);
      if (joint_optimum && result.makespan_ticks <= *joint_optimum) {
        reached_joint_objective = true;
        first_joint_time_seconds = result.timing.total_seconds;
      }
      ApplyScheduleMetrics(row, instance.instance, result.schedule);
      row.operation_order_changed =
          !OrdersEqual(result.machine_orders, canonical_orders);
      row.partition_changed = result.split != uniform_split;
      if (algorithm.algorithm == BenchmarkAlgorithm::kPartitionOnly &&
          row.operation_order_changed) {
        throw Error("PARTITION_ONLY_ORDER_MUTATION canonical order changed");
      }
      if (row.ablation_mode == "partition-only") {
        row.operation_order_fixed_validation_passed =
            !row.operation_order_changed;
      }
      if (row.ablation_mode != "joint" && row.partition_changed &&
          row.operation_order_changed) {
        throw Error("ONLY_JOINT_MAY_CHANGE_PARTITION_AND_ORDER");
      }
      ValidateAblationLabel(row);
      ValidateSchedule(instance.instance, row, result.schedule,
                       activation_options, std::nullopt,
                       &result.activation_cap_constraints);
    } else if (algorithm.algorithm == BenchmarkAlgorithm::kJointCpSat) {
      JointOptimizerOptions options;
      options.num_workers = workers;
      options.random_seed = seed;
      options.time_limit_seconds = timeout;
      options.require_optimal = false;
      options.activation_options = activation_options;
      if (algorithm.use_bfs_hints)
        options.use_bfs_hints = *algorithm.use_bfs_hints;
      const JointOptimizationResult result =
          OptimizeJointSplitAndScheduleCpSat(instance.instance, options);
      row.status = result.status;
      row.proven_optimal = result.proven_optimal;
      row.makespan = result.makespan_ticks;
      row.simulated_iteration_time = result.makespan_ticks;
      row.final_objective = result.makespan_ticks;
      row.objective = result.solver_objective_ticks;
      row.best_bound = result.best_bound_ticks;
      row.split = JoinTicks(result.split);
      row.worker_local_operation_order =
          JoinOrders(instance.instance, result.machine_orders);
      row.solver_deterministic_time = result.deterministic_time;
      row.cp_sat_models_solved = result.cp_sat_models_solved;
      row.branches = result.branches;
      row.conflicts = result.conflicts;
      row.time_to_first_feasible_seconds =
          result.time_to_first_feasible_seconds;
      row.time_to_best_incumbent_seconds =
          result.time_to_best_incumbent_seconds;
      row.first_feasible_objective = result.first_feasible_objective;
      row.incumbent_improvement_count = result.incumbent_improvement_count;
      ApplyJointProvenance(row, result);
      ApplyTiming(row, result.timing);
      row.canonical = BuildCanonicalResultMetadata(
          instance.instance,
          BenchmarkRequestContext(row, algorithm, timeout,
                                  result.joint_budget_seconds,
                                  requested_command, executable_name),
          SemanticsForJointUnrestrictedNoOverlap(result, false),
          OutcomeFromJointResult(result), result.split, result.machine_orders);
      if (joint_optimum && result.makespan_ticks <= *joint_optimum) {
        reached_joint_objective = true;
        first_joint_time_seconds = result.timing.total_seconds;
      }
      ApplyScheduleMetrics(row, instance.instance, result.schedule);
      row.partition_changed = result.split != UniformSplit(instance.instance);
      row.operation_order_changed = !OrdersEqual(
          result.machine_orders, BreadthFirstOrders(instance.instance));
      ValidateAblationLabel(row);
      ValidateSchedule(instance.instance, row, result.schedule,
                       activation_options, std::nullopt,
                       &result.activation_cap_constraints);
    } else if (algorithm.algorithm == BenchmarkAlgorithm::kScheduleOnly) {
      const FixedPartitionResolution fixed_partition =
          ResolveFixedPartition(instance.instance, algorithm, workers, seed,
                                timeout, row_deadline, activation_options);
      const std::vector<Tick> &fixed_split = fixed_partition.split;
      row.supplied_split = JoinTicks(fixed_split);
      row.fixed_partition_source = fixed_partition.reference_source;
      const MachineOrders canonical_orders =
          BreadthFirstOrders(instance.instance);
      const EvaluationResult canonical =
          EvaluateSchedule(instance.instance, fixed_split, canonical_orders);
      if (!canonical.schedule.ok()) {
        throw Error("canonical fixed-partition schedule is invalid");
      }

      JointOptimizerOptions options;
      options.num_workers = workers;
      options.random_seed = seed;
      options.time_limit_seconds =
          timeout > 0.0 ? row_deadline.clamp_solver_limit(0.0) : 0.0;
      options.require_optimal = false;
      options.activation_options = activation_options;
      if (algorithm.use_bfs_hints)
        options.use_bfs_hints = *algorithm.use_bfs_hints;
      JointOptimizationResult result =
          timeout > 0.0 && row_deadline.expired()
              ? BuildScheduleOnlyFixedSplitDeadlineFallback(
                    instance.instance, fixed_split, options,
                    row_deadline.elapsed_seconds(),
                    "global_deadline_expired_before_cp_sat")
              : OptimizeScheduleForFixedSplitCpSat(instance.instance,
                                                   fixed_split, options);
      result.cp_sat_models_solved += fixed_partition.cp_sat_models_solved;
      row.status = result.status;
      row.proven_optimal = result.proven_optimal;
      row.makespan = result.makespan_ticks;
      row.simulated_iteration_time = result.makespan_ticks;
      row.initial_objective = canonical.schedule.makespan;
      row.first_feasible_objective = result.first_feasible_objective;
      row.final_objective = result.makespan_ticks;
      row.objective = result.solver_objective_ticks;
      row.best_bound = result.best_bound_ticks;
      row.split = JoinTicks(result.split);
      row.worker_local_operation_order =
          JoinOrders(instance.instance, result.machine_orders);
      row.solver_deterministic_time = result.deterministic_time;
      row.cp_sat_models_solved = result.cp_sat_models_solved;
      row.branches = result.branches;
      row.conflicts = result.conflicts;
      row.time_to_first_feasible_seconds =
          result.time_to_first_feasible_seconds;
      row.time_to_best_incumbent_seconds =
          result.time_to_best_incumbent_seconds;
      row.incumbent_improvement_count = result.incumbent_improvement_count;
      ApplyJointProvenance(row, result);
      ApplyTiming(row, result.timing);
      row.optimized_bfs_seconds = fixed_partition.runtime_seconds;
      CanonicalOutcome outcome = OutcomeFromJointResult(result);
      if (fixed_partition.runtime_seconds > 0.0) {
        outcome.reference_runtime_seconds = fixed_partition.runtime_seconds;
      }
      if (fixed_partition.phase_limit_seconds > 0.0) {
        outcome.phase_budget.reference_phase_limit_seconds =
            fixed_partition.phase_limit_seconds;
      }
      CanonicalPhaseBudget partition_phase;
      partition_phase.phase = "fixed_partition_reference";
      partition_phase.effective_limit_seconds =
          fixed_partition.phase_limit_seconds > 0.0
              ? std::optional<double>(fixed_partition.phase_limit_seconds)
              : std::nullopt;
      partition_phase.runtime_seconds =
          fixed_partition.runtime_seconds > 0.0
              ? std::optional<double>(fixed_partition.runtime_seconds)
              : std::nullopt;
      partition_phase.status =
          fixed_partition.status.empty()
              ? std::nullopt
              : std::optional<std::string>(fixed_partition.status);
      outcome.phase_budget.phases.insert(outcome.phase_budget.phases.begin(),
                                         partition_phase);
      row.canonical = BuildCanonicalResultMetadata(
          instance.instance,
          BenchmarkRequestContext(row, algorithm, timeout,
                                  result.joint_budget_seconds,
                                  requested_command, executable_name),
          SemanticsForScheduleOnlyFixedSplit(fixed_partition.reference_source,
                                             result, false),
          outcome, result.split, result.machine_orders);
      if (joint_optimum && result.makespan_ticks <= *joint_optimum) {
        reached_joint_objective = true;
        first_joint_time_seconds = result.timing.total_seconds;
      }
      ApplyScheduleMetrics(row, instance.instance, result.schedule);
      row.partition_changed = result.split != fixed_split;
      row.operation_order_changed =
          !OrdersEqual(result.machine_orders, canonical_orders);
      row.partition_fixed_validation_passed = !row.partition_changed;
      if (row.partition_changed) {
        throw Error("SCHEDULE_ONLY_PARTITION_MUTATION fixed partition changed");
      }
      if (row.ablation_mode != "joint" && row.partition_changed &&
          row.operation_order_changed) {
        throw Error("ONLY_JOINT_MAY_CHANGE_PARTITION_AND_ORDER");
      }
      ValidateAblationLabel(row);
      ValidateSchedule(instance.instance, row, result.schedule,
                       activation_options, fixed_split,
                       &result.activation_cap_constraints);
    } else if (algorithm.algorithm ==
                   BenchmarkAlgorithm::kSequentialPartitionThenSchedule ||
               algorithm.algorithm ==
                   BenchmarkAlgorithm::kAlternatingPartitionSchedule) {
      AlternatingOptimizerOptions options;
      options.num_workers = workers;
      options.random_seed = seed;
      options.time_limit_seconds = timeout;
      options.require_optimal = false;
      options.max_rounds = algorithm.alternating_max_rounds;
      options.activation_options = activation_options;
      if (algorithm.use_bfs_hints) {
        options.use_bfs_hints = *algorithm.use_bfs_hints;
      }
      const AlternatingOptimizationResult result =
          algorithm.algorithm ==
                  BenchmarkAlgorithm::kSequentialPartitionThenSchedule
              ? OptimizeSequentialPartitionThenSchedule(instance.instance,
                                                        options)
              : OptimizeAlternatingPartitionSchedule(instance.instance,
                                                     options);
      row.status = result.status;
      row.proven_optimal = result.proven_optimal;
      row.makespan = result.makespan_ticks;
      row.simulated_iteration_time = result.makespan_ticks;
      row.initial_objective = result.initial_makespan;
      row.first_feasible_objective = result.initial_makespan;
      row.final_objective = result.makespan_ticks;
      row.objective = static_cast<double>(result.makespan_ticks);
      row.best_bound = result.best_bound_ticks;
      row.split = JoinTicks(result.split);
      row.worker_local_operation_order =
          JoinOrders(instance.instance, result.machine_orders);
      row.cp_sat_models_solved = result.cp_sat_models_solved;
      ApplyTiming(row, result.timing);
      row.time_to_first_feasible_seconds = 0.0;
      row.time_to_best_incumbent_seconds = result.wall_time_seconds;
      row.canonical = BuildCanonicalResultMetadata(
          instance.instance,
          BenchmarkRequestContext(row, algorithm, timeout, timeout,
                                  requested_command, executable_name),
          algorithm.algorithm ==
                  BenchmarkAlgorithm::kSequentialPartitionThenSchedule
              ? SemanticsForSequentialPartitionThenSchedule(
                    result.cp_sat_models_solved)
              : SemanticsForAlternatingPartitionSchedule(
                    result.cp_sat_models_solved),
          OutcomeFromAlternatingResult(result), result.split,
          result.machine_orders);
      ApplyAlternatingCanonicalFields(result, row.canonical);
      if (joint_optimum && result.makespan_ticks <= *joint_optimum) {
        reached_joint_objective = true;
        first_joint_time_seconds = result.wall_time_seconds;
      }
      ApplyScheduleMetrics(row, instance.instance, result.schedule);
      row.partition_changed = result.split != UniformSplit(instance.instance);
      row.operation_order_changed = !OrdersEqual(
          result.machine_orders, BreadthFirstOrders(instance.instance));
      ValidateAblationLabel(row);
      ValidateSchedule(instance.instance, row, result.schedule,
                       activation_options, std::nullopt,
                       &result.activation_cap_constraints);
    } else if (algorithm.algorithm == BenchmarkAlgorithm::kCanonicalSlackPipe) {
      SlackPipeOptions options;
      options.num_workers = workers;
      options.random_seed = seed;
      options.time_limit_seconds = timeout;
      options.require_optimal = false;
      options.split_mode = algorithm.split_mode;
      options.move_budget = algorithm.move_budget;
      options.move_budget_provided = algorithm.move_budget_provided;
      options.per_stage_delta = algorithm.per_stage_delta;
      options.worker_move_budget = algorithm.worker_move_budget;
      options.worker_move_budget_provided =
          algorithm.worker_move_budget_provided;
      options.per_worker_delta = algorithm.per_worker_delta;
      options.activation_options = activation_options;
      if (algorithm.use_bfs_hints)
        options.use_bfs_hints = *algorithm.use_bfs_hints;
      const SlackPipeResult result =
          SolveCanonicalSlackPipe(instance.instance, options);
      row.status = result.status;
      if (row.status.empty() && !result.diagnostic.empty()) {
        row.status = "UNAVAILABLE";
        row.failure = result.diagnostic;
      }
      row.proven_optimal = result.proven_optimal;
      row.makespan = result.makespan_ticks;
      row.simulated_iteration_time = result.makespan_ticks;
      row.initial_objective = result.bfs.makespan_ticks;
      row.first_feasible_objective = result.first_feasible_objective;
      row.final_objective = result.makespan_ticks;
      row.objective = result.solver_objective_ticks;
      row.best_bound = result.best_bound_ticks;
      row.split = JoinTicks(result.schedule.split);
      row.worker_local_operation_order =
          JoinOrders(instance.instance, result.machine_orders);
      row.branches = result.branches;
      row.conflicts = result.conflicts;
      row.cp_sat_models_solved = result.cp_sat_models_solved;
      ApplyTiming(row, result.timing);
      row.uniform_bfs_seconds = result.bfs.timing.total_seconds;
      row.time_to_first_feasible_seconds =
          result.time_to_first_feasible_seconds;
      row.time_to_best_incumbent_seconds =
          result.time_to_best_incumbent_seconds;
      row.incumbent_improvement_count = result.incumbent_improvement_count;
      ApplySlackPipeJointProvenance(row, result);
      row.canonical = BuildCanonicalResultMetadata(
          instance.instance,
          BenchmarkRequestContext(row, algorithm, timeout,
                                  result.joint_remaining_budget_seconds,
                                  requested_command, executable_name),
          SemanticsForSlackPipe(algorithm.split_mode, result, false),
          OutcomeFromSlackPipeResult(result), result.split,
          result.machine_orders);
      row.baseline_worker_layers = JoinTicks(result.baseline_worker_layers);
      row.final_worker_layers = JoinTicks(result.final_worker_layers);
      row.worker_balance_l1 = result.worker_balance_l1;
      row.worker_balance_max_deviation = result.worker_balance_max_deviation;
      row.stage_split_l1 = SplitL1(result.split, result.bfs.split);
      row.stage_split_max_deviation =
          SplitMaxDeviation(result.split, result.bfs.split);
      row.proven_global_optimal = result.proven_global_optimal;
      row.global_certificate = result.global_certificate;
      row.time_to_global_certificate_seconds =
          result.proven_global_optimal ? result.total_wall_time_seconds : 0.0;
      if (joint_optimum && result.makespan_ticks <= *joint_optimum) {
        reached_joint_objective = true;
        first_joint_time_seconds = result.timing.total_seconds;
      }
      ApplyScheduleMetrics(row, instance.instance, result.schedule);
      ValidateSchedule(instance.instance, row, result.schedule,
                       activation_options,
                       SlackPipeModeFixesFullPartition(algorithm.split_mode)
                           ? std::optional<std::vector<Tick>>(result.bfs.split)
                           : std::nullopt,
                       &result.activation_cap_constraints);
    }
    row.total_planning_seconds = Since(total_start);
    row.external_total_seconds = row.total_planning_seconds;
    row.orchestration_seconds = std::max(
        0.0, row.external_total_seconds -
                 (row.bfs_seconds + row.model_build_seconds +
                  row.external_solver_seconds + row.extraction_seconds +
                  row.canonicalization_seconds + row.policy_seconds));
    if (row.best_bound > 0.0 && row.makespan > 0) {
      row.certified_gap =
          std::max(0.0, (static_cast<double>(row.makespan) - row.best_bound) /
                            static_cast<double>(row.makespan));
    }
    ApplyGap(row, joint_optimum);
    if (row.reaches_joint_objective || reached_joint_objective) {
      row.time_to_first_joint_objective_seconds =
          first_joint_time_seconds > 0.0 ? first_joint_time_seconds
                                         : row.total_planning_seconds;
    }
    row.completed = true;
  } catch (const std::exception &error) {
    row.status = "CRASHED";
    row.failure = error.what();
    row.total_planning_seconds = Since(total_start);
    row.external_total_seconds = row.total_planning_seconds;
    row.completed = true;
  }
  if (row.canonical.outcome.reported_status.value_or("").empty() &&
      row.status == "CRASHED") {
    CanonicalOutcome outcome;
    outcome.solver_status_raw = "CRASHED";
    outcome.reported_status = "CRASHED";
    outcome.feasible = false;
    outcome.optimal = false;
    outcome.total_runtime_seconds = row.total_planning_seconds;
    outcome.result_validation_passed = false;
    outcome.result_validation_error = row.failure;
    row.canonical = BuildCanonicalResultMetadata(
        instance.instance,
        BenchmarkRequestContext(row, algorithm, timeout, timeout,
                                requested_command, executable_name),
        PlannedSemanticsForAlgorithm(algorithm), outcome, std::nullopt,
        std::nullopt);
  }
  row.process_cpu_micros = ProcessCpuMicros() - cpu_start;
  row.peak_rss_kb = PeakRssKb();
  return row;
}

void WriteSummary(const std::filesystem::path &path,
                  const std::vector<BenchmarkRow> &rows) {
  std::ofstream out(path, std::ios::trunc);
  out << "algorithm,configuration,runtime_metric,count,successful_count,"
         "timeout_count,"
         "optimal_count,median,mean,standard_deviation,minimum,maximum,p25,p75,"
         "p90,geometric_mean_positive_runtime_ratio\n";
  std::map<std::string, std::vector<BenchmarkRow>> groups;
  for (const BenchmarkRow &row : rows) {
    groups[ToString(row.algorithm) + "|" + row.configuration].push_back(row);
  }
  for (const auto &entry : groups) {
    const SummaryStats stats = ComputeSummaryStats(entry.second);
    const std::size_t split = entry.first.find('|');
    out << entry.first.substr(0, split) << ','
        << CsvEscape(entry.first.substr(split + 1))
        << ",external_total_seconds," << stats.count << ','
        << stats.successful_count << ',' << stats.timeout_count << ','
        << stats.optimal_count << ',' << stats.median << ',' << stats.mean
        << ',' << stats.standard_deviation << ',' << stats.minimum << ','
        << stats.maximum << ',' << stats.p25 << ',' << stats.p75 << ','
        << stats.p90 << ',' << stats.geometric_mean_positive_runtime_ratio
        << '\n';
  }
}

}  // namespace

std::string ToString(BenchmarkAlgorithm algorithm) {
  switch (algorithm) {
    case BenchmarkAlgorithm::kUniformFixedOrder:
      return kUniformBreadthFirstMethod;
    case BenchmarkAlgorithm::kOptimizedBfs:
      return "optimized-bfs";
    case BenchmarkAlgorithm::kJointCpSat:
      return "joint-unrestricted-no-overlap";
    case BenchmarkAlgorithm::kPartitionOnly:
      return "partition-only-fixed-order";
    case BenchmarkAlgorithm::kScheduleOnly:
      return "schedule-only-uniform";
    case BenchmarkAlgorithm::kSequentialPartitionThenSchedule:
      return "sequential-partition-then-schedule";
    case BenchmarkAlgorithm::kAlternatingPartitionSchedule:
      return "alternating-partition-schedule";
    case BenchmarkAlgorithm::kCanonicalSlackPipe:
      return "canonical-slackpipe";
  }
  return "unknown";
}

BenchmarkAlgorithm ParseBenchmarkAlgorithm(const std::string &text) {
  const std::string canonical = CanonicalizeEvaluationMethodName(text);
  if (canonical == kUniformBreadthFirstMethod) {
    return BenchmarkAlgorithm::kUniformFixedOrder;
  }
  if (text == "optimized-bfs" || text == "bfs") {
    return BenchmarkAlgorithm::kOptimizedBfs;
  }
  if (canonical == "joint-unrestricted-no-overlap" || text == "joint-cpsat") {
    return BenchmarkAlgorithm::kJointCpSat;
  }
  if (canonical == "partition-only-fixed-order") {
    return BenchmarkAlgorithm::kPartitionOnly;
  }
  if (canonical == "schedule-only-uniform" ||
      text == "schedule-only-load-balanced" ||
      text == "schedule-only-balanced" || text == "schedule-only-fixed-split") {
    return BenchmarkAlgorithm::kScheduleOnly;
  }
  if (canonical == "sequential-partition-then-schedule") {
    return BenchmarkAlgorithm::kSequentialPartitionThenSchedule;
  }
  if (canonical == "alternating-partition-schedule") {
    return BenchmarkAlgorithm::kAlternatingPartitionSchedule;
  }
  if (text == "slackpipe" || text == "canonical-slackpipe" ||
      text == "one-shot-slackpipe") {
    return BenchmarkAlgorithm::kCanonicalSlackPipe;
  }
  throw Error("unknown benchmark algorithm: " + text);
}

std::string ToString(ExperimentMode mode) {
  switch (mode) {
    case ExperimentMode::kLatency:
      return "latency";
    case ExperimentMode::kThroughput:
      return "throughput";
    case ExperimentMode::kAlgorithmComparison:
      return "algorithm-comparison";
  }
  return "unknown";
}

ExperimentMode ParseExperimentMode(const std::string &text) {
  if (text == "latency") return ExperimentMode::kLatency;
  if (text == "throughput") return ExperimentMode::kThroughput;
  if (text == "algorithm-comparison")
    return ExperimentMode::kAlgorithmComparison;
  throw Error("unknown experiment mode: " + text);
}

BenchmarkConfig ParseBenchmarkConfigFile(const std::string &path) {
  const std::string text = ReadFile(path);
  BenchmarkConfig config;
  if (path.size() >= 4 && path.substr(path.size() - 4) == ".csv") {
    std::stringstream ss(text);
    std::string header;
    if (!std::getline(ss, header)) throw Error("empty csv config");
    const std::vector<std::string> columns = Split(header, ',');
    std::string line;
    while (std::getline(ss, line)) {
      if (Trim(line).empty()) continue;
      const std::vector<std::string> cells = Split(line, ',');
      std::map<std::string, std::string> row;
      for (std::size_t i = 0; i < columns.size() && i < cells.size(); ++i) {
        row[columns[i]] = cells[i];
      }
      config.instances.push_back(InstanceFromFields(row));
    }
    config.algorithms.push_back(AlgorithmFromText("joint-cpsat"));
    return config;
  }
  config.repetitions =
      static_cast<int>(ExtractLong(text, "repetitions", config.repetitions));
  config.warmups =
      static_cast<int>(ExtractLong(text, "warmups", config.warmups));
  config.parallel_instances = static_cast<int>(
      ExtractLong(text, "parallel_instances", config.parallel_instances));
  config.configurable_cpu_budget = static_cast<int>(ExtractLong(
      text, "configurable_cpu_budget", config.configurable_cpu_budget));
  config.allow_oversubscription = ExtractBool(text, "allow_oversubscription",
                                              config.allow_oversubscription);
  config.derive_schedule_only_best = ExtractBool(
      text, "derive_schedule_only_best", config.derive_schedule_only_best);
  config.timeout_seconds =
      ExtractDouble(text, "timeout_seconds", config.timeout_seconds);
  config.random_seed_base = static_cast<int>(
      ExtractLong(text, "random_seed_base", config.random_seed_base));
  config.mode =
      ParseExperimentMode(ExtractString(text, "mode", ToString(config.mode)));
  config.cp_sat_workers =
      ExtractIntArray(text, "cp_sat_workers", config.cp_sat_workers);
  config.activation_options.model = ParseActivationModel(ExtractString(
      text, "activation_model", ToCliString(config.activation_options.model)));
  config.activation_options.activation_units_per_layer =
      ExtractLong(text, "activation_units_per_layer",
                  config.activation_options.activation_units_per_layer);
  config.activation_options.explicit_stage_activation_units = ExtractTickArray(
      text, "explicit_stage_activation_units",
      config.activation_options.explicit_stage_activation_units);
  if (HasKey(text, "activation_bytes_per_unit")) {
    config.activation_options.activation_bytes_per_unit =
        ExtractDouble(text, "activation_bytes_per_unit", 0.0);
  }
  config.activation_options.cap_mode = ParseActivationCapMode(
      ExtractString(text, "activation_cap_mode",
                    ToCliString(config.activation_options.cap_mode)));
  config.activation_options.activation_cap_units =
      ExtractTickArray(text, "activation_cap_units",
                       config.activation_options.activation_cap_units);
  config.activation_options.enforce_activation_cap =
      ExtractBool(text, "enforce_activation_cap",
                  config.activation_options.enforce_activation_cap);
  if (HasKey(text, "activation_cap_enforcement")) {
    const std::string enforcement =
        ExtractString(text, "activation_cap_enforcement", "posthoc-only");
    if (enforcement == "solver") {
      config.activation_options.enforce_activation_cap = true;
    } else if (enforcement == "posthoc-only" || enforcement == "posthoc_only" ||
               enforcement == "posthoc") {
      config.activation_options.enforce_activation_cap = false;
    } else {
      throw Error("unknown activation_cap_enforcement: " + enforcement);
    }
  }
  config.activation_options.emit_activation_trace =
      ExtractBool(text, "emit_activation_trace",
                  config.activation_options.emit_activation_trace);
  for (const std::string &suite : ExtractStringArray(text, "suites")) {
    const std::vector<InstanceSpec> generated = GenerateNamedSuite(suite);
    config.instances.insert(config.instances.end(), generated.begin(),
                            generated.end());
  }
  for (const std::string &object : ExtractObjects(text, "instances")) {
    std::map<std::string, std::string> row;
    for (const std::string key :
         {"name", "B", "N", "J", "L", "ratio_num", "ratio_den", "min_layers",
          "communication_ticks"}) {
      if (key == "name") {
        const std::string value = ExtractString(object, key, "");
        if (!value.empty()) row[key] = value;
      } else {
        const long value =
            ExtractLong(object, key, std::numeric_limits<long>::min());
        if (value != std::numeric_limits<long>::min()) {
          row[key] = std::to_string(value);
        }
      }
    }
    config.instances.push_back(InstanceFromFields(row));
  }
  for (const std::string &algorithm : ExtractStringArray(text, "algorithms")) {
    config.algorithms.push_back(AlgorithmFromText(algorithm));
  }
  for (const std::string &object : ExtractObjects(text, "algorithm_configs")) {
    config.algorithms.push_back(AlgorithmFromObject(object));
  }
  if (config.algorithms.empty()) {
    if (config.mode == ExperimentMode::kAlgorithmComparison) {
      config.algorithms.push_back(AlgorithmFromText("joint-cpsat"));
      const bool has_virtualized =
          std::any_of(config.instances.begin(), config.instances.end(),
                      [](const InstanceSpec &spec) {
                        return spec.instance.stages > spec.instance.workers;
                      });
      if (has_virtualized) {
        config.algorithms.push_back(AlgorithmFromObject(
            "{\"algorithm\":\"slackpipe\",\"configuration\":\"worker-fixed\","
            "\"split_mode\":\"worker-fixed\"}"));
        for (Index budget : {1, 2, 4}) {
          std::ostringstream object;
          object << "{\"algorithm\":\"slackpipe\",\"configuration\":"
                 << "\"worker-local-" << budget
                 << "\",\"split_mode\":\"worker-local\","
                 << "\"worker_move_budget\":" << budget << "}";
          config.algorithms.push_back(AlgorithmFromObject(object.str()));
        }
        config.algorithms.push_back(AlgorithmFromObject(
            "{\"algorithm\":\"slackpipe\",\"configuration\":\"global\","
            "\"split_mode\":\"global\"}"));
      } else {
        config.algorithms.push_back(AlgorithmFromText("slackpipe"));
      }
    } else {
      config.algorithms.push_back(AlgorithmFromText("joint-cpsat"));
    }
  }
  if (config.instances.empty())
    throw Error("benchmark config has no instances");
  for (const InstanceSpec &instance : config.instances) {
    ValidateActivationOptions(instance.instance, config.activation_options);
  }
  return config;
}

std::vector<InstanceSpec> GenerateNamedSuite(const std::string &suite) {
  std::vector<InstanceSpec> out;
  auto add = [&](std::string name, Index b, Index n, Index j, Index l) {
    InstanceSpec spec;
    spec.name = std::move(name);
    spec.instance.microbatches = b;
    spec.instance.stages = n;
    spec.instance.workers = j;
    spec.instance.total_layers = l;
    spec.instance.backward_ratio_num = 2;
    spec.instance.backward_ratio_den = 1;
    spec.instance.min_layers = 1;
    spec.instance.Validate();
    out.push_back(spec);
  };
  if (suite == "exact-small") {
    add("exact-small-4-4-4-33", 4, 4, 4, 33);
    add("exact-small-6-4-4-48", 6, 4, 4, 48);
  } else if (suite == "scaling-B") {
    for (Index b : {2, 4, 6, 8}) {
      add("scaling-B-" + std::to_string(b), b, 4, 4, 48);
    }
  } else if (suite == "scaling-N") {
    for (Index n : {2, 3, 4, 5}) {
      add("scaling-N-" + std::to_string(n), 4, n, n, n * 12);
    }
  } else if (suite == "virtualization") {
    add("virtualization-6-8-4-64", 6, 8, 4, 64);
    add("virtualization-8-10-4-80", 8, 10, 4, 80);
  } else if (suite == "split-scale") {
    for (Index l : {32, 48, 64, 96}) {
      add("split-scale-" + std::to_string(l), 6, 4, 4, l);
    }
  } else {
    throw Error("unknown benchmark suite: " + suite);
  }
  return out;
}

std::string DeterministicRunId(
    const InstanceSpec &instance, const AlgorithmSpec &algorithm,
    int repetition, int seed, int workers, double timeout_seconds,
    const ActivationAnalysisOptions &activation_options) {
  std::ostringstream canonical;
  canonical
      << instance.name << '|' << instance.instance.microbatches << '|'
      << instance.instance.stages << '|' << instance.instance.workers << '|'
      << instance.instance.total_layers << '|' << instance.instance.min_layers
      << '|' << instance.instance.backward_ratio_num << '/'
      << instance.instance.backward_ratio_den << '|'
      << instance.instance.communication_ticks << '|'
      << ToString(algorithm.algorithm) << '|' << algorithm.configuration << '|'
      << algorithm.fixed_partition_source << '|'
      << JoinTicks(algorithm.fixed_split) << '|'
      << (algorithm.use_bfs_hints
              ? (*algorithm.use_bfs_hints ? "hints=true" : "hints=false")
              : "hints=default")
      << '|' << ToString(algorithm.split_mode) << '|' << algorithm.move_budget
      << '|'
      << (algorithm.per_stage_delta ? std::to_string(*algorithm.per_stage_delta)
                                    : "none")
      << '|' << algorithm.worker_move_budget << '|'
      << (algorithm.per_worker_delta
              ? std::to_string(*algorithm.per_worker_delta)
              : "none")
      << '|' << repetition << '|' << seed << '|' << workers << '|'
      << timeout_seconds << '|' << ToString(activation_options.model) << '|'
      << activation_options.activation_units_per_layer << '|'
      << JoinTicks(activation_options.explicit_stage_activation_units) << '|'
      << (activation_options.activation_bytes_per_unit
              ? std::to_string(*activation_options.activation_bytes_per_unit)
              : "bytes=null")
      << '|' << ToString(activation_options.cap_mode) << '|'
      << JoinTicks(activation_options.activation_cap_units) << '|'
      << (activation_options.enforce_activation_cap ? "enforce=true"
                                                    : "enforce=false")
      << '|'
      << (activation_options.emit_activation_trace ? "trace=true"
                                                   : "trace=false");
  return HashHex(canonical.str());
}

GapMetrics CalculateGap(std::optional<Tick> joint_optimum, Tick candidate) {
  GapMetrics gap;
  if (!joint_optimum || *joint_optimum <= 0 || candidate <= 0) return gap;
  gap.joint_optimum_when_available = true;
  gap.absolute_gap_to_joint = static_cast<double>(candidate - *joint_optimum);
  gap.relative_gap_to_joint =
      gap.absolute_gap_to_joint / static_cast<double>(*joint_optimum);
  gap.reaches_joint_objective = candidate <= *joint_optimum;
  return gap;
}

void ValidateCpuBudget(const BenchmarkConfig &config, int logical_cpus) {
  if (logical_cpus <= 0) throw Error("logical CPU count must be positive");
  const int budget = config.configurable_cpu_budget > 0
                         ? config.configurable_cpu_budget
                         : std::max(1, logical_cpus - 2);
  for (int workers : config.cp_sat_workers) {
    if (workers <= 0) throw Error("cp_sat_workers must be positive");
    if (workers > logical_cpus) {
      throw Error("cp_sat_workers exceeds available logical CPUs");
    }
    if (!config.allow_oversubscription &&
        config.parallel_instances * workers > budget) {
      throw Error("parallel_instances * cp_sat_workers exceeds CPU budget");
    }
  }
}

std::map<std::string, bool> LoadCompleteRunIds(
    const std::string &results_jsonl_path) {
  std::map<std::string, bool> ids;
  std::ifstream in(results_jsonl_path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line.back() != '}') continue;
    if (line.find("\"completed\":true") == std::string::npos) continue;
    const std::string id = ExtractString(line, "run_id", "");
    if (!id.empty()) ids[id] = true;
  }
  return ids;
}

void RecoverIncrementalOutputs(const std::string &output_dir) {
  const std::filesystem::path jsonl =
      std::filesystem::path(output_dir) / "results.jsonl";
  std::ifstream in(jsonl);
  if (!in) return;
  std::vector<std::string> complete;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '}') complete.push_back(line);
  }
  in.close();
  std::ofstream out(jsonl, std::ios::trunc);
  for (const std::string &kept : complete) out << kept << '\n';
}

SummaryStats ComputeSummaryStats(const std::vector<BenchmarkRow> &rows) {
  SummaryStats stats;
  stats.count = static_cast<int>(rows.size());
  std::vector<double> runtimes;
  for (const BenchmarkRow &row : rows) {
    if (row.status != "CRASHED") ++stats.successful_count;
    if (TimeoutLike(row.status) && !row.proven_optimal) ++stats.timeout_count;
    if (row.proven_optimal || row.proven_global_optimal) ++stats.optimal_count;
    const double runtime = row.external_total_seconds > 0.0
                               ? row.external_total_seconds
                               : row.total_planning_seconds;
    if (runtime > 0.0) {
      runtimes.push_back(runtime);
    }
  }
  if (runtimes.empty()) return stats;
  std::sort(runtimes.begin(), runtimes.end());
  auto percentile = [&](double p) {
    const double index = p * static_cast<double>(runtimes.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(index));
    const auto hi = static_cast<std::size_t>(std::ceil(index));
    if (lo == hi) return runtimes[lo];
    return runtimes[lo] +
           (runtimes[hi] - runtimes[lo]) * (index - static_cast<double>(lo));
  };
  stats.minimum = runtimes.front();
  stats.maximum = runtimes.back();
  stats.median = percentile(0.5);
  stats.p25 = percentile(0.25);
  stats.p75 = percentile(0.75);
  stats.p90 = percentile(0.90);
  stats.mean = std::accumulate(runtimes.begin(), runtimes.end(), 0.0) /
               static_cast<double>(runtimes.size());
  double variance = 0.0;
  for (double runtime : runtimes) {
    variance += (runtime - stats.mean) * (runtime - stats.mean);
  }
  stats.standard_deviation =
      std::sqrt(variance / static_cast<double>(runtimes.size()));
  return stats;
}

std::map<std::string, double> ComputePairedRuntimeRatios(
    const std::vector<BenchmarkRow> &rows, BenchmarkAlgorithm numerator,
    BenchmarkAlgorithm denominator) {
  std::map<std::string, double> numerators;
  std::map<std::string, double> denominators;
  for (const BenchmarkRow &row : rows) {
    const std::string key =
        row.instance_name + ":" + std::to_string(row.repetition) + ":" +
        std::to_string(row.seed) + ":" + std::to_string(row.cp_sat_workers);
    const double runtime = row.external_total_seconds > 0.0
                               ? row.external_total_seconds
                               : row.total_planning_seconds;
    if (row.algorithm == numerator) numerators[key] = runtime;
    if (row.algorithm == denominator) {
      denominators[key] = runtime;
    }
  }
  std::map<std::string, double> ratios;
  double log_sum = 0.0;
  int positive = 0;
  for (const auto &entry : numerators) {
    const auto it = denominators.find(entry.first);
    if (it != denominators.end() && entry.second > 0.0 && it->second > 0.0) {
      const double ratio = entry.second / it->second;
      ratios[entry.first] = ratio;
      log_sum += std::log(ratio);
      ++positive;
    }
  }
  if (positive > 0) ratios["__geomean__"] = std::exp(log_sum / positive);
  return ratios;
}

std::string HostMetadataJson() {
  utsname uts{};
  const bool have_uts = uname(&uts) == 0;
  const unsigned logical = std::max(1u, std::thread::hardware_concurrency());
  std::string cpu_model = "unknown";
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpuinfo, line)) {
    if (line.find("model name") != std::string::npos) {
      const std::size_t colon = line.find(':');
      if (colon != std::string::npos) cpu_model = Trim(line.substr(colon + 1));
      break;
    }
  }
  long ram_kb = 0;
  std::ifstream meminfo("/proc/meminfo");
  while (std::getline(meminfo, line)) {
    if (line.rfind("MemTotal:", 0) == 0) {
      std::stringstream ss(line.substr(9));
      ss >> ram_kb;
      break;
    }
  }
  std::ostringstream out;
  out << "{\n";
  out << "  \"cpu_model\": \"" << JsonEscape(cpu_model) << "\",\n";
  out << "  \"physical_cores\": \"unknown\",\n";
  out << "  \"logical_cores\": " << logical << ",\n";
  out << "  \"ram_kb\": " << ram_kb << ",\n";
  out << "  \"os\": \""
      << (have_uts ? JsonEscape(std::string(uts.sysname) + " " + uts.release)
                   : "unknown")
      << "\",\n";
  out << "  \"compiler_version\": \"" << JsonEscape(__VERSION__) << "\",\n";
  out << "  \"build_type\": ";
  WriteJsonOptionalString(out, CurrentBuildType());
  out << ",\n";
  out << "  \"ortools_enabled\": "
      << (SLACKPIPE_HAVE_ORTOOLS ? "true" : "false") << ",\n";
  out << "  \"ortools_version\": \"unknown\",\n";
  out << "  \"git_commit\": ";
  WriteJsonOptionalString(out, CurrentGitCommit());
  out << ",\n";
  std::optional<std::string> dirty_scope;
  out << "  \"dirty_tree\": ";
  WriteJsonOptionalBool(out, CurrentGitDirty(&dirty_scope));
  out << ",\n";
  out << "  \"dirty_tree_scope\": ";
  WriteJsonOptionalString(out, dirty_scope);
  out << "\n";
  out << "}\n";
  return out.str();
}

std::string RowComparisonKey(const BenchmarkRow &row) {
  return row.instance_name + ":" + std::to_string(row.repetition) + ":" +
         std::to_string(row.seed) + ":" + std::to_string(row.cp_sat_workers);
}

void PopulateGlobalGaps(std::vector<BenchmarkRow> &rows) {
  std::map<std::string, Tick> global_objectives;
  for (const BenchmarkRow &row : rows) {
    if (row.algorithm == BenchmarkAlgorithm::kCanonicalSlackPipe &&
        row.configuration == "global" && row.makespan > 0) {
      global_objectives[RowComparisonKey(row)] = row.makespan;
    }
  }
  for (BenchmarkRow &row : rows) {
    const auto it = global_objectives.find(RowComparisonKey(row));
    if (it == global_objectives.end() || it->second <= 0 || row.makespan <= 0) {
      continue;
    }
    row.absolute_gap_to_global = static_cast<double>(row.makespan - it->second);
    row.relative_gap_to_global =
        row.absolute_gap_to_global / static_cast<double>(it->second);
  }
}

bool IsSuccessfulScheduleRow(const BenchmarkRow &row) {
  return row.makespan > 0 && (row.status == "OPTIMAL" ||
                              row.status == "FEASIBLE" || row.proven_optimal);
}

bool IsPredefinedScheduleOnlySource(const std::string &source) {
  return source == "uniform" || source == "load-balanced" ||
         source == "partition-only" ||
         source == "partition_only_within_global_budget";
}

void AppendBestScheduleOnlyRows(std::vector<BenchmarkRow> &rows) {
  std::map<std::string, BenchmarkRow> best_by_group;
  for (const BenchmarkRow &row : rows) {
    if (row.algorithm != BenchmarkAlgorithm::kScheduleOnly ||
        row.configuration == "schedule-only-best" ||
        row.ablation_mode != "schedule-only" ||
        !IsPredefinedScheduleOnlySource(row.fixed_partition_source) ||
        !IsSuccessfulScheduleRow(row)) {
      continue;
    }
    const std::string key = RowComparisonKey(row);
    auto it = best_by_group.find(key);
    if (it == best_by_group.end() || row.makespan < it->second.makespan ||
        (row.makespan == it->second.makespan &&
         row.fixed_partition_source < it->second.fixed_partition_source)) {
      best_by_group[key] = row;
    }
  }

  for (auto &entry : best_by_group) {
    BenchmarkRow best = entry.second;
    best.configuration = "schedule-only-best";
    best.run_id = HashHex(best.run_id + "|schedule-only-best");
    best.ablation_mode = "schedule-only";
    best.partition_search_enabled = false;
    best.schedule_search_enabled = true;
    best.partition_fixed_validation_passed = !best.partition_changed;
    best.label_validation_passed = false;
    ValidateAblationLabel(best);
    if (best.partition_changed || best.partition_search_enabled ||
        !best.schedule_search_enabled) {
      throw Error("SCHEDULE_ONLY_BEST_RESTRICTION_MISMATCH");
    }
    rows.push_back(std::move(best));
  }
}

void ValidateScheduleOnlyPartitionOnlySource(
    const std::vector<BenchmarkRow> &rows) {
  std::map<std::string, std::string> partition_only_splits;
  for (const BenchmarkRow &row : rows) {
    if (row.algorithm == BenchmarkAlgorithm::kPartitionOnly &&
        row.ablation_mode == "partition-only" && IsSuccessfulScheduleRow(row)) {
      partition_only_splits[RowComparisonKey(row)] = row.split;
    }
  }
  for (const BenchmarkRow &row : rows) {
    if (row.algorithm != BenchmarkAlgorithm::kScheduleOnly ||
        row.configuration == "schedule-only-best" ||
        row.fixed_partition_source != "partition-only" ||
        !IsSuccessfulScheduleRow(row)) {
      continue;
    }
    const auto it = partition_only_splits.find(RowComparisonKey(row));
    if (it != partition_only_splits.end() && row.supplied_split != it->second) {
      throw Error("SCHEDULE_ONLY_PARTITION_ONLY_SOURCE_MISMATCH");
    }
  }
}

void ValidateUtilizationRankings(std::vector<BenchmarkRow> &rows) {
  std::map<std::string, std::vector<std::size_t>> ablation_groups;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const BenchmarkRow &row = rows[i];
    const bool included = row.ablation_mode == "joint" ||
                          row.ablation_mode == "partition-only" ||
                          (row.ablation_mode == "schedule-only" &&
                           row.configuration == "schedule-only-best");
    if (included && IsSuccessfulScheduleRow(row)) {
      ablation_groups[RowComparisonKey(row)].push_back(i);
    }
  }

  for (const auto &entry : ablation_groups) {
    const std::vector<std::size_t> &indexes = entry.second;
    for (std::size_t lhs_index = 0; lhs_index < indexes.size(); ++lhs_index) {
      for (std::size_t rhs_index = lhs_index + 1; rhs_index < indexes.size();
           ++rhs_index) {
        const BenchmarkRow &lhs = rows[indexes[lhs_index]];
        const BenchmarkRow &rhs = rows[indexes[rhs_index]];
        if (lhs.simulated_iteration_time < rhs.simulated_iteration_time &&
            !(lhs.pipeline_utilization > rhs.pipeline_utilization)) {
          throw Error(
              "UTILIZATION_RANKING_MISMATCH lower iteration time did "
              "not produce higher utilization");
        }
        if (lhs.simulated_iteration_time > rhs.simulated_iteration_time &&
            !(lhs.pipeline_utilization < rhs.pipeline_utilization)) {
          throw Error(
              "UTILIZATION_RANKING_MISMATCH higher iteration time did "
              "not produce lower utilization");
        }
        if (lhs.simulated_iteration_time == rhs.simulated_iteration_time &&
            lhs.pipeline_utilization != rhs.pipeline_utilization) {
          throw Error(
              "UTILIZATION_RANKING_MISMATCH equal iteration time did "
              "not produce equal utilization");
        }
      }
    }
    for (std::size_t index : indexes) {
      rows[index].utilization_ranking_validation_passed = true;
    }
  }
}

std::string ResultsCsvHeader() {
  return "run_id,instance,algorithm,configuration,repetition,seed,"
         "cp_sat_workers,status,ablation_mode,split_mode,"
         "fixed_partition_source,supplied_split,"
         "worker_move_budget,per_worker_delta,"
         "schema_version,budget_policy_version,"
         "reference_phase_limit_seconds,reference_runtime_seconds,"
         "schedule_solver_effective_limit_seconds,"
         "time_to_best_solution_seconds,"
         "validation_runtime_seconds,result_validation_error,"
         "requested_method,canonical_method,"
         "actual_solver_path,partition_decision,schedule_decision,"
         "full_partition_fixed,worker_aggregate_loads_fixed,"
         "partition_optimized,schedule_optimized,"
         "predecessor_candidate_restriction_requested,"
         "predecessor_candidate_restriction_active,"
         "fixed_order_partition_backend_requested,"
         "fixed_order_partition_backend_effective,"
         "estimated_partition_count,enumeration_safety_threshold,"
         "cp_sat_launched,"
         "solver_status_raw,reported_status,fallback_reason,"
         "returned_solution_source,communication_model,"
         "canonical_git_commit,canonical_git_dirty,"
         "completed,proven_optimal,makespan,simulated_iteration_time,"
         "pipeline_utilization,per_worker_busy_time,per_worker_idle_time,"
         "maximum_worker_load,pipeline_fill_time,pipeline_drain_time,"
         "communication_blocked_time,initial_objective,"
         "first_feasible_objective,final_objective,objective,"
         "best_bound,certified_gap,split,worker_local_operation_order,"
         "canonical_validation_passed,partition_search_enabled,"
         "schedule_search_enabled,partition_changed,operation_order_changed,"
         "partition_fixed_validation_passed,"
         "operation_order_fixed_validation_passed,label_validation_passed,"
         "hints_requested,hints_effective,hint_source,hint_scope,"
         "hint_complete_for_basic_model,hint_complete_for_full_model,"
         "hinted_layer_variable_count,hinted_operation_variable_count,"
         "hinted_scalar_variable_count,hinted_auxiliary_variable_count,"
         "hinted_total_variable_count,fallback_available,fallback_used,"
         "fallback_source,solution_source,bfs_incumbent_method_requested,"
         "bfs_incumbent_method_effective,"
         "incumbent_method_requested,incumbent_method_effective,"
         "incumbent_source,incumbent_feasible,incumbent_primary_objective,"
         "incumbent_hybrid_min_slack,incumbent_baseline_primary_objective,"
         "incumbent_baseline_hybrid_min_slack,"
         "incumbent_improved_over_baseline,"
         "incumbent_hybrid_stage_scores,"
         "incumbent_hybrid_bottleneck_stages,horizon_source,"
         "hint_budget_seconds,hint_elapsed_seconds,hint_iterations,"
         "hint_candidates_generated,hint_candidates_simulated,"
         "hint_partition_moves_accepted,hint_interleaving_moves_accepted,"
         "hint_deadline_reached,hint_termination_reason,"
         "utilization_ranking_validation_passed,"
         "joint_optimum_when_available,absolute_gap_to_joint,"
         "relative_gap_to_joint,absolute_gap_to_global,"
         "relative_gap_to_global,reaches_joint_objective,"
         "proven_global_optimal,"
         "global_certificate,external_total_seconds,model_build_seconds,"
         "external_solver_seconds,ortools_wall_time_seconds,"
         "cp_sat_solve_seconds,solver_internal_wall_seconds,"
         "solver_deterministic_time,extraction_verification_seconds,"
         "extraction_seconds,canonicalization_seconds,"
         "policy_seconds,serialization_seconds,"
         "bfs_seconds,uniform_bfs_seconds,optimized_bfs_seconds,"
         "time_to_first_feasible_seconds,time_to_best_incumbent_seconds,"
         "incumbent_improvement_count,baseline_worker_layers,"
         "final_worker_layers,"
         "worker_balance_l1,worker_balance_max_deviation,"
         "stage_split_l1,stage_split_max_deviation,"
         "orchestration_seconds,total_planning_seconds,"
         "time_to_first_joint_objective_seconds,"
         "time_to_global_certificate_seconds,process_cpu_micros,peak_rss_kb,"
         "activation_model,activation_units_per_layer,"
         "explicit_stage_activation_units,activation_cap_mode,"
         "activation_cap_units_per_worker,activation_cap_enforced,"
         "activation_cap_enforced_in_solver,"
         "activation_cap_enforcement_requested,"
         "activation_cap_enforcement_mode,"
         "activation_cap_solver_supported,"
         "activation_cap_solver_support_level,"
         "activation_cap_constraints_added,"
         "activation_retained_interval_count,"
         "activation_cumulative_constraint_count,"
         "activation_variable_demand_count,"
         "activation_fixed_demand_count,"
         "activation_constraint_build_runtime_seconds,"
         "incumbent_rejected_for_activation_cap,"
         "activation_model_validation_agreement,"
         "activation_model_disagreement_details,"
         "activation_cap_formulation_version,"
         "activation_cap_satisfied,"
         "maximum_worker_peak_activation_units,"
         "global_simultaneous_peak_activation_units,"
         "activation_peak_ratio_to_uniform,"
         "activation_cap_derivation_hash,"
         "cp_sat_models_solved,model_build_seconds_per_model,"
         "solver_seconds_per_model,branches,conflicts,failure\n";
}

std::string ToCsvRow(const BenchmarkRow &row) {
  std::ostringstream out;
  out << row.run_id << ',' << CsvEscape(row.instance_name) << ','
      << ToString(row.algorithm) << ',' << CsvEscape(row.configuration) << ','
      << row.repetition << ',' << row.seed << ',' << row.cp_sat_workers << ','
      << row.status << ',' << CsvEscape(row.ablation_mode) << ','
      << CsvEscape(row.split_mode) << ','
      << CsvEscape(row.fixed_partition_source) << ','
      << CsvEscape(row.supplied_split) << ',' << row.worker_move_budget << ',';
  if (row.per_worker_delta) {
    out << *row.per_worker_delta;
  }
  out << ',' << row.canonical.schema_version << ','
      << row.canonical.budget_policy_version << ','
      << OptionalNumberForCsv(
             row.canonical.outcome.phase_budget.reference_phase_limit_seconds)
      << ','
      << OptionalNumberForCsv(row.canonical.outcome.reference_runtime_seconds)
      << ','
      << OptionalNumberForCsv(row.canonical.outcome.phase_budget
                                  .schedule_solver_effective_limit_seconds)
      << ','
      << OptionalNumberForCsv(
             row.canonical.outcome.time_to_best_solution_seconds)
      << ','
      << OptionalNumberForCsv(row.canonical.outcome.validation_runtime_seconds)
      << ','
      << CsvEscape(OptionalStringForCsv(
             row.canonical.outcome.result_validation_error))
      << ',' << CsvEscape(OptionalStringForCsv(row.canonical.requested_method))
      << ',' << CsvEscape(row.canonical.canonical_method) << ','
      << CsvEscape(row.canonical.actual_solver_path) << ','
      << CsvEscape(row.canonical.semantics.partition_decision) << ','
      << CsvEscape(row.canonical.semantics.schedule_decision) << ','
      << (row.canonical.semantics.full_partition_fixed ? "true" : "false")
      << ','
      << (row.canonical.semantics.worker_aggregate_loads_fixed ? "true"
                                                               : "false")
      << ',' << (row.canonical.semantics.partition_optimized ? "true" : "false")
      << ',' << (row.canonical.semantics.schedule_optimized ? "true" : "false")
      << ','
      << (row.canonical.semantics.predecessor_candidate_restriction_requested
              ? "true"
              : "false")
      << ','
      << (row.canonical.semantics.predecessor_candidate_restriction_active
              ? "true"
              : "false")
      << ','
      << CsvEscape(OptionalStringForCsv(
             row.canonical.semantics.fixed_order_partition_backend_requested))
      << ','
      << CsvEscape(OptionalStringForCsv(
             row.canonical.semantics.fixed_order_partition_backend_effective))
      << ','
      << OptionalNumberForCsv(row.canonical.semantics.estimated_partition_count)
      << ','
      << OptionalNumberForCsv(
             row.canonical.semantics.enumeration_safety_threshold)
      << ',' << OptionalBoolForCsv(row.canonical.semantics.cp_sat_launched)
      << ','
      << CsvEscape(
             OptionalStringForCsv(row.canonical.outcome.solver_status_raw))
      << ','
      << CsvEscape(OptionalStringForCsv(row.canonical.outcome.reported_status))
      << ','
      << CsvEscape(OptionalStringForCsv(row.canonical.outcome.fallback_reason))
      << ','
      << CsvEscape(OptionalStringForCsv(
             row.canonical.outcome.returned_solution_source))
      << ',' << CsvEscape(row.canonical.communication_model) << ','
      << CsvEscape(OptionalStringForCsv(row.canonical.git_commit)) << ','
      << OptionalBoolForCsv(row.canonical.git_dirty) << ','
      << (row.completed ? "true" : "false") << ','
      << (row.proven_optimal ? "true" : "false") << ',' << row.makespan << ','
      << row.simulated_iteration_time << ',' << row.pipeline_utilization << ','
      << CsvEscape(row.per_worker_busy_time) << ','
      << CsvEscape(row.per_worker_idle_time) << ',' << row.maximum_worker_load
      << ',' << row.pipeline_fill_time << ',' << row.pipeline_drain_time << ',';
  if (row.communication_blocked_time) {
    out << *row.communication_blocked_time;
  }
  out << ',' << row.initial_objective << ',' << row.first_feasible_objective
      << ',' << row.final_objective << ',' << row.objective << ','
      << row.best_bound << ',' << row.certified_gap << ','
      << CsvEscape(row.split) << ','
      << CsvEscape(row.worker_local_operation_order) << ','
      << (row.canonical_validation_passed ? "true" : "false") << ','
      << (row.partition_search_enabled ? "true" : "false") << ','
      << (row.schedule_search_enabled ? "true" : "false") << ','
      << (row.partition_changed ? "true" : "false") << ','
      << (row.operation_order_changed ? "true" : "false") << ','
      << (row.partition_fixed_validation_passed ? "true" : "false") << ','
      << (row.operation_order_fixed_validation_passed ? "true" : "false") << ','
      << (row.label_validation_passed ? "true" : "false") << ','
      << (row.hints_requested ? "true" : "false") << ','
      << (row.hints_effective ? "true" : "false") << ','
      << CsvEscape(row.hint_source) << ',' << CsvEscape(row.hint_scope) << ','
      << (row.hint_complete_for_basic_model ? "true" : "false") << ','
      << (row.hint_complete_for_full_model ? "true" : "false") << ','
      << row.hinted_layer_variable_count << ','
      << row.hinted_operation_variable_count << ','
      << row.hinted_scalar_variable_count << ','
      << row.hinted_auxiliary_variable_count << ','
      << row.hinted_total_variable_count << ','
      << (row.fallback_available ? "true" : "false") << ','
      << (row.fallback_used ? "true" : "false") << ','
      << CsvEscape(row.fallback_source) << ',' << CsvEscape(row.solution_source)
      << ',' << CsvEscape(row.bfs_incumbent_method_requested) << ','
      << CsvEscape(row.bfs_incumbent_method_effective) << ','
      << CsvEscape(row.incumbent_method_requested) << ','
      << CsvEscape(row.incumbent_method_effective) << ','
      << CsvEscape(row.incumbent_source) << ','
      << (row.incumbent_feasible ? "true" : "false") << ','
      << row.incumbent_primary_objective << ','
      << row.incumbent_hybrid_min_slack << ','
      << row.incumbent_baseline_primary_objective << ','
      << row.incumbent_baseline_hybrid_min_slack << ','
      << (row.incumbent_improved_over_baseline ? "true" : "false") << ','
      << CsvEscape(row.incumbent_hybrid_stage_scores) << ','
      << CsvEscape(row.incumbent_hybrid_bottleneck_stages) << ','
      << CsvEscape(row.horizon_source) << ',' << row.hint_budget_seconds << ','
      << row.hint_elapsed_seconds << ',' << row.hint_iterations << ','
      << row.hint_candidates_generated << ',' << row.hint_candidates_simulated
      << ',' << row.hint_partition_moves_accepted << ','
      << row.hint_interleaving_moves_accepted << ','
      << (row.hint_deadline_reached ? "true" : "false") << ','
      << CsvEscape(row.hint_termination_reason) << ','
      << (row.utilization_ranking_validation_passed ? "true" : "false") << ','
      << (row.joint_optimum_when_available ? "true" : "false") << ','
      << row.absolute_gap_to_joint << ',' << row.relative_gap_to_joint << ','
      << row.absolute_gap_to_global << ',' << row.relative_gap_to_global << ','
      << (row.reaches_joint_objective ? "true" : "false") << ','
      << (row.proven_global_optimal ? "true" : "false") << ','
      << CsvEscape(row.global_certificate) << ',' << row.external_total_seconds
      << ',' << row.model_build_seconds << ',' << row.external_solver_seconds
      << ',' << row.ortools_wall_time_seconds << ',' << row.cp_sat_solve_seconds
      << ',' << row.solver_internal_wall_seconds << ','
      << row.solver_deterministic_time << ','
      << row.extraction_verification_seconds << ',' << row.extraction_seconds
      << ',' << row.canonicalization_seconds << ',' << row.policy_seconds << ','
      << row.serialization_seconds << ',' << row.bfs_seconds << ','
      << row.uniform_bfs_seconds << ',' << row.optimized_bfs_seconds << ','
      << row.time_to_first_feasible_seconds << ','
      << row.time_to_best_incumbent_seconds << ','
      << row.incumbent_improvement_count << ','
      << CsvEscape(row.baseline_worker_layers) << ','
      << CsvEscape(row.final_worker_layers) << ',' << row.worker_balance_l1
      << ',' << row.worker_balance_max_deviation << ',' << row.stage_split_l1
      << ',' << row.stage_split_max_deviation << ','
      << row.orchestration_seconds << ',' << row.total_planning_seconds << ','
      << row.time_to_first_joint_objective_seconds << ','
      << row.time_to_global_certificate_seconds << ',' << row.process_cpu_micros
      << ',' << row.peak_rss_kb << ',' << CsvEscape(row.activation_model) << ','
      << row.activation_units_per_layer << ','
      << CsvEscape(row.explicit_stage_activation_units) << ','
      << CsvEscape(row.activation_cap_mode) << ','
      << CsvEscape(row.activation_cap_units_per_worker) << ','
      << (row.activation_cap_enforced ? "true" : "false") << ','
      << (row.activation_cap_enforced_in_solver ? "true" : "false") << ','
      << (row.activation_cap_enforcement_requested ? "true" : "false") << ','
      << CsvEscape(row.activation_cap_enforcement_mode) << ','
      << (row.activation_cap_solver_supported ? "true" : "false") << ','
      << CsvEscape(row.activation_cap_solver_support_level) << ','
      << (row.activation_cap_constraints_added ? "true" : "false") << ','
      << row.activation_retained_interval_count << ','
      << row.activation_cumulative_constraint_count << ','
      << row.activation_variable_demand_count << ','
      << row.activation_fixed_demand_count << ','
      << row.activation_constraint_build_runtime_seconds << ','
      << (row.incumbent_rejected_for_activation_cap ? "true" : "false") << ','
      << OptionalBoolForCsv(row.activation_model_validation_agreement) << ','
      << CsvEscape(row.activation_model_disagreement_details) << ','
      << row.activation_cap_formulation_version << ','
      << OptionalBoolForCsv(row.activation_cap_satisfied) << ','
      << row.maximum_worker_peak_activation_units << ','
      << row.global_simultaneous_peak_activation_units << ','
      << OptionalNumberForCsv(row.activation_peak_ratio_to_uniform) << ','
      << CsvEscape(row.activation_cap_derivation_hash) << ','
      << row.cp_sat_models_solved << ',' << row.model_build_seconds_per_model
      << ',' << row.solver_seconds_per_model << ',' << row.branches << ','
      << row.conflicts << ',' << CsvEscape(row.failure) << '\n';
  return out.str();
}

std::string ToJsonLine(const BenchmarkRow &row) {
  std::ostringstream out;
  out << "{"
      << CompactJsonForJsonLine(
             CanonicalResultTopLevelJsonFields(row.canonical, "", true))
      << "\"run_id\":\"" << JsonEscape(row.run_id) << "\",\"instance\":\""
      << JsonEscape(row.instance_name) << "\",\"algorithm\":\""
      << ToString(row.algorithm) << "\",\"configuration\":\""
      << JsonEscape(row.configuration) << "\",\"repetition\":" << row.repetition
      << ",\"seed\":" << row.seed
      << ",\"cp_sat_workers\":" << row.cp_sat_workers << ",\"status\":\""
      << JsonEscape(row.status) << "\",\"ablation_mode\":\""
      << JsonEscape(row.ablation_mode) << "\",\"split_mode\":\""
      << JsonEscape(row.split_mode) << "\",\"fixed_partition_source\":\""
      << JsonEscape(row.fixed_partition_source) << "\",\"supplied_split\":\""
      << JsonEscape(row.supplied_split)
      << "\",\"worker_move_budget\":" << row.worker_move_budget
      << ",\"per_worker_delta\":";
  if (row.per_worker_delta) {
    out << *row.per_worker_delta;
  } else {
    out << "null";
  }
  out << ",\"completed\":" << (row.completed ? "true" : "false")
      << ",\"proven_optimal\":" << (row.proven_optimal ? "true" : "false")
      << ",\"makespan\":" << row.makespan
      << ",\"simulated_iteration_time\":" << row.simulated_iteration_time
      << ",\"pipeline_utilization\":" << row.pipeline_utilization
      << ",\"per_worker_busy_time\":\"" << JsonEscape(row.per_worker_busy_time)
      << "\",\"per_worker_idle_time\":\""
      << JsonEscape(row.per_worker_idle_time)
      << "\",\"maximum_worker_load\":" << row.maximum_worker_load
      << ",\"pipeline_fill_time\":" << row.pipeline_fill_time
      << ",\"pipeline_drain_time\":" << row.pipeline_drain_time
      << ",\"communication_blocked_time\":";
  if (row.communication_blocked_time) {
    out << *row.communication_blocked_time;
  } else {
    out << "null";
  }
  out << ",\"initial_objective\":" << row.initial_objective
      << ",\"first_feasible_objective\":" << row.first_feasible_objective
      << ",\"final_objective\":" << row.final_objective
      << ",\"objective\":" << row.objective
      << ",\"best_bound\":" << row.best_bound
      << ",\"certified_gap\":" << row.certified_gap << ",\"split\":\""
      << JsonEscape(row.split) << "\",\"worker_local_operation_order\":\""
      << JsonEscape(row.worker_local_operation_order)
      << "\",\"canonical_validation_passed\":"
      << (row.canonical_validation_passed ? "true" : "false")
      << ",\"partition_search_enabled\":"
      << (row.partition_search_enabled ? "true" : "false")
      << ",\"schedule_search_enabled\":"
      << (row.schedule_search_enabled ? "true" : "false")
      << ",\"partition_changed\":" << (row.partition_changed ? "true" : "false")
      << ",\"operation_order_changed\":"
      << (row.operation_order_changed ? "true" : "false")
      << ",\"partition_fixed_validation_passed\":"
      << (row.partition_fixed_validation_passed ? "true" : "false")
      << ",\"operation_order_fixed_validation_passed\":"
      << (row.operation_order_fixed_validation_passed ? "true" : "false")
      << ",\"label_validation_passed\":"
      << (row.label_validation_passed ? "true" : "false")
      << ",\"hints_requested\":" << (row.hints_requested ? "true" : "false")
      << ",\"hints_effective\":" << (row.hints_effective ? "true" : "false")
      << ",\"hint_source\":\"" << JsonEscape(row.hint_source)
      << "\",\"hint_scope\":\"" << JsonEscape(row.hint_scope)
      << "\",\"hint_complete_for_basic_model\":"
      << (row.hint_complete_for_basic_model ? "true" : "false")
      << ",\"hint_complete_for_full_model\":"
      << (row.hint_complete_for_full_model ? "true" : "false")
      << ",\"hinted_layer_variable_count\":" << row.hinted_layer_variable_count
      << ",\"hinted_operation_variable_count\":"
      << row.hinted_operation_variable_count
      << ",\"hinted_scalar_variable_count\":"
      << row.hinted_scalar_variable_count
      << ",\"hinted_auxiliary_variable_count\":"
      << row.hinted_auxiliary_variable_count
      << ",\"hinted_total_variable_count\":" << row.hinted_total_variable_count
      << ",\"fallback_available\":"
      << (row.fallback_available ? "true" : "false")
      << ",\"fallback_used\":" << (row.fallback_used ? "true" : "false")
      << ",\"fallback_source\":\"" << JsonEscape(row.fallback_source)
      << "\",\"solution_source\":\"" << JsonEscape(row.solution_source)
      << "\",\"bfs_incumbent_method_requested\":\""
      << JsonEscape(row.bfs_incumbent_method_requested)
      << "\",\"bfs_incumbent_method_effective\":\""
      << JsonEscape(row.bfs_incumbent_method_effective)
      << "\",\"incumbent_method_requested\":\""
      << JsonEscape(row.incumbent_method_requested)
      << "\",\"incumbent_method_effective\":\""
      << JsonEscape(row.incumbent_method_effective)
      << "\",\"incumbent_source\":\"" << JsonEscape(row.incumbent_source)
      << "\",\"incumbent_feasible\":"
      << (row.incumbent_feasible ? "true" : "false")
      << ",\"incumbent_primary_objective\":" << row.incumbent_primary_objective
      << ",\"incumbent_hybrid_min_slack\":" << row.incumbent_hybrid_min_slack
      << ",\"incumbent_baseline_primary_objective\":"
      << row.incumbent_baseline_primary_objective
      << ",\"incumbent_baseline_hybrid_min_slack\":"
      << row.incumbent_baseline_hybrid_min_slack
      << ",\"incumbent_improved_over_baseline\":"
      << (row.incumbent_improved_over_baseline ? "true" : "false")
      << ",\"incumbent_hybrid_stage_scores\":\""
      << JsonEscape(row.incumbent_hybrid_stage_scores)
      << "\",\"incumbent_hybrid_bottleneck_stages\":\""
      << JsonEscape(row.incumbent_hybrid_bottleneck_stages)
      << "\",\"horizon_source\":\"" << JsonEscape(row.horizon_source)
      << "\",\"hint_budget_seconds\":" << row.hint_budget_seconds
      << ",\"hint_elapsed_seconds\":" << row.hint_elapsed_seconds
      << ",\"hint_iterations\":" << row.hint_iterations
      << ",\"hint_candidates_generated\":" << row.hint_candidates_generated
      << ",\"hint_candidates_simulated\":" << row.hint_candidates_simulated
      << ",\"hint_partition_moves_accepted\":"
      << row.hint_partition_moves_accepted
      << ",\"hint_interleaving_moves_accepted\":"
      << row.hint_interleaving_moves_accepted << ",\"hint_deadline_reached\":"
      << (row.hint_deadline_reached ? "true" : "false")
      << ",\"hint_termination_reason\":\""
      << JsonEscape(row.hint_termination_reason) << "\""
      << ",\"utilization_ranking_validation_passed\":"
      << (row.utilization_ranking_validation_passed ? "true" : "false")
      << ",\"joint_optimum_when_available\":"
      << (row.joint_optimum_when_available ? "true" : "false")
      << ",\"absolute_gap_to_joint\":" << row.absolute_gap_to_joint
      << ",\"relative_gap_to_joint\":" << row.relative_gap_to_joint
      << ",\"absolute_gap_to_global\":" << row.absolute_gap_to_global
      << ",\"relative_gap_to_global\":" << row.relative_gap_to_global
      << ",\"reaches_joint_objective\":"
      << (row.reaches_joint_objective ? "true" : "false")
      << ",\"proven_global_optimal\":"
      << (row.proven_global_optimal ? "true" : "false")
      << ",\"global_certificate\":\"" << JsonEscape(row.global_certificate)
      << "\",\"external_total_seconds\":" << row.external_total_seconds
      << ",\"model_build_seconds\":" << row.model_build_seconds
      << ",\"external_solver_seconds\":" << row.external_solver_seconds
      << ",\"ortools_wall_time_seconds\":" << row.ortools_wall_time_seconds
      << ",\"cp_sat_solve_seconds\":" << row.cp_sat_solve_seconds
      << ",\"solver_internal_wall_seconds\":"
      << row.solver_internal_wall_seconds
      << ",\"solver_deterministic_time\":" << row.solver_deterministic_time
      << ",\"extraction_verification_seconds\":"
      << row.extraction_verification_seconds
      << ",\"extraction_seconds\":" << row.extraction_seconds
      << ",\"canonicalization_seconds\":" << row.canonicalization_seconds
      << ",\"policy_seconds\":" << row.policy_seconds
      << ",\"serialization_seconds\":" << row.serialization_seconds
      << ",\"bfs_seconds\":" << row.bfs_seconds
      << ",\"uniform_bfs_seconds\":" << row.uniform_bfs_seconds
      << ",\"optimized_bfs_seconds\":" << row.optimized_bfs_seconds
      << ",\"time_to_first_feasible_seconds\":"
      << row.time_to_first_feasible_seconds
      << ",\"time_to_best_incumbent_seconds\":"
      << row.time_to_best_incumbent_seconds
      << ",\"incumbent_improvement_count\":" << row.incumbent_improvement_count
      << ",\"baseline_worker_layers\":\""
      << JsonEscape(row.baseline_worker_layers)
      << "\",\"final_worker_layers\":\"" << JsonEscape(row.final_worker_layers)
      << "\"" << ",\"worker_balance_l1\":" << row.worker_balance_l1
      << ",\"worker_balance_max_deviation\":"
      << row.worker_balance_max_deviation
      << ",\"stage_split_l1\":" << row.stage_split_l1
      << ",\"stage_split_max_deviation\":" << row.stage_split_max_deviation
      << ",\"orchestration_seconds\":" << row.orchestration_seconds
      << ",\"total_planning_seconds\":" << row.total_planning_seconds
      << ",\"time_to_first_joint_objective_seconds\":"
      << row.time_to_first_joint_objective_seconds
      << ",\"time_to_global_certificate_seconds\":"
      << row.time_to_global_certificate_seconds
      << ",\"process_cpu_micros\":" << row.process_cpu_micros
      << ",\"peak_rss_kb\":" << row.peak_rss_kb << ",\"activation_model\":\""
      << JsonEscape(row.activation_model)
      << "\",\"activation_units_per_layer\":" << row.activation_units_per_layer
      << ",\"explicit_stage_activation_units\":\""
      << JsonEscape(row.explicit_stage_activation_units)
      << "\",\"activation_cap_mode\":\"" << JsonEscape(row.activation_cap_mode)
      << "\",\"activation_cap_units_per_worker\":\""
      << JsonEscape(row.activation_cap_units_per_worker)
      << "\",\"activation_cap_enforced\":"
      << (row.activation_cap_enforced ? "true" : "false")
      << ",\"activation_cap_enforced_in_solver\":"
      << (row.activation_cap_enforced_in_solver ? "true" : "false")
      << ",\"activation_cap_enforcement_requested\":"
      << (row.activation_cap_enforcement_requested ? "true" : "false")
      << ",\"activation_cap_enforcement_mode\":\""
      << JsonEscape(row.activation_cap_enforcement_mode)
      << "\",\"activation_cap_solver_supported\":"
      << (row.activation_cap_solver_supported ? "true" : "false")
      << ",\"activation_cap_solver_support_level\":\""
      << JsonEscape(row.activation_cap_solver_support_level)
      << "\",\"activation_cap_constraints_added\":"
      << (row.activation_cap_constraints_added ? "true" : "false")
      << ",\"activation_retained_interval_count\":"
      << row.activation_retained_interval_count
      << ",\"activation_cumulative_constraint_count\":"
      << row.activation_cumulative_constraint_count
      << ",\"activation_variable_demand_count\":"
      << row.activation_variable_demand_count
      << ",\"activation_fixed_demand_count\":"
      << row.activation_fixed_demand_count
      << ",\"activation_constraint_build_runtime_seconds\":"
      << row.activation_constraint_build_runtime_seconds
      << ",\"incumbent_rejected_for_activation_cap\":"
      << (row.incumbent_rejected_for_activation_cap ? "true" : "false")
      << ",\"activation_model_validation_agreement\":";
  if (row.activation_model_validation_agreement) {
    out << (*row.activation_model_validation_agreement ? "true" : "false");
  } else {
    out << "null";
  }
  out << ",\"activation_model_disagreement_details\":\""
      << JsonEscape(row.activation_model_disagreement_details)
      << "\",\"activation_cap_formulation_version\":"
      << row.activation_cap_formulation_version
      << ",\"activation_cap_satisfied\":";
  if (row.activation_cap_satisfied) {
    out << (*row.activation_cap_satisfied ? "true" : "false");
  } else {
    out << "null";
  }
  out << ",\"maximum_worker_peak_activation_units\":"
      << row.maximum_worker_peak_activation_units
      << ",\"global_simultaneous_peak_activation_units\":"
      << row.global_simultaneous_peak_activation_units
      << ",\"activation_peak_ratio_to_uniform\":";
  if (row.activation_peak_ratio_to_uniform) {
    out << *row.activation_peak_ratio_to_uniform;
  } else {
    out << "null";
  }
  out << ",\"activation_cap_derivation_hash\":\""
      << JsonEscape(row.activation_cap_derivation_hash) << "\""
      << ",\"cp_sat_models_solved\":" << row.cp_sat_models_solved
      << ",\"model_build_seconds_per_model\":"
      << row.model_build_seconds_per_model
      << ",\"solver_seconds_per_model\":" << row.solver_seconds_per_model
      << ",\"branches\":" << row.branches << ",\"conflicts\":" << row.conflicts
      << ",\"failure\":\"" << JsonEscape(row.failure) << "\"}";
  return out.str();
}

std::vector<BenchmarkRow> RunBenchmark(const BenchmarkConfig &config,
                                       const std::string &output_dir) {
  if (config.repetitions <= 0) throw Error("repetitions must be positive");
  if (config.warmups < 0) throw Error("warmups must be non-negative");
  if (config.parallel_instances <= 0) {
    throw Error("parallel_instances must be positive");
  }
  const int logical =
      static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  ValidateCpuBudget(config, logical);
  std::filesystem::create_directories(output_dir);
  WriteTextFile((std::filesystem::path(output_dir) / "metadata.json").string(),
                HostMetadataJson());
  RecoverIncrementalOutputs(output_dir);
  const std::filesystem::path csv_path =
      std::filesystem::path(output_dir) / "results.csv";
  const std::filesystem::path jsonl_path =
      std::filesystem::path(output_dir) / "results.jsonl";
  const bool write_header = !std::filesystem::exists(csv_path) ||
                            std::filesystem::file_size(csv_path) == 0;
  std::ofstream csv(csv_path, std::ios::app);
  std::ofstream jsonl(jsonl_path, std::ios::app);
  std::ofstream failures(std::filesystem::path(output_dir) / "failures.jsonl",
                         std::ios::app);
  if (write_header) csv << ResultsCsvHeader();
  const std::map<std::string, bool> complete =
      config.resume ? LoadCompleteRunIds(jsonl_path.string())
                    : std::map<std::string, bool>{};
  std::vector<BenchmarkRow> measured;
  std::mutex emit_mutex;
  auto emit = [&](BenchmarkRow row) {
    std::lock_guard<std::mutex> lock(emit_mutex);
    const auto serialization_started = Clock::now();
    const std::string csv_row = ToCsvRow(row);
    const std::string json_row = ToJsonLine(row);
    (void)csv_row;
    (void)json_row;
    row.serialization_seconds =
        std::chrono::duration<double>(Clock::now() - serialization_started)
            .count();
    const std::string final_csv_row = ToCsvRow(row);
    const std::string final_json_row = ToJsonLine(row);
    csv << final_csv_row;
    csv.flush();
    jsonl << final_json_row << '\n';
    jsonl.flush();
    if (!row.failure.empty() || row.status == "CRASHED") {
      failures << final_json_row << '\n';
      failures.flush();
    }
    measured.push_back(row);
  };
  for (const InstanceSpec &instance : config.instances) {
    for (int workers : config.cp_sat_workers) {
      for (int warmup = 0; warmup < config.warmups; ++warmup) {
        const int seed = config.random_seed_base + warmup;
        const std::optional<Tick> joint = std::nullopt;
        for (const AlgorithmSpec &algorithm : config.algorithms) {
          (void)RunOne(instance, algorithm, -1 - warmup, seed, workers,
                       config.timeout_seconds, joint, config.requested_command,
                       config.executable_name, config.activation_options);
        }
      }
      for (int repetition = 0; repetition < config.repetitions; ++repetition) {
        const int seed = config.random_seed_base + repetition;
        const std::optional<Tick> joint = std::nullopt;
        std::vector<std::future<BenchmarkRow>> futures;
        for (const AlgorithmSpec &algorithm : config.algorithms) {
          const BenchmarkRow probe =
              BaseRow(instance, algorithm, repetition, seed, workers,
                      config.timeout_seconds, config.requested_command,
                      config.executable_name, config.activation_options);
          if (complete.count(probe.run_id) != 0) continue;
          futures.push_back(
              std::async(std::launch::async, [&, algorithm, joint] {
                return RunOne(instance, algorithm, repetition, seed, workers,
                              config.timeout_seconds, joint,
                              config.requested_command, config.executable_name,
                              config.activation_options);
              }));
          if (static_cast<int>(futures.size()) >= config.parallel_instances) {
            emit(futures.front().get());
            futures.erase(futures.begin());
          }
        }
        for (std::future<BenchmarkRow> &future : futures) emit(future.get());
      }
    }
  }
  const std::filesystem::path summary_path =
      std::filesystem::path(output_dir) / "summary.csv";
  if (!measured.empty() || !std::filesystem::exists(summary_path)) {
    if (config.derive_schedule_only_best) {
      AppendBestScheduleOnlyRows(measured);
      ValidateScheduleOnlyPartitionOnlySource(measured);
    }
    ValidateUtilizationRankings(measured);
    PopulateGlobalGaps(measured);
    csv.close();
    jsonl.close();
    std::ofstream rewritten_csv(csv_path, std::ios::trunc);
    rewritten_csv << ResultsCsvHeader();
    std::ofstream rewritten_jsonl(jsonl_path, std::ios::trunc);
    for (const BenchmarkRow &row : measured) {
      rewritten_csv << ToCsvRow(row);
      rewritten_jsonl << ToJsonLine(row) << '\n';
    }
    WriteSummary(summary_path, measured);
  }
  return measured;
}

}  // namespace slackpipe::benchmark
