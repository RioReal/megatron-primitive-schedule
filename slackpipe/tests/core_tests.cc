#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "slackpipe/activation_analyzer.h"
#include "slackpipe/alternating_solver.h"
#include "slackpipe/bfs_solver.h"
#include "slackpipe/breadth_first.h"
#include "slackpipe/cost_profile.h"
#include "slackpipe/dag_evaluator.h"
#include "slackpipe/deadline.h"
#include "slackpipe/evaluation_method.h"
#include "slackpipe/fixed_order_partition_solver.h"
#include "slackpipe/hybrid_slack.h"
#include "slackpipe/interleaving_stats.h"
#include "slackpipe/io.h"
#include "slackpipe/joint_solver.h"
#include "slackpipe/one_f_one_b.h"
#include "slackpipe/operation.h"
#include "slackpipe/plan_export.h"
#include "slackpipe/pressure_pruning.h"
#include "slackpipe/result_schema.h"
#include "slackpipe/result_validator.h"
#include "slackpipe/slackpipe_solver.h"
#include "gtest/gtest.h"

namespace {

slackpipe::Instance BaseInstance() {
  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 3;
  instance.workers = 2;
  instance.total_layers = 9;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;
  return instance;
}

slackpipe::Instance SanityInstance() {
  slackpipe::Instance instance;
  instance.microbatches = 8;
  instance.stages = 4;
  instance.workers = 4;
  instance.total_layers = 8;
  instance.min_layers = 1;
  return instance;
}

bool Solved(const std::string &status) {
  return status == "OPTIMAL" || status == "FEASIBLE";
}

std::size_t CountSubstring(const std::string &text, const std::string &needle) {
  std::size_t count = 0;
  std::size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

void ExpectDoubleVectorNear(const std::vector<double> &actual,
                            const std::vector<double> &expected,
                            double tolerance = 1e-9) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    EXPECT_TRUE(std::abs(actual[i] - expected[i]) <= tolerance);
  }
}

std::vector<slackpipe::OperationId> SameWorkerBasePredecessors(
    const slackpipe::Instance &instance, slackpipe::OperationId id) {
  const slackpipe::OperationView view =
      slackpipe::DecodeOperation(instance, id);
  std::vector<slackpipe::OperationId> predecessors;
  const std::optional<slackpipe::OperationId> data =
      slackpipe::DataPredecessor(instance, id);
  if (data &&
      slackpipe::DecodeOperation(instance, *data).worker == view.worker) {
    predecessors.push_back(*data);
  }
  const std::optional<slackpipe::OperationId> fifo =
      slackpipe::FifoPredecessor(instance, id);
  if (fifo &&
      slackpipe::DecodeOperation(instance, *fifo).worker == view.worker) {
    predecessors.push_back(*fifo);
  }
  return predecessors;
}

std::vector<std::vector<slackpipe::OperationId>> WorkerLocalExtensions(
    const slackpipe::Instance &instance, slackpipe::Index worker) {
  std::vector<slackpipe::OperationId> worker_ops;
  for (slackpipe::Index id = 0; id < instance.OperationCount(); ++id) {
    const slackpipe::OperationId op{id};
    if (slackpipe::DecodeOperation(instance, op).worker == worker) {
      worker_ops.push_back(op);
    }
  }

  std::vector<char> placed(static_cast<std::size_t>(instance.OperationCount()),
                           false);
  std::vector<slackpipe::OperationId> current;
  std::vector<std::vector<slackpipe::OperationId>> extensions;
  std::function<void()> visit = [&]() {
    if (current.size() == worker_ops.size()) {
      extensions.push_back(current);
      return;
    }
    for (slackpipe::OperationId candidate : worker_ops) {
      if (placed[static_cast<std::size_t>(candidate.value)]) continue;
      bool ready = true;
      for (slackpipe::OperationId predecessor :
           SameWorkerBasePredecessors(instance, candidate)) {
        if (!placed[static_cast<std::size_t>(predecessor.value)]) {
          ready = false;
          break;
        }
      }
      if (!ready) continue;
      placed[static_cast<std::size_t>(candidate.value)] = true;
      current.push_back(candidate);
      visit();
      current.pop_back();
      placed[static_cast<std::size_t>(candidate.value)] = false;
    }
  };
  visit();
  return extensions;
}

std::vector<slackpipe::MachineOrders> ExhaustiveMachineOrders(
    const slackpipe::Instance &instance) {
  std::vector<std::vector<std::vector<slackpipe::OperationId>>>
      extensions_by_worker;
  for (slackpipe::Index worker = 0; worker < instance.workers; ++worker) {
    extensions_by_worker.push_back(WorkerLocalExtensions(instance, worker));
  }

  slackpipe::MachineOrders current(static_cast<std::size_t>(instance.workers));
  std::vector<slackpipe::MachineOrders> orders;
  std::function<void(slackpipe::Index)> choose_worker =
      [&](slackpipe::Index worker) {
        if (worker == instance.workers) {
          orders.push_back(current);
          return;
        }
        for (const std::vector<slackpipe::OperationId> &extension :
             extensions_by_worker[static_cast<std::size_t>(worker)]) {
          current[static_cast<std::size_t>(worker)] = extension;
          choose_worker(worker + 1);
        }
      };
  choose_worker(0);
  return orders;
}

std::vector<slackpipe::SerializedWorkerPredecessorRecord> DerivedRecords(
    const slackpipe::Instance &instance,
    const slackpipe::MachineOrders &orders) {
  const slackpipe::MachinePredecessors predecessors =
      slackpipe::ExtractMachinePredecessors(instance, orders);
  std::vector<slackpipe::SerializedWorkerPredecessorRecord> records;
  for (const auto &[id_value, predecessor] : predecessors) {
    const slackpipe::OperationId id{id_value};
    const slackpipe::OperationView view =
        slackpipe::DecodeOperation(instance, id);
    records.push_back(slackpipe::SerializedWorkerPredecessorRecord{
        id, slackpipe::OperationName(instance, id), predecessor,
        slackpipe::OperationName(instance, predecessor), view.worker});
  }
  return records;
}

slackpipe::SerializedOperationRecord OperationRecord(
    const slackpipe::Instance &instance, slackpipe::Index microbatch,
    slackpipe::Index position, slackpipe::Tick start, slackpipe::Tick end,
    const std::vector<slackpipe::Tick> &split) {
  const slackpipe::OperationId id =
      slackpipe::EncodeOperation(instance, microbatch, position);
  const slackpipe::OperationView view =
      slackpipe::DecodeOperation(instance, id);
  slackpipe::SerializedOperationRecord record;
  record.id = id;
  record.name = slackpipe::OperationName(instance, id);
  record.microbatch = view.microbatch;
  record.operation_position = view.chain_index;
  record.stage = view.stage;
  record.phase = view.backward ? "B" : "F";
  record.worker = view.worker;
  record.start = start;
  record.end = end;
  record.duration = instance.Duration(view.stage, view.backward, split);
  return record;
}

slackpipe::ResultValidationInput ManualValidatorInput() {
  slackpipe::Instance instance;
  instance.microbatches = 1;
  instance.stages = 2;
  instance.workers = 2;
  instance.total_layers = 4;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;
  instance.communication_ticks = 0;
  const std::vector<slackpipe::Tick> split{2, 2};
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);

  slackpipe::ResultValidationInput input;
  input.instance = instance;
  input.selected_partition = split;
  input.worker_orders = orders;
  input.derived_worker_predecessors = DerivedRecords(instance, orders);
  input.require_serialized_worker_predecessors = true;
  input.serialized_operations =
      std::vector<slackpipe::SerializedOperationRecord>{
          OperationRecord(instance, 0, 0, 0, 2, split),
          OperationRecord(instance, 0, 1, 2, 4, split),
          OperationRecord(instance, 0, 2, 4, 8, split),
          OperationRecord(instance, 0, 3, 8, 12, split)};
  input.require_serialized_operations = true;
  input.reported_makespan = 12;
  input.feasible_claimed = true;
  input.solver_status_raw = "FEASIBLE";
  input.reported_status = "FEASIBLE";
  input.method_name = "test";
  return input;
}

slackpipe::ResultValidationInput EvaluatedValidatorInput(
    const slackpipe::Instance &instance,
    const std::vector<slackpipe::Tick> &split,
    const slackpipe::MachineOrders &orders) {
  const slackpipe::EvaluationResult evaluated =
      slackpipe::EvaluateSchedule(instance, split, orders);
  slackpipe::ResultValidationInput input =
      slackpipe::ValidationInputFromSchedule(instance, evaluated.schedule,
                                             "FEASIBLE", "test");
  input.derived_worker_predecessors = DerivedRecords(instance, orders);
  input.require_serialized_worker_predecessors = true;
  input.require_serialized_operations = true;
  return input;
}

slackpipe::Instance OneStageActivationInstance(slackpipe::Index microbatches) {
  slackpipe::Instance instance;
  instance.microbatches = microbatches;
  instance.stages = 1;
  instance.workers = 1;
  instance.total_layers = 1;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;
  return instance;
}

slackpipe::OperationId OneStageOp(const slackpipe::Instance &instance,
                                  slackpipe::Index microbatch, bool backward) {
  return slackpipe::EncodeOperation(instance, microbatch, backward ? 1 : 0);
}

struct ExhaustiveOracleResult {
  bool feasible = false;
  slackpipe::Tick makespan = std::numeric_limits<slackpipe::Tick>::max();
  std::vector<slackpipe::Tick> split;
  slackpipe::MachineOrders orders;
  slackpipe::Index candidates_checked = 0;
  slackpipe::Index feasible_candidates = 0;
};

bool SatisfiesActivationOptions(
    const slackpipe::Instance &instance,
    const slackpipe::ScheduleSolution &schedule,
    const std::optional<slackpipe::ActivationAnalysisOptions> &options) {
  if (!options) return true;
  const slackpipe::ActivationAnalysisResult analysis =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(instance, schedule,
                                                            *options);
  return analysis.passed && (!options->enforce_activation_cap ||
                             analysis.activation_cap_satisfied.value_or(false));
}

void ConsiderOracleCandidate(
    const slackpipe::Instance &instance,
    const std::vector<slackpipe::Tick> &split,
    const slackpipe::MachineOrders &orders,
    const std::optional<slackpipe::ActivationAnalysisOptions>
        &activation_options,
    ExhaustiveOracleResult &oracle) {
  ++oracle.candidates_checked;
  const slackpipe::EvaluationResult evaluated =
      slackpipe::EvaluateSchedule(instance, split, orders);
  if (!evaluated.schedule.ok()) return;
  if (!SatisfiesActivationOptions(instance, evaluated.schedule,
                                  activation_options))
    return;
  ++oracle.feasible_candidates;
  if (!oracle.feasible || evaluated.schedule.makespan < oracle.makespan ||
      (evaluated.schedule.makespan == oracle.makespan &&
       std::lexicographical_compare(split.begin(), split.end(),
                                    oracle.split.begin(),
                                    oracle.split.end()))) {
    oracle.feasible = true;
    oracle.makespan = evaluated.schedule.makespan;
    oracle.split = split;
    oracle.orders = orders;
  }
}

ExhaustiveOracleResult ExhaustiveFixedOrderOracle(
    const slackpipe::Instance &instance, const slackpipe::MachineOrders &orders,
    const std::optional<slackpipe::ActivationAnalysisOptions>
        &activation_options = std::nullopt) {
  ExhaustiveOracleResult oracle;
  slackpipe::EnumerateValidSplits(
      instance, [&](const std::vector<slackpipe::Tick> &split) {
        ConsiderOracleCandidate(instance, split, orders, activation_options,
                                oracle);
      });
  return oracle;
}

ExhaustiveOracleResult ExhaustiveFixedSplitOrderOracle(
    const slackpipe::Instance &instance,
    const std::vector<slackpipe::Tick> &split,
    const std::optional<slackpipe::ActivationAnalysisOptions>
        &activation_options = std::nullopt) {
  ExhaustiveOracleResult oracle;
  for (const slackpipe::MachineOrders &orders :
       ExhaustiveMachineOrders(instance)) {
    ConsiderOracleCandidate(instance, split, orders, activation_options,
                            oracle);
  }
  return oracle;
}

ExhaustiveOracleResult ExhaustiveJointOracle(
    const slackpipe::Instance &instance,
    const std::optional<slackpipe::ActivationAnalysisOptions>
        &activation_options = std::nullopt) {
  ExhaustiveOracleResult oracle;
  const std::vector<slackpipe::MachineOrders> orders =
      ExhaustiveMachineOrders(instance);
  slackpipe::EnumerateValidSplits(
      instance, [&](const std::vector<slackpipe::Tick> &split) {
        for (const slackpipe::MachineOrders &order : orders) {
          ConsiderOracleCandidate(instance, split, order, activation_options,
                                  oracle);
        }
      });
  return oracle;
}

void ExpectScheduleIndependentlyValid(
    const slackpipe::Instance &instance,
    const slackpipe::ScheduleSolution &schedule, const std::string &status,
    const std::string &method = "ortools-smoke") {
  const slackpipe::ResultValidationResult validation =
      slackpipe::ValidateScheduleSolutionIndependent(instance, schedule, status,
                                                     method);
  EXPECT_TRUE(validation.passed);
}

void ExpectSamePositionFifoRespected(const slackpipe::Instance &instance,
                                     const slackpipe::MachineOrders &orders) {
  ASSERT_EQ(orders.size(), static_cast<std::size_t>(instance.workers));
  const slackpipe::Index positions =
      slackpipe::OperationPositionCount(instance);
  for (slackpipe::Index worker = 0; worker < instance.workers; ++worker) {
    std::vector<slackpipe::Index> last_microbatch(
        static_cast<std::size_t>(positions), -1);
    for (slackpipe::OperationId id : orders[static_cast<std::size_t>(worker)]) {
      const slackpipe::OperationView view =
          slackpipe::DecodeOperation(instance, id);
      ASSERT_EQ(view.worker, worker);
      slackpipe::Index &last =
          last_microbatch[static_cast<std::size_t>(view.chain_index)];
      EXPECT_TRUE(last < view.microbatch);
      last = view.microbatch;
    }
  }
}

slackpipe::ActivationAnalysisOptions ExplicitActivationCap(
    std::vector<slackpipe::Tick> cap_units) {
  slackpipe::ActivationAnalysisOptions options;
  options.model = slackpipe::ActivationModel::kCount;
  options.cap_mode = slackpipe::ActivationCapMode::kExplicit;
  options.activation_cap_units = std::move(cap_units);
  options.enforce_activation_cap = true;
  return options;
}

slackpipe::ActivationAnalysisOptions UniformBaselineActivationCap(
    std::vector<slackpipe::Tick> partition,
    std::vector<slackpipe::Tick> cap_units) {
  slackpipe::ActivationUniformBaseline baseline;
  baseline.partition = std::move(partition);
  baseline.cap_units_per_worker = cap_units;
  baseline.baseline_run_id = "activation-uniform-test";
  baseline.cap_derivation_hash = "activation-uniform-test-hash";
  baseline.method_contract_hash = "activation-uniform-test-contract";

  slackpipe::ActivationAnalysisOptions options;
  options.model = slackpipe::ActivationModel::kLinearInStageLayers;
  options.activation_units_per_layer = 1;
  options.cap_mode = slackpipe::ActivationCapMode::kUniformBaseline;
  options.activation_cap_units = std::move(cap_units);
  options.enforce_activation_cap = true;
  options.uniform_baseline = std::move(baseline);
  return options;
}

slackpipe::Instance TinyTwoStageActivationTradeoffInstance() {
  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 2;
  instance.workers = 2;
  instance.total_layers = 2;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;
  instance.communication_ticks = 0;
  return instance;
}

slackpipe::Instance CalFallbackReplayBlockerInstance() {
  slackpipe::Instance instance;
  instance.microbatches = 8;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 32;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;
  instance.communication_ticks = 8;
  return instance;
}

slackpipe::Instance CalJointFifoFallbackBlockerInstance() {
  slackpipe::Instance instance;
  instance.microbatches = 8;
  instance.stages = 8;
  instance.workers = 2;
  instance.total_layers = 64;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;
  instance.communication_ticks = 2;
  return instance;
}

slackpipe::Instance CalJointEqualMemoryFallbackBlockerInstance() {
  slackpipe::Instance instance = CalJointFifoFallbackBlockerInstance();
  instance.microbatches = 4;
  instance.communication_ticks = 8;
  return instance;
}

}  // namespace

TEST(OperationId, UsesCanonicalOperationPositionIndexing) {
  const slackpipe::Instance instance = SanityInstance();
  EXPECT_EQ(slackpipe::OperationPositionCount(instance), 8);

  for (slackpipe::Index n = 0; n < 8; ++n) {
    const slackpipe::OperationId id =
        slackpipe::EncodeOperation(instance, 5, n);
    const slackpipe::OperationView view =
        slackpipe::DecodeOperation(instance, id);
    EXPECT_EQ(view.chain_index, n);
    EXPECT_EQ(view.stage, n < 4 ? n : 7 - n);
    EXPECT_EQ(view.worker, view.stage % instance.workers);
  }
}

TEST(ActivationAnalyzer, OneZeroLengthLifetimeConsumesNoCapacity) {
  const slackpipe::Instance instance = OneStageActivationInstance(1);
  const std::vector<slackpipe::Tick> split{1};
  slackpipe::MachineOrders orders(1);
  orders[0] = {OneStageOp(instance, 0, false), OneStageOp(instance, 0, true)};
  const slackpipe::EvaluationResult evaluated =
      slackpipe::EvaluateSchedule(instance, split, orders);
  ASSERT_TRUE(evaluated.schedule.ok());

  const slackpipe::ActivationAnalysisResult analysis =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          instance, evaluated.schedule, slackpipe::ActivationAnalysisOptions{});

  EXPECT_TRUE(analysis.passed);
  EXPECT_EQ(analysis.total_activation_lifetimes, 1);
  EXPECT_EQ(analysis.zero_length_activation_lifetimes, 1);
  ASSERT_EQ(analysis.per_worker.size(), std::size_t{1});
  EXPECT_EQ(analysis.per_worker[0].peak_live_activation_count, 0);
  EXPECT_EQ(analysis.per_worker[0].peak_activation_units, 0);
  EXPECT_FALSE(analysis.per_worker[0].peak_activation_bytes);
}

TEST(ActivationAnalyzer, ReleaseBeforeAcquireAtEqualTimestamp) {
  const slackpipe::Instance instance = OneStageActivationInstance(2);
  const std::vector<slackpipe::Tick> split{1};
  slackpipe::MachineOrders orders(1);
  orders[0] = {OneStageOp(instance, 0, false), OneStageOp(instance, 1, false),
               OneStageOp(instance, 0, true), OneStageOp(instance, 1, true)};
  const slackpipe::EvaluationResult evaluated =
      slackpipe::EvaluateSchedule(instance, split, orders);
  ASSERT_TRUE(evaluated.schedule.ok());

  const slackpipe::ActivationAnalysisResult analysis =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          instance, evaluated.schedule, slackpipe::ActivationAnalysisOptions{});

  ASSERT_EQ(analysis.per_worker.size(), std::size_t{1});
  EXPECT_EQ(analysis.per_worker[0].peak_live_activation_count, 1);
  EXPECT_EQ(analysis.per_worker[0].peak_activation_units, 1);
  EXPECT_EQ(analysis.per_worker[0].time_of_peak, 1);
  EXPECT_EQ(analysis.per_worker[0].activation_unit_time_area, 3);
}

TEST(ActivationAnalyzer, SameProblemDifferentOrdersCanChangePeakMemory) {
  const slackpipe::Instance instance = OneStageActivationInstance(3);
  const std::vector<slackpipe::Tick> split{1};

  slackpipe::MachineOrders low_memory(1);
  low_memory[0] = {
      OneStageOp(instance, 0, false), OneStageOp(instance, 0, true),
      OneStageOp(instance, 1, false), OneStageOp(instance, 2, false),
      OneStageOp(instance, 1, true),  OneStageOp(instance, 2, true)};
  const slackpipe::EvaluationResult low =
      slackpipe::EvaluateSchedule(instance, split, low_memory);
  ASSERT_TRUE(low.schedule.ok());

  slackpipe::MachineOrders high_memory(1);
  high_memory[0] = {
      OneStageOp(instance, 0, false), OneStageOp(instance, 1, false),
      OneStageOp(instance, 2, false), OneStageOp(instance, 0, true),
      OneStageOp(instance, 1, true),  OneStageOp(instance, 2, true)};
  const slackpipe::EvaluationResult high =
      slackpipe::EvaluateSchedule(instance, split, high_memory);
  ASSERT_TRUE(high.schedule.ok());

  const slackpipe::ActivationAnalysisResult low_analysis =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          instance, low.schedule, slackpipe::ActivationAnalysisOptions{});
  slackpipe::ActivationAnalysisOptions trace_options;
  trace_options.emit_activation_trace = true;
  const slackpipe::ActivationAnalysisResult high_analysis =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          instance, high.schedule, trace_options);

  EXPECT_EQ(low_analysis.per_worker[0].peak_activation_units, 1);
  EXPECT_EQ(high_analysis.per_worker[0].peak_activation_units, 2);
  EXPECT_EQ(high_analysis.per_worker[0].peak_live_activation_count, 2);
  EXPECT_FALSE(high_analysis.event_trace.empty());
}

TEST(ActivationAnalyzer, LinearUnitsExplicitUnitsAndBytes) {
  slackpipe::Instance instance = OneStageActivationInstance(3);
  instance.total_layers = 3;
  const std::vector<slackpipe::Tick> split{3};
  slackpipe::MachineOrders orders(1);
  orders[0] = {OneStageOp(instance, 0, false), OneStageOp(instance, 1, false),
               OneStageOp(instance, 2, false), OneStageOp(instance, 0, true),
               OneStageOp(instance, 1, true),  OneStageOp(instance, 2, true)};
  const slackpipe::EvaluationResult evaluated =
      slackpipe::EvaluateSchedule(instance, split, orders);
  ASSERT_TRUE(evaluated.schedule.ok());

  slackpipe::ActivationAnalysisOptions linear;
  linear.activation_units_per_layer = 2;
  linear.activation_bytes_per_unit = 2.5;
  const slackpipe::ActivationAnalysisResult linear_analysis =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          instance, evaluated.schedule, linear);
  EXPECT_EQ(linear_analysis.per_worker[0].peak_activation_units, 12);
  ASSERT_TRUE(linear_analysis.per_worker[0].peak_activation_bytes);
  EXPECT_TRUE(std::abs(*linear_analysis.per_worker[0].peak_activation_bytes -
                       30.0) <= 1e-9);
  EXPECT_TRUE(
      std::abs(*linear_analysis.global.maximum_worker_peak_activation_bytes -
               30.0) <= 1e-9);

  slackpipe::ActivationAnalysisOptions explicit_units;
  explicit_units.model = slackpipe::ActivationModel::kExplicitStageUnits;
  explicit_units.explicit_stage_activation_units = {7};
  const slackpipe::ActivationAnalysisResult explicit_analysis =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          instance, evaluated.schedule, explicit_units);
  EXPECT_EQ(explicit_analysis.per_worker[0].peak_activation_units, 14);
}

TEST(ActivationAnalyzer, ExplicitCapViolationDiagnosticsAndUniformRatio) {
  const slackpipe::Instance instance = OneStageActivationInstance(3);
  const std::vector<slackpipe::Tick> split{1};
  slackpipe::MachineOrders orders(1);
  orders[0] = {OneStageOp(instance, 0, false), OneStageOp(instance, 1, false),
               OneStageOp(instance, 2, false), OneStageOp(instance, 0, true),
               OneStageOp(instance, 1, true),  OneStageOp(instance, 2, true)};
  const slackpipe::EvaluationResult evaluated =
      slackpipe::EvaluateSchedule(instance, split, orders);
  ASSERT_TRUE(evaluated.schedule.ok());

  slackpipe::ActivationAnalysisOptions options;
  options.cap_mode = slackpipe::ActivationCapMode::kExplicit;
  options.activation_cap_units = {1};
  const slackpipe::ActivationAnalysisResult analysis =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          instance, evaluated.schedule, options);

  ASSERT_TRUE(analysis.activation_cap_satisfied);
  EXPECT_FALSE(*analysis.activation_cap_satisfied);
  ASSERT_TRUE(analysis.activation_cap_violation);
  EXPECT_EQ(analysis.activation_cap_violation->worker, 0);
  EXPECT_EQ(analysis.activation_cap_violation->cap_units, 1);
  EXPECT_EQ(analysis.activation_cap_violation->observed_peak_units, 2);
  EXPECT_EQ(analysis.activation_cap_violation->first_violating_timestamp, 2);
  EXPECT_EQ(analysis.activation_cap_violation->live_activations.size(),
            std::size_t{2});
  EXPECT_FALSE(analysis.activation_peak_ratio_to_uniform
                   .maximum_worker_peak_units_ratio);
  EXPECT_FALSE(analysis.activation_peak_ratio_to_uniform.warnings.empty());
}

TEST(ActivationAnalyzer, ResolvesScalarAndVectorCaps) {
  slackpipe::Instance instance;
  instance.microbatches = 1;
  instance.stages = 2;
  instance.workers = 2;
  instance.total_layers = 2;
  instance.min_layers = 1;

  const std::vector<slackpipe::Tick> scalar =
      slackpipe::ResolveExplicitActivationCap(instance, {5});
  ASSERT_EQ(scalar.size(), std::size_t{2});
  EXPECT_EQ(scalar[0], 5);
  EXPECT_EQ(scalar[1], 5);

  const std::vector<slackpipe::Tick> vector =
      slackpipe::ResolveExplicitActivationCap(instance, {3, 7});
  ASSERT_EQ(vector.size(), std::size_t{2});
  EXPECT_EQ(vector[0], 3);
  EXPECT_EQ(vector[1], 7);
  EXPECT_THROW(
      (void)slackpipe::ResolveExplicitActivationCap(instance, {1, 2, 3}),
      slackpipe::Error);
}

TEST(ActivationAnalyzer, EnforcedCapRequiresConfiguredCap) {
  const slackpipe::Instance instance = OneStageActivationInstance(1);
  slackpipe::ActivationAnalysisOptions options;
  options.enforce_activation_cap = true;

  EXPECT_THROW(slackpipe::ValidateActivationOptions(instance, options),
               slackpipe::Error);
}

TEST(ActivationCapSolverSupport, ReportsUnsupportedVariableDemandCombination) {
  if (slackpipe::ActivationCapSolverSupport() ==
      slackpipe::ActivationCapSolverSupportLevel::kVariableDemands) {
    return;
  }
  slackpipe::ActivationAnalysisOptions options;
  options.cap_mode = slackpipe::ActivationCapMode::kExplicit;
  options.activation_cap_units = {10};
  options.enforce_activation_cap = true;
  options.model = slackpipe::ActivationModel::kLinearInStageLayers;

  EXPECT_FALSE(slackpipe::ActivationCapSolverCanEnforce(options, true));
  const std::string reason =
      slackpipe::ActivationCapSolverUnsupportedReason(options, true);
  if (slackpipe::ActivationCapSolverSupport() ==
      slackpipe::ActivationCapSolverSupportLevel::kNone) {
    EXPECT_TRUE(reason.find("cumulative") != std::string::npos);
  } else {
    EXPECT_TRUE(reason.find("variable cumulative") != std::string::npos);
  }
}

TEST(ActivationCapSolverSupport, ReportsNoSupportWithoutOrTools) {
  if (slackpipe::OrToolsCompiledIn()) {
    return;
  }
  EXPECT_FALSE(slackpipe::CumulativeConstraintCompiledIn());
  EXPECT_FALSE(slackpipe::VariableCumulativeDemandCompiledIn());
  EXPECT_EQ(slackpipe::ToString(slackpipe::ActivationCapSolverSupport()),
            std::string("none"));
  EXPECT_EQ(slackpipe::ActivationCapSolverSupportLevelForBuild(),
            std::string("none"));

  slackpipe::ActivationAnalysisOptions options;
  options.cap_mode = slackpipe::ActivationCapMode::kExplicit;
  options.activation_cap_units = {10};
  options.enforce_activation_cap = true;
  EXPECT_FALSE(slackpipe::ActivationCapSolverCanEnforce(options, true));
  EXPECT_TRUE(slackpipe::ActivationCapSolverUnsupportedReason(options, true)
                  .find("cumulative") != std::string::npos);
}

TEST(ActivationAnalyzer, UniformBaselineDerivationIsStable) {
  const slackpipe::Instance instance = OneStageActivationInstance(3);
  slackpipe::ActivationAnalysisOptions options;
  options.cap_mode = slackpipe::ActivationCapMode::kUniformBaseline;

  const slackpipe::ActivationUniformBaseline first =
      slackpipe::DeriveUniformActivationBaseline(instance, options);
  const slackpipe::ActivationUniformBaseline second =
      slackpipe::DeriveUniformActivationBaseline(instance, options);

  EXPECT_TRUE(first.errors.empty());
  ASSERT_EQ(first.partition.size(), std::size_t{1});
  EXPECT_EQ(first.partition[0], 1);
  ASSERT_EQ(first.cap_units_per_worker.size(), std::size_t{1});
  EXPECT_EQ(first.cap_units_per_worker[0], 0);
  EXPECT_FALSE(first.cap_derivation_hash.empty());
  EXPECT_EQ(first.cap_derivation_hash, second.cap_derivation_hash);
  EXPECT_EQ(first.baseline_run_id, second.baseline_run_id);
}

TEST(ActivationAnalyzer, PeakRatioToUniformUsesNonzeroDenominators) {
  slackpipe::Instance instance;
  instance.microbatches = 1;
  instance.stages = 2;
  instance.workers = 2;
  instance.total_layers = 2;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;
  const std::vector<slackpipe::Tick> split = slackpipe::UniformSplit(instance);
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);
  const slackpipe::EvaluationResult evaluated =
      slackpipe::EvaluateSchedule(instance, split, orders);
  ASSERT_TRUE(evaluated.schedule.ok());

  const slackpipe::ActivationAnalysisResult analysis =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          instance, evaluated.schedule, slackpipe::ActivationAnalysisOptions{});

  ASSERT_TRUE(analysis.activation_peak_ratio_to_uniform
                  .maximum_worker_peak_units_ratio);
  EXPECT_TRUE(std::abs(*analysis.activation_peak_ratio_to_uniform
                            .maximum_worker_peak_units_ratio -
                       1.0) <= 1e-9);
  ASSERT_EQ(analysis.activation_peak_ratio_to_uniform
                .per_worker_peak_units_ratio.size(),
            std::size_t{2});
  ASSERT_TRUE(
      analysis.activation_peak_ratio_to_uniform.per_worker_peak_units_ratio[0]);
  EXPECT_TRUE(std::abs(*analysis.activation_peak_ratio_to_uniform
                            .per_worker_peak_units_ratio[0] -
                       1.0) <= 1e-9);
}

TEST(ActivationAnalyzer, EnforcedCapViolationInvalidatesCanonicalOutcome) {
  const slackpipe::Instance instance = OneStageActivationInstance(3);
  const std::vector<slackpipe::Tick> split{1};
  slackpipe::MachineOrders orders(1);
  orders[0] = {OneStageOp(instance, 0, false), OneStageOp(instance, 1, false),
               OneStageOp(instance, 2, false), OneStageOp(instance, 0, true),
               OneStageOp(instance, 1, true),  OneStageOp(instance, 2, true)};
  const slackpipe::EvaluationResult evaluated =
      slackpipe::EvaluateSchedule(instance, split, orders);
  ASSERT_TRUE(evaluated.schedule.ok());

  slackpipe::CanonicalResultMetadata canonical =
      slackpipe::BuildCanonicalResultMetadata(
          instance, slackpipe::CanonicalRequestContext{},
          slackpipe::SemanticsForUniformFixedOrderBaseline(),
          slackpipe::OutcomeFromSchedule(evaluated.schedule, "FEASIBLE"), split,
          orders);
  const slackpipe::ResultValidationResult validation =
      slackpipe::ValidateScheduleSolutionIndependent(instance,
                                                     evaluated.schedule);
  ASSERT_TRUE(validation.passed);
  slackpipe::ApplyResultValidation(canonical.outcome, validation);

  slackpipe::ActivationAnalysisOptions options;
  options.cap_mode = slackpipe::ActivationCapMode::kExplicit;
  options.activation_cap_units = {1};
  options.enforce_activation_cap = true;
  const slackpipe::ActivationAnalysisResult activation =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          instance, evaluated.schedule, options);
  slackpipe::ApplyActivationAnalysis(canonical, activation);

  ASSERT_TRUE(canonical.activation_analysis);
  ASSERT_TRUE(canonical.activation_analysis->activation_cap_satisfied);
  EXPECT_FALSE(*canonical.activation_analysis->activation_cap_satisfied);
  ASSERT_TRUE(canonical.outcome.reported_status);
  EXPECT_EQ(*canonical.outcome.reported_status, "INVALID_RESULT");
  ASSERT_TRUE(canonical.outcome.feasible);
  EXPECT_FALSE(*canonical.outcome.feasible);
  EXPECT_FALSE(canonical.outcome.makespan);
  EXPECT_FALSE(canonical.outcome.result_validation_passed);
}

TEST(OperationId, MapsBackwardOperationsToStageWorkers) {
  slackpipe::Instance instance;
  instance.microbatches = 1;
  instance.stages = 8;
  instance.workers = 4;
  instance.total_layers = 8;
  instance.min_layers = 1;

  const std::vector<slackpipe::Index> expected_stage{0, 1, 2, 3, 4, 5, 6, 7,
                                                     7, 6, 5, 4, 3, 2, 1, 0};
  const std::vector<slackpipe::Index> expected_worker{0, 1, 2, 3, 0, 1, 2, 3,
                                                      3, 2, 1, 0, 3, 2, 1, 0};
  for (slackpipe::Index n = 0; n < 16; ++n) {
    EXPECT_EQ(slackpipe::StageForOperationPosition(instance, n),
              expected_stage[static_cast<std::size_t>(n)]);
    EXPECT_EQ(slackpipe::WorkerForOperationPosition(instance, n),
              expected_worker[static_cast<std::size_t>(n)]);
  }
}

TEST(OperationId, BackwardOperationWorkerDoesNotUseRawModulo) {
  slackpipe::Instance instance;
  instance.microbatches = 1;
  instance.stages = 8;
  instance.workers = 4;
  instance.total_layers = 8;
  instance.min_layers = 1;

  const slackpipe::Index n = 8;
  EXPECT_EQ(slackpipe::StageForOperationPosition(instance, n), 7);
  EXPECT_EQ(slackpipe::WorkerForOperationPosition(instance, n), 3);
  EXPECT_EQ(n % instance.workers, 0);
  EXPECT_NE(slackpipe::WorkerForOperationPosition(instance, n),
            n % instance.workers);
}

TEST(BreadthFirst, ProducesOneOperationPerWorkerListEntry) {
  const slackpipe::Instance instance = BaseInstance();
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);
  std::set<slackpipe::Index> seen;
  for (const auto &order : orders) {
    for (slackpipe::OperationId id : order) seen.insert(id.value);
  }
  EXPECT_EQ(static_cast<slackpipe::Index>(seen.size()),
            instance.OperationCount());
}

TEST(BreadthFirst, UsesBreadthFirstWavefrontNotExactInterleavedOneFOneB) {
  slackpipe::Instance instance;
  instance.microbatches = 3;
  instance.stages = 2;
  instance.workers = 1;
  instance.total_layers = 4;
  instance.min_layers = 1;
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);
  ASSERT_EQ(orders.size(), std::size_t{1});

  std::vector<std::string> names;
  for (slackpipe::OperationId id : orders.front()) {
    names.push_back(slackpipe::OperationName(instance, id));
  }
  const std::vector<std::string> exact_1f1b_prefix{"F0(b0)", "F1(b0)", "B1(b0)",
                                                   "F0(b1)"};
  EXPECT_NE(std::vector<std::string>(names.begin(), names.begin() + 4),
            exact_1f1b_prefix);
  EXPECT_EQ(slackpipe::CanonicalizeEvaluationMethodName("eval-bfs"),
            std::string(slackpipe::kUniformBreadthFirstMethod));
}

TEST(InterleavedOneFOneB, HasExactOneFOneBPrefixAndDiffersFromBreadthFirst) {
  slackpipe::Instance instance;
  instance.microbatches = 3;
  instance.stages = 2;
  instance.workers = 1;
  instance.total_layers = 4;
  instance.min_layers = 1;

  const slackpipe::MachineOrders orders =
      slackpipe::InterleavedOneFOneBOrders(instance);
  ASSERT_EQ(orders.size(), std::size_t{1});

  std::vector<std::string> names;
  for (slackpipe::OperationId id : orders.front()) {
    names.push_back(slackpipe::OperationName(instance, id));
  }
  const std::vector<std::string> exact_1f1b_prefix{"F0(b0)", "F1(b0)", "B1(b0)",
                                                   "F0(b1)"};
  const std::vector<std::string> actual_prefix(names.begin(),
                                               names.begin() + 4);
  EXPECT_TRUE(actual_prefix == exact_1f1b_prefix);
  EXPECT_NE(orders, slackpipe::BreadthFirstOrders(instance));
}

TEST(InterleavedOneFOneB, ValidatesSmallCyclicCases) {
  std::vector<slackpipe::Instance> cases;
  {
    slackpipe::Instance instance;
    instance.microbatches = 4;
    instance.stages = 2;
    instance.workers = 2;
    instance.total_layers = 16;
    instance.min_layers = 1;
    cases.push_back(instance);
  }
  {
    slackpipe::Instance instance;
    instance.microbatches = 8;
    instance.stages = 4;
    instance.workers = 4;
    instance.total_layers = 32;
    instance.min_layers = 1;
    cases.push_back(instance);
  }
  {
    slackpipe::Instance instance;
    instance.microbatches = 4;
    instance.stages = 4;
    instance.workers = 2;
    instance.total_layers = 32;
    instance.min_layers = 1;
    cases.push_back(instance);
  }
  {
    slackpipe::Instance instance;
    instance.microbatches = 8;
    instance.stages = 8;
    instance.workers = 4;
    instance.total_layers = 64;
    instance.min_layers = 1;
    cases.push_back(instance);
  }

  for (const slackpipe::Instance &instance : cases) {
    const std::vector<slackpipe::Tick> split =
        slackpipe::UniformSplit(instance);
    const slackpipe::MachineOrders orders =
        slackpipe::InterleavedOneFOneBOrders(instance);
    const slackpipe::EvaluationResult evaluated =
        slackpipe::EvaluateSchedule(instance, split, orders);
    ASSERT_TRUE(evaluated.schedule.ok());
    std::set<slackpipe::Index> seen;
    for (slackpipe::Index worker = 0; worker < instance.workers; ++worker) {
      for (slackpipe::OperationId id :
           orders[static_cast<std::size_t>(worker)]) {
        const slackpipe::OperationView view =
            slackpipe::DecodeOperation(instance, id);
        EXPECT_EQ(view.worker, worker);
        seen.insert(id.value);
      }
    }
    EXPECT_EQ(static_cast<slackpipe::Index>(seen.size()),
              instance.OperationCount());
    ExpectSamePositionFifoRespected(instance, orders);
    slackpipe::ResultValidationInput input =
        slackpipe::ValidationInputFromSchedule(instance, evaluated.schedule,
                                               "FEASIBLE", "1f1b-test");
    input.derived_worker_predecessors = DerivedRecords(instance, orders);
    input.require_serialized_worker_predecessors = true;
    input.require_serialized_operations = true;
    ASSERT_TRUE(input.reported_makespan.has_value());
    EXPECT_EQ(*input.reported_makespan, evaluated.schedule.makespan);
    const slackpipe::ResultValidationResult validation =
        slackpipe::ValidateResult(input);
    EXPECT_TRUE(validation.passed);
    const slackpipe::ResultValidationResult independent_validation =
        slackpipe::ValidateScheduleSolutionIndependent(
            instance, evaluated.schedule, "FEASIBLE", "1f1b-test");
    EXPECT_TRUE(independent_validation.passed);
    const slackpipe::ActivationAnalysisResult activation =
        slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
            instance, evaluated.schedule,
            slackpipe::ActivationAnalysisOptions{});
    EXPECT_TRUE(activation.passed);
    EXPECT_EQ(activation.total_activation_lifetimes,
              instance.microbatches * instance.stages);
  }
}

TEST(EvaluationMethodRegistry, ContainsInterleavedOneFOneBContract) {
  const slackpipe::EvaluationMethodDefinition *definition =
      slackpipe::FindEvaluationMethod(
          slackpipe::kUniformInterleavedOneFOneBMethod);
  ASSERT_TRUE(definition != nullptr);
  EXPECT_EQ(definition->canonical_name,
            std::string(slackpipe::kUniformInterleavedOneFOneBMethod));
  EXPECT_EQ(definition->schedule_decision, "fixed_interleaved_1f1b");
  EXPECT_TRUE(definition->deterministic);
  EXPECT_FALSE(definition->requires_ortools);
  EXPECT_EQ(slackpipe::CanonicalizeEvaluationMethodName("eval-1f1b"),
            std::string(slackpipe::kUniformInterleavedOneFOneBMethod));
  EXPECT_EQ(slackpipe::EvaluationMethodContractHash(*definition),
            "0f2b9aaaa049fd5b");

  const std::string json = slackpipe::DescribeEvaluationMethodJson(
      slackpipe::kUniformInterleavedOneFOneBMethod);
  EXPECT_NE(json.find("\"fixed_schedule_rule\": "
                      "\"interleaved_1f1b_stage_local_streams\""),
            std::string::npos);
  EXPECT_NE(json.find("\"method_contract_hash\": \"0f2b9aaaa049fd5b\""),
            std::string::npos);
}

TEST(Deadline, RemainingTimeIsMonotonicAndNonNegative) {
  slackpipe::Deadline deadline(0.02);
  const double first = deadline.remaining_seconds();
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  const double second = deadline.remaining_seconds();
  EXPECT_TRUE(first >= second);
  std::this_thread::sleep_for(std::chrono::milliseconds(25));
  EXPECT_EQ(deadline.remaining_seconds(), 0.0);
  EXPECT_TRUE(deadline.expired());
}

TEST(Deadline, ChildPhaseCannotExceedParentRemaining) {
  slackpipe::Deadline deadline(1.0);
  const double remaining = deadline.remaining_seconds();
  const double child = deadline.clamp_solver_limit(10.0);
  EXPECT_TRUE(child <= remaining);
  EXPECT_TRUE(child > 0.0);

  slackpipe::Deadline unbounded(0.0);
  EXPECT_FALSE(unbounded.bounded());
  EXPECT_EQ(unbounded.clamp_solver_limit(0.0), 0.0);
  EXPECT_EQ(unbounded.remaining_seconds_for_reporting(), 0.0);
}

TEST(EvaluationMethodRegistry, DescribesCanonicalControlsAndPhaseCaps) {
  const slackpipe::EvaluationMethodDefinition *alternating =
      slackpipe::FindEvaluationMethod("alternating-partition-schedule");
  ASSERT_TRUE(alternating != nullptr);
  EXPECT_TRUE(alternating->partition_optimized);
  EXPECT_TRUE(alternating->schedule_optimized);
  EXPECT_TRUE(alternating->requires_ortools);
  EXPECT_EQ(slackpipe::AlternatingPhaseLimitSeconds(8.0, 4), 2.0);
  EXPECT_EQ(slackpipe::AlternatingPhaseLimitSeconds(8.0, 0), 0.0);

  const std::string json =
      slackpipe::DescribeEvaluationMethodJson("schedule-only-partition-only");
  EXPECT_TRUE(json.find("\"canonical_name\": "
                        "\"sequential-partition-then-schedule\"") !=
              std::string::npos);
  EXPECT_TRUE(json.find("\"evaluation_method_version\": 1") !=
              std::string::npos);
}

TEST(EvaluationMethodRegistry, ControlledFieldCheckerRejectsMismatch) {
  slackpipe::MethodCompatibilityRecord a;
  a.method_name = "uniform-breadth-first";
  a.micro_batches = 2;
  a.logical_stages = 2;
  a.physical_workers = 2;
  a.total_layers = 4;
  a.requested_time_limit_seconds = 10.0;
  a.solver_threads = 1;
  slackpipe::MethodCompatibilityRecord b = a;
  b.method_name = "joint-unrestricted-no-overlap";
  b.total_layers = 5;
  const slackpipe::MethodCompatibilityCheck check =
      slackpipe::CheckMethodCompatibility({a, b}, false);
  EXPECT_FALSE(check.passed);
  EXPECT_TRUE(check.message.find("L mismatch") != std::string::npos);
}

TEST(ResultValidator, AcceptsValidHandBfsSchedule) {
  const slackpipe::ResultValidationResult validation =
      slackpipe::ValidateResult(ManualValidatorInput());
  EXPECT_TRUE(validation.passed);
  ASSERT_TRUE(validation.reconstructed_makespan.has_value());
  ASSERT_TRUE(validation.serialized_makespan.has_value());
  EXPECT_EQ(*validation.reconstructed_makespan, 12);
  EXPECT_EQ(*validation.serialized_makespan, 12);
  EXPECT_EQ(validation.data_edge_count, 3);
  EXPECT_EQ(validation.fifo_edge_count, 0);
  EXPECT_EQ(validation.worker_edge_count, 1);
}

TEST(ResultValidator, AcceptsValidNonBfsWorkerOrder) {
  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 1;
  instance.workers = 1;
  instance.total_layers = 2;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  const std::vector<slackpipe::Tick> split{2};
  slackpipe::MachineOrders orders(1);
  orders[0] = {slackpipe::EncodeOperation(instance, 0, 0),
               slackpipe::EncodeOperation(instance, 1, 0),
               slackpipe::EncodeOperation(instance, 0, 1),
               slackpipe::EncodeOperation(instance, 1, 1)};

  slackpipe::ResultValidationInput input;
  input.instance = instance;
  input.selected_partition = split;
  input.worker_orders = orders;
  input.derived_worker_predecessors = DerivedRecords(instance, orders);
  input.require_serialized_worker_predecessors = true;
  input.reported_makespan = 12;
  input.feasible_claimed = true;

  const slackpipe::ResultValidationResult validation =
      slackpipe::ValidateResult(input);
  EXPECT_TRUE(validation.passed);
  EXPECT_EQ(validation.worker_edge_count, 1);
  ASSERT_TRUE(validation.reconstructed_makespan.has_value());
  EXPECT_EQ(*validation.reconstructed_makespan, 12);
}

TEST(ResultValidator, RejectsPartitionErrors) {
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.selected_partition = std::nullopt;
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code, "partition_missing");
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.selected_partition = std::vector<slackpipe::Tick>{4};
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "partition_length_mismatch");
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.selected_partition = std::vector<slackpipe::Tick>{0, 4};
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "partition_empty_stage");
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.selected_partition = std::vector<slackpipe::Tick>{2, 3};
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "partition_sum_mismatch");
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.fixed_partition_reference = std::vector<slackpipe::Tick>{3, 1};
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "partition_fixed_value_mismatch");
  }
}

TEST(ResultValidator, RejectsOperationInventoryErrors) {
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.worker_orders->front().pop_back();
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code, "operation_missing");
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.worker_orders->front().push_back(
        input.worker_orders->front().front());
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "operation_duplicate");
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.worker_orders->front().push_back(
        slackpipe::EncodeOperation(input.instance, 0, 1));
    input.worker_orders->at(1).erase(input.worker_orders->at(1).begin());
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "worker_order_wrong_worker");
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.worker_orders->front().front() = slackpipe::OperationId{999};
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code, "operation_unknown");
  }
}

TEST(ResultValidator, RejectsSerializedOperationIdentityAndDurationErrors) {
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.serialized_operations->at(1).name = "F0(b0)";
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "operation_name_malformed");
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.serialized_operations->at(1).stage = 0;
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "operation_stage_mismatch");
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.serialized_operations->at(1).phase = "B";
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "operation_phase_mismatch");
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.serialized_operations->at(1).worker = 0;
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "operation_worker_mismatch");
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.serialized_operations->at(1).duration = 7;
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "operation_duration_mismatch");
  }
}

TEST(ResultValidator, RejectsDependencyTimingErrors) {
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.serialized_operations->at(1).start = 1;
    input.serialized_operations->at(1).end = 3;
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "data_dependency_violation");
  }
  {
    slackpipe::Instance instance;
    instance.microbatches = 2;
    instance.stages = 1;
    instance.workers = 1;
    instance.total_layers = 2;
    instance.min_layers = 1;
    instance.backward_ratio_num = 2;
    const std::vector<slackpipe::Tick> split{2};
    const slackpipe::MachineOrders orders =
        slackpipe::BreadthFirstOrders(instance);
    slackpipe::ResultValidationInput input =
        EvaluatedValidatorInput(instance, split, orders);
    input.serialized_operations->at(2).start = 1;
    input.serialized_operations->at(2).end = 3;
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "fifo_dependency_violation");
  }
  {
    slackpipe::Instance instance;
    instance.microbatches = 2;
    instance.stages = 1;
    instance.workers = 1;
    instance.total_layers = 2;
    instance.min_layers = 1;
    instance.backward_ratio_num = 2;
    const std::vector<slackpipe::Tick> split{2};
    slackpipe::MachineOrders orders(1);
    orders[0] = {slackpipe::EncodeOperation(instance, 0, 0),
                 slackpipe::EncodeOperation(instance, 1, 0),
                 slackpipe::EncodeOperation(instance, 0, 1),
                 slackpipe::EncodeOperation(instance, 1, 1)};
    slackpipe::ResultValidationInput input =
        EvaluatedValidatorInput(instance, split, orders);
    input.serialized_operations->at(1).start = 3;
    input.serialized_operations->at(1).end = 7;
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "worker_serialization_violation");
  }
}

TEST(ResultValidator, RejectsCyclesAndBadDerivedPredecessors) {
  slackpipe::Instance instance;
  instance.microbatches = 1;
  instance.stages = 1;
  instance.workers = 1;
  instance.total_layers = 2;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  const std::vector<slackpipe::Tick> split{2};
  slackpipe::MachineOrders cyclic_orders(1);
  cyclic_orders[0] = {slackpipe::EncodeOperation(instance, 0, 1),
                      slackpipe::EncodeOperation(instance, 0, 0)};

  slackpipe::ResultValidationInput cyclic;
  cyclic.instance = instance;
  cyclic.selected_partition = split;
  cyclic.worker_orders = cyclic_orders;
  cyclic.reported_makespan = 6;
  cyclic.feasible_claimed = true;

  const slackpipe::ResultValidationResult first =
      slackpipe::ValidateResult(cyclic);
  const slackpipe::ResultValidationResult second =
      slackpipe::ValidateResult(cyclic);
  EXPECT_EQ(first.error_code, "dependency_cycle");
  ASSERT_EQ(first.cycle_witness.size(), second.cycle_witness.size());
  for (std::size_t i = 0; i < first.cycle_witness.size(); ++i) {
    EXPECT_EQ(first.cycle_witness[i], second.cycle_witness[i]);
  }

  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.derived_worker_predecessors->at(0).predecessor_id =
        input.derived_worker_predecessors->at(0).operation_id;
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "malformed_self_edge");
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.derived_worker_predecessors->clear();
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "derived_predecessor_mismatch");
  }
}

TEST(ResultValidator, RejectsCommunicationAndObjectiveErrors) {
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.communication_model = "alpha_beta";
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "unsupported_communication_model");
  }
  {
    slackpipe::Instance instance;
    instance.microbatches = 1;
    instance.stages = 2;
    instance.workers = 2;
    instance.total_layers = 4;
    instance.min_layers = 1;
    instance.backward_ratio_num = 2;
    instance.communication_ticks = 3;
    const std::vector<slackpipe::Tick> split{2, 2};
    const slackpipe::MachineOrders orders =
        slackpipe::BreadthFirstOrders(instance);
    slackpipe::ResultValidationInput input =
        EvaluatedValidatorInput(instance, split, orders);
    input.serialized_operations->at(1).start = 2;
    input.serialized_operations->at(1).end = 4;
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "data_dependency_violation");
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.reported_makespan = 13;
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "reported_makespan_mismatch");
  }
}

TEST(ResultValidator, HandlesFallbackAndInfeasibleSemantics) {
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.solver_status_raw = "UNKNOWN";
    input.reported_status = "FEASIBLE";
    EXPECT_TRUE(slackpipe::ValidateResult(input).passed);
  }
  {
    slackpipe::ResultValidationInput input = ManualValidatorInput();
    input.solver_status_raw = "NOT_RUN";
    input.reported_status = "FEASIBLE";
    EXPECT_TRUE(slackpipe::ValidateResult(input).passed);
  }
  {
    slackpipe::ResultValidationInput input;
    input.instance = ManualValidatorInput().instance;
    input.feasible_claimed = false;
    const slackpipe::ResultValidationResult validation =
        slackpipe::ValidateResult(input);
    EXPECT_TRUE(validation.passed);
    EXPECT_FALSE(validation.warnings.empty());
  }
  {
    slackpipe::ResultValidationInput input;
    input.instance = ManualValidatorInput().instance;
    input.feasible_claimed = false;
    input.reported_makespan = 12;
    EXPECT_EQ(slackpipe::ValidateResult(input).error_code,
              "infeasible_has_makespan");
  }
}

TEST(ResultValidator, CrossChecksSmallSchedulesWithDagEvaluatorInTestsOnly) {
  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 2;
  instance.workers = 2;
  instance.total_layers = 4;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  const std::vector<slackpipe::Tick> split{2, 2};
  const std::vector<slackpipe::MachineOrders> orders =
      ExhaustiveMachineOrders(instance);

  int checked = 0;
  for (const slackpipe::MachineOrders &order : orders) {
    const slackpipe::EvaluationResult evaluated =
        slackpipe::EvaluateSchedule(instance, split, order);
    if (!evaluated.schedule.ok()) continue;
    slackpipe::ResultValidationInput input =
        slackpipe::ValidationInputFromSchedule(instance, evaluated.schedule,
                                               "FEASIBLE", "cross-check");
    input.derived_worker_predecessors = DerivedRecords(instance, order);
    input.require_serialized_worker_predecessors = true;
    input.require_serialized_operations = true;
    const slackpipe::ResultValidationResult validation =
        slackpipe::ValidateResult(input);
    ASSERT_TRUE(validation.passed);
    ASSERT_TRUE(validation.reconstructed_makespan.has_value());
    EXPECT_EQ(*validation.reconstructed_makespan, evaluated.schedule.makespan);
    if (++checked == 3) break;
  }
  EXPECT_EQ(checked, 3);
}

TEST(ResultSchema, PartitionOnlyBfsSerializesFixedSchedule) {
  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 2;
  instance.workers = 2;
  instance.total_layers = 4;
  instance.min_layers = 1;

  slackpipe::BfsSplitOptimizationResult result =
      slackpipe::OptimizeBfsSplitEnumerate(instance);
  slackpipe::CanonicalRequestContext request;
  request.requested_method = "optimize-bfs";
  request.requested_command = "slackpipe_cli --algorithm optimize-bfs";
  request.requested_time_limit_seconds = 0.0;
  request.effective_time_limit_seconds = 0.0;
  request.random_seed = 1;
  request.solver_threads = 1;
  result.canonical = slackpipe::BuildCanonicalResultMetadata(
      instance, request, slackpipe::SemanticsForPartitionOnlyBfs(result.method),
      slackpipe::OutcomeFromBfsResult(result), result.split,
      result.machine_orders);
  const slackpipe::ActivationAnalysisResult activation =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          instance, result.schedule, slackpipe::ActivationAnalysisOptions{});
  slackpipe::ApplyActivationAnalysis(*result.canonical, activation);

  const std::string json = slackpipe::ToJson(instance, result);
  EXPECT_TRUE(json.find("\"schema_version\": 1") != std::string::npos);
  EXPECT_TRUE(json.find("\"budget_policy_version\": 1") != std::string::npos);
  EXPECT_TRUE(json.find("\"evaluation_method_version\": 1") !=
              std::string::npos);
  EXPECT_TRUE(json.find("\"phase_budget\": {") != std::string::npos);
  EXPECT_TRUE(
      json.find("\"canonical_method\": \"partition-only-fixed-order\"") !=
      std::string::npos);
  EXPECT_TRUE(json.find("\"method_contract_hash\": \"") != std::string::npos);
  EXPECT_TRUE(json.find("\"partition_decision\": \"optimized_global\"") !=
              std::string::npos);
  EXPECT_TRUE(json.find("\"schedule_decision\": \"fixed_breadth_first\"") !=
              std::string::npos);
  EXPECT_TRUE(json.find("\"partition_optimized\": true") != std::string::npos);
  EXPECT_TRUE(json.find("\"schedule_optimized\": false") != std::string::npos);
  EXPECT_TRUE(
      json.find("\"communication_model\": \"constant_inter_worker_delay\"") !=
      std::string::npos);
  EXPECT_TRUE(json.find("\"communication_alpha\": null") != std::string::npos);
  EXPECT_TRUE(json.find("\"activation_analysis_version\": 1") !=
              std::string::npos);
  EXPECT_TRUE(json.find("\"activation_model\": \"linear_in_stage_layers\"") !=
              std::string::npos);
  EXPECT_TRUE(json.find("\"activation_memory_metrics\": {") !=
              std::string::npos);
  EXPECT_TRUE(json.find("\"activation_cap_enforced_in_solver\": false") !=
              std::string::npos);
}

TEST(ResultSchema, NotRunFallbackIsDistinctFromUnknown) {
  const slackpipe::Instance instance = BaseInstance();
  const std::vector<slackpipe::Tick> split = slackpipe::UniformSplit(instance);
  slackpipe::JointOptimizerOptions options;
  options.require_optimal = false;
  const slackpipe::JointOptimizationResult result =
      slackpipe::BuildScheduleOnlyFixedSplitDeadlineFallback(
          instance, split, options, 0.001,
          "global_deadline_expired_before_cp_sat");
  EXPECT_EQ(result.joint_status, std::string("NOT_RUN"));
  EXPECT_EQ(result.cp_sat_models_solved, 0);
  EXPECT_TRUE(result.fallback_used);

  const slackpipe::CanonicalResultMetadata metadata =
      slackpipe::BuildCanonicalResultMetadata(
          instance, slackpipe::CanonicalRequestContext{},
          slackpipe::SemanticsForScheduleOnlyFixedSplit(
              "partition_only_within_global_budget", result, false),
          slackpipe::OutcomeFromJointResult(result), result.split,
          result.machine_orders);
  const std::string json = slackpipe::CanonicalResultToJson(metadata, "  ");
  EXPECT_TRUE(json.find("\"solver_status_raw\": \"NOT_RUN\"") !=
              std::string::npos);
  EXPECT_TRUE(json.find("\"reported_status\": \"FEASIBLE\"") !=
              std::string::npos);
  EXPECT_TRUE(json.find("\"fallback_reason\": "
                        "\"global_deadline_expired_before_cp_sat\"") !=
              std::string::npos);
  EXPECT_TRUE(json.find("\"cp_sat_models_solved\": 0") != std::string::npos);
}

TEST(ResultSchema, ScheduleOnlySerializesFixedPartitionOptimizedSchedule) {
  const slackpipe::Instance instance = BaseInstance();
  const std::vector<slackpipe::Tick> split{3, 3, 3};
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);
  slackpipe::EvaluationResult evaluated =
      slackpipe::EvaluateSchedule(instance, split, orders);
  ASSERT_TRUE(evaluated.schedule.ok());

  slackpipe::JointOptimizationResult result;
  result.split = split;
  result.makespan_ticks = evaluated.schedule.makespan;
  result.best_bound_ticks = evaluated.schedule.makespan;
  result.status = "OPTIMAL";
  result.joint_status = "OPTIMAL";
  result.proven_optimal = true;
  result.schedule = evaluated.schedule;
  result.machine_orders = orders;
  result.machine_predecessors =
      slackpipe::ExtractMachinePredecessors(instance, orders);
  result.solution_source = "joint_cpsat";
  result.horizon_source = "schedule-only-hint";

  const slackpipe::CanonicalResultMetadata metadata =
      slackpipe::BuildCanonicalResultMetadata(
          instance, slackpipe::CanonicalRequestContext{},
          slackpipe::SemanticsForScheduleOnlyFixedSplit("uniform", result,
                                                        false),
          slackpipe::OutcomeFromJointResult(result), result.split,
          result.machine_orders);
  const std::string json = slackpipe::CanonicalResultToJson(metadata, "  ");
  EXPECT_TRUE(json.find("\"canonical_method\": \"schedule-only-uniform\"") !=
              std::string::npos);
  EXPECT_TRUE(json.find("\"partition_decision\": \"fixed_uniform\"") !=
              std::string::npos);
  EXPECT_TRUE(json.find("\"schedule_decision\": \"optimized_no_overlap\"") !=
              std::string::npos);
  EXPECT_TRUE(json.find("\"full_partition_fixed\": true") != std::string::npos);
  EXPECT_TRUE(json.find("\"partition_optimized\": false") != std::string::npos);
  EXPECT_TRUE(json.find("\"schedule_optimized\": true") != std::string::npos);
}

TEST(ResultSchema, JointNoOverlapFallbackPreservesRawStatus) {
  const slackpipe::Instance instance = BaseInstance();
  const std::vector<slackpipe::Tick> split = slackpipe::UniformSplit(instance);
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);
  slackpipe::EvaluationResult evaluated =
      slackpipe::EvaluateSchedule(instance, split, orders);
  ASSERT_TRUE(evaluated.schedule.ok());

  slackpipe::JointOptimizationResult result;
  result.split = split;
  result.makespan_ticks = evaluated.schedule.makespan;
  result.best_bound_ticks = evaluated.schedule.makespan - 1;
  result.status = "FEASIBLE";
  result.joint_status = "UNKNOWN";
  result.proven_optimal = false;
  result.schedule = evaluated.schedule;
  result.machine_orders = orders;
  result.solution_source = "bfs_incumbent_fallback";
  result.fallback_used = true;
  result.diagnostic = "returned BFS incumbent after CP-SAT status=UNKNOWN";
  result.horizon_source = "hybrid_slack_incumbent";

  const slackpipe::CanonicalResultMetadata metadata =
      slackpipe::BuildCanonicalResultMetadata(
          instance, slackpipe::CanonicalRequestContext{},
          slackpipe::SemanticsForJointUnrestrictedNoOverlap(result, true),
          slackpipe::OutcomeFromJointResult(result), result.split,
          result.machine_orders);
  const std::string json = slackpipe::CanonicalResultToJson(metadata, "  ");
  EXPECT_TRUE(
      json.find("\"canonical_method\": \"joint-unrestricted-no-overlap\"") !=
      std::string::npos);
  EXPECT_TRUE(json.find("\"schedule_decision\": \"optimized_no_overlap\"") !=
              std::string::npos);
  EXPECT_TRUE(
      json.find("\"predecessor_candidate_restriction_requested\": true") !=
      std::string::npos);
  EXPECT_TRUE(
      json.find("\"predecessor_candidate_restriction_active\": false") !=
      std::string::npos);
  EXPECT_TRUE(
      json.find(
          "\"predecessor_candidate_rule\": \"unrestricted_no_overlap\"") !=
      std::string::npos);
  EXPECT_TRUE(json.find("\"solver_status_raw\": \"UNKNOWN\"") !=
              std::string::npos);
  EXPECT_TRUE(json.find("\"reported_status\": \"FEASIBLE\"") !=
              std::string::npos);
  EXPECT_TRUE(json.find("\"fallback_used\": true") != std::string::npos);
  EXPECT_TRUE(json.find("\"returned_solution_source\": \"incumbent\"") !=
              std::string::npos);
  EXPECT_TRUE(json.find("\"best_objective_bound\": ") != std::string::npos);
}

TEST(ResultSchema, WorkerFixedMeansWorkerAggregateFixed) {
  const slackpipe::Instance instance = BaseInstance();
  const std::vector<slackpipe::Tick> split = slackpipe::UniformSplit(instance);
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);
  slackpipe::EvaluationResult evaluated =
      slackpipe::EvaluateSchedule(instance, split, orders);
  ASSERT_TRUE(evaluated.schedule.ok());

  slackpipe::SlackPipeResult result;
  result.split = split;
  result.makespan_ticks = evaluated.schedule.makespan;
  result.best_bound_ticks = evaluated.schedule.makespan;
  result.status = "FEASIBLE";
  result.joint_status = "FEASIBLE";
  result.schedule = evaluated.schedule;
  result.machine_orders = orders;
  result.initial_split_method = "hybrid-slack";
  result.joint_solution_source = "joint_cpsat";

  const slackpipe::CanonicalResultMetadata metadata =
      slackpipe::BuildCanonicalResultMetadata(
          instance, slackpipe::CanonicalRequestContext{},
          slackpipe::SemanticsForSlackPipe(
              slackpipe::SlackPipeSplitMode::kWorkerFixed, result, false),
          slackpipe::OutcomeFromSlackPipeResult(result), result.split,
          result.machine_orders);
  EXPECT_EQ(metadata.canonical_method,
            std::string("canonical-slackpipe-worker-aggregate-fixed"));
  EXPECT_EQ(metadata.semantics.partition_decision,
            std::string("optimized_worker_aggregate_fixed"));
  EXPECT_FALSE(metadata.semantics.full_partition_fixed);
  EXPECT_TRUE(metadata.semantics.worker_aggregate_loads_fixed);
  EXPECT_TRUE(metadata.semantics.partition_optimized);
}

TEST(ResultSchema, DerivedPredecessorsMatchWorkerOrder) {
  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 2;
  instance.workers = 1;
  instance.total_layers = 4;
  instance.min_layers = 1;
  const std::vector<slackpipe::Tick> split{2, 2};
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);
  slackpipe::EvaluationResult evaluated =
      slackpipe::EvaluateSchedule(instance, split, orders);
  ASSERT_TRUE(evaluated.schedule.ok());

  const slackpipe::CanonicalResultMetadata metadata =
      slackpipe::BuildCanonicalResultMetadata(
          instance, slackpipe::CanonicalRequestContext{},
          slackpipe::SemanticsForBfsEvaluate(),
          slackpipe::OutcomeFromSchedule(evaluated.schedule, "FEASIBLE"), split,
          orders);
  const slackpipe::MachinePredecessors expected =
      slackpipe::ExtractMachinePredecessors(instance, orders);
  ASSERT_EQ(metadata.derived_worker_predecessors.size(), expected.size());
  for (const auto &edge : metadata.derived_worker_predecessors) {
    const auto it = expected.find(edge.operation_id);
    ASSERT_TRUE(it != expected.end());
    EXPECT_EQ(edge.predecessor_id, it->second.value);
    EXPECT_EQ(edge.operation,
              slackpipe::OperationName(
                  instance, slackpipe::OperationId{edge.operation_id}));
    EXPECT_EQ(edge.predecessor,
              slackpipe::OperationName(
                  instance, slackpipe::OperationId{edge.predecessor_id}));
  }
}

TEST(Evaluator, AppliesFifoPerOperationPosition) {
  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 2;
  instance.workers = 2;
  instance.total_layers = 4;
  instance.min_layers = 1;

  const slackpipe::MachinePredecessors empty_pred_n;
  const slackpipe::EvaluationResult result =
      slackpipe::EvaluateScheduleWithPredecessors(
          instance, std::vector<slackpipe::Tick>{2, 2}, empty_pred_n);
  ASSERT_TRUE(result.schedule.ok());

  for (slackpipe::Index n = 0; n < 2 * instance.stages; ++n) {
    const slackpipe::OperationId prev =
        slackpipe::EncodeOperation(instance, 0, n);
    const slackpipe::OperationId next =
        slackpipe::EncodeOperation(instance, 1, n);
    const auto &prev_op = result.schedule.operations_by_id[prev.value];
    const auto &next_op = result.schedule.operations_by_id[next.value];
    EXPECT_TRUE(next_op.start >= prev_op.end);
  }

  const slackpipe::OperationId b0_f1 =
      slackpipe::EncodeOperation(instance, 0, 1);
  const slackpipe::OperationId b1_f0 =
      slackpipe::EncodeOperation(instance, 1, 0);
  const auto &op_b0_f1 = result.schedule.operations_by_id[b0_f1.value];
  const auto &op_b1_f0 = result.schedule.operations_by_id[b1_f0.value];
  EXPECT_TRUE(op_b1_f0.start < op_b0_f1.end);
}

TEST(Evaluator, FifoOffAllowsSamePositionWorkerSerialization) {
  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 1;
  instance.workers = 1;
  instance.total_layers = 1;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;

  const std::vector<slackpipe::Tick> split{1};
  slackpipe::MachineOrders orders(1);
  orders[0] = {
      slackpipe::EncodeOperation(instance, 1, 0),
      slackpipe::EncodeOperation(instance, 0, 0),
      slackpipe::EncodeOperation(instance, 0, 1),
      slackpipe::EncodeOperation(instance, 1, 1),
  };

  const slackpipe::MachinePredecessors fifo_on_predecessors =
      slackpipe::ExtractMachinePredecessors(instance, orders);
  const slackpipe::MachinePredecessors fifo_off_predecessors =
      slackpipe::ExtractMachinePredecessors(instance, orders, false);
  EXPECT_TRUE(fifo_off_predecessors.size() > fifo_on_predecessors.size());

  const slackpipe::EvaluationResult fifo_off =
      slackpipe::EvaluateSchedule(instance, split, orders, false);
  ASSERT_TRUE(fifo_off.schedule.ok());
  const slackpipe::ResultValidationResult fifo_off_validation =
      slackpipe::ValidateScheduleSolutionIndependent(
          instance, fifo_off.schedule, "FEASIBLE", "fifo-off-test", false);
  EXPECT_TRUE(fifo_off_validation.passed);
  EXPECT_EQ(fifo_off_validation.fifo_edge_count, 0);
  EXPECT_TRUE(fifo_off_validation.worker_edge_count > 0);

  const slackpipe::ResultValidationResult fifo_on_validation =
      slackpipe::ValidateScheduleSolutionIndependent(
          instance, fifo_off.schedule, "FEASIBLE", "fifo-on-test");
  EXPECT_FALSE(fifo_on_validation.passed);
  EXPECT_EQ(fifo_on_validation.error_code, "worker_order_reverses_fifo");
}

TEST(Evaluator, MachinePredecessorsAreSameWorkerDifferentPosition) {
  const slackpipe::Instance instance = BaseInstance();
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);
  const slackpipe::MachinePredecessors predecessors =
      slackpipe::ExtractMachinePredecessors(instance, orders);

  for (const auto &[id_value, predecessor] : predecessors) {
    const slackpipe::OperationView current =
        slackpipe::DecodeOperation(instance, slackpipe::OperationId{id_value});
    const slackpipe::OperationView pred =
        slackpipe::DecodeOperation(instance, predecessor);
    EXPECT_EQ(pred.worker, current.worker);
    EXPECT_NE(pred.chain_index, current.chain_index);
    EXPECT_FALSE(slackpipe::IsDataPredecessor(
        instance, predecessor, slackpipe::OperationId{id_value}));
  }
}

TEST(Evaluator, ComputesPrimaryAblationMetrics) {
  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 2;
  instance.workers = 2;
  instance.total_layers = 4;
  instance.min_layers = 1;
  instance.backward_ratio_num = 1;
  instance.backward_ratio_den = 1;
  instance.communication_ticks = 3;

  const std::vector<slackpipe::Tick> split{2, 2};
  const slackpipe::EvaluationResult evaluated = slackpipe::EvaluateSchedule(
      instance, split, slackpipe::BreadthFirstOrders(instance));
  ASSERT_TRUE(evaluated.schedule.ok());
  EXPECT_EQ(evaluated.schedule.makespan, 28);

  const slackpipe::ScheduleMetrics metrics =
      slackpipe::ComputeScheduleMetrics(instance, evaluated.schedule);
  EXPECT_EQ(metrics.simulated_iteration_time, 28);
  EXPECT_EQ(metrics.total_useful_work, 32);
  ASSERT_EQ(metrics.per_worker_busy_time.size(), std::size_t{2});
  ASSERT_EQ(metrics.per_worker_idle_time.size(), std::size_t{2});
  EXPECT_EQ(metrics.per_worker_busy_time[0], 16);
  EXPECT_EQ(metrics.per_worker_busy_time[1], 16);
  EXPECT_EQ(metrics.per_worker_idle_time[0], 12);
  EXPECT_EQ(metrics.per_worker_idle_time[1], 12);
  EXPECT_EQ(metrics.max_worker_load, 16);
  EXPECT_EQ(metrics.pipeline_utilization, 32.0 / (2.0 * 28.0));
  ASSERT_TRUE(metrics.communication_blocked_time.has_value());
  EXPECT_TRUE(*metrics.communication_blocked_time > 0);
}

TEST(SlackPipeSplit, LoadBalancedSplitBalancesWorkerAggregates) {
  slackpipe::Instance instance;
  instance.microbatches = 1;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 10;
  instance.min_layers = 1;

  const std::vector<slackpipe::Tick> split =
      slackpipe::LoadBalancedSplit(instance);
  slackpipe::ValidateSplit(instance, split);
  const std::vector<slackpipe::Tick> loads =
      slackpipe::WorkerLayerTotals(instance, split);
  ASSERT_EQ(loads.size(), std::size_t{2});
  EXPECT_EQ(loads[0], 5);
  EXPECT_EQ(loads[1], 5);
}

TEST(PrimaryAblationModes, PartitionOnlyKeepsCanonicalOrder) {
  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 2;
  instance.workers = 2;
  instance.total_layers = 4;
  instance.min_layers = 1;
  instance.backward_ratio_num = 1;
  instance.backward_ratio_den = 1;
  instance.communication_ticks = 3;

  const slackpipe::BfsSplitOptimizationResult result =
      slackpipe::OptimizeBfsSplitEnumerate(instance);
  ASSERT_TRUE(result.proven_optimal);
  EXPECT_EQ(result.makespan_ticks, 28);
  EXPECT_TRUE(result.machine_orders == slackpipe::BreadthFirstOrders(instance));
}

TEST(PrimaryAblationModes, FixedOrderPartitionPreservesNonBfsOrder) {
  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 1;
  instance.workers = 1;
  instance.total_layers = 3;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;

  slackpipe::MachineOrders orders(1);
  orders[0] = {slackpipe::EncodeOperation(instance, 0, 0),
               slackpipe::EncodeOperation(instance, 1, 0),
               slackpipe::EncodeOperation(instance, 0, 1),
               slackpipe::EncodeOperation(instance, 1, 1)};
  ASSERT_TRUE(orders != slackpipe::BreadthFirstOrders(instance));

  slackpipe::BfsSplitOptimizerOptions options;
  options.require_optimal = false;
  const slackpipe::BfsSplitOptimizationResult result =
      slackpipe::OptimizePartitionForFixedOrderEnumerate(instance, orders,
                                                         options);
  EXPECT_EQ(result.status, std::string("OPTIMAL"));
  EXPECT_TRUE(result.machine_orders == orders);
  EXPECT_TRUE(result.schedule.ok());
  const slackpipe::ResultValidationResult validation =
      slackpipe::ValidateScheduleSolutionIndependent(
          instance, result.schedule, result.status,
          "partition-only-fixed-order");
  EXPECT_TRUE(validation.passed);
}

TEST(PrimaryAblationModes, ScheduleOnlyKeepsFixedPartition) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 2;
  instance.workers = 2;
  instance.total_layers = 4;
  instance.min_layers = 1;
  instance.backward_ratio_num = 1;
  instance.backward_ratio_den = 1;
  instance.communication_ticks = 3;

  const std::vector<slackpipe::Tick> fixed_split{2, 2};
  slackpipe::JointOptimizerOptions options;
  options.time_limit_seconds = 10.0;
  options.num_workers = 1;
  options.random_seed = 1;
  options.require_optimal = true;

  const slackpipe::JointOptimizationResult result =
      slackpipe::OptimizeScheduleForFixedSplitCpSat(instance, fixed_split,
                                                    options);
  ASSERT_TRUE(result.proven_optimal);
  EXPECT_TRUE(result.split == fixed_split);
  EXPECT_EQ(result.makespan_ticks, 26);
  EXPECT_TRUE(result.schedule.ok());
}

TEST(PrimaryAblationModes,
     ExhaustiveTinyInstanceShowsOneDimensionalOptimizationIsInsufficient) {
  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 2;
  instance.workers = 2;
  instance.total_layers = 4;
  instance.min_layers = 1;
  instance.backward_ratio_num = 1;
  instance.backward_ratio_den = 1;
  instance.communication_ticks = 3;

  std::vector<std::vector<slackpipe::Tick>> splits;
  slackpipe::EnumerateValidSplits(
      instance, [&](const std::vector<slackpipe::Tick> &split) {
        splits.push_back(split);
      });
  ASSERT_EQ(splits.size(), std::size_t{3});
  const std::vector<slackpipe::MachineOrders> orders =
      ExhaustiveMachineOrders(instance);
  ASSERT_EQ(orders.size(), std::size_t{980});

  const slackpipe::MachineOrders canonical =
      slackpipe::BreadthFirstOrders(instance);
  const std::vector<slackpipe::Tick> fixed_split =
      slackpipe::UniformSplit(instance);
  slackpipe::Tick partition_only_best =
      std::numeric_limits<slackpipe::Tick>::max();
  slackpipe::Tick schedule_only_best =
      std::numeric_limits<slackpipe::Tick>::max();
  slackpipe::Tick joint_best = std::numeric_limits<slackpipe::Tick>::max();
  int legal_schedule_only_states = 0;
  int legal_joint_states = 0;

  for (const std::vector<slackpipe::Tick> &split : splits) {
    const slackpipe::EvaluationResult evaluated =
        slackpipe::EvaluateSchedule(instance, split, canonical);
    ASSERT_TRUE(evaluated.schedule.ok());
    partition_only_best =
        std::min(partition_only_best, evaluated.schedule.makespan);
  }

  for (const slackpipe::MachineOrders &order : orders) {
    const slackpipe::EvaluationResult evaluated =
        slackpipe::EvaluateSchedule(instance, fixed_split, order);
    if (evaluated.schedule.ok()) {
      ++legal_schedule_only_states;
      schedule_only_best =
          std::min(schedule_only_best, evaluated.schedule.makespan);
    }
  }

  for (const std::vector<slackpipe::Tick> &split : splits) {
    for (const slackpipe::MachineOrders &order : orders) {
      const slackpipe::EvaluationResult evaluated =
          slackpipe::EvaluateSchedule(instance, split, order);
      if (evaluated.schedule.ok()) {
        ++legal_joint_states;
        joint_best = std::min(joint_best, evaluated.schedule.makespan);
      }
    }
  }

  EXPECT_EQ(legal_schedule_only_states, 84);
  EXPECT_EQ(legal_joint_states, 252);
  EXPECT_EQ(partition_only_best, 28);
  EXPECT_EQ(schedule_only_best, 26);
  EXPECT_EQ(joint_best, 24);
  EXPECT_TRUE(joint_best < partition_only_best);
  EXPECT_TRUE(joint_best < schedule_only_best);
}

TEST(Evaluator, RejectsCombinedPredNCycle) {
  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 2;
  instance.workers = 1;
  instance.total_layers = 4;
  instance.min_layers = 1;

  const std::vector<slackpipe::Tick> split{2, 2};
  const slackpipe::OperationId b1_f0 =
      slackpipe::EncodeOperation(instance, 1, 0);
  const slackpipe::OperationId b0_f1 =
      slackpipe::EncodeOperation(instance, 0, 1);

  slackpipe::MachinePredecessors first;
  first.emplace(b0_f1.value, b1_f0);
  EXPECT_TRUE(
      slackpipe::EvaluateScheduleWithPredecessors(instance, split, first)
          .schedule.ok());

  slackpipe::MachinePredecessors second;
  second.emplace(b1_f0.value, b0_f1);
  EXPECT_TRUE(
      slackpipe::EvaluateScheduleWithPredecessors(instance, split, second)
          .schedule.ok());

  slackpipe::MachinePredecessors combined;
  combined.emplace(b0_f1.value, b1_f0);
  combined.emplace(b1_f0.value, b0_f1);
  const slackpipe::EvaluationResult result =
      slackpipe::EvaluateScheduleWithPredecessors(instance, split, combined);
  EXPECT_FALSE(result.schedule.ok());
}

TEST(PredNCandidates, PrunesToNeighboringOperationPositions) {
  const slackpipe::Instance instance = SanityInstance();
  const slackpipe::OperationId target =
      slackpipe::EncodeOperation(instance, 5, 4);
  const std::vector<slackpipe::OperationId> candidates =
      slackpipe::PredNCandidates(instance, target);

  std::set<slackpipe::Index> candidate_microbatches;
  for (slackpipe::OperationId candidate : candidates) {
    const slackpipe::OperationView view =
        slackpipe::DecodeOperation(instance, candidate);
    EXPECT_EQ(view.chain_index, 3);
    candidate_microbatches.insert(view.microbatch);
  }

  const std::vector<slackpipe::Index> actual(candidate_microbatches.begin(),
                                             candidate_microbatches.end());
  const std::vector<slackpipe::Index> expected{0, 1, 2, 3, 4, 6, 7};
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual[i], expected[i]);
  }
}

TEST(BfsSplitOptimizer, EnumeratesSplitsWithCurrentEvaluator) {
  slackpipe::Instance instance = BaseInstance();
  slackpipe::BfsSplitOptimizerOptions options;
  slackpipe::SearchStats stats;
  options.search_stats = &stats;
  const slackpipe::BfsSplitOptimizationResult result =
      slackpipe::OptimizeBfsSplitEnumerate(instance, options);
  EXPECT_EQ(result.status, "OPTIMAL");
  EXPECT_TRUE(result.proven_optimal);
  EXPECT_TRUE(result.schedule.ok());
  EXPECT_TRUE(stats.enumerative_search);
  EXPECT_TRUE(stats.stage_partitions_enumerated);
  const std::int64_t checked_splits =
      static_cast<std::int64_t>(result.checked_splits);
  EXPECT_EQ(stats.stage_partitions_theoretical, checked_splits);
  EXPECT_EQ(stats.stage_partitions_visited, checked_splits);
  EXPECT_EQ(stats.candidate_schedules_deterministically_evaluated,
            checked_splits);
  EXPECT_EQ(stats.candidate_schedules_accepted, checked_splits);
  EXPECT_FALSE(stats.interleave_orders_enumerated);
}

TEST(JointCpSat, ExtractedOrdersReplayInDeterministicEvaluator) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;

  slackpipe::JointOptimizerOptions options;
  options.time_limit_seconds = 10.0;
  options.num_workers = 1;
  options.random_seed = 1;
  options.require_optimal = false;
  options.log_search_progress = false;

  const slackpipe::JointOptimizationResult result =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(instance, options);
  ASSERT_TRUE(result.status == "OPTIMAL" || result.status == "FEASIBLE");
  ASSERT_TRUE(!result.split.empty());
  ASSERT_EQ(result.machine_orders.size(),
            static_cast<std::size_t>(instance.workers));

  const slackpipe::EvaluationResult replay = slackpipe::EvaluateSchedule(
      instance, result.split, result.machine_orders);
  std::string validation_errors;
  for (const std::string &error : replay.schedule.validation_errors) {
    validation_errors += error;
    validation_errors += '\n';
  }
  (void)validation_errors;
  ASSERT_TRUE(replay.schedule.ok());
  EXPECT_EQ(replay.schedule.makespan, result.makespan_ticks);

  const std::size_t missing_position = std::numeric_limits<std::size_t>::max();
  std::vector<int> seen(static_cast<std::size_t>(instance.OperationCount()), 0);
  std::vector<std::size_t> order_position(
      static_cast<std::size_t>(instance.OperationCount()), missing_position);
  for (slackpipe::Index worker = 0; worker < instance.workers; ++worker) {
    const auto &order = result.machine_orders[static_cast<std::size_t>(worker)];
    for (std::size_t position = 0; position < order.size(); ++position) {
      const slackpipe::OperationId id = order[position];
      ASSERT_TRUE(id.value >= 0);
      ASSERT_TRUE(id.value < instance.OperationCount());
      const slackpipe::OperationView view =
          slackpipe::DecodeOperation(instance, id);
      EXPECT_EQ(view.worker, worker);
      ++seen[static_cast<std::size_t>(id.value)];
      order_position[static_cast<std::size_t>(id.value)] = position;
    }
  }
  for (slackpipe::Index id = 0; id < instance.OperationCount(); ++id) {
    EXPECT_EQ(seen[static_cast<std::size_t>(id)], 1);
    EXPECT_NE(order_position[static_cast<std::size_t>(id)], missing_position);
  }

  for (slackpipe::Index b = 0; b + 1 < instance.microbatches; ++b) {
    for (slackpipe::Index n = 0;
         n < slackpipe::OperationPositionCount(instance); ++n) {
      const slackpipe::OperationId prev =
          slackpipe::EncodeOperation(instance, b, n);
      const slackpipe::OperationId next =
          slackpipe::EncodeOperation(instance, b + 1, n);
      const slackpipe::OperationView prev_view =
          slackpipe::DecodeOperation(instance, prev);
      const slackpipe::OperationView next_view =
          slackpipe::DecodeOperation(instance, next);
      ASSERT_EQ(prev_view.worker, next_view.worker);
      EXPECT_TRUE(order_position[static_cast<std::size_t>(prev.value)] <
                  order_position[static_cast<std::size_t>(next.value)]);
    }
  }
}

TEST(JointCpSat, RangeCostProfileDurationsReplayInDeterministicEvaluator) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;
  instance.profile_prefix_forward_ticks = {0,   10,  100, 110, 200,
                                           210, 300, 310, 400};
  instance.profile_prefix_backward_ticks = {0,   20,  200, 220, 400,
                                            420, 600, 620, 800};
  instance.profile_role_forward_bias_ticks = {1, 2, 3};
  instance.profile_role_backward_bias_ticks = {4, 5, 6};
  instance.cost_profile_schema_version = "slackpipe.cost_profile.v2";

  slackpipe::JointOptimizerOptions options;
  options.time_limit_seconds = 10.0;
  options.num_workers = 1;
  options.random_seed = 1;
  options.require_optimal = false;
  options.log_search_progress = false;

  const slackpipe::JointOptimizationResult result =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(instance, options);
  ASSERT_TRUE(result.status == "OPTIMAL" || result.status == "FEASIBLE");
  ASSERT_TRUE(!result.split.empty());
  slackpipe::ValidateSplit(instance, result.split);

  const slackpipe::EvaluationResult replay = slackpipe::EvaluateSchedule(
      instance, result.split, result.machine_orders);
  ASSERT_TRUE(replay.schedule.ok());
  EXPECT_EQ(replay.schedule.makespan, result.makespan_ticks);

  const std::vector<slackpipe::Tick> split_a{1, 3, 1, 3};
  const std::vector<slackpipe::Tick> split_b{3, 1, 3, 1};
  ASSERT_EQ(split_a[1], split_b[2]);
  EXPECT_NE(instance.Duration(1, false, split_a),
            instance.Duration(2, false, split_b));
}

TEST(SlackPipeSplitMode, FixedModeKeepsBfsReferenceSplit) {
  if (!slackpipe::IsSlackPipeSolverAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 16;
  instance.min_layers = 1;

  slackpipe::SlackPipeOptions options;
  options.split_mode = slackpipe::SlackPipeSplitMode::kFixed;
  options.time_limit_seconds = 5.0;
  options.num_workers = 1;
  options.random_seed = 1;
  options.require_optimal = false;

  const slackpipe::SlackPipeResult result =
      slackpipe::SolveCanonicalSlackPipe(instance, options);
  ASSERT_TRUE(result.status == "OPTIMAL" || result.status == "FEASIBLE");
  EXPECT_TRUE(result.split == result.bfs.split);
  EXPECT_TRUE(result.mode_validation_passed);
  EXPECT_EQ(result.effective_split_mode, std::string("fixed"));
}

TEST(SlackPipeSplitMode, FixedModeKeepsKnownProbeReferenceSplit) {
  if (!slackpipe::IsSlackPipeSolverAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 8;
  instance.stages = 8;
  instance.workers = 4;
  instance.total_layers = 64;
  instance.min_layers = 1;

  slackpipe::SlackPipeOptions options;
  options.split_mode = slackpipe::SlackPipeSplitMode::kFixed;
  options.time_limit_seconds = 10.0;
  options.num_workers = 1;
  options.random_seed = 1;
  options.require_optimal = false;

  const slackpipe::SlackPipeResult result =
      slackpipe::SolveCanonicalSlackPipe(instance, options);
  ASSERT_TRUE(Solved(result.status));
  EXPECT_TRUE(result.split == result.bfs.split);
  EXPECT_TRUE(result.mode_validation_passed);
}

TEST(SlackPipeSplitMode, WorkerLocalZeroBudgetPreservesWorkerLoads) {
  if (!slackpipe::IsSlackPipeSolverAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 16;
  instance.min_layers = 1;

  slackpipe::SlackPipeOptions options;
  options.split_mode = slackpipe::SlackPipeSplitMode::kWorkerLocal;
  options.worker_move_budget = 0;
  options.worker_move_budget_provided = true;
  options.time_limit_seconds = 10.0;
  options.num_workers = 1;
  options.random_seed = 1;
  options.require_optimal = false;

  const slackpipe::SlackPipeResult result =
      slackpipe::SolveCanonicalSlackPipe(instance, options);
  ASSERT_TRUE(Solved(result.status));
  EXPECT_TRUE(result.mode_validation_passed);
  EXPECT_TRUE(result.final_worker_layers == result.baseline_worker_layers);
  EXPECT_EQ(result.worker_balance_l1, 0);
}

TEST(SlackPipeSplitMode, PredicateMatchesCpSatRestrictionOnTinySplits) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 1;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;
  const std::vector<slackpipe::Tick> reference{2, 2, 2, 2};

  std::vector<slackpipe::SlackPipeOptions> cases;
  slackpipe::SlackPipeOptions fixed;
  fixed.split_mode = slackpipe::SlackPipeSplitMode::kFixed;
  cases.push_back(fixed);

  slackpipe::SlackPipeOptions global;
  global.split_mode = slackpipe::SlackPipeSplitMode::kGlobal;
  cases.push_back(global);

  slackpipe::SlackPipeOptions local_zero;
  local_zero.split_mode = slackpipe::SlackPipeSplitMode::kLocal;
  local_zero.move_budget = 0;
  local_zero.move_budget_provided = true;
  cases.push_back(local_zero);

  slackpipe::SlackPipeOptions local_budget;
  local_budget.split_mode = slackpipe::SlackPipeSplitMode::kLocal;
  local_budget.move_budget = 1;
  local_budget.move_budget_provided = true;
  local_budget.per_stage_delta = 1;
  cases.push_back(local_budget);

  slackpipe::SlackPipeOptions worker_fixed;
  worker_fixed.split_mode = slackpipe::SlackPipeSplitMode::kWorkerFixed;
  cases.push_back(worker_fixed);

  slackpipe::SlackPipeOptions worker_zero;
  worker_zero.split_mode = slackpipe::SlackPipeSplitMode::kWorkerLocal;
  worker_zero.worker_move_budget = 0;
  worker_zero.worker_move_budget_provided = true;
  cases.push_back(worker_zero);

  slackpipe::SlackPipeOptions worker_budget;
  worker_budget.split_mode = slackpipe::SlackPipeSplitMode::kWorkerLocal;
  worker_budget.worker_move_budget = 1;
  worker_budget.worker_move_budget_provided = true;
  worker_budget.per_worker_delta = 1;
  cases.push_back(worker_budget);

  for (const slackpipe::SlackPipeOptions &options : cases) {
    const slackpipe::PartitionRestriction restriction =
        slackpipe::MakePartitionRestriction(instance, reference, options);
    slackpipe::EnumerateValidSplits(
        instance, [&](const std::vector<slackpipe::Tick> &split) {
          const bool predicate_accepts = slackpipe::SplitSatisfiesSlackPipeMode(
              instance, split, reference, options);
          const bool cpsat_accepts =
              slackpipe::CpSatPartitionRestrictionAcceptsSplitForTesting(
                  instance, split, restriction);
          EXPECT_EQ(predicate_accepts, cpsat_accepts);
        });
  }
}

TEST(SlackPipeSplitMode, WorkerLocalNeighborhoodMonotonicity) {
  if (!slackpipe::IsSlackPipeSolverAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 3;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 12;
  instance.min_layers = 1;

  slackpipe::SlackPipeOptions tight;
  tight.split_mode = slackpipe::SlackPipeSplitMode::kWorkerLocal;
  tight.worker_move_budget = 0;
  tight.worker_move_budget_provided = true;
  tight.time_limit_seconds = 10.0;
  tight.num_workers = 1;
  tight.random_seed = 1;
  tight.require_optimal = true;

  slackpipe::SlackPipeOptions relaxed = tight;
  relaxed.worker_move_budget = 2;

  const slackpipe::SlackPipeResult tight_result =
      slackpipe::SolveCanonicalSlackPipe(instance, tight);
  const slackpipe::SlackPipeResult relaxed_result =
      slackpipe::SolveCanonicalSlackPipe(instance, relaxed);

  ASSERT_TRUE(tight_result.proven_optimal);
  ASSERT_TRUE(relaxed_result.proven_optimal);
  EXPECT_TRUE(relaxed_result.makespan_ticks <= tight_result.makespan_ticks);
}

TEST(SlackPipeSplitMode, GlobalMatchesUnrestrictedJointOptimum) {
  if (!slackpipe::IsSlackPipeSolverAvailable() ||
      !slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;

  slackpipe::JointOptimizerOptions joint_options;
  joint_options.time_limit_seconds = 10.0;
  joint_options.num_workers = 1;
  joint_options.random_seed = 1;
  joint_options.require_optimal = true;

  slackpipe::SlackPipeOptions slack_options;
  slack_options.split_mode = slackpipe::SlackPipeSplitMode::kGlobal;
  slack_options.time_limit_seconds = joint_options.time_limit_seconds;
  slack_options.num_workers = joint_options.num_workers;
  slack_options.random_seed = joint_options.random_seed;
  slack_options.require_optimal = joint_options.require_optimal;

  const slackpipe::JointOptimizationResult joint =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(instance, joint_options);
  const slackpipe::SlackPipeResult slack =
      slackpipe::SolveCanonicalSlackPipe(instance, slack_options);

  ASSERT_TRUE(joint.proven_optimal);
  ASSERT_TRUE(slack.proven_optimal);
  EXPECT_EQ(slack.makespan_ticks, joint.makespan_ticks);
}

TEST(SlackPipeSplitMode, InvalidOptionsAreRejected) {
  slackpipe::Instance instance;
  instance.microbatches = 1;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;

  slackpipe::SlackPipeOptions negative;
  negative.split_mode = slackpipe::SlackPipeSplitMode::kLocal;
  negative.move_budget = -1;
  negative.move_budget_provided = true;
  EXPECT_THROW((void)slackpipe::SolveCanonicalSlackPipe(instance, negative),
               slackpipe::Error);

  slackpipe::SlackPipeOptions global_with_delta;
  global_with_delta.split_mode = slackpipe::SlackPipeSplitMode::kGlobal;
  global_with_delta.per_stage_delta = 1;
  EXPECT_THROW(
      (void)slackpipe::SolveCanonicalSlackPipe(instance, global_with_delta),
      slackpipe::Error);

  slackpipe::SlackPipeOptions fixed_with_budget;
  fixed_with_budget.split_mode = slackpipe::SlackPipeSplitMode::kFixed;
  fixed_with_budget.move_budget = 1;
  fixed_with_budget.move_budget_provided = true;
  EXPECT_THROW(
      (void)slackpipe::SolveCanonicalSlackPipe(instance, fixed_with_budget),
      slackpipe::Error);

  slackpipe::SlackPipeOptions local_with_worker_budget;
  local_with_worker_budget.split_mode = slackpipe::SlackPipeSplitMode::kLocal;
  local_with_worker_budget.worker_move_budget = 1;
  local_with_worker_budget.worker_move_budget_provided = true;
  EXPECT_THROW((void)slackpipe::SolveCanonicalSlackPipe(
                   instance, local_with_worker_budget),
               slackpipe::Error);

  slackpipe::SlackPipeOptions worker_local_with_stage_delta;
  worker_local_with_stage_delta.split_mode =
      slackpipe::SlackPipeSplitMode::kWorkerLocal;
  worker_local_with_stage_delta.per_stage_delta = 1;
  EXPECT_THROW((void)slackpipe::SolveCanonicalSlackPipe(
                   instance, worker_local_with_stage_delta),
               slackpipe::Error);
}

TEST(SlackPipeSplitMode, PressurePruningPreservesActiveRestriction) {
  if (!slackpipe::IsSlackPipeSolverAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 16;
  instance.min_layers = 1;

  slackpipe::SlackPipeOptions options;
  options.split_mode = slackpipe::SlackPipeSplitMode::kFixed;
  options.time_limit_seconds = 10.0;
  options.num_workers = 1;
  options.random_seed = 1;
  options.require_optimal = false;
  options.pressure_pruning.enabled = true;
  options.pressure_pruning.partition_top_k = 1;

  const slackpipe::SlackPipeResult result =
      slackpipe::SolveCanonicalSlackPipe(instance, options);
  ASSERT_TRUE(Solved(result.status));
  EXPECT_TRUE(result.pressure_pruning_stats.enabled);
  EXPECT_TRUE(result.split == result.bfs.split);
  EXPECT_TRUE(result.mode_validation_passed);
}

TEST(WorkerBalanceConstraint, ComputesPercentBounds) {
  slackpipe::Instance instance;
  instance.microbatches = 1;
  instance.stages = 8;
  instance.workers = 4;
  instance.total_layers = 64;
  instance.min_layers = 1;

  const slackpipe::WorkerBalanceConstraintResult disabled =
      slackpipe::ComputeWorkerBalanceConstraint(instance, -1.0);
  EXPECT_FALSE(disabled.enabled);

  const slackpipe::WorkerBalanceConstraintResult zero =
      slackpipe::ComputeWorkerBalanceConstraint(instance, 0.0);
  EXPECT_TRUE(zero.enabled);
  EXPECT_EQ(zero.tolerance_layers, 0);
  EXPECT_EQ(zero.lower_bound, 16);
  EXPECT_EQ(zero.upper_bound, 16);

  const slackpipe::WorkerBalanceConstraintResult three =
      slackpipe::ComputeWorkerBalanceConstraint(instance, 3.0);
  EXPECT_TRUE(three.enabled);
  EXPECT_EQ(three.tolerance_layers, 2);
  EXPECT_EQ(three.lower_bound, 14);
  EXPECT_EQ(three.upper_bound, 18);

  const slackpipe::WorkerBalanceConstraintResult five =
      slackpipe::ComputeWorkerBalanceConstraint(instance, 5.0);
  EXPECT_TRUE(five.enabled);
  EXPECT_EQ(five.tolerance_layers, 4);
  EXPECT_EQ(five.lower_bound, 12);
  EXPECT_EQ(five.upper_bound, 20);

  const slackpipe::WorkerBalanceConstraintResult ten =
      slackpipe::ComputeWorkerBalanceConstraint(instance, 10.0);
  EXPECT_TRUE(ten.enabled);
  EXPECT_EQ(ten.tolerance_layers, 7);
  EXPECT_EQ(ten.lower_bound, 9);
  EXPECT_EQ(ten.upper_bound, 23);
}

TEST(WorkerBalanceConstraint, JointCpSatJsonReportsEnabledConstraint) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;

  slackpipe::JointOptimizerOptions options;
  options.time_limit_seconds = 10.0;
  options.num_workers = 1;
  options.random_seed = 1;
  options.require_optimal = false;
  options.log_search_progress = false;
  options.worker_balance_pruning = true;
  options.worker_balance_tolerance_percent = 0.0;

  const slackpipe::JointOptimizationResult result =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(instance, options);
  ASSERT_TRUE(result.status == "OPTIMAL" || result.status == "FEASIBLE");
  EXPECT_TRUE(result.worker_balance_constraint.enabled);
  EXPECT_EQ(result.worker_balance_constraint.tolerance_layers, 0);
  EXPECT_EQ(result.worker_balance_constraint.lower_bound, 4);
  EXPECT_EQ(result.worker_balance_constraint.upper_bound, 4);

  const std::vector<slackpipe::Tick> loads =
      slackpipe::WorkerLayerTotals(instance, result.split);
  ASSERT_EQ(loads.size(), std::size_t{2});
  for (slackpipe::Tick load : loads) {
    EXPECT_TRUE(load >= result.worker_balance_constraint.lower_bound);
    EXPECT_TRUE(load <= result.worker_balance_constraint.upper_bound);
  }

  const std::string json = slackpipe::ToJson(instance, result);
  EXPECT_TRUE(json.find("\"worker_balance_constraint\":") != std::string::npos);
  EXPECT_TRUE(json.find("\"worker_balance_pruning_requested\": true") !=
              std::string::npos);
  EXPECT_TRUE(json.find("\"worker_balance_pruning_effective\": true") !=
              std::string::npos);
  EXPECT_TRUE(json.find("\"enabled\": true") != std::string::npos);
  EXPECT_TRUE(json.find("\"tolerance_percent\": 0") != std::string::npos);
  EXPECT_TRUE(json.find("\"tolerance_layers\": 0") != std::string::npos);
  EXPECT_TRUE(json.find("\"lower_bound\": 4") != std::string::npos);
  EXPECT_TRUE(json.find("\"upper_bound\": 4") != std::string::npos);
  EXPECT_TRUE(json.find("\"worker_aggregate_layer_loads\":") !=
              std::string::npos);
}

TEST(WorkerBalanceConstraint, JointCpSatPruningFlagGatesConstraint) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;

  slackpipe::JointOptimizerOptions off;
  off.time_limit_seconds = 1.0;
  off.num_workers = 1;
  off.random_seed = 1;
  off.require_optimal = false;
  off.worker_balance_pruning = false;

  const slackpipe::JointOptimizationResult unpruned =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(instance, off);
  ASSERT_TRUE(Solved(unpruned.status));
  EXPECT_TRUE(unpruned.worker_balance_pruning_requested == false);
  EXPECT_FALSE(unpruned.worker_balance_pruning_effective);
  EXPECT_FALSE(unpruned.worker_balance_constraint.enabled);

  slackpipe::JointOptimizerOptions missing_tolerance = off;
  missing_tolerance.worker_balance_pruning = true;
  EXPECT_THROW(
      {
        (void)slackpipe::OptimizeJointSplitAndScheduleCpSat(instance,
                                                            missing_tolerance);
      },
      slackpipe::Error);

  slackpipe::JointOptimizerOptions contradictory = off;
  contradictory.worker_balance_tolerance_layers = 0;
  EXPECT_THROW(
      {
        (void)slackpipe::OptimizeJointSplitAndScheduleCpSat(instance,
                                                            contradictory);
      },
      slackpipe::Error);
}

TEST(SearchStats, JointCpSatReportsEncodedSpacesAsNonEnumerated) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;

  slackpipe::SearchStats stats;
  slackpipe::JointOptimizerOptions options;
  options.time_limit_seconds = 10.0;
  options.num_workers = 1;
  options.random_seed = 1;
  options.require_optimal = false;
  options.log_search_progress = false;
  options.search_stats = &stats;

  const slackpipe::JointOptimizationResult result =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(instance, options);
  ASSERT_TRUE(result.status == "OPTIMAL" || result.status == "FEASIBLE");
  EXPECT_EQ(stats.algorithm, "optimize-joint");
  EXPECT_FALSE(stats.enumerative_search);
  EXPECT_TRUE(stats.stage_partitions_theoretical_available);
  EXPECT_EQ(stats.stage_partitions_theoretical, 35);
  EXPECT_FALSE(stats.stage_partitions_enumerated);
  EXPECT_EQ(stats.stage_partitions_visited, 0);
  EXPECT_EQ(stats.stage_partitions_kept, 0);
  EXPECT_FALSE(stats.interleave_orders_enumerated);
  EXPECT_EQ(stats.interleave_orders_visited, 0);
  EXPECT_EQ(stats.interleave_orders_evaluated, 0);
  EXPECT_EQ(stats.candidate_schedules_extracted, 1);
  EXPECT_EQ(stats.candidate_schedules_deterministically_evaluated, 1);
  EXPECT_EQ(stats.candidate_schedules_accepted, 1);
  EXPECT_EQ(stats.candidate_schedules_rejected, 0);
  EXPECT_TRUE(stats.cp_sat_available);

  const std::string json = slackpipe::ToJson(instance, stats);
  EXPECT_TRUE(json.find("\"algorithm\": \"optimize-joint\"") !=
              std::string::npos);
  EXPECT_TRUE(json.find("\"enumerative_search\": false") != std::string::npos);
  EXPECT_TRUE(json.find("\"stage_level_partition\":") != std::string::npos);
  EXPECT_TRUE(json.find("\"enumerated\": false") != std::string::npos);
  EXPECT_TRUE(json.find("\"visited\": null") != std::string::npos);
  EXPECT_TRUE(json.find("Stage partitions are encoded as CP-SAT variables") !=
              std::string::npos);
  EXPECT_TRUE(
      json.find("Worker-local orders are encoded in the CP-SAT model") !=
      std::string::npos);
  EXPECT_TRUE(json.find("\"candidate_schedules\":") != std::string::npos);
  EXPECT_TRUE(json.find("\"deterministically_evaluated\": 1") !=
              std::string::npos);
  EXPECT_TRUE(json.find("\"cp_sat\":") != std::string::npos);
}

TEST(HybridSlack, ScoreArithmeticMatchesFig1Examples) {
  slackpipe::Instance instance;
  instance.microbatches = 1;
  instance.stages = 8;
  instance.workers = 4;
  instance.min_layers = 1;
  instance.backward_ratio_den = 1;
  instance.backward_ratio_num = 2;

  auto expect_scores = [&](std::vector<slackpipe::Tick> split,
                           double expected_min,
                           std::vector<double> expected_forward,
                           std::vector<double> expected_backward,
                           std::vector<double> expected_stage) {
    instance.total_layers = 0;
    for (slackpipe::Tick value : split) instance.total_layers += value;
    const slackpipe::HybridSlackScores scores =
        slackpipe::ComputeHybridSlackScores(instance, split);
    EXPECT_TRUE(std::abs(scores.partition_min_slack - expected_min) < 1e-9);
    ExpectDoubleVectorNear(scores.forward_stage_scores, expected_forward);
    ExpectDoubleVectorNear(scores.backward_stage_scores, expected_backward);
    ExpectDoubleVectorNear(scores.stage_scores, expected_stage);
    for (slackpipe::Index s = 0; s < instance.stages; ++s) {
      EXPECT_TRUE(
          std::abs(scores.operation_scores[static_cast<std::size_t>(s)] -
                   expected_forward[static_cast<std::size_t>(s)]) < 1e-9);
      const slackpipe::Index backward_position = 2 * instance.stages - 1 - s;
      EXPECT_TRUE(std::abs(scores.operation_scores[static_cast<std::size_t>(
                               backward_position)] -
                           expected_backward[static_cast<std::size_t>(s)]) <
                  1e-9);
    }
  };

  expect_scores({1, 1, 1, 1, 1, 1, 1, 1}, 4.0,
                {4.0, 4.0, 4.0, 4.0, 13.0, 10.0, 7.0, 4.0},
                {4.0, 4.0, 4.0, 4.0, 8.5, 7.0, 5.5, 4.0},
                {4.0, 4.0, 4.0, 4.0, 8.5, 7.0, 5.5, 4.0});
  expect_scores({3, 5, 8, 8, 14, 11, 8, 7}, 5.125,
                {8.0, 7.0, 5.125, 5.125, 8.071428571428571, 7.818181818181818,
                 7.75, 5.714285714285714},
                {8.0, 7.0, 5.125, 5.125, 5.285714285714286, 5.7727272727272725,
                 6.4375, 5.714285714285714},
                {8.0, 7.0, 5.125, 5.125, 5.285714285714286, 5.7727272727272725,
                 6.4375, 5.714285714285714});
  expect_scores(
      {42, 84, 101, 109, 221, 172, 145, 126}, 5.26984126984127,
      {8.0, 6.130952380952381, 5.97029702970297, 5.935779816513762,
       8.343891402714933, 8.232558139534884, 7.068965517241379,
       5.26984126984127},
      {8.0, 6.130952380952381, 5.97029702970297, 5.935779816513762,
       5.33710407239819, 5.869186046511628, 5.76551724137931, 5.26984126984127},
      {8.0, 6.130952380952381, 5.97029702970297, 5.935779816513762,
       5.33710407239819, 5.869186046511628, 5.76551724137931,
       5.26984126984127});
}

TEST(HybridSlack, AlmostZeroBudgetReturnsUniformBreadthFirstIncumbent) {
  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;
  instance.backward_ratio_den = 1;
  instance.backward_ratio_num = 2;

  slackpipe::HybridSlackIncumbentOptions options;
  options.time_limit_seconds = 1e-12;
  const slackpipe::BfsSplitOptimizationResult result =
      slackpipe::BuildHybridSlackIncumbent(instance, options);
  EXPECT_EQ(result.status, std::string("FEASIBLE"));
  EXPECT_FALSE(result.proven_optimal);
  EXPECT_TRUE(result.schedule.ok());
  EXPECT_TRUE(result.split == slackpipe::UniformSplit(instance));
  EXPECT_TRUE(result.machine_orders == slackpipe::BreadthFirstOrders(instance));
  EXPECT_TRUE(result.hint_deadline_reached);
  EXPECT_EQ(result.hint_termination_reason, std::string("deadline"));
}

TEST(HybridSlack, PartitionMoveCanImproveHybridScoreAndPreserveBounds) {
  slackpipe::Instance instance;
  instance.microbatches = 1;
  instance.stages = 8;
  instance.workers = 4;
  instance.total_layers = 64;
  instance.min_layers = 1;
  instance.backward_ratio_den = 1;
  instance.backward_ratio_num = 2;
  const std::vector<slackpipe::Tick> split{3, 5, 8, 8, 14, 11, 8, 7};
  const slackpipe::HybridSlackScores scores =
      slackpipe::ComputeHybridSlackScores(instance, split);
  const std::vector<std::vector<slackpipe::Tick>> candidates =
      slackpipe::GenerateHybridSlackPartitionMoveCandidates(
          instance, split, scores, std::nullopt, 64);

  bool found_improving = false;
  for (const std::vector<slackpipe::Tick> &candidate : candidates) {
    slackpipe::ValidateSplit(instance, candidate);
    const slackpipe::HybridSlackScores candidate_scores =
        slackpipe::ComputeHybridSlackScores(instance, candidate);
    if (candidate_scores.partition_min_slack > scores.partition_min_slack) {
      found_improving = true;
      slackpipe::Tick sum = 0;
      for (slackpipe::Tick value : candidate) {
        EXPECT_TRUE(value >= instance.min_layers);
        sum += value;
      }
      EXPECT_EQ(sum, instance.total_layers);
      break;
    }
  }
  EXPECT_TRUE(found_improving);
}

TEST(HybridSlack, InterleavingMoveChangesMachineOrdersAndCanRemainFeasible) {
  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;
  instance.backward_ratio_den = 1;
  instance.backward_ratio_num = 2;
  const std::vector<slackpipe::Tick> split = slackpipe::UniformSplit(instance);
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);
  const slackpipe::HybridSlackScores scores =
      slackpipe::ComputeHybridSlackScores(instance, split);
  const std::vector<slackpipe::MachineOrders> candidates =
      slackpipe::GenerateHybridSlackInterleavingMoveCandidates(
          instance, split, orders, scores, 128);

  bool found_feasible_changed = false;
  for (const slackpipe::MachineOrders &candidate : candidates) {
    if (candidate == orders) continue;
    const slackpipe::EvaluationResult evaluated =
        slackpipe::EvaluateSchedule(instance, split, candidate);
    if (evaluated.schedule.ok()) {
      found_feasible_changed = true;
      break;
    }
  }
  EXPECT_TRUE(found_feasible_changed);
}

TEST(HybridSlack, ObjectiveProtectionRejectsWorseProxyImprovement) {
  slackpipe::HybridSlackCandidateSummary incumbent;
  incumbent.primary_objective = 10;
  incumbent.hybrid_min_slack = 4.0;
  incumbent.hybrid_stage_scores = {4.0, 5.0};
  incumbent.split = {1, 1};

  slackpipe::HybridSlackCandidateSummary candidate = incumbent;
  candidate.primary_objective = 11;
  candidate.hybrid_min_slack = 5.0;
  candidate.hybrid_stage_scores = {5.0, 6.0};
  EXPECT_FALSE(slackpipe::HybridSlackCandidateBetter(candidate, incumbent));
}

TEST(HybridSlack, DeterministicForSameSmallInstance) {
  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 12;
  instance.min_layers = 1;
  instance.backward_ratio_den = 1;
  instance.backward_ratio_num = 2;
  slackpipe::HybridSlackIncumbentOptions options;
  options.max_iterations = 16;

  const slackpipe::BfsSplitOptimizationResult first =
      slackpipe::BuildHybridSlackIncumbent(instance, options);
  const slackpipe::BfsSplitOptimizationResult second =
      slackpipe::BuildHybridSlackIncumbent(instance, options);
  EXPECT_TRUE(first.split == second.split);
  EXPECT_TRUE(first.machine_orders == second.machine_orders);
  EXPECT_EQ(first.makespan_ticks, second.makespan_ticks);
  EXPECT_TRUE(std::abs(first.hybrid_min_slack - second.hybrid_min_slack) <
              1e-9);
}

TEST(HybridSlack, RespectsFixedAndWorkerLocalRestrictions) {
  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 12;
  instance.min_layers = 1;
  instance.backward_ratio_den = 1;
  instance.backward_ratio_num = 2;
  const std::vector<slackpipe::Tick> reference{2, 4, 2, 4};

  slackpipe::PartitionRestriction fixed;
  fixed.mode = slackpipe::SlackPipeSplitMode::kFixed;
  fixed.reference_split = reference;
  slackpipe::HybridSlackIncumbentOptions fixed_options;
  fixed_options.partition_restriction = fixed;
  const slackpipe::BfsSplitOptimizationResult fixed_result =
      slackpipe::BuildHybridSlackIncumbent(instance, fixed_options);
  EXPECT_TRUE(fixed_result.split == reference);

  slackpipe::PartitionRestriction local_zero;
  local_zero.mode = slackpipe::SlackPipeSplitMode::kLocal;
  local_zero.reference_split = reference;
  local_zero.move_budget = 0;
  local_zero.per_stage_delta = 0;
  slackpipe::HybridSlackIncumbentOptions local_options;
  local_options.partition_restriction = local_zero;
  const slackpipe::BfsSplitOptimizationResult local_result =
      slackpipe::BuildHybridSlackIncumbent(instance, local_options);
  EXPECT_TRUE(local_result.split == reference);

  slackpipe::PartitionRestriction worker_local;
  worker_local.mode = slackpipe::SlackPipeSplitMode::kWorkerLocal;
  worker_local.reference_split = reference;
  worker_local.worker_move_budget = 1;
  worker_local.per_worker_delta = 1;
  slackpipe::HybridSlackIncumbentOptions worker_options;
  worker_options.partition_restriction = worker_local;
  const slackpipe::BfsSplitOptimizationResult worker_result =
      slackpipe::BuildHybridSlackIncumbent(instance, worker_options);
  EXPECT_TRUE(slackpipe::SplitSatisfiesPartitionRestriction(
      instance, worker_result.split, worker_local));
}

TEST(HybridSlack, TimedJointAutoUsesHybridAndLeavesJointBudget) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;
  instance.backward_ratio_den = 1;
  instance.backward_ratio_num = 2;

  slackpipe::JointOptimizerOptions options;
  options.time_limit_seconds = 0.2;
  options.num_workers = 1;
  options.require_optimal = false;
  const slackpipe::JointOptimizationResult result =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(instance, options);
  ASSERT_TRUE(Solved(result.status));
  EXPECT_EQ(result.incumbent_method_requested, std::string("slack"));
  EXPECT_EQ(result.incumbent_method_effective, std::string("slack"));
  EXPECT_EQ(result.bfs_incumbent_method_requested, std::string("auto"));
  EXPECT_EQ(result.bfs_incumbent_method_effective, std::string("hybrid-slack"));
  EXPECT_TRUE(result.incumbent_feasible);
  EXPECT_TRUE(result.hint_budget_seconds <= 0.02 + 1e-9);
  EXPECT_TRUE(result.hint_elapsed_seconds <= 0.02 + 0.05);
  EXPECT_TRUE(result.joint_budget_seconds <= 0.2);
  EXPECT_TRUE(result.joint_budget_seconds >= 0.17);
  EXPECT_TRUE(result.horizon_source == result.incumbent_source);
}

TEST(JointHintPolicy, DirectJointDefaultHintsIncumbentAndCounts) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;

  slackpipe::JointOptimizerOptions options;
  options.time_limit_seconds = 2.0;
  options.num_workers = 1;
  options.random_seed = 1;
  options.require_optimal = false;

  const slackpipe::JointOptimizationResult result =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(instance, options);
  ASSERT_TRUE(Solved(result.status));
  EXPECT_EQ(result.bfs_incumbent_method_requested, std::string("auto"));
  EXPECT_EQ(result.bfs_incumbent_method_effective, std::string("hybrid-slack"));
  EXPECT_EQ(result.incumbent_method_requested, std::string("slack"));
  EXPECT_EQ(result.incumbent_method_effective, std::string("slack"));
  EXPECT_EQ(result.incumbent_source, std::string("hybrid_slack_incumbent"));
  EXPECT_EQ(result.horizon_source, std::string("hybrid_slack_incumbent"));
  EXPECT_TRUE(result.incumbent_bound_requested);
  EXPECT_TRUE(result.incumbent_bound_effective);
  EXPECT_EQ(result.incumbent_bound_horizon,
            result.bfs_incumbent.makespan_ticks);
  EXPECT_TRUE(result.incumbent_feasible);
  EXPECT_TRUE(result.incumbent_found);
  EXPECT_TRUE(result.incumbent_valid);
  EXPECT_EQ(result.incumbent_makespan, result.bfs_incumbent.makespan_ticks);
  EXPECT_EQ(result.incumbent_primary_objective,
            result.bfs_incumbent.makespan_ticks);
  EXPECT_TRUE(result.incumbent_hybrid_min_slack > 0.0);
  EXPECT_EQ(result.incumbent_hybrid_stage_scores.size(),
            static_cast<std::size_t>(instance.stages));
  EXPECT_TRUE(result.hint_budget_seconds <= 0.21);
  EXPECT_TRUE(result.hint_elapsed_seconds <= result.hint_budget_seconds + 0.05);
  EXPECT_FALSE(result.hint_termination_reason.empty());
  EXPECT_TRUE(result.hints_requested);
  EXPECT_TRUE(result.hints_effective);
  EXPECT_TRUE(result.incumbent_hints_requested);
  EXPECT_TRUE(result.incumbent_hints_effective);
  EXPECT_EQ(result.hint_source, std::string("hybrid_slack_incumbent"));
  EXPECT_EQ(result.hint_scope, std::string("basic_integer_variables"));
  EXPECT_TRUE(result.hint_complete_for_basic_model);
  EXPECT_TRUE(result.hint_complete_for_full_model);
  EXPECT_EQ(result.hinted_layer_variable_count, 4);
  EXPECT_EQ(result.hinted_operation_variable_count, 3 * (4 * 2 * 4));
  EXPECT_EQ(result.hinted_scalar_variable_count, 1);
  EXPECT_EQ(result.hinted_auxiliary_variable_count, 0);
  EXPECT_EQ(result.hinted_total_variable_count, 101);
  EXPECT_TRUE(result.fallback_available);
  EXPECT_TRUE(result.fallback_enabled);
  EXPECT_EQ(result.fallback_source, std::string("hybrid_slack_incumbent"));

  const std::string json = slackpipe::ToJson(instance, result);
  EXPECT_TRUE(json.find("\"incumbent_method_effective\": \"slack\"") !=
              std::string::npos);
  EXPECT_TRUE(
      json.find("\"bfs_incumbent_method_effective\": \"hybrid-slack\"") !=
      std::string::npos);
  EXPECT_TRUE(json.find("\"incumbent_method_requested\": \"slack\"") !=
              std::string::npos);
  EXPECT_TRUE(
      json.find("\"incumbent_method_requested_normalized\": \"slack\"") !=
      std::string::npos);
  EXPECT_TRUE(json.find("\"incumbent_source\": \"hybrid_slack_incumbent\"") !=
              std::string::npos);
  EXPECT_TRUE(json.find("\"hints_requested\": true") != std::string::npos);
  EXPECT_TRUE(json.find("\"hints_effective\": true") != std::string::npos);
  EXPECT_TRUE(json.find("\"hinted_total_variable_count\": 101") !=
              std::string::npos);
  EXPECT_TRUE(json.find("\"incumbent_bound_requested\": true") !=
              std::string::npos);
  EXPECT_TRUE(json.find("\"incumbent_bound_effective\": true") !=
              std::string::npos);
  EXPECT_TRUE(json.find("\"fallback_enabled\": true") != std::string::npos);
}

TEST(JointHintPolicy, DirectJointNoHintsKeepsIncumbentAndFallback) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;

  slackpipe::JointOptimizerOptions options;
  options.time_limit_seconds = 2.0;
  options.num_workers = 1;
  options.random_seed = 1;
  options.require_optimal = false;
  options.use_bfs_hints = false;

  const slackpipe::JointOptimizationResult result =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(instance, options);
  ASSERT_TRUE(Solved(result.status));
  EXPECT_TRUE(Solved(result.incumbent_status));
  EXPECT_EQ(result.bfs_incumbent_method_effective, std::string("hybrid-slack"));
  EXPECT_EQ(result.incumbent_source, std::string("hybrid_slack_incumbent"));
  EXPECT_TRUE(result.incumbent_feasible);
  EXPECT_FALSE(result.hints_requested);
  EXPECT_FALSE(result.hints_effective);
  EXPECT_FALSE(result.incumbent_hints_requested);
  EXPECT_FALSE(result.incumbent_hints_effective);
  EXPECT_EQ(result.hint_source, std::string("none"));
  EXPECT_EQ(result.hint_scope, std::string("none"));
  EXPECT_FALSE(result.hint_complete_for_basic_model);
  EXPECT_EQ(result.hinted_total_variable_count, 0);
  EXPECT_TRUE(result.fallback_available);
  EXPECT_TRUE(result.fallback_enabled);
  EXPECT_EQ(result.fallback_source, std::string("hybrid_slack_incumbent"));
}

TEST(JointMechanismControls, ExplicitProductionMatchesLegacyDefaults) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 2;
  instance.workers = 2;
  instance.total_layers = 4;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;

  slackpipe::JointOptimizerOptions legacy;
  legacy.time_limit_seconds = 1.0;
  legacy.num_workers = 1;
  legacy.random_seed = 1;
  legacy.require_optimal = false;

  slackpipe::JointOptimizerOptions explicit_options = legacy;
  explicit_options.worker_balance_pruning = false;
  explicit_options.incumbent_method = "slack";
  explicit_options.incumbent_bound = true;
  explicit_options.incumbent_hints = true;

  const slackpipe::JointOptimizationResult old_result =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(instance, legacy);
  const slackpipe::JointOptimizationResult explicit_result =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(instance, explicit_options);

  ASSERT_TRUE(Solved(old_result.status));
  ASSERT_TRUE(Solved(explicit_result.status));
  EXPECT_EQ(old_result.incumbent_method_effective,
            explicit_result.incumbent_method_effective);
  EXPECT_EQ(old_result.bfs_incumbent_method_effective,
            explicit_result.bfs_incumbent_method_effective);
  EXPECT_EQ(old_result.incumbent_budget_seconds,
            explicit_result.incumbent_budget_seconds);
  EXPECT_EQ(old_result.incumbent_makespan, explicit_result.incumbent_makespan);
  EXPECT_EQ(old_result.horizon_source, explicit_result.horizon_source);
  EXPECT_EQ(old_result.incumbent_bound_horizon,
            explicit_result.incumbent_bound_horizon);
  EXPECT_EQ(old_result.hinted_total_variable_count,
            explicit_result.hinted_total_variable_count);
  EXPECT_FALSE(explicit_result.worker_balance_pruning_effective);
  EXPECT_EQ(old_result.makespan_ticks, explicit_result.makespan_ticks);
}

TEST(JointMechanismControls, CanonicalIncumbentUsesUniformBreadthFirstOnly) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;

  slackpipe::JointOptimizerOptions options;
  options.time_limit_seconds = 2.0;
  options.num_workers = 1;
  options.random_seed = 1;
  options.require_optimal = false;
  options.incumbent_method = "canonical";

  const slackpipe::JointOptimizationResult result =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(instance, options);
  ASSERT_TRUE(Solved(result.status));
  EXPECT_EQ(result.incumbent_method_requested, std::string("canonical"));
  EXPECT_EQ(result.incumbent_method_effective, std::string("canonical"));
  EXPECT_TRUE(result.bfs_incumbent.split == slackpipe::UniformSplit(instance));
  EXPECT_TRUE(result.bfs_incumbent.machine_orders ==
              slackpipe::BreadthFirstOrders(instance));
  EXPECT_EQ(result.hint_candidates_simulated, 0);
  EXPECT_EQ(result.hint_partition_moves_accepted, 0);
  EXPECT_EQ(result.hint_interleaving_moves_accepted, 0);
}

TEST(JointMechanismControls, NoIncumbentDisablesBoundHintsAndFallback) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 2;
  instance.workers = 2;
  instance.total_layers = 4;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;

  slackpipe::JointOptimizerOptions options;
  options.time_limit_seconds = 1.0;
  options.num_workers = 1;
  options.random_seed = 1;
  options.require_optimal = false;
  options.incumbent_method = "none";
  options.incumbent_bound = false;
  options.incumbent_hints = false;

  const slackpipe::JointOptimizationResult result =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(instance, options);
  ASSERT_TRUE(Solved(result.status));
  EXPECT_EQ(result.incumbent_method_effective, std::string("none"));
  EXPECT_EQ(result.incumbent_status, std::string("NOT_RUN"));
  EXPECT_FALSE(result.incumbent_found);
  EXPECT_FALSE(result.incumbent_valid);
  EXPECT_EQ(result.incumbent_makespan, 0);
  EXPECT_FALSE(result.incumbent_bound_effective);
  EXPECT_EQ(result.horizon_source, std::string("conservative"));
  EXPECT_FALSE(result.incumbent_hints_effective);
  EXPECT_EQ(result.hinted_total_variable_count, 0);
  EXPECT_FALSE(result.fallback_enabled);
  EXPECT_FALSE(result.fallback_available);
  EXPECT_FALSE(result.fallback_used);
  EXPECT_EQ(result.final_solution_source, std::string("cpsat"));
  EXPECT_TRUE(result.solver_solution_available);
}

TEST(JointMechanismControls, NoBoundUsesConservativeHorizonWithHints) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;

  slackpipe::JointOptimizerOptions options;
  options.time_limit_seconds = 2.0;
  options.num_workers = 1;
  options.random_seed = 1;
  options.require_optimal = false;
  options.incumbent_method = "slack";
  options.incumbent_bound = false;
  options.incumbent_hints = true;

  const slackpipe::JointOptimizationResult result =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(instance, options);
  ASSERT_TRUE(Solved(result.status));
  EXPECT_TRUE(result.incumbent_valid);
  EXPECT_FALSE(result.incumbent_bound_effective);
  EXPECT_EQ(result.incumbent_bound_horizon, 0);
  EXPECT_EQ(result.horizon_source, std::string("conservative"));
  EXPECT_TRUE(result.incumbent_hints_effective);
  EXPECT_TRUE(result.hinted_total_variable_count > 0);
}

TEST(JointMechanismControls, NoHintsEmitsZeroIncumbentHints) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;

  slackpipe::JointOptimizerOptions options;
  options.time_limit_seconds = 2.0;
  options.num_workers = 1;
  options.random_seed = 1;
  options.require_optimal = false;
  options.incumbent_method = "slack";
  options.incumbent_hints = false;

  const slackpipe::JointOptimizationResult result =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(instance, options);
  ASSERT_TRUE(Solved(result.status));
  EXPECT_TRUE(result.incumbent_valid);
  EXPECT_TRUE(result.incumbent_bound_effective);
  EXPECT_FALSE(result.incumbent_hints_requested);
  EXPECT_FALSE(result.incumbent_hints_effective);
  EXPECT_EQ(result.hinted_total_variable_count, 0);
}

TEST(JointHintPolicy, ScheduleOnlyHonorsHintToggleAndFixedSplit) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;
  const std::vector<slackpipe::Tick> fixed_split =
      slackpipe::UniformSplit(instance);

  slackpipe::JointOptimizerOptions enabled;
  enabled.time_limit_seconds = 2.0;
  enabled.num_workers = 1;
  enabled.random_seed = 1;
  enabled.require_optimal = false;
  const slackpipe::JointOptimizationResult hinted =
      slackpipe::OptimizeScheduleForFixedSplitCpSat(instance, fixed_split,
                                                    enabled);
  ASSERT_TRUE(Solved(hinted.status));
  EXPECT_TRUE(hinted.split == fixed_split);
  EXPECT_EQ(hinted.bfs_incumbent_method_requested, std::string("fixed"));
  EXPECT_EQ(hinted.incumbent_source,
            std::string("schedule_only_fixed_partition"));
  EXPECT_TRUE(hinted.hints_requested);
  EXPECT_TRUE(hinted.hints_effective);
  EXPECT_EQ(hinted.hinted_total_variable_count, 101);
  EXPECT_TRUE(hinted.fallback_available);

  slackpipe::JointOptimizerOptions disabled = enabled;
  disabled.use_bfs_hints = false;
  const slackpipe::JointOptimizationResult unhinted =
      slackpipe::OptimizeScheduleForFixedSplitCpSat(instance, fixed_split,
                                                    disabled);
  ASSERT_TRUE(Solved(unhinted.status));
  EXPECT_TRUE(unhinted.split == fixed_split);
  EXPECT_FALSE(unhinted.hints_requested);
  EXPECT_FALSE(unhinted.hints_effective);
  EXPECT_EQ(unhinted.hinted_total_variable_count, 0);
  EXPECT_TRUE(unhinted.fallback_available);
}

TEST(JointHintPolicy, SlackPipeHonorsInnerHintToggle) {
  if (!slackpipe::IsSlackPipeSolverAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;

  slackpipe::SlackPipeOptions enabled;
  enabled.time_limit_seconds = 2.0;
  enabled.num_workers = 1;
  enabled.random_seed = 1;
  enabled.require_optimal = false;
  const slackpipe::SlackPipeResult hinted =
      slackpipe::SolveCanonicalSlackPipe(instance, enabled);
  ASSERT_TRUE(Solved(hinted.status));
  EXPECT_EQ(hinted.joint_incumbent_method_effective, std::string("slack"));
  EXPECT_EQ(hinted.joint_bfs_incumbent_method_effective,
            std::string("hybrid-slack"));
  EXPECT_EQ(hinted.joint_incumbent_source,
            std::string("hybrid_slack_incumbent_override"));
  EXPECT_TRUE(hinted.joint_incumbent_feasible);
  EXPECT_TRUE(hinted.joint_hints_requested);
  EXPECT_TRUE(hinted.joint_hints_effective);
  EXPECT_EQ(hinted.joint_hint_source,
            std::string("hybrid_slack_incumbent_override"));
  EXPECT_EQ(hinted.joint_hinted_total_variable_count, 101);
  EXPECT_TRUE(hinted.joint_fallback_available);
  EXPECT_TRUE(hinted.reference_budget_seconds <= 0.21);
  EXPECT_TRUE(hinted.joint_remaining_budget_seconds <= 2.0);

  slackpipe::SlackPipeOptions disabled = enabled;
  disabled.use_bfs_hints = false;
  const slackpipe::SlackPipeResult unhinted =
      slackpipe::SolveCanonicalSlackPipe(instance, disabled);
  ASSERT_TRUE(Solved(unhinted.status));
  EXPECT_FALSE(unhinted.joint_hints_requested);
  EXPECT_FALSE(unhinted.joint_hints_effective);
  EXPECT_EQ(unhinted.joint_hinted_total_variable_count, 0);
  EXPECT_TRUE(unhinted.joint_fallback_available);

  const std::string json = slackpipe::ToJson(instance, unhinted);
  EXPECT_TRUE(json.find("\"joint_hints_requested\": false") !=
              std::string::npos);
  EXPECT_TRUE(json.find("\"joint_hints_effective\": false") !=
              std::string::npos);
}

TEST(JointHintPolicy, TimedExplicitEnumerationIsRejected) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;

  slackpipe::JointOptimizerOptions joint_options;
  joint_options.time_limit_seconds = 1.0;
  joint_options.num_workers = 1;
  joint_options.require_optimal = false;
  joint_options.bfs_method = "enumerate";
  EXPECT_THROW(
      {
        const slackpipe::JointOptimizationResult result =
            slackpipe::OptimizeJointSplitAndScheduleCpSat(instance,
                                                          joint_options);
        (void)result;
      },
      slackpipe::Error);

  slackpipe::SlackPipeOptions slackpipe_options;
  slackpipe_options.time_limit_seconds = 1.0;
  slackpipe_options.num_workers = 1;
  slackpipe_options.require_optimal = false;
  slackpipe_options.bfs_method = "enumerate";
  EXPECT_THROW(
      {
        const slackpipe::SlackPipeResult result =
            slackpipe::SolveCanonicalSlackPipe(instance, slackpipe_options);
        (void)result;
      },
      slackpipe::Error);
}

TEST(JointHintPolicy, AuxiliaryVariablesMakeFullModelHintPartial) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;

  slackpipe::JointOptimizerOptions options;
  options.time_limit_seconds = 2.0;
  options.num_workers = 1;
  options.random_seed = 1;
  options.require_optimal = false;
  options.pressure_pruning.enabled = true;
  options.pressure_pruning.partition_top_k = 2;

  const slackpipe::JointOptimizationResult result =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(instance, options);
  ASSERT_TRUE(Solved(result.status));
  EXPECT_TRUE(result.hints_requested);
  EXPECT_TRUE(result.hints_effective);
  EXPECT_TRUE(result.hint_complete_for_basic_model);
  EXPECT_FALSE(result.hint_complete_for_full_model);
  EXPECT_TRUE(result.auxiliary_variable_count > 0);
  EXPECT_EQ(result.hinted_auxiliary_variable_count, 0);
  EXPECT_EQ(result.hinted_layer_variable_count, 4);
  EXPECT_EQ(result.hinted_operation_variable_count, 3 * (4 * 2 * 4));
  EXPECT_EQ(result.hinted_total_variable_count, 101);
}

TEST(PressurePruning, PrefersBalancedPartitions) {
  slackpipe::Instance instance;
  instance.microbatches = 1;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;

  slackpipe::PressurePruningOptions options;
  const double balanced = slackpipe::ComputePartitionPressure(
      instance, std::vector<slackpipe::Tick>{2, 2, 2, 2}, options);
  const double unbalanced = slackpipe::ComputePartitionPressure(
      instance, std::vector<slackpipe::Tick>{5, 1, 1, 1}, options);
  EXPECT_TRUE(balanced < unbalanced);
}

TEST(PressurePruning, IncreasesForNearbyCoLocatedHeavyStages) {
  slackpipe::Instance instance;
  instance.microbatches = 1;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 10;
  instance.min_layers = 1;

  slackpipe::PressurePruningOptions options;
  const double co_located_heavy = slackpipe::ComputePartitionPressure(
      instance, std::vector<slackpipe::Tick>{4, 1, 4, 1}, options);
  const double split_worker_heavy = slackpipe::ComputePartitionPressure(
      instance, std::vector<slackpipe::Tick>{4, 4, 1, 1}, options);
  EXPECT_TRUE(co_located_heavy > split_worker_heavy);
}

TEST(PressurePruning, PartitionPruningPreservesAtLeastOneCandidate) {
  slackpipe::Instance instance;
  instance.microbatches = 1;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;

  slackpipe::PressurePruningOptions options;
  options.enabled = true;
  options.partition_top_k = 0;
  options.partition_epsilon = 0.0;
  slackpipe::PressurePruningStats stats;
  const std::vector<slackpipe::PressurePartitionCandidate> kept =
      slackpipe::SelectPressurePartitions(instance, options, &stats);

  EXPECT_FALSE(kept.empty());
  EXPECT_TRUE(stats.partitions_before > 0);
  EXPECT_TRUE(stats.partitions_after >= 1);
  EXPECT_TRUE(stats.pressure_cutoff_available);
}

TEST(PressurePruning, BeamGeneratorReturnsValidStrictlyIncreasingSplits) {
  slackpipe::Instance instance;
  instance.microbatches = 1;
  instance.stages = 8;
  instance.workers = 4;
  instance.total_layers = 64;
  instance.min_layers = 1;

  slackpipe::PressurePruningOptions options;
  options.beam_width = 16;
  options.branch_width = 8;
  options.generated_partitions = 32;

  const std::vector<slackpipe::PressurePartitionCandidate> candidates =
      slackpipe::GeneratePressureBeamPartitions(instance, options);
  ASSERT_TRUE(!candidates.empty());
  for (const slackpipe::PressurePartitionCandidate &candidate : candidates) {
    ASSERT_EQ(candidate.split.size(),
              static_cast<std::size_t>(instance.stages));
    slackpipe::Tick cumulative = 0;
    slackpipe::Tick previous = 0;
    for (slackpipe::Tick size : candidate.split) {
      EXPECT_TRUE(size >= instance.min_layers);
      cumulative += size;
      EXPECT_TRUE(cumulative > previous);
      previous = cumulative;
    }
    EXPECT_EQ(cumulative, instance.total_layers);
  }
}

TEST(PressurePruning, BeamGeneratorIncludesUniformSplit) {
  slackpipe::Instance instance;
  instance.microbatches = 1;
  instance.stages = 8;
  instance.workers = 4;
  instance.total_layers = 64;
  instance.min_layers = 1;

  slackpipe::PressurePruningOptions options;
  options.beam_width = 8;
  options.branch_width = 4;
  options.generated_partitions = 16;
  const std::vector<slackpipe::PressurePartitionCandidate> candidates =
      slackpipe::GeneratePressureBeamPartitions(instance, options);

  const std::vector<slackpipe::Tick> uniform{8, 8, 8, 8, 8, 8, 8, 8};
  bool found_uniform = false;
  for (const slackpipe::PressurePartitionCandidate &candidate : candidates) {
    if (candidate.split == uniform) found_uniform = true;
  }
  EXPECT_TRUE(found_uniform);
}

TEST(PressurePruning, BeamGeneratorDeduplicatesAndBoundsGeneratedCount) {
  slackpipe::Instance instance;
  instance.microbatches = 1;
  instance.stages = 8;
  instance.workers = 4;
  instance.total_layers = 64;
  instance.min_layers = 1;

  slackpipe::PressurePruningOptions options;
  options.beam_width = 64;
  options.branch_width = 32;
  options.generated_partitions = 20;
  const std::vector<slackpipe::PressurePartitionCandidate> candidates =
      slackpipe::GeneratePressureBeamPartitions(instance, options);

  EXPECT_TRUE(candidates.size() <= 20);
  std::set<std::vector<slackpipe::Tick>> unique_splits;
  for (const slackpipe::PressurePartitionCandidate &candidate : candidates) {
    unique_splits.insert(candidate.split);
  }
  EXPECT_EQ(unique_splits.size(), candidates.size());
}

TEST(PressurePruning, LargeInstanceUsesBeamWithoutExhaustiveEnumeration) {
  slackpipe::Instance instance;
  instance.microbatches = 1;
  instance.stages = 8;
  instance.workers = 4;
  instance.total_layers = 64;
  instance.min_layers = 1;

  slackpipe::PressurePruningOptions options;
  options.enabled = true;
  options.partition_top_k = 64;
  options.beam_width = 64;
  options.branch_width = 16;
  slackpipe::PressurePruningStats stats;
  const std::vector<slackpipe::PressurePartitionCandidate> kept =
      slackpipe::SelectPressurePartitions(instance, options, &stats);

  EXPECT_FALSE(kept.empty());
  EXPECT_EQ(stats.generation_mode, "beam");
  EXPECT_TRUE(stats.exhaustive_enumeration_skipped);
  EXPECT_TRUE(stats.generated_partitions <= 512);
  EXPECT_TRUE(stats.partitions_after <= 64);
}

TEST(PressurePruning, AnchorsAreProtectedFromTopKPruning) {
  slackpipe::Instance instance;
  instance.microbatches = 1;
  instance.stages = 8;
  instance.workers = 4;
  instance.total_layers = 64;
  instance.min_layers = 1;

  slackpipe::PressurePruningOptions options;
  options.enabled = true;
  options.partition_top_k = 1;
  options.beam_width = 16;
  options.branch_width = 8;

  slackpipe::PressurePruningAnchors anchors;
  anchors.has_incumbent_split = true;
  anchors.incumbent_split = {1, 1, 1, 1, 1, 1, 1, 57};
  anchors.has_uniform_split = true;
  anchors.uniform_split = slackpipe::BuildUniformPressureSplit(instance);
  anchors.has_cost_balanced_split = true;
  anchors.cost_balanced_split =
      slackpipe::BuildCostBalancedPressureSplit(instance);

  slackpipe::PressurePruningStats stats;
  const std::vector<slackpipe::PressurePartitionCandidate> kept =
      slackpipe::SelectPressurePartitions(instance, options, &stats, anchors);

  EXPECT_TRUE(stats.incumbent_split_included);
  EXPECT_TRUE(stats.uniform_split_included);
  EXPECT_TRUE(stats.cost_balanced_split_included);
  EXPECT_TRUE(stats.anchor_splits_added >= 2);
  EXPECT_TRUE(stats.selected_splits_total >= stats.anchor_splits_added);

  bool found_incumbent = false;
  bool found_uniform = false;
  for (const slackpipe::PressurePartitionCandidate &candidate : kept) {
    if (candidate.split == anchors.incumbent_split) found_incumbent = true;
    if (candidate.split == anchors.uniform_split) found_uniform = true;
  }
  EXPECT_TRUE(found_incumbent);
  EXPECT_TRUE(found_uniform);
}

TEST(PressurePruning, JointCpSatLargeInstanceDoesNotPruneAwayFeasibility) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 8;
  instance.stages = 8;
  instance.workers = 4;
  instance.total_layers = 64;
  instance.min_layers = 1;

  slackpipe::JointOptimizerOptions options;
  options.time_limit_seconds = 30.0;
  options.num_workers = 1;
  options.random_seed = 1;
  options.require_optimal = false;
  options.log_search_progress = false;
  options.pressure_pruning.enabled = true;
  options.pressure_pruning.partition_top_k = 64;
  options.pressure_pruning.beam_width = 128;
  options.pressure_pruning.branch_width = 16;

  const slackpipe::JointOptimizationResult result =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(instance, options);
  EXPECT_TRUE(result.status != "INFEASIBLE");
  EXPECT_TRUE(result.pressure_pruning_stats.incumbent_split_included);
  EXPECT_TRUE(result.pressure_pruning_stats.uniform_split_included);
  EXPECT_TRUE(result.pressure_pruning_stats.cost_balanced_split_included);
}

TEST(PressurePruning, DisabledCpSatPathMatchesDefaultBehavior) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;

  slackpipe::JointOptimizerOptions baseline_options;
  baseline_options.time_limit_seconds = 10.0;
  baseline_options.num_workers = 1;
  baseline_options.random_seed = 1;
  baseline_options.require_optimal = false;
  baseline_options.log_search_progress = false;

  slackpipe::JointOptimizerOptions disabled_options = baseline_options;
  disabled_options.pressure_pruning.enabled = false;
  disabled_options.pressure_pruning.partition_top_k = 1;
  disabled_options.pressure_pruning.partition_epsilon = 0.0;
  disabled_options.pressure_pruning.predecessor_top_k = 1;

  const slackpipe::JointOptimizationResult baseline =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(instance, baseline_options);
  const slackpipe::JointOptimizationResult disabled =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(instance, disabled_options);

  ASSERT_TRUE(baseline.status == "OPTIMAL" || baseline.status == "FEASIBLE");
  ASSERT_EQ(disabled.status, baseline.status);
  EXPECT_EQ(disabled.makespan_ticks, baseline.makespan_ticks);
  EXPECT_TRUE(disabled.split == baseline.split);
  EXPECT_FALSE(disabled.pressure_pruning_stats.enabled);
}

TEST(InterleavingStats, CountsAllAdjacentDirectionPairs) {
  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 2;
  instance.workers = 1;
  instance.total_layers = 4;
  instance.min_layers = 1;

  slackpipe::ScheduleSolution schedule;
  schedule.split = {2, 2};
  schedule.makespan = 8;
  schedule.operations_by_id.resize(
      static_cast<std::size_t>(instance.OperationCount()));

  auto set_op = [&](slackpipe::Index microbatch, slackpipe::Index chain_index,
                    slackpipe::Tick start, slackpipe::Tick end) {
    const slackpipe::OperationId id =
        slackpipe::EncodeOperation(instance, microbatch, chain_index);
    const slackpipe::OperationView view =
        slackpipe::DecodeOperation(instance, id);
    schedule.operations_by_id[static_cast<std::size_t>(id.value)] =
        slackpipe::ScheduledOperation{id, start, end, end - start, view.worker};
  };

  set_op(0, 0, 0, 1);  // F0
  set_op(0, 1, 1, 2);  // F1 => F_to_F
  set_op(0, 2, 2, 3);  // B1 => F_to_B
  set_op(1, 1, 3, 4);  // F1 => B_to_F
  set_op(1, 2, 4, 5);  // B1 => F_to_B
  set_op(1, 3, 5, 6);  // B0 => B_to_B
  set_op(1, 0, 6, 7);  // F0 => B_to_F
  set_op(0, 3, 7, 8);  // B0 => F_to_B

  const slackpipe::InterleavingStats stats =
      slackpipe::CollectInterleavingStats(instance, schedule);
  ASSERT_EQ(stats.events.size(), std::size_t{7});

  std::vector<std::string> kinds;
  std::vector<bool> switches;
  for (const slackpipe::InterleavingEvent &event : stats.events) {
    kinds.push_back(event.kind);
    switches.push_back(event.is_direction_switch);
    EXPECT_EQ(event.gap_ticks, 0);
  }

  const std::vector<std::string> expected_kinds{
      "F_to_F", "F_to_B", "B_to_F", "F_to_B", "B_to_B", "B_to_F", "F_to_B"};
  const std::vector<bool> expected_switches{false, true, true, true,
                                            false, true, true};
  ASSERT_EQ(kinds.size(), expected_kinds.size());
  ASSERT_EQ(switches.size(), expected_switches.size());
  for (std::size_t i = 0; i < expected_kinds.size(); ++i) {
    EXPECT_EQ(kinds[i], expected_kinds[i]);
    EXPECT_EQ(switches[i], expected_switches[i]);
  }

  EXPECT_EQ(stats.events[0].prev_stage_position, 0);
  EXPECT_EQ(stats.events[0].next_stage_position, 1);
  EXPECT_EQ(stats.events[0].prev_order_index, 0);
  EXPECT_EQ(stats.events[0].next_order_index, 1);
  EXPECT_EQ(stats.events[0].num_worker_ops, 8);
  EXPECT_EQ(stats.events[0].prev_order_fraction, 0.0);
  EXPECT_EQ(stats.events[0].next_order_fraction, 1.0 / 7.0);
  EXPECT_FALSE(stats.events[0].forward_b.has_value());
  EXPECT_FALSE(stats.events[0].backward_b.has_value());

  EXPECT_EQ(stats.events[6].prev_order_index, 6);
  EXPECT_EQ(stats.events[6].next_order_index, 7);
  EXPECT_EQ(stats.events[6].num_worker_ops, 8);
  EXPECT_EQ(stats.events[6].prev_order_fraction, 6.0 / 7.0);
  EXPECT_EQ(stats.events[6].next_order_fraction, 1.0);

  ASSERT_TRUE(stats.events[1].forward_n.has_value());
  EXPECT_EQ(*stats.events[1].forward_n, 1);
  ASSERT_TRUE(stats.events[1].forward_stage_position.has_value());
  EXPECT_EQ(*stats.events[1].forward_stage_position, 1);
  ASSERT_TRUE(stats.events[1].backward_n.has_value());
  EXPECT_EQ(*stats.events[1].backward_n, 1);
  ASSERT_TRUE(stats.events[1].backward_stage_position.has_value());
  EXPECT_EQ(*stats.events[1].backward_stage_position, 1);

  EXPECT_EQ(stats.events[4].kind, "B_to_B");
  EXPECT_FALSE(stats.events[4].is_direction_switch);
  EXPECT_FALSE(stats.events[4].forward_n.has_value());
  EXPECT_FALSE(stats.events[4].backward_n.has_value());

  const std::string csv = slackpipe::ToInterleavingEventsCsv(
      instance, slackpipe::InterleavingRunMetadata{"OK", 8, std::nullopt},
      stats);
  EXPECT_TRUE(csv.find("is_direction_switch") != std::string::npos);
  EXPECT_TRUE(csv.find("prev_stage_position") != std::string::npos);
  EXPECT_TRUE(csv.find("next_stage_position") != std::string::npos);
  EXPECT_TRUE(csv.find("prev_order_index") != std::string::npos);
  EXPECT_TRUE(csv.find("next_order_index") != std::string::npos);
  EXPECT_TRUE(csv.find("num_worker_ops") != std::string::npos);
  EXPECT_TRUE(csv.find("prev_order_fraction") != std::string::npos);
  EXPECT_TRUE(csv.find("next_order_fraction") != std::string::npos);
  EXPECT_TRUE(csv.find("F_to_F") != std::string::npos);
  EXPECT_TRUE(csv.find("F_to_B") != std::string::npos);
  EXPECT_TRUE(csv.find("B_to_F") != std::string::npos);
  EXPECT_TRUE(csv.find("B_to_B") != std::string::npos);

  const std::string summary_csv = slackpipe::ToInterleavingSummaryCsv(
      instance, slackpipe::InterleavingRunMetadata{"OK", 8, std::nullopt},
      stats);
  EXPECT_TRUE(summary_csv.find("prev_stage_position") != std::string::npos);
  EXPECT_TRUE(summary_csv.find("next_stage_position") != std::string::npos);
}

TEST(FixedOrderPartitionBackend, ExplicitEnumerateRecordsProvenance) {
  slackpipe::Instance instance = BaseInstance();
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);
  slackpipe::BfsSplitOptimizerOptions options;
  options.fixed_order_partition_backend = "enumerate";
  options.require_optimal = false;

  const slackpipe::BfsSplitOptimizationResult result =
      slackpipe::OptimizePartitionForFixedOrder(instance, orders, options);

  EXPECT_EQ(result.status, std::string("OPTIMAL"));
  EXPECT_TRUE(result.proven_optimal);
  EXPECT_TRUE(result.schedule.ok());
  EXPECT_EQ(result.fixed_order_partition_backend_requested,
            std::string("enumerate"));
  EXPECT_EQ(result.fixed_order_partition_backend_effective,
            std::string("enumerate"));
  EXPECT_FALSE(result.cp_sat_launched);
  EXPECT_EQ(result.cp_sat_models_solved, 0);
  EXPECT_TRUE(result.estimated_partition_count_available);
  EXPECT_EQ(result.estimated_partition_count,
            slackpipe::CountValidSplitsCapped(instance, 1000000));
}

TEST(FixedOrderPartitionBackend,
     EnumerateEnforcesActivationCapBeforeObjectiveComparison) {
  slackpipe::Instance instance;
  instance.microbatches = 1;
  instance.stages = 2;
  instance.workers = 2;
  instance.total_layers = 4;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;
  instance.communication_ticks = 0;
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);
  const slackpipe::ActivationAnalysisOptions cap =
      UniformBaselineActivationCap({2, 2}, {2, 0});

  const ExhaustiveOracleResult oracle =
      ExhaustiveFixedOrderOracle(instance, orders, cap);
  ASSERT_TRUE(oracle.feasible);
  EXPECT_EQ(oracle.candidates_checked, 3);
  EXPECT_EQ(oracle.feasible_candidates, 2);
  EXPECT_EQ(oracle.makespan, 12);
  EXPECT_TRUE((oracle.split == std::vector<slackpipe::Tick>{1, 3}));

  slackpipe::BfsSplitOptimizerOptions options;
  options.fixed_order_partition_backend = "enumerate";
  options.require_optimal = false;
  options.activation_options = cap;
  const slackpipe::BfsSplitOptimizationResult result =
      slackpipe::OptimizePartitionForFixedOrder(instance, orders, options);

  ASSERT_EQ(result.status, std::string("OPTIMAL"));
  EXPECT_TRUE(result.proven_optimal);
  EXPECT_TRUE(result.enumeration_proved_optimal);
  EXPECT_EQ(result.optimality_proof_source,
            std::string("exhaustive_enumeration"));
  EXPECT_EQ(result.checked_splits, 3);
  EXPECT_EQ(result.enumeration_candidates_total, 3);
  EXPECT_EQ(result.enumeration_candidates_valid_schedule, 3);
  EXPECT_EQ(result.enumeration_candidates_cap_feasible, 2);
  EXPECT_EQ(result.enumeration_candidates_cap_rejected, 1);
  EXPECT_EQ(result.makespan_ticks, oracle.makespan);
  EXPECT_EQ(result.best_bound_ticks, result.makespan_ticks);
  EXPECT_TRUE(result.split == oracle.split);
  EXPECT_TRUE(result.activation_cap_constraints.enforced_by_enumeration);
  EXPECT_FALSE(result.activation_cap_constraints.constraints_added);

  const slackpipe::ActivationAnalysisResult analysis =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          instance, result.schedule, cap);
  ASSERT_TRUE(analysis.activation_cap_satisfied);
  EXPECT_TRUE(*analysis.activation_cap_satisfied);
  slackpipe::ActivationAnalysisResult with_metadata = analysis;
  slackpipe::ApplyActivationCapConstraintMetadata(
      with_metadata, result.activation_cap_constraints, "exact_enumeration");
  EXPECT_TRUE(with_metadata.activation_cap_enforced_by_enumeration);
  EXPECT_FALSE(with_metadata.activation_cap_enforced_in_solver);
  EXPECT_EQ(with_metadata.activation_cap_enforcement_mode,
            std::string("exact_enumeration"));
  ExpectScheduleIndependentlyValid(instance, result.schedule, result.status,
                                   "partition-only-fixed-order");
}

TEST(FixedOrderPartitionBackend,
     EnumerateChoosesSlowerCapFeasibleSplitWhenUncappedBestViolatesCap) {
  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 2;
  instance.workers = 2;
  instance.total_layers = 16;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;
  instance.communication_ticks = 0;
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);
  const slackpipe::ActivationAnalysisOptions cap =
      UniformBaselineActivationCap({8, 8}, {16, 0});

  slackpipe::BfsSplitOptimizerOptions uncapped_options;
  uncapped_options.fixed_order_partition_backend = "enumerate";
  uncapped_options.require_optimal = false;
  const slackpipe::BfsSplitOptimizationResult uncapped =
      slackpipe::OptimizePartitionForFixedOrder(instance, orders,
                                                uncapped_options);
  ASSERT_EQ(uncapped.status, std::string("OPTIMAL"));
  EXPECT_EQ(uncapped.makespan_ticks, 68);
  EXPECT_TRUE((uncapped.split == std::vector<slackpipe::Tick>{10, 6}));

  const ExhaustiveOracleResult capped_oracle =
      ExhaustiveFixedOrderOracle(instance, orders, cap);
  ASSERT_TRUE(capped_oracle.feasible);
  EXPECT_EQ(capped_oracle.makespan, 72);

  slackpipe::BfsSplitOptimizerOptions capped_options = uncapped_options;
  capped_options.activation_options = cap;
  const slackpipe::BfsSplitOptimizationResult capped =
      slackpipe::OptimizePartitionForFixedOrder(instance, orders,
                                                capped_options);

  ASSERT_EQ(capped.status, std::string("OPTIMAL"));
  EXPECT_TRUE(capped.proven_optimal);
  EXPECT_EQ(capped.makespan_ticks, capped_oracle.makespan);
  EXPECT_TRUE(capped.split == capped_oracle.split);
  EXPECT_TRUE(capped.makespan_ticks > uncapped.makespan_ticks);
  EXPECT_TRUE(capped.split != uncapped.split);
  EXPECT_TRUE(capped.enumeration_candidates_cap_rejected > 0);
  EXPECT_EQ(capped.enumeration_candidates_cap_feasible,
            static_cast<std::uint64_t>(capped_oracle.feasible_candidates));
  const slackpipe::ActivationAnalysisResult analysis =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          instance, capped.schedule, cap);
  ASSERT_TRUE(analysis.activation_cap_satisfied);
  EXPECT_TRUE(*analysis.activation_cap_satisfied);
}

TEST(FixedOrderPartitionBackend,
     EnumerateReportsInfeasibleWhenNoSplitMeetsCap) {
  const slackpipe::Instance instance = TinyTwoStageActivationTradeoffInstance();
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);

  slackpipe::BfsSplitOptimizerOptions options;
  options.fixed_order_partition_backend = "enumerate";
  options.require_optimal = false;
  options.activation_options = ExplicitActivationCap({0, 0});
  const slackpipe::BfsSplitOptimizationResult result =
      slackpipe::OptimizePartitionForFixedOrder(instance, orders, options);

  EXPECT_EQ(result.status, std::string("INFEASIBLE"));
  EXPECT_FALSE(result.proven_optimal);
  EXPECT_TRUE(result.enumeration_proved_optimal);
  EXPECT_EQ(result.optimality_proof_source,
            std::string("exhaustive_enumeration"));
  EXPECT_TRUE(result.split.empty());
  EXPECT_TRUE(result.schedule.operations_by_id.empty());
  EXPECT_EQ(result.makespan_ticks, 0);
  EXPECT_EQ(result.enumeration_candidates_total, 1);
  EXPECT_EQ(result.enumeration_candidates_valid_schedule, 1);
  EXPECT_EQ(result.enumeration_candidates_cap_feasible, 0);
  EXPECT_EQ(result.enumeration_candidates_cap_rejected, 1);
}

TEST(FixedOrderPartitionBackend, SplitCountEstimateCapsSafely) {
  slackpipe::Instance small;
  small.microbatches = 1;
  small.stages = 3;
  small.workers = 1;
  small.total_layers = 5;
  small.min_layers = 1;
  EXPECT_EQ(slackpipe::CountValidSplitsCapped(small, 1000), 6);

  slackpipe::Instance large;
  large.microbatches = 1;
  large.stages = 64;
  large.workers = 1;
  large.total_layers = 1000;
  large.min_layers = 1;
  EXPECT_EQ(slackpipe::CountValidSplitsCapped(large, 123), 123);
}

TEST(FixedOrderPartitionBackend, ExplicitCpsatUnavailableWithoutOrTools) {
  if (slackpipe::IsCpSatFixedOrderPartitionOptimizerAvailable()) {
    return;
  }
  slackpipe::Instance instance = BaseInstance();
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);
  slackpipe::BfsSplitOptimizerOptions options;
  options.fixed_order_partition_backend = "cpsat";
  options.require_optimal = false;

  const slackpipe::BfsSplitOptimizationResult result =
      slackpipe::OptimizePartitionForFixedOrder(instance, orders, options);

  EXPECT_EQ(result.status, std::string("UNAVAILABLE"));
  EXPECT_EQ(result.solver_status_raw, std::string("UNAVAILABLE"));
  EXPECT_EQ(result.fixed_order_partition_backend_requested,
            std::string("cpsat"));
  EXPECT_EQ(result.fixed_order_partition_backend_effective,
            std::string("unavailable"));
  EXPECT_FALSE(result.cp_sat_launched);
  EXPECT_EQ(result.cp_sat_models_solved, 0);
  EXPECT_TRUE(result.split.empty());
  EXPECT_FALSE(result.fallback_used);
}

TEST(FixedOrderPartitionBackend,
     EnforcedActivationCapUnavailableWithoutOrTools) {
  if (slackpipe::IsCpSatFixedOrderPartitionOptimizerAvailable()) {
    return;
  }
  slackpipe::Instance instance = BaseInstance();
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);
  slackpipe::BfsSplitOptimizerOptions options;
  options.fixed_order_partition_backend = "auto";
  options.enumeration_threshold = 1000;
  options.require_optimal = false;
  options.activation_options.cap_mode = slackpipe::ActivationCapMode::kExplicit;
  options.activation_options.activation_cap_units = {100};
  options.activation_options.enforce_activation_cap = true;

  const slackpipe::BfsSplitOptimizationResult result =
      slackpipe::OptimizePartitionForFixedOrder(instance, orders, options);

  EXPECT_EQ(result.status, std::string("UNAVAILABLE"));
  EXPECT_EQ(result.activation_cap_constraints.model_support_level,
            std::string("none"));
  EXPECT_FALSE(result.activation_cap_constraints.solver_supported);
  EXPECT_FALSE(result.activation_cap_constraints.constraints_added);
  EXPECT_TRUE(result.activation_cap_constraints.unsupported_reason.find(
                  "cumulative") != std::string::npos);
}

TEST(FixedOrderPartitionBackend,
     AutoEnumeratesOnlyBelowSafetyThresholdWithoutOrTools) {
  if (slackpipe::IsCpSatFixedOrderPartitionOptimizerAvailable()) {
    return;
  }
  slackpipe::Instance instance = BaseInstance();
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);

  slackpipe::BfsSplitOptimizerOptions safe_options;
  safe_options.fixed_order_partition_backend = "auto";
  safe_options.enumeration_threshold = 1000;
  safe_options.require_optimal = false;
  const slackpipe::BfsSplitOptimizationResult safe =
      slackpipe::OptimizePartitionForFixedOrder(instance, orders, safe_options);
  EXPECT_EQ(safe.status, std::string("OPTIMAL"));
  EXPECT_EQ(safe.fixed_order_partition_backend_requested, std::string("auto"));
  EXPECT_EQ(safe.fixed_order_partition_backend_effective,
            std::string("enumerate"));

  slackpipe::BfsSplitOptimizerOptions unsafe_options;
  unsafe_options.fixed_order_partition_backend = "auto";
  unsafe_options.enumeration_threshold = 0;
  unsafe_options.require_optimal = false;
  const slackpipe::BfsSplitOptimizationResult unsafe =
      slackpipe::OptimizePartitionForFixedOrder(instance, orders,
                                                unsafe_options);
  EXPECT_EQ(unsafe.status, std::string("UNAVAILABLE"));
  EXPECT_EQ(unsafe.fixed_order_partition_backend_requested,
            std::string("auto"));
  EXPECT_EQ(unsafe.fixed_order_partition_backend_effective,
            std::string("unavailable"));
  EXPECT_FALSE(unsafe.estimated_partition_count_available);
  EXPECT_EQ(unsafe.estimated_partition_count, 1);
}

TEST(FixedOrderPartitionBackend, ProvenanceSerializesInCanonicalJson) {
  slackpipe::Instance instance = BaseInstance();
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);
  slackpipe::BfsSplitOptimizerOptions options;
  options.fixed_order_partition_backend = "enumerate";
  options.require_optimal = false;
  slackpipe::BfsSplitOptimizationResult result =
      slackpipe::OptimizePartitionForFixedOrder(instance, orders, options);
  result.canonical = slackpipe::BuildCanonicalResultMetadata(
      instance, slackpipe::CanonicalRequestContext{},
      slackpipe::SemanticsForPartitionOnlyFixedOrder(result),
      slackpipe::OutcomeFromBfsResult(result), result.split,
      result.machine_orders);

  const std::string json = slackpipe::ToJson(instance, result);
  EXPECT_TRUE(json.find("\"fixed_order_partition_backend_requested\": "
                        "\"enumerate\"") != std::string::npos);
  EXPECT_TRUE(json.find("\"fixed_order_partition_backend_effective\": "
                        "\"enumerate\"") != std::string::npos);
  EXPECT_TRUE(json.find("\"estimated_partition_count\"") != std::string::npos);
  EXPECT_TRUE(json.find("\"cp_sat_launched\": false") != std::string::npos);
  EXPECT_TRUE(json.find("\"optimality_proof_source\": "
                        "\"exhaustive_enumeration\"") != std::string::npos);
  EXPECT_TRUE(json.find("\"enumeration_proved_optimal\": true") !=
              std::string::npos);
  EXPECT_TRUE(json.find("\"enumeration_candidates_total\"") !=
              std::string::npos);
}

TEST(FixedOrderPartitionCpSat, MatchesEnumerationOnTinyInstanceWhenAvailable) {
  if (!slackpipe::IsCpSatFixedOrderPartitionOptimizerAvailable()) {
    return;
  }
  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 3;
  instance.workers = 2;
  instance.total_layers = 7;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;
  instance.communication_ticks = 1;
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);

  slackpipe::BfsSplitOptimizerOptions options;
  options.fixed_order_partition_backend = "cpsat";
  options.time_limit_seconds = 5.0;
  options.num_workers = 1;
  options.random_seed = 7;
  options.require_optimal = false;

  const slackpipe::BfsSplitOptimizationResult cpsat =
      slackpipe::OptimizePartitionForFixedOrder(instance, orders, options);
  const slackpipe::BfsSplitOptimizationResult enumerate =
      slackpipe::OptimizePartitionForFixedOrderEnumerate(instance, orders,
                                                         options);

  ASSERT_TRUE(Solved(cpsat.status));
  ASSERT_TRUE(Solved(enumerate.status));
  EXPECT_EQ(cpsat.makespan_ticks, enumerate.makespan_ticks);
  EXPECT_TRUE(cpsat.machine_orders == orders);
  EXPECT_TRUE(cpsat.schedule.ok());
  EXPECT_TRUE(cpsat.cp_sat_launched);
  EXPECT_EQ(cpsat.cp_sat_models_solved, 1);
  EXPECT_EQ(cpsat.fixed_order_partition_backend_effective,
            std::string("cpsat"));
  const slackpipe::ResultValidationResult validation =
      slackpipe::ValidateScheduleSolutionIndependent(
          instance, cpsat.schedule, cpsat.status, "partition-only-fixed-order");
  EXPECT_TRUE(validation.passed);
}

TEST(OrToolsCapabilities, ReportsCumulativeSupportConsistently) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  EXPECT_TRUE(slackpipe::OrToolsCompiledIn());
  EXPECT_TRUE(slackpipe::CumulativeConstraintCompiledIn());
  EXPECT_TRUE(slackpipe::ActivationCapCumulativeConstraintSupported());
  EXPECT_EQ(slackpipe::CumulativeConstraintCompiledIn(),
            slackpipe::ActivationCapCumulativeConstraintSupported());
  EXPECT_EQ(slackpipe::VariableCumulativeDemandCompiledIn(),
            slackpipe::ActivationCapVariableCumulativeDemandSupported());
  EXPECT_EQ(slackpipe::ActivationCapSolverSupportLevelForBuild(),
            slackpipe::ToString(slackpipe::ActivationCapSolverSupport()));
  EXPECT_NE(slackpipe::ActivationCapSolverSupportLevelForBuild(),
            std::string("none"));
}

TEST(OrToolsFixedOrderPartitionCpSat, MatchesExhaustiveFixedOrderOracles) {
  if (!slackpipe::IsCpSatFixedOrderPartitionOptimizerAvailable()) {
    return;
  }

  struct Case {
    slackpipe::Instance instance;
    slackpipe::MachineOrders orders;
  };
  std::vector<Case> cases;

  {
    slackpipe::Instance instance;
    instance.microbatches = 1;
    instance.stages = 2;
    instance.workers = 2;
    instance.total_layers = 4;
    instance.min_layers = 1;
    instance.backward_ratio_num = 2;
    instance.backward_ratio_den = 1;
    instance.communication_ticks = 0;
    cases.push_back(Case{instance, slackpipe::BreadthFirstOrders(instance)});
  }
  {
    slackpipe::Instance instance;
    instance.microbatches = 2;
    instance.stages = 3;
    instance.workers = 2;
    instance.total_layers = 5;
    instance.min_layers = 1;
    instance.backward_ratio_num = 2;
    instance.backward_ratio_den = 1;
    instance.communication_ticks = 1;
    cases.push_back(Case{instance, slackpipe::BreadthFirstOrders(instance)});
  }
  {
    slackpipe::Instance instance;
    instance.microbatches = 3;
    instance.stages = 4;
    instance.workers = 2;
    instance.total_layers = 6;
    instance.min_layers = 1;
    instance.backward_ratio_num = 2;
    instance.backward_ratio_den = 1;
    instance.communication_ticks = 0;
    cases.push_back(Case{instance, slackpipe::BreadthFirstOrders(instance)});
  }
  {
    slackpipe::Instance instance;
    instance.microbatches = 2;
    instance.stages = 1;
    instance.workers = 1;
    instance.total_layers = 3;
    instance.min_layers = 1;
    instance.backward_ratio_num = 2;
    instance.backward_ratio_den = 1;
    instance.communication_ticks = 0;
    slackpipe::MachineOrders orders(1);
    orders[0] = {slackpipe::EncodeOperation(instance, 0, 0),
                 slackpipe::EncodeOperation(instance, 1, 0),
                 slackpipe::EncodeOperation(instance, 0, 1),
                 slackpipe::EncodeOperation(instance, 1, 1)};
    ASSERT_TRUE(orders != slackpipe::BreadthFirstOrders(instance));
    cases.push_back(Case{instance, orders});
  }

  for (const Case &test_case : cases) {
    const ExhaustiveOracleResult oracle =
        ExhaustiveFixedOrderOracle(test_case.instance, test_case.orders);
    ASSERT_TRUE(oracle.feasible);

    slackpipe::BfsSplitOptimizerOptions options;
    options.fixed_order_partition_backend = "cpsat";
    options.time_limit_seconds = 5.0;
    options.num_workers = 1;
    options.random_seed = 11;
    options.require_optimal = false;

    const slackpipe::BfsSplitOptimizationResult cpsat =
        slackpipe::OptimizePartitionForFixedOrderCpSat(
            test_case.instance, test_case.orders, options);
    ASSERT_EQ(cpsat.status, std::string("OPTIMAL"));
    EXPECT_TRUE(cpsat.proven_optimal);
    EXPECT_EQ(cpsat.makespan_ticks, oracle.makespan);
    EXPECT_TRUE(cpsat.machine_orders == test_case.orders);
    EXPECT_EQ(cpsat.fixed_order_partition_backend_effective,
              std::string("cpsat"));
    EXPECT_TRUE(cpsat.cp_sat_launched);
    EXPECT_EQ(cpsat.cp_sat_models_solved, 1);
    ExpectScheduleIndependentlyValid(test_case.instance, cpsat.schedule,
                                     cpsat.status,
                                     "partition-only-fixed-order");
  }
}

TEST(OrToolsFixedOrderPartitionCpSat,
     RejectsMalformedAndCyclicOrdersBeforeSolve) {
  if (!slackpipe::IsCpSatFixedOrderPartitionOptimizerAvailable()) {
    return;
  }

  slackpipe::Instance malformed_instance;
  malformed_instance.microbatches = 1;
  malformed_instance.stages = 1;
  malformed_instance.workers = 1;
  malformed_instance.total_layers = 1;
  malformed_instance.min_layers = 1;
  slackpipe::MachineOrders malformed(1);
  malformed[0] = {slackpipe::EncodeOperation(malformed_instance, 0, 0)};

  slackpipe::BfsSplitOptimizerOptions options;
  options.fixed_order_partition_backend = "cpsat";
  options.time_limit_seconds = 5.0;
  options.num_workers = 1;
  options.random_seed = 1;
  EXPECT_THROW((void)slackpipe::OptimizePartitionForFixedOrderCpSat(
                   malformed_instance, malformed, options),
               slackpipe::Error);

  slackpipe::MachineOrders cyclic(1);
  cyclic[0] = {slackpipe::EncodeOperation(malformed_instance, 0, 1),
               slackpipe::EncodeOperation(malformed_instance, 0, 0)};
  EXPECT_THROW((void)slackpipe::OptimizePartitionForFixedOrderCpSat(
                   malformed_instance, cyclic, options),
               slackpipe::Error);
}

TEST(OrToolsScheduleOnlyCpSat,
     MatchesOrderOracleAndEnforcesActivationCapAsConstraint) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  const slackpipe::Instance instance = TinyTwoStageActivationTradeoffInstance();
  const std::vector<slackpipe::Tick> fixed_split{1, 1};
  const slackpipe::ActivationAnalysisOptions cap =
      ExplicitActivationCap({1, 1});
  const ExhaustiveOracleResult uncapped_oracle =
      ExhaustiveFixedSplitOrderOracle(instance, fixed_split);
  const ExhaustiveOracleResult capped_oracle =
      ExhaustiveFixedSplitOrderOracle(instance, fixed_split, cap);
  ASSERT_TRUE(uncapped_oracle.feasible);
  ASSERT_TRUE(capped_oracle.feasible);
  ASSERT_EQ(uncapped_oracle.makespan, 9);
  ASSERT_EQ(capped_oracle.makespan, 12);
  ASSERT_TRUE(uncapped_oracle.makespan < capped_oracle.makespan);

  slackpipe::JointOptimizerOptions uncapped_options;
  uncapped_options.time_limit_seconds = 5.0;
  uncapped_options.num_workers = 1;
  uncapped_options.random_seed = 17;
  uncapped_options.require_optimal = false;
  const slackpipe::JointOptimizationResult uncapped =
      slackpipe::OptimizeScheduleForFixedSplitCpSat(instance, fixed_split,
                                                    uncapped_options);
  ASSERT_EQ(uncapped.status, std::string("OPTIMAL"));
  EXPECT_TRUE(uncapped.proven_optimal);
  EXPECT_TRUE(uncapped.split == fixed_split);
  EXPECT_EQ(uncapped.makespan_ticks, uncapped_oracle.makespan);
  const slackpipe::ActivationAnalysisResult uncapped_as_capped =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          instance, uncapped.schedule, cap);
  ASSERT_TRUE(uncapped_as_capped.activation_cap_satisfied);
  EXPECT_FALSE(*uncapped_as_capped.activation_cap_satisfied);

  slackpipe::JointOptimizerOptions capped_options = uncapped_options;
  capped_options.activation_options = cap;
  const slackpipe::JointOptimizationResult capped =
      slackpipe::OptimizeScheduleForFixedSplitCpSat(instance, fixed_split,
                                                    capped_options);
  ASSERT_EQ(capped.status, std::string("OPTIMAL"));
  EXPECT_TRUE(capped.proven_optimal);
  EXPECT_TRUE(capped.split == fixed_split);
  EXPECT_EQ(capped.makespan_ticks, capped_oracle.makespan);
  EXPECT_EQ(capped.num_workers, 1);
  EXPECT_TRUE(capped.joint_budget_seconds <= capped_options.time_limit_seconds);
  EXPECT_TRUE(capped.activation_cap_constraints.solver_supported);
  EXPECT_TRUE(capped.activation_cap_constraints.constraints_added);
  EXPECT_EQ(capped.activation_cap_constraints.retained_interval_count,
            instance.microbatches * instance.stages);
  EXPECT_EQ(capped.activation_cap_constraints.cumulative_constraint_count,
            instance.workers);
  EXPECT_EQ(capped.activation_cap_constraints.fixed_demand_count,
            instance.microbatches * instance.stages);

  const slackpipe::ActivationAnalysisResult capped_analysis =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          instance, capped.schedule, cap,
          capped.activation_cap_constraints.constraints_added);
  slackpipe::ActivationAnalysisResult capped_analysis_with_metadata =
      capped_analysis;
  slackpipe::ApplyActivationCapConstraintMetadata(
      capped_analysis_with_metadata, capped.activation_cap_constraints);
  ASSERT_TRUE(capped_analysis.activation_cap_satisfied);
  EXPECT_TRUE(*capped_analysis.activation_cap_satisfied);
  EXPECT_TRUE(capped_analysis_with_metadata.activation_cap_enforced_in_solver);
  EXPECT_EQ(capped_analysis_with_metadata.activation_cap_enforcement_mode,
            std::string("solver"));
  ExpectScheduleIndependentlyValid(instance, capped.schedule, capped.status,
                                   "schedule-only-uniform");

  const slackpipe::CanonicalSemantics semantics =
      slackpipe::SemanticsForScheduleOnlyFixedSplit("uniform", capped, false);
  EXPECT_FALSE(semantics.partition_optimized);
  EXPECT_TRUE(semantics.schedule_optimized);
  EXPECT_TRUE(semantics.full_partition_fixed);
  EXPECT_FALSE(semantics.predecessor_candidate_restriction_active);
  ASSERT_TRUE(semantics.predecessor_candidate_rule);
  EXPECT_EQ(*semantics.predecessor_candidate_rule,
            std::string("unrestricted_no_overlap"));
}

TEST(OrToolsJointCpSat, MatchesExhaustiveJointOracleAndImprovesBothDimensions) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 2;
  instance.workers = 2;
  instance.total_layers = 4;
  instance.min_layers = 1;
  instance.backward_ratio_num = 1;
  instance.backward_ratio_den = 1;
  instance.communication_ticks = 3;

  const ExhaustiveOracleResult oracle = ExhaustiveJointOracle(instance);
  ASSERT_TRUE(oracle.feasible);
  ASSERT_EQ(oracle.candidates_checked, 2940);
  ASSERT_EQ(oracle.makespan, 24);

  slackpipe::JointOptimizerOptions options;
  options.time_limit_seconds = 10.0;
  options.num_workers = 1;
  options.random_seed = 19;
  options.require_optimal = false;
  const slackpipe::JointOptimizationResult result =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(instance, options);
  ASSERT_EQ(result.status, std::string("OPTIMAL"));
  EXPECT_TRUE(result.proven_optimal);
  EXPECT_EQ(result.makespan_ticks, oracle.makespan);
  slackpipe::ValidateSplit(instance, result.split);
  EXPECT_TRUE(result.schedule.ok());
  ExpectScheduleIndependentlyValid(instance, result.schedule, result.status,
                                   "joint-unrestricted-no-overlap");

  const std::vector<slackpipe::Tick> uniform_split =
      slackpipe::UniformSplit(instance);
  const slackpipe::MachineOrders bfs = slackpipe::BreadthFirstOrders(instance);
  const slackpipe::EvaluationResult uniform_bfs =
      slackpipe::EvaluateSchedule(instance, uniform_split, bfs);
  ASSERT_TRUE(uniform_bfs.schedule.ok());
  EXPECT_TRUE(result.makespan_ticks < uniform_bfs.schedule.makespan);
  EXPECT_TRUE(result.split != uniform_split);
  EXPECT_TRUE(result.machine_orders != bfs);

  const slackpipe::CanonicalSemantics semantics =
      slackpipe::SemanticsForJointUnrestrictedNoOverlap(result, false);
  EXPECT_TRUE(semantics.partition_optimized);
  EXPECT_TRUE(semantics.schedule_optimized);
  EXPECT_FALSE(semantics.predecessor_candidate_restriction_active);
  ASSERT_TRUE(semantics.predecessor_candidate_rule);
  EXPECT_EQ(*semantics.predecessor_candidate_rule,
            std::string("unrestricted_no_overlap"));
  EXPECT_TRUE(result.cp_sat_models_solved >= 1);
  EXPECT_TRUE(result.joint_budget_seconds <= options.time_limit_seconds);
}

TEST(OrToolsJointCpSat, RejectsInvalidHybridIncumbentOnCalFifoFallbackBlocker) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  const slackpipe::Instance instance = CalJointFifoFallbackBlockerInstance();
  slackpipe::JointOptimizerOptions options;
  options.time_limit_seconds = 0.5;
  options.num_workers = 1;
  options.random_seed = 0;
  options.require_optimal = false;
  options.symmetry_break_f0_fifo = true;

  const slackpipe::JointOptimizationResult result =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(instance, options);

  ASSERT_TRUE(Solved(result.status));
  EXPECT_NE(result.status, std::string("INVALID_RESULT"));
  ASSERT_TRUE(result.schedule.ok());
  ExpectScheduleIndependentlyValid(instance, result.schedule, result.status,
                                   "joint-unrestricted-no-overlap");
  ExpectSamePositionFifoRespected(instance, result.machine_orders);

  const slackpipe::EvaluationResult replay = slackpipe::EvaluateSchedule(
      instance, result.split, result.machine_orders);
  ASSERT_TRUE(replay.schedule.ok());
  EXPECT_EQ(replay.schedule.makespan, result.makespan_ticks);
  EXPECT_TRUE(result.makespan_ticks <= result.bfs_incumbent.makespan_ticks);
  EXPECT_TRUE(result.fallback_available);
  EXPECT_EQ(result.fallback_source,
            std::string("deterministic_uniform_breadth_first_incumbent"));
  EXPECT_EQ(result.incumbent_source,
            std::string("deterministic_uniform_breadth_first_incumbent"));
  EXPECT_EQ(result.incumbent_method_effective, std::string("slack"));
  EXPECT_EQ(result.bfs_incumbent_method_effective,
            std::string("deterministic-uniform-bfs-fallback"));
  EXPECT_TRUE(result.incumbent_feasible);
}

TEST(OrToolsActivationCap,
     FixedOrderPartitionDemandModelsMatchCappedEnumeration) {
  if (!slackpipe::IsCpSatFixedOrderPartitionOptimizerAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 2;
  instance.workers = 2;
  instance.total_layers = 4;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;
  instance.communication_ticks = 1;
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);

  auto run_case = [&](slackpipe::ActivationAnalysisOptions cap,
                      bool partition_linear_demand) {
    const ExhaustiveOracleResult oracle =
        ExhaustiveFixedOrderOracle(instance, orders, cap);
    ASSERT_TRUE(oracle.feasible);

    slackpipe::BfsSplitOptimizerOptions options;
    options.fixed_order_partition_backend = "cpsat";
    options.time_limit_seconds = 5.0;
    options.num_workers = 1;
    options.random_seed = 23;
    options.require_optimal = false;
    options.activation_options = cap;
    const slackpipe::BfsSplitOptimizationResult result =
        slackpipe::OptimizePartitionForFixedOrderCpSat(instance, orders,
                                                       options);
    if (partition_linear_demand &&
        !slackpipe::ActivationCapVariableCumulativeDemandSupported()) {
      EXPECT_EQ(result.status, std::string("UNAVAILABLE"));
      EXPECT_TRUE(result.activation_cap_constraints.unsupported_reason.find(
                      "variable cumulative") != std::string::npos);
      return;
    }
    ASSERT_EQ(result.status, std::string("OPTIMAL"));
    EXPECT_EQ(result.makespan_ticks, oracle.makespan);
    EXPECT_TRUE(result.activation_cap_constraints.solver_supported);
    EXPECT_TRUE(result.activation_cap_constraints.constraints_added);
    if (partition_linear_demand) {
      EXPECT_EQ(result.activation_cap_constraints.variable_demand_count,
                instance.microbatches * instance.stages);
    } else {
      EXPECT_EQ(result.activation_cap_constraints.fixed_demand_count,
                instance.microbatches * instance.stages);
    }
    const slackpipe::ActivationAnalysisResult analysis =
        slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
            instance, result.schedule, cap,
            result.activation_cap_constraints.constraints_added);
    ASSERT_TRUE(analysis.activation_cap_satisfied);
    EXPECT_TRUE(*analysis.activation_cap_satisfied);
    ExpectScheduleIndependentlyValid(instance, result.schedule, result.status,
                                     "partition-only-fixed-order");
  };

  slackpipe::ActivationAnalysisOptions count_cap =
      ExplicitActivationCap({100, 100});
  count_cap.model = slackpipe::ActivationModel::kCount;
  run_case(count_cap, false);

  slackpipe::ActivationAnalysisOptions explicit_cap =
      ExplicitActivationCap({100, 100});
  explicit_cap.model = slackpipe::ActivationModel::kExplicitStageUnits;
  explicit_cap.explicit_stage_activation_units = {3, 5};
  run_case(explicit_cap, false);

  slackpipe::ActivationAnalysisOptions linear_cap =
      ExplicitActivationCap({100, 100});
  linear_cap.model = slackpipe::ActivationModel::kLinearInStageLayers;
  linear_cap.activation_units_per_layer = 2;
  run_case(linear_cap, true);
}

TEST(OrToolsActivationCap,
     FixedOrderPartitionVariableDemandMatchesProductionBlockerOracle) {
  if (!slackpipe::IsCpSatFixedOrderPartitionOptimizerAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 2;
  instance.workers = 2;
  instance.total_layers = 16;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;
  instance.communication_ticks = 0;
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);
  const slackpipe::ActivationAnalysisOptions cap =
      UniformBaselineActivationCap({8, 8}, {16, 0});

  const ExhaustiveOracleResult uncapped_oracle =
      ExhaustiveFixedOrderOracle(instance, orders);
  ASSERT_TRUE(uncapped_oracle.feasible);
  EXPECT_EQ(uncapped_oracle.makespan, 68);
  EXPECT_TRUE((uncapped_oracle.split == std::vector<slackpipe::Tick>{10, 6}));
  const ExhaustiveOracleResult capped_oracle =
      ExhaustiveFixedOrderOracle(instance, orders, cap);
  ASSERT_TRUE(capped_oracle.feasible);
  EXPECT_EQ(capped_oracle.candidates_checked, 15);
  EXPECT_EQ(capped_oracle.makespan, 72);

  slackpipe::ActivationCapModelDebugDump debug_dump;
  slackpipe::BfsSplitOptimizerOptions options;
  options.fixed_order_partition_backend = "cpsat";
  options.time_limit_seconds = 30.0;
  options.num_workers = 1;
  options.random_seed = 0;
  options.require_optimal = false;
  options.activation_options = cap;
  options.activation_cap_model_debug_dump = &debug_dump;
  const slackpipe::BfsSplitOptimizationResult result =
      slackpipe::OptimizePartitionForFixedOrderCpSat(instance, orders, options);

  if (!slackpipe::ActivationCapVariableCumulativeDemandSupported()) {
    EXPECT_EQ(result.status, std::string("UNAVAILABLE"));
    EXPECT_TRUE(result.activation_cap_constraints.unsupported_reason.find(
                    "variable cumulative") != std::string::npos);
    return;
  }

  ASSERT_EQ(result.status, std::string("OPTIMAL"));
  EXPECT_TRUE(result.proven_optimal);
  EXPECT_NE(result.status, std::string("INVALID_RESULT"));
  EXPECT_EQ(result.makespan_ticks, capped_oracle.makespan);
  EXPECT_TRUE(result.activation_cap_constraints.solver_supported);
  EXPECT_TRUE(result.activation_cap_constraints.constraints_added);
  EXPECT_EQ(result.activation_cap_constraints.variable_demand_count,
            instance.microbatches * instance.stages);
  EXPECT_EQ(result.activation_cap_constraints.fixed_demand_count, 0);
  EXPECT_EQ(result.activation_cap_constraints.retained_interval_count,
            instance.microbatches * instance.stages);
  EXPECT_EQ(result.activation_cap_constraints.cumulative_constraint_count,
            instance.workers);

  slackpipe::Index worker0_stage0_terms = 0;
  for (const slackpipe::ActivationCapModelTerm &term : debug_dump.terms) {
    if (term.worker == 0 && term.stage == 0) {
      ++worker0_stage0_terms;
      EXPECT_EQ(term.capacity_units, 16);
      EXPECT_EQ(term.demand_type, std::string("variable"));
      EXPECT_TRUE(term.demand_source.find("layers_0") != std::string::npos);
      EXPECT_TRUE(term.included_in_cumulative);
    }
  }
  EXPECT_EQ(worker0_stage0_terms, instance.microbatches);

  ExpectScheduleIndependentlyValid(instance, result.schedule, result.status,
                                   "partition-only-fixed-order");
  const slackpipe::ActivationAnalysisResult analysis =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          instance, result.schedule, cap,
          result.activation_cap_constraints.constraints_added);
  ASSERT_TRUE(analysis.activation_cap_satisfied);
  EXPECT_TRUE(*analysis.activation_cap_satisfied);
  ASSERT_TRUE(analysis.per_worker.size() >= std::size_t{1});
  EXPECT_TRUE(analysis.per_worker[0].peak_activation_units <= 16);

  std::vector<slackpipe::ActivationLifetime> stage0_lifetimes;
  for (const slackpipe::ActivationLifetime &lifetime : analysis.lifetimes) {
    if (lifetime.identity.worker == 0 && lifetime.identity.stage == 0) {
      stage0_lifetimes.push_back(lifetime);
    }
  }
  ASSERT_EQ(stage0_lifetimes.size(), std::size_t{2});
  const bool stage0_overlap =
      stage0_lifetimes[0].start < stage0_lifetimes[1].end &&
      stage0_lifetimes[1].start < stage0_lifetimes[0].end;
  if (stage0_overlap) {
    EXPECT_TRUE(2 * result.split[0] <= 16);
  }
}

TEST(OrToolsActivationCap,
     JointRetainsUniformCapFallbackOnCalEqualMemoryBlocker) {
  if (!slackpipe::IsJointOptimizerAvailable() ||
      !slackpipe::ActivationCapVariableCumulativeDemandSupported()) {
    return;
  }

  const slackpipe::Instance instance =
      CalJointEqualMemoryFallbackBlockerInstance();
  const std::vector<slackpipe::Tick> uniform =
      slackpipe::UniformSplit(instance);
  const slackpipe::MachineOrders bfs = slackpipe::BreadthFirstOrders(instance);
  const slackpipe::EvaluationResult uniform_eval =
      slackpipe::EvaluateSchedule(instance, uniform, bfs);
  ASSERT_TRUE(uniform_eval.schedule.ok());
  EXPECT_EQ(uniform_eval.schedule.makespan, 456);

  const slackpipe::ActivationAnalysisOptions cap =
      UniformBaselineActivationCap(uniform, {120, 96});
  const slackpipe::ActivationAnalysisResult baseline_analysis =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          instance, uniform_eval.schedule, cap);
  ASSERT_TRUE(baseline_analysis.activation_cap_satisfied);
  EXPECT_TRUE(*baseline_analysis.activation_cap_satisfied);
  ASSERT_EQ(baseline_analysis.per_worker.size(), std::size_t{2});
  EXPECT_EQ(baseline_analysis.per_worker[0].peak_activation_units, 120);
  EXPECT_EQ(baseline_analysis.per_worker[1].peak_activation_units, 96);

  slackpipe::JointOptimizerOptions options;
  options.time_limit_seconds = 0.5;
  options.num_workers = 1;
  options.random_seed = 0;
  options.require_optimal = false;
  options.symmetry_break_f0_fifo = true;
  options.activation_options = cap;

  const slackpipe::JointOptimizationResult result =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(instance, options);

  ASSERT_TRUE(Solved(result.status));
  EXPECT_NE(result.status, std::string("INVALID_RESULT"));
  ASSERT_TRUE(result.schedule.ok());
  ExpectScheduleIndependentlyValid(instance, result.schedule, result.status,
                                   "joint-unrestricted-no-overlap");
  ExpectSamePositionFifoRespected(instance, result.machine_orders);

  const slackpipe::EvaluationResult replay = slackpipe::EvaluateSchedule(
      instance, result.split, result.machine_orders);
  ASSERT_TRUE(replay.schedule.ok());
  EXPECT_EQ(replay.schedule.makespan, result.makespan_ticks);
  EXPECT_TRUE(result.fallback_available);
  EXPECT_EQ(result.fallback_source,
            std::string("deterministic_uniform_breadth_first_incumbent"));
  EXPECT_EQ(result.incumbent_source,
            std::string("deterministic_uniform_breadth_first_incumbent"));
  EXPECT_TRUE(result.bfs_incumbent.split == uniform);
  EXPECT_EQ(result.bfs_incumbent.makespan_ticks,
            uniform_eval.schedule.makespan);
  EXPECT_TRUE(result.incumbent_feasible);
  EXPECT_TRUE(result.diagnostic.find("no cap-feasible incumbent") ==
              std::string::npos);

  const slackpipe::ActivationAnalysisResult analysis =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          instance, result.schedule, cap,
          result.activation_cap_constraints.constraints_added);
  ASSERT_TRUE(analysis.activation_cap_satisfied);
  EXPECT_TRUE(*analysis.activation_cap_satisfied);
  ASSERT_EQ(analysis.per_worker.size(), std::size_t{2});
  EXPECT_TRUE(analysis.per_worker[0].peak_activation_units <= 120);
  EXPECT_TRUE(analysis.per_worker[1].peak_activation_units <= 96);
  EXPECT_TRUE(result.activation_cap_constraints.solver_supported);
  EXPECT_TRUE(result.activation_cap_constraints.constraints_added);
}

TEST(OrToolsActivationCap, JointCapFeasibleAndImpossibleCap) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  const slackpipe::Instance feasible_instance =
      TinyTwoStageActivationTradeoffInstance();
  slackpipe::JointOptimizerOptions feasible_options;
  feasible_options.time_limit_seconds = 5.0;
  feasible_options.num_workers = 1;
  feasible_options.random_seed = 29;
  feasible_options.require_optimal = false;
  feasible_options.activation_options = ExplicitActivationCap({1, 1});
  const slackpipe::JointOptimizationResult feasible =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(feasible_instance,
                                                    feasible_options);
  ASSERT_EQ(feasible.status, std::string("OPTIMAL"));
  EXPECT_TRUE(feasible.activation_cap_constraints.solver_supported);
  EXPECT_TRUE(feasible.activation_cap_constraints.constraints_added);
  EXPECT_EQ(feasible.activation_cap_constraints.retained_interval_count,
            feasible_instance.microbatches * feasible_instance.stages);
  const slackpipe::ActivationAnalysisResult analysis =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          feasible_instance, feasible.schedule,
          feasible_options.activation_options,
          feasible.activation_cap_constraints.constraints_added);
  ASSERT_TRUE(analysis.activation_cap_satisfied);
  EXPECT_TRUE(*analysis.activation_cap_satisfied);
  ExpectScheduleIndependentlyValid(feasible_instance, feasible.schedule,
                                   feasible.status,
                                   "joint-unrestricted-no-overlap");

  slackpipe::Instance variable_instance = feasible_instance;
  variable_instance.total_layers = 4;
  slackpipe::ActivationAnalysisOptions variable_cap =
      ExplicitActivationCap({2, 2});
  variable_cap.model = slackpipe::ActivationModel::kLinearInStageLayers;
  variable_cap.activation_units_per_layer = 1;
  const ExhaustiveOracleResult variable_oracle =
      ExhaustiveJointOracle(variable_instance, variable_cap);
  ASSERT_TRUE(variable_oracle.feasible);

  slackpipe::JointOptimizerOptions variable_options = feasible_options;
  variable_options.random_seed = 30;
  variable_options.activation_options = variable_cap;
  const slackpipe::JointOptimizationResult variable =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(variable_instance,
                                                    variable_options);
  if (!slackpipe::ActivationCapVariableCumulativeDemandSupported()) {
    EXPECT_EQ(variable.status, std::string("UNAVAILABLE"));
  } else {
    ASSERT_EQ(variable.status, std::string("OPTIMAL"));
    EXPECT_EQ(variable.makespan_ticks, variable_oracle.makespan);
    EXPECT_TRUE(variable.activation_cap_constraints.solver_supported);
    EXPECT_TRUE(variable.activation_cap_constraints.constraints_added);
    EXPECT_EQ(variable.activation_cap_constraints.variable_demand_count,
              variable_instance.microbatches * variable_instance.stages);
    const slackpipe::ActivationAnalysisResult variable_analysis =
        slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
            variable_instance, variable.schedule, variable_cap,
            variable.activation_cap_constraints.constraints_added);
    ASSERT_TRUE(variable_analysis.activation_cap_satisfied);
    EXPECT_TRUE(*variable_analysis.activation_cap_satisfied);
    ExpectScheduleIndependentlyValid(variable_instance, variable.schedule,
                                     variable.status,
                                     "joint-unrestricted-no-overlap");
  }

  slackpipe::Instance impossible_instance;
  impossible_instance.microbatches = 1;
  impossible_instance.stages = 2;
  impossible_instance.workers = 2;
  impossible_instance.total_layers = 2;
  impossible_instance.min_layers = 1;
  impossible_instance.backward_ratio_num = 2;
  impossible_instance.backward_ratio_den = 1;
  impossible_instance.communication_ticks = 0;
  slackpipe::JointOptimizerOptions impossible_options = feasible_options;
  impossible_options.activation_options = ExplicitActivationCap({0, 0});
  const slackpipe::JointOptimizationResult impossible =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(impossible_instance,
                                                    impossible_options);
  EXPECT_EQ(impossible.status, std::string("NO_VALID_SOLUTION"));
  EXPECT_EQ(impossible.joint_status, std::string("INFEASIBLE"));
  EXPECT_FALSE(impossible.solver_solution_available);
  EXPECT_FALSE(impossible.final_solution_available);
  EXPECT_EQ(impossible.final_solution_source, std::string("none"));
  EXPECT_EQ(impossible.no_solution_reason,
            std::string("solver_no_feasible_solution"));
  EXPECT_TRUE(impossible.schedule.operations_by_id.empty());
  EXPECT_TRUE(impossible.activation_cap_constraints.constraints_added);
}

TEST(OrToolsActivationCap, SequentialAndAlternatingRecordCapConstrainedPhases) {
  if (!slackpipe::IsJointOptimizerAvailable() ||
      !slackpipe::IsCpSatFixedOrderPartitionOptimizerAvailable()) {
    return;
  }

  const slackpipe::Instance instance = TinyTwoStageActivationTradeoffInstance();
  slackpipe::AlternatingOptimizerOptions options;
  options.time_limit_seconds = 5.0;
  options.num_workers = 1;
  options.random_seed = 31;
  options.require_optimal = false;
  options.fixed_order_partition_backend = "cpsat";
  options.max_rounds = 1;
  options.activation_options = ExplicitActivationCap({1, 1});

  const slackpipe::AlternatingOptimizationResult sequential =
      slackpipe::OptimizeSequentialPartitionThenSchedule(instance, options);
  ASSERT_TRUE(Solved(sequential.status));
  EXPECT_TRUE(sequential.activation_cap_constraints.solver_supported);
  EXPECT_TRUE(sequential.activation_cap_constraints.constraints_added);
  ASSERT_EQ(sequential.alternating_trace.size(), std::size_t{2});
  for (const slackpipe::CanonicalAlternatingTraceEntry &entry :
       sequential.alternating_trace) {
    EXPECT_TRUE(entry.activation_cap_requested);
    EXPECT_TRUE(entry.activation_cap_supported);
    EXPECT_TRUE(entry.activation_cap_constraints_added);
    ASSERT_TRUE(entry.solver_threads);
    EXPECT_EQ(*entry.solver_threads, 1);
  }
  const slackpipe::ActivationAnalysisResult sequential_analysis =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          instance, sequential.schedule, options.activation_options,
          sequential.activation_cap_constraints.constraints_added);
  ASSERT_TRUE(sequential_analysis.activation_cap_satisfied);
  EXPECT_TRUE(*sequential_analysis.activation_cap_satisfied);

  const slackpipe::AlternatingOptimizationResult alternating =
      slackpipe::OptimizeAlternatingPartitionSchedule(instance, options);
  ASSERT_TRUE(Solved(alternating.status));
  EXPECT_TRUE(alternating.activation_cap_constraints.solver_supported);
  EXPECT_TRUE(alternating.activation_cap_constraints.constraints_added);
  EXPECT_FALSE(alternating.alternating_trace.empty());
  const slackpipe::ActivationAnalysisResult alternating_analysis =
      slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
          instance, alternating.schedule, options.activation_options,
          alternating.activation_cap_constraints.constraints_added);
  ASSERT_TRUE(alternating_analysis.activation_cap_satisfied);
  EXPECT_TRUE(*alternating_analysis.activation_cap_satisfied);
}

TEST(OrToolsActivationCap,
     SequentialAndAlternatingReplayProductionFallbackDeterministically) {
  if (!slackpipe::IsJointOptimizerAvailable() ||
      !slackpipe::IsCpSatFixedOrderPartitionOptimizerAvailable() ||
      !slackpipe::ActivationCapVariableCumulativeDemandSupported()) {
    return;
  }

  const slackpipe::Instance instance = CalFallbackReplayBlockerInstance();
  const slackpipe::ActivationAnalysisOptions cap =
      UniformBaselineActivationCap({8, 8, 8, 8}, {72, 40});

  slackpipe::AlternatingOptimizerOptions options;
  options.time_limit_seconds = 30.0;
  options.num_workers = 1;
  options.random_seed = 0;
  options.require_optimal = false;
  options.fixed_order_partition_backend = "cpsat";
  options.max_rounds = 4;
  options.activation_options = cap;

  auto expect_replay_valid =
      [&](const slackpipe::AlternatingOptimizationResult &result,
          const std::string &method) {
        EXPECT_TRUE(Solved(result.status));
        ASSERT_TRUE(Solved(result.status));
        EXPECT_NE(result.status, std::string("INVALID_RESULT"));
        ASSERT_TRUE(result.schedule.ok());
        ASSERT_TRUE(!result.split.empty());
        ASSERT_EQ(result.machine_orders.size(),
                  static_cast<std::size_t>(instance.workers));

        const slackpipe::EvaluationResult replay = slackpipe::EvaluateSchedule(
            instance, result.split, result.machine_orders);
        ASSERT_TRUE(replay.schedule.ok());
        EXPECT_EQ(replay.schedule.makespan, result.makespan_ticks);

        ExpectScheduleIndependentlyValid(instance, result.schedule,
                                         result.status, method);
        const slackpipe::ActivationAnalysisResult analysis =
            slackpipe::AnalyzeActivationMemoryWithUniformBaseline(
                instance, result.schedule, cap,
                result.activation_cap_constraints.constraints_added);
        ASSERT_TRUE(analysis.activation_cap_satisfied);
        EXPECT_TRUE(*analysis.activation_cap_satisfied);
        ASSERT_EQ(analysis.per_worker.size(),
                  static_cast<std::size_t>(instance.workers));
        EXPECT_TRUE(analysis.per_worker[0].peak_activation_units <= 72);
        EXPECT_TRUE(analysis.per_worker[1].peak_activation_units <= 40);
        EXPECT_TRUE(result.activation_cap_constraints.solver_supported);
        EXPECT_TRUE(result.activation_cap_constraints.constraints_added);
        EXPECT_TRUE(result.cp_sat_models_solved >= 1);

        for (const slackpipe::CanonicalAlternatingTraceEntry &entry :
             result.alternating_trace) {
          EXPECT_TRUE(entry.activation_cap_requested);
          EXPECT_TRUE(entry.activation_cap_supported);
          EXPECT_TRUE(entry.activation_cap_constraints_added);
          if (entry.accepted) {
            EXPECT_TRUE(entry.validation_passed);
            ASSERT_TRUE(entry.activation_cap_satisfied);
            EXPECT_TRUE(*entry.activation_cap_satisfied);
            EXPECT_FALSE(entry.selected_partition.empty());
          }
        }
      };

  const slackpipe::AlternatingOptimizationResult sequential =
      slackpipe::OptimizeSequentialPartitionThenSchedule(instance, options);
  expect_replay_valid(sequential, "sequential-partition-then-schedule");

  const slackpipe::AlternatingOptimizationResult alternating =
      slackpipe::OptimizeAlternatingPartitionSchedule(instance, options);
  expect_replay_valid(alternating, "alternating-partition-schedule");
}

TEST(OrToolsBudgetProvenance, TinyJointRecordsBudgetAndFallbackFields) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 2;
  instance.workers = 2;
  instance.total_layers = 4;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;
  instance.communication_ticks = 1;

  slackpipe::JointOptimizerOptions options;
  options.time_limit_seconds = 1.0;
  options.num_workers = 1;
  options.random_seed = 37;
  options.require_optimal = false;
  const slackpipe::JointOptimizationResult result =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(instance, options);
  ASSERT_TRUE(Solved(result.status));
  EXPECT_TRUE(result.incumbent_budget_seconds <=
              options.time_limit_seconds *
                      slackpipe::kSlackPipePreparationBudgetFraction +
                  1e-9);
  EXPECT_TRUE(result.joint_budget_seconds <= options.time_limit_seconds);
  EXPECT_TRUE(result.wall_time_seconds >= result.timing.model_build_seconds);
  EXPECT_TRUE(result.wall_time_seconds >= result.timing.solver_seconds);
  EXPECT_TRUE(result.cp_sat_models_solved >= 1);
  EXPECT_TRUE(result.time_to_first_feasible_seconds <=
              result.time_to_best_incumbent_seconds + 1e-9);
  EXPECT_EQ(result.num_workers, options.num_workers);
  EXPECT_EQ(result.symmetry_break_f0_fifo, options.symmetry_break_f0_fifo);

  const std::vector<slackpipe::Tick> fixed_split{1, 1};
  slackpipe::JointOptimizerOptions fallback_options;
  fallback_options.num_workers = 1;
  fallback_options.random_seed = 41;
  fallback_options.require_optimal = false;
  fallback_options.activation_options = ExplicitActivationCap({0, 0});
  const slackpipe::JointOptimizationResult fallback =
      slackpipe::BuildScheduleOnlyFixedSplitDeadlineFallback(
          TinyTwoStageActivationTradeoffInstance(), fixed_split,
          fallback_options, 0.01, "unit_test_deadline_expired");
  EXPECT_EQ(fallback.status, std::string("NOT_RUN"));
  EXPECT_EQ(fallback.joint_status, std::string("NOT_RUN"));
  EXPECT_FALSE(fallback.fallback_available);
  EXPECT_FALSE(fallback.fallback_used);
  EXPECT_TRUE(fallback.diagnostic.find("unit_test_deadline_expired") !=
              std::string::npos);
}

TEST(OrToolsFifoOrdering, ConstraintCountsDefaultEquivalenceAndTinyOptimum) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  slackpipe::Instance instance;
  instance.microbatches = 2;
  instance.stages = 2;
  instance.workers = 2;
  instance.total_layers = 4;
  instance.min_layers = 1;
  instance.backward_ratio_num = 2;
  instance.backward_ratio_den = 1;
  instance.communication_ticks = 1;

  slackpipe::JointOptimizerOptions base;
  base.time_limit_seconds = 10.0;
  base.num_workers = 1;
  base.random_seed = 123;
  base.require_optimal = true;
  base.incumbent_method = "none";
  base.incumbent_bound = false;
  base.incumbent_hints = false;
  base.use_bfs_hints = false;

  slackpipe::JointOptimizerOptions explicit_on = base;
  explicit_on.fifo_ordering = true;
  const slackpipe::JointOptimizationResult legacy =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(instance, base);
  const slackpipe::JointOptimizationResult on =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(instance, explicit_on);

  ASSERT_EQ(legacy.status, std::string("OPTIMAL"));
  ASSERT_EQ(on.status, std::string("OPTIMAL"));
  const slackpipe::Index expected_fifo_constraints =
      (instance.microbatches - 1) *
          slackpipe::OperationPositionCount(instance) +
      (instance.microbatches - 1);
  EXPECT_EQ(legacy.fifo_constraint_count, expected_fifo_constraints);
  EXPECT_EQ(on.fifo_constraint_count, expected_fifo_constraints);
  EXPECT_TRUE(legacy.fifo_ordering_requested);
  EXPECT_TRUE(legacy.fifo_ordering_effective);
  EXPECT_TRUE(legacy.split == on.split);
  EXPECT_EQ(legacy.makespan_ticks, on.makespan_ticks);
  EXPECT_EQ(legacy.solver_objective_ticks, on.solver_objective_ticks);
  EXPECT_EQ(legacy.best_bound_ticks, on.best_bound_ticks);
  EXPECT_EQ(legacy.incumbent_source, on.incumbent_source);

  slackpipe::JointOptimizerOptions off_options = base;
  off_options.fifo_ordering = false;
  const slackpipe::JointOptimizationResult off =
      slackpipe::OptimizeJointSplitAndScheduleCpSat(instance, off_options);
  ASSERT_EQ(off.status, std::string("OPTIMAL"));
  EXPECT_FALSE(off.fifo_ordering_requested);
  EXPECT_FALSE(off.fifo_ordering_effective);
  EXPECT_EQ(off.fifo_constraint_count, 0);
  EXPECT_FALSE(off.symmetry_break_f0_fifo);
  EXPECT_EQ(on.makespan_ticks, off.makespan_ticks);
  EXPECT_EQ(on.solver_objective_ticks, off.solver_objective_ticks);

  const std::string json = slackpipe::ToJson(instance, off);
  EXPECT_NE(json.find("\"fifo_ordering_requested\": false"), std::string::npos);
  EXPECT_NE(json.find("\"fifo_ordering_effective\": false"), std::string::npos);
  EXPECT_NE(json.find("\"fifo_constraint_count\": 0"), std::string::npos);
}

TEST(MegatronPlanExport, EmitsRequiredSchemaAndStructuredOperations) {
  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;
  instance.backward_ratio_num = 1;
  instance.backward_ratio_den = 1;

  const std::vector<slackpipe::Tick> split = slackpipe::UniformSplit(instance);
  const slackpipe::MachineOrders orders =
      slackpipe::BreadthFirstOrders(instance);
  const slackpipe::EvaluationResult evaluated =
      slackpipe::EvaluateSchedule(instance, split, orders);
  ASSERT_TRUE(evaluated.schedule.ok());

  const std::string json = slackpipe::ToMegatronSlackPipePlanJson(
      instance, evaluated.schedule, "FEASIBLE");

  EXPECT_NE(json.find("\"schema_version\": \"slackpipe.plan.v1\""),
            std::string::npos);
  EXPECT_NE(json.find("\"num_microbatches\": 4"), std::string::npos);
  EXPECT_NE(json.find("\"num_stages\": 4"), std::string::npos);
  EXPECT_NE(json.find("\"num_workers\": 2"), std::string::npos);
  EXPECT_NE(json.find("\"num_layers\": 8"), std::string::npos);
  EXPECT_NE(json.find("\"layer_split\": [2, 2, 2, 2]"), std::string::npos);
  EXPECT_NE(json.find("\"stage_to_worker\": [0, 1, 0, 1]"), std::string::npos);
  EXPECT_NE(json.find("\"solver_status\": \"FEASIBLE\""), std::string::npos);
  EXPECT_NE(json.find("\"predicted_makespan\": 36"), std::string::npos);
  EXPECT_NE(json.find("\"forward_costs\": [2, 2, 2, 2]"), std::string::npos);
  EXPECT_NE(json.find("\"backward_costs\": [2, 2, 2, 2]"), std::string::npos);
  EXPECT_NE(json.find("{\"kind\": \"F\", \"microbatch\": 0, \"stage\": 0}"),
            std::string::npos);
  EXPECT_NE(json.find("{\"kind\": \"B\", \"microbatch\": 3, \"stage\": 0}"),
            std::string::npos);
  EXPECT_EQ(
      CountSubstring(json, "{\"kind\":"),
      static_cast<std::size_t>(2 * instance.microbatches * instance.stages));
}

TEST(MegatronPlanExport, RejectsDuplicateOrMissingOperations) {
  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;

  const std::vector<slackpipe::Tick> split = slackpipe::UniformSplit(instance);
  const slackpipe::EvaluationResult evaluated = slackpipe::EvaluateSchedule(
      instance, split, slackpipe::BreadthFirstOrders(instance));
  ASSERT_TRUE(evaluated.schedule.ok());

  slackpipe::ScheduleSolution duplicate = evaluated.schedule;
  duplicate.orders[0].push_back(duplicate.orders[0].front());
  EXPECT_THROW(
      slackpipe::ValidateMegatronSlackPipePlanExport(instance, duplicate),
      slackpipe::Error);

  slackpipe::ScheduleSolution missing = evaluated.schedule;
  missing.orders[0].pop_back();
  EXPECT_THROW(
      slackpipe::ValidateMegatronSlackPipePlanExport(instance, missing),
      slackpipe::Error);
}

TEST(MegatronPlanExport, AtomicWriteLeavesNoFinalFileOnValidationFailure) {
  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;

  const std::filesystem::path final_path =
      std::filesystem::temp_directory_path() /
      "slackpipe_plan_export_failure.plan.json";
  const std::filesystem::path tmp_path = final_path.string() + ".tmp";
  std::error_code ignored;
  std::filesystem::remove(final_path, ignored);
  std::filesystem::remove(tmp_path, ignored);

  slackpipe::ScheduleSolution invalid;
  invalid.split = slackpipe::UniformSplit(instance);
  invalid.orders.assign(static_cast<std::size_t>(instance.workers), {});

  EXPECT_THROW(slackpipe::WriteMegatronSlackPipePlanFile(
                   final_path.string(), instance, invalid, "FEASIBLE"),
               slackpipe::Error);
  EXPECT_FALSE(std::filesystem::exists(final_path));
  EXPECT_FALSE(std::filesystem::exists(tmp_path));
}

TEST(CostProfile, AppliesSharedSlopeAndStageBiasDurations) {
  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 16;
  instance.min_layers = 1;

  const std::filesystem::path profile_path =
      std::filesystem::temp_directory_path() /
      "slackpipe_cost_profile_unit.json";
  std::ofstream out(profile_path);
  out << "{\n"
      << "  \"schema_version\": \"slackpipe.cost_profile.v1\",\n"
      << "  \"calibration_partition\": [1, 4, 7, 4],\n"
      << "  \"a_fwd\": 0.25,\n"
      << "  \"a_bwd\": 0.5,\n"
      << "  \"bias_fwd\": [0.01, 0.02, 0.03, 0.04],\n"
      << "  \"bias_bwd\": [0.05, 0.06, 0.07, 0.08]\n"
      << "}\n";
  out.close();
  ASSERT_TRUE(out.good());

  slackpipe::ApplyCostProfileFile(instance, profile_path.string());

  EXPECT_TRUE(instance.HasCostProfile());
  const std::vector<slackpipe::Tick> split{1, 2, 3, 10};
  EXPECT_EQ(instance.Duration(0, false, split), 260);
  EXPECT_EQ(instance.Duration(1, false, split), 520);
  EXPECT_EQ(instance.Duration(2, true, split), 1570);
  EXPECT_EQ(instance.Duration(3, true, split), 5080);
  EXPECT_EQ(instance.cost_profile_units, std::string("microseconds"));

  std::error_code ignored;
  std::filesystem::remove(profile_path, ignored);
}

TEST(CostProfile, AppliesRangePrefixAndRoleBiasDurations) {
  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 3;
  instance.workers = 2;
  instance.total_layers = 4;
  instance.min_layers = 1;

  const std::filesystem::path profile_path =
      std::filesystem::temp_directory_path() /
      "slackpipe_cost_profile_v2_unit.json";
  std::ofstream out(profile_path);
  out << "{\n"
      << "  \"schema_version\": \"slackpipe.cost_profile.v2\",\n"
      << "  \"model_manifest_hash\": \"manifest-sha\",\n"
      << "  \"cost_profile_hash\": \"profile-sha\",\n"
      << "  \"prefix_forward_us\": [0, 100, 300, 700, 800],\n"
      << "  \"prefix_backward_us\": [0, 200, 500, 1100, 1300],\n"
      << "  \"stage_role_bias_us\": {\n"
      << "    \"first\": {\"forward\": 10, \"backward\": 40},\n"
      << "    \"middle\": {\"forward\": 20, \"backward\": 50},\n"
      << "    \"last\": {\"forward\": 30, \"backward\": 60}\n"
      << "  }\n"
      << "}\n";
  out.close();
  ASSERT_TRUE(out.good());

  slackpipe::ApplyCostProfileFile(instance, profile_path.string());

  EXPECT_TRUE(instance.HasCostProfile());
  EXPECT_TRUE(instance.HasRangeCostProfile());
  EXPECT_FALSE(instance.HasAffineCostProfile());
  EXPECT_EQ(instance.cost_profile_hash, std::string("profile-sha"));
  EXPECT_EQ(instance.model_manifest_hash, std::string("manifest-sha"));

  const std::vector<slackpipe::Tick> split{1, 2, 1};
  EXPECT_EQ(instance.StageBeginLayer(1, split), 1);
  EXPECT_EQ(instance.StageEndLayer(1, split), 3);
  EXPECT_EQ(instance.Duration(0, false, split), 110);
  EXPECT_EQ(instance.Duration(1, false, split), 620);
  EXPECT_EQ(instance.Duration(2, true, split), 260);
  EXPECT_EQ(instance.ForwardDuration(1, 1, 3), 620);
  EXPECT_EQ(instance.BackwardDuration(0, 0, 1), 240);

  std::error_code ignored;
  std::filesystem::remove(profile_path, ignored);
}

TEST(MegatronPlanExport, CostProfileMetadataUsesMeasuredDurations) {
  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;
  instance.profile_forward_slope_ticks = {100, 100, 100, 100};
  instance.profile_backward_slope_ticks = {200, 200, 200, 200};
  instance.profile_forward_bias_ticks = {1, 2, 3, 4};
  instance.profile_backward_bias_ticks = {5, 6, 7, 8};
  instance.cost_profile_path = "/tmp/profile.json";
  instance.cost_profile_schema_version = "slackpipe.cost_profile.v1";
  instance.cost_profile_units = "microseconds";

  const std::vector<slackpipe::Tick> split = slackpipe::UniformSplit(instance);
  const slackpipe::EvaluationResult evaluated = slackpipe::EvaluateSchedule(
      instance, split, slackpipe::BreadthFirstOrders(instance));
  ASSERT_TRUE(evaluated.schedule.ok());

  const std::string json = slackpipe::ToMegatronSlackPipePlanJson(
      instance, evaluated.schedule, "FEASIBLE");

  EXPECT_NE(json.find("\"kind\": \"cost_profile\""), std::string::npos);
  EXPECT_NE(json.find("\"path\": \"/tmp/profile.json\""), std::string::npos);
  EXPECT_NE(json.find("\"tick_scale\": \"microseconds\""), std::string::npos);
  EXPECT_NE(json.find("\"forward_costs\": [201, 202, 203, 204]"),
            std::string::npos);
  EXPECT_NE(json.find("\"backward_costs\": [405, 406, 407, 408]"),
            std::string::npos);
  EXPECT_EQ(
      CountSubstring(json, "{\"kind\":"),
      static_cast<std::size_t>(2 * instance.microbatches * instance.stages));
}

TEST(MegatronPlanExport, RangeCostProfileEmitsPlanV2RangesAndHashes) {
  slackpipe::Instance instance;
  instance.microbatches = 4;
  instance.stages = 4;
  instance.workers = 2;
  instance.total_layers = 8;
  instance.min_layers = 1;
  instance.profile_prefix_forward_ticks = {0,   10,  30,  60, 100,
                                           150, 210, 280, 360};
  instance.profile_prefix_backward_ticks = {0,   20,  60,  120, 200,
                                            300, 420, 560, 720};
  instance.profile_role_forward_bias_ticks = {1, 2, 3};
  instance.profile_role_backward_bias_ticks = {4, 5, 6};
  instance.cost_profile_path = "/tmp/profile-v2.json";
  instance.cost_profile_schema_version = "slackpipe.cost_profile.v2";
  instance.cost_profile_hash = "profile-sha";
  instance.model_manifest_hash = "manifest-sha";
  instance.cost_profile_units = "microseconds";

  const std::vector<slackpipe::Tick> split{1, 2, 1, 4};
  const slackpipe::EvaluationResult evaluated = slackpipe::EvaluateSchedule(
      instance, split, slackpipe::BreadthFirstOrders(instance));
  ASSERT_TRUE(evaluated.schedule.ok());

  const std::string json = slackpipe::ToMegatronSlackPipePlanJson(
      instance, evaluated.schedule, "FEASIBLE");

  EXPECT_NE(json.find("\"schema_version\": \"slackpipe.plan.v2\""),
            std::string::npos);
  EXPECT_NE(json.find("\"layer_split\": [1, 2, 1, 4]"), std::string::npos);
  EXPECT_NE(json.find("\"stage_layer_ranges\": [{\"begin\": 0, \"end\": 1}, "
                      "{\"begin\": 1, \"end\": 3}, {\"begin\": 3, \"end\": 4}, "
                      "{\"begin\": 4, \"end\": 8}]"),
            std::string::npos);
  EXPECT_NE(json.find("\"model_manifest_hash\": \"manifest-sha\""),
            std::string::npos);
  EXPECT_NE(json.find("\"cost_profile_hash\": \"profile-sha\""),
            std::string::npos);
  EXPECT_NE(
      json.find("\"cost_profile_version\": \"slackpipe.cost_profile.v2\""),
      std::string::npos);
  EXPECT_NE(json.find("\"forward_costs\": [11, 52, 42, 263]"),
            std::string::npos);
  EXPECT_NE(json.find("\"backward_costs\": [24, 105, 85, 526]"),
            std::string::npos);
}
