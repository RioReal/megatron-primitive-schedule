#include <filesystem>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "slackpipe/benchmark.h"
#include "slackpipe/bfs_solver.h"
#include "slackpipe/breadth_first.h"
#include "slackpipe/dag_evaluator.h"
#include "slackpipe/joint_solver.h"
#include "slackpipe/operation.h"

namespace {

std::string TempPath(const std::string &name) {
  return (std::filesystem::temp_directory_path() / name).string();
}

slackpipe::benchmark::InstanceSpec TinyInstance() {
  slackpipe::benchmark::InstanceSpec spec;
  spec.name = "tiny";
  spec.instance.microbatches = 1;
  spec.instance.stages = 2;
  spec.instance.workers = 2;
  spec.instance.total_layers = 4;
  spec.instance.min_layers = 1;
  spec.instance.backward_ratio_num = 2;
  spec.instance.backward_ratio_den = 1;
  return spec;
}

std::vector<slackpipe::Tick> ParseSplitList(const std::string &text) {
  std::vector<slackpipe::Tick> values;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ';')) {
    if (!token.empty()) values.push_back(std::stoll(token));
  }
  return values;
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

struct ExhaustiveOracleResult {
  bool feasible = false;
  slackpipe::Tick makespan = std::numeric_limits<slackpipe::Tick>::max();
  std::vector<slackpipe::Tick> split;
  slackpipe::MachineOrders orders;
  slackpipe::Index candidates_checked = 0;
  slackpipe::Index feasible_candidates = 0;
};

void ConsiderOracleCandidate(const slackpipe::Instance &instance,
                             const std::vector<slackpipe::Tick> &split,
                             const slackpipe::MachineOrders &orders,
                             ExhaustiveOracleResult &oracle) {
  ++oracle.candidates_checked;
  const slackpipe::EvaluationResult evaluated =
      slackpipe::EvaluateSchedule(instance, split, orders);
  if (!evaluated.schedule.ok()) return;
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
    const slackpipe::Instance &instance,
    const slackpipe::MachineOrders &orders) {
  ExhaustiveOracleResult oracle;
  slackpipe::EnumerateValidSplits(
      instance, [&](const std::vector<slackpipe::Tick> &split) {
        ConsiderOracleCandidate(instance, split, orders, oracle);
      });
  return oracle;
}

ExhaustiveOracleResult ExhaustiveFixedSplitOrderOracle(
    const slackpipe::Instance &instance,
    const std::vector<slackpipe::Tick> &split) {
  ExhaustiveOracleResult oracle;
  for (const slackpipe::MachineOrders &orders :
       ExhaustiveMachineOrders(instance)) {
    ConsiderOracleCandidate(instance, split, orders, oracle);
  }
  return oracle;
}

ExhaustiveOracleResult ExhaustiveJointOracle(
    const slackpipe::Instance &instance) {
  ExhaustiveOracleResult oracle;
  const std::vector<slackpipe::MachineOrders> orders =
      ExhaustiveMachineOrders(instance);
  slackpipe::EnumerateValidSplits(
      instance, [&](const std::vector<slackpipe::Tick> &split) {
        for (const slackpipe::MachineOrders &order : orders) {
          ConsiderOracleCandidate(instance, split, order, oracle);
        }
      });
  return oracle;
}

void ExpectUtilizationConsistent(
    const slackpipe::Instance &instance,
    const slackpipe::benchmark::BenchmarkRow &row) {
  ASSERT_TRUE(row.makespan > 0);
  const double expected =
      static_cast<double>(slackpipe::TotalUsefulWork(instance)) /
      static_cast<double>(instance.workers * row.makespan);
  EXPECT_TRUE(std::abs(row.pipeline_utilization - expected) <= 1e-12);
}

}  // namespace

TEST(BenchmarkConfig, ParsesJsonAndCsv) {
  const std::string json = TempPath("slackpipe-benchmark-config.json");
  {
    std::ofstream out(json);
    out << R"({
      "mode": "algorithm-comparison",
      "repetitions": 2,
      "warmups": 1,
      "cp_sat_workers": [1, 2],
      "instances": [{"name":"a","B":4,"N":4,"J":4,"L":33,
        "ratio_num":2,"ratio_den":1,"min_layers":1}],
      "algorithm_configs": [
        {"algorithm":"slackpipe","configuration":"canonical-local",
         "split_mode":"local","move_budget":2}
      ]
    })";
  }
  const slackpipe::benchmark::BenchmarkConfig parsed =
      slackpipe::benchmark::ParseBenchmarkConfigFile(json);
  EXPECT_EQ(parsed.repetitions, 2);
  EXPECT_EQ(parsed.warmups, 1);
  ASSERT_EQ(parsed.cp_sat_workers.size(), std::size_t{2});
  ASSERT_EQ(parsed.instances.size(), std::size_t{1});
  ASSERT_EQ(parsed.algorithms.size(), std::size_t{1});

  const std::string csv = TempPath("slackpipe-benchmark-config.csv");
  {
    std::ofstream out(csv);
    out << "name,B,N,J,L,ratio_num,ratio_den,min_layers\n";
    out << "b,2,2,2,4,2,1,1\n";
  }
  EXPECT_EQ(
      slackpipe::benchmark::ParseBenchmarkConfigFile(csv).instances.size(),
      std::size_t{1});
}

TEST(BenchmarkConfig, CanonicalSlackPipeAliasesAndDefaultComparison) {
  EXPECT_TRUE(slackpipe::benchmark::ParseBenchmarkAlgorithm("slackpipe") ==
              slackpipe::benchmark::BenchmarkAlgorithm::kCanonicalSlackPipe);
  EXPECT_TRUE(slackpipe::benchmark::ParseBenchmarkAlgorithm("partition-only") ==
              slackpipe::benchmark::BenchmarkAlgorithm::kPartitionOnly);
  EXPECT_TRUE(
      slackpipe::benchmark::ParseBenchmarkAlgorithm("schedule-only-uniform") ==
      slackpipe::benchmark::BenchmarkAlgorithm::kScheduleOnly);
  const std::string json =
      TempPath("slackpipe-benchmark-default-comparison.json");
  {
    std::ofstream out(json);
    out << R"({
      "mode": "algorithm-comparison",
      "instances": [{"name":"a","B":2,"N":2,"J":2,"L":4,
        "ratio_num":2,"ratio_den":1,"min_layers":1}]
    })";
  }
  const slackpipe::benchmark::BenchmarkConfig parsed =
      slackpipe::benchmark::ParseBenchmarkConfigFile(json);
  ASSERT_EQ(parsed.algorithms.size(), std::size_t{2});
  EXPECT_TRUE(parsed.algorithms[0].algorithm ==
              slackpipe::benchmark::BenchmarkAlgorithm::kJointCpSat);
  EXPECT_TRUE(parsed.algorithms[1].algorithm ==
              slackpipe::benchmark::BenchmarkAlgorithm::kCanonicalSlackPipe);

  const std::string virtualized =
      TempPath("slackpipe-benchmark-virtualized-comparison.json");
  {
    std::ofstream out(virtualized);
    out << R"({
      "mode": "algorithm-comparison",
      "instances": [{"name":"v","B":8,"N":8,"J":4,"L":64,
        "ratio_num":2,"ratio_den":1,"min_layers":1}]
    })";
  }
  const slackpipe::benchmark::BenchmarkConfig virtualized_parsed =
      slackpipe::benchmark::ParseBenchmarkConfigFile(virtualized);
  ASSERT_EQ(virtualized_parsed.algorithms.size(), std::size_t{6});
  EXPECT_EQ(virtualized_parsed.algorithms[1].configuration,
            std::string("worker-fixed"));
  EXPECT_EQ(virtualized_parsed.algorithms[2].configuration,
            std::string("worker-local-1"));
  EXPECT_EQ(virtualized_parsed.algorithms[3].configuration,
            std::string("worker-local-2"));
  EXPECT_EQ(virtualized_parsed.algorithms[4].configuration,
            std::string("worker-local-4"));
  EXPECT_EQ(virtualized_parsed.algorithms[5].configuration,
            std::string("global"));
}

TEST(BenchmarkSerialization, EmitsCanonicalSchemaFields) {
  slackpipe::benchmark::BenchmarkRow row;
  row.run_id = "row-1";
  row.instance_name = "tiny";
  row.algorithm = slackpipe::benchmark::BenchmarkAlgorithm::kJointCpSat;
  row.configuration = "joint";
  row.repetition = 0;
  row.seed = 1;
  row.cp_sat_workers = 1;
  row.status = "FEASIBLE";
  row.ablation_mode = "joint";

  const slackpipe::benchmark::InstanceSpec instance = TinyInstance();
  slackpipe::JointOptimizationResult result;
  result.status = "FEASIBLE";
  result.joint_status = "UNKNOWN";
  result.fallback_used = true;
  result.diagnostic = "returned BFS incumbent after CP-SAT status=UNKNOWN";
  result.solution_source = "bfs_incumbent_fallback";
  result.best_bound_ticks = 1;
  row.canonical = slackpipe::BuildCanonicalResultMetadata(
      instance.instance, slackpipe::CanonicalRequestContext{},
      slackpipe::SemanticsForJointUnrestrictedNoOverlap(result, false),
      slackpipe::OutcomeFromJointResult(result), std::nullopt, std::nullopt);

  const std::string json = slackpipe::benchmark::ToJsonLine(row);
  EXPECT_TRUE(json.find("\"schema_version\": 1") != std::string::npos);
  EXPECT_TRUE(json.find("\"budget_policy_version\": 1") != std::string::npos);
  EXPECT_TRUE(
      json.find("\"canonical_method\": \"joint-unrestricted-no-overlap\"") !=
      std::string::npos);
  EXPECT_TRUE(json.find("\"solver_status_raw\": \"UNKNOWN\"") !=
              std::string::npos);
  EXPECT_TRUE(json.find("\"reported_status\": \"FEASIBLE\"") !=
              std::string::npos);
  EXPECT_EQ(json.find('\n'), std::string::npos);

  const std::string header = slackpipe::benchmark::ResultsCsvHeader();
  EXPECT_TRUE(header.find("canonical_method") != std::string::npos);
  EXPECT_TRUE(header.find("budget_policy_version") != std::string::npos);
  EXPECT_TRUE(header.find("predecessor_candidate_restriction_active") !=
              std::string::npos);
  const std::string csv = slackpipe::benchmark::ToCsvRow(row);
  EXPECT_TRUE(csv.find("joint-unrestricted-no-overlap") != std::string::npos);
}

TEST(BenchmarkSerialization, GitProvenanceDoesNotUseUnknownPlaceholder) {
  const std::string metadata = slackpipe::benchmark::HostMetadataJson();
  EXPECT_TRUE(metadata.find("\"git_commit\": \"unknown\"") ==
              std::string::npos);
  EXPECT_TRUE(metadata.find("\"dirty_tree\": \"unknown\"") ==
              std::string::npos);
}

TEST(BenchmarkRun, ScheduleOnlyBestRowsEnforceRestrictions) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  slackpipe::benchmark::BenchmarkConfig config;
  config.mode = slackpipe::benchmark::ExperimentMode::kAlgorithmComparison;
  config.timeout_seconds = 10.0;
  config.random_seed_base = 1;
  config.cp_sat_workers = {1};
  config.parallel_instances = 1;
  config.allow_oversubscription = true;

  slackpipe::benchmark::InstanceSpec instance;
  instance.name = "ablation-witness";
  instance.instance.microbatches = 4;
  instance.instance.stages = 2;
  instance.instance.workers = 2;
  instance.instance.total_layers = 4;
  instance.instance.min_layers = 1;
  instance.instance.backward_ratio_num = 1;
  instance.instance.backward_ratio_den = 1;
  instance.instance.communication_ticks = 3;
  config.instances.push_back(instance);

  slackpipe::benchmark::AlgorithmSpec joint_spec;
  joint_spec.algorithm = slackpipe::benchmark::BenchmarkAlgorithm::kJointCpSat;
  joint_spec.configuration = "joint";
  config.algorithms.push_back(joint_spec);

  slackpipe::benchmark::AlgorithmSpec partition_spec;
  partition_spec.algorithm =
      slackpipe::benchmark::BenchmarkAlgorithm::kPartitionOnly;
  partition_spec.configuration = "partition-only";
  config.algorithms.push_back(partition_spec);

  slackpipe::benchmark::AlgorithmSpec schedule_uniform;
  schedule_uniform.algorithm =
      slackpipe::benchmark::BenchmarkAlgorithm::kScheduleOnly;
  schedule_uniform.configuration = "schedule-only-uniform";
  schedule_uniform.fixed_partition_source = "uniform";
  config.algorithms.push_back(schedule_uniform);

  slackpipe::benchmark::AlgorithmSpec schedule_balanced = schedule_uniform;
  schedule_balanced.configuration = "schedule-only-load-balanced";
  schedule_balanced.fixed_partition_source = "load-balanced";
  config.algorithms.push_back(schedule_balanced);

  slackpipe::benchmark::AlgorithmSpec schedule_partition = schedule_uniform;
  schedule_partition.configuration = "schedule-only-partition-only";
  schedule_partition.fixed_partition_source = "partition-only";
  config.algorithms.push_back(schedule_partition);

  const std::string dir = TempPath("slackpipe-benchmark-primary-ablation");
  std::filesystem::remove_all(dir);
  const std::vector<slackpipe::benchmark::BenchmarkRow> rows =
      slackpipe::benchmark::RunBenchmark(config, dir);
  ASSERT_EQ(rows.size(), std::size_t{6});

  std::map<std::string, slackpipe::benchmark::BenchmarkRow> by_config;
  for (const slackpipe::benchmark::BenchmarkRow &row : rows) {
    by_config[row.configuration] = row;
    EXPECT_TRUE(row.completed);
    EXPECT_TRUE(row.status == "OPTIMAL" || row.status == "FEASIBLE");
    EXPECT_TRUE(row.label_validation_passed);
    EXPECT_TRUE(row.canonical_validation_passed);
    EXPECT_EQ(row.simulated_iteration_time, row.makespan);
    EXPECT_TRUE(row.pipeline_utilization > 0.0);
    EXPECT_FALSE(row.worker_local_operation_order.empty());
  }

  ASSERT_TRUE(by_config.count("joint") != 0);
  ASSERT_TRUE(by_config.count("partition-only") != 0);
  ASSERT_TRUE(by_config.count("schedule-only-uniform") != 0);
  ASSERT_TRUE(by_config.count("schedule-only-load-balanced") != 0);
  ASSERT_TRUE(by_config.count("schedule-only-partition-only") != 0);
  ASSERT_TRUE(by_config.count("schedule-only-best") != 0);

  const auto &joint = by_config.at("joint");
  const auto &partition_only = by_config.at("partition-only");
  const auto &schedule_partition_row =
      by_config.at("schedule-only-partition-only");
  const auto &schedule_best = by_config.at("schedule-only-best");
  const slackpipe::Instance &model = instance.instance;
  const std::vector<slackpipe::Tick> schedule_partition_split =
      ParseSplitList(schedule_partition_row.split);
  const std::vector<slackpipe::Tick> schedule_best_split =
      ParseSplitList(schedule_best.split);

  const slackpipe::MachineOrders breadth_first =
      slackpipe::BreadthFirstOrders(model);
  const ExhaustiveOracleResult fixed_order_oracle =
      ExhaustiveFixedOrderOracle(model, breadth_first);
  ASSERT_TRUE(fixed_order_oracle.feasible);
  ASSERT_EQ(fixed_order_oracle.makespan, 28);
  const slackpipe::EvaluationResult schedule_partition_fixed_order =
      slackpipe::EvaluateSchedule(model, schedule_partition_split,
                                  breadth_first);
  ASSERT_TRUE(schedule_partition_fixed_order.schedule.ok());
  EXPECT_EQ(schedule_partition_fixed_order.schedule.makespan,
            fixed_order_oracle.makespan);

  const ExhaustiveOracleResult schedule_partition_order_oracle =
      ExhaustiveFixedSplitOrderOracle(model, schedule_partition_split);
  ASSERT_TRUE(schedule_partition_order_oracle.feasible);
  ASSERT_EQ(schedule_partition_order_oracle.makespan, 24);
  const ExhaustiveOracleResult schedule_best_order_oracle =
      ExhaustiveFixedSplitOrderOracle(model, schedule_best_split);
  ASSERT_TRUE(schedule_best_order_oracle.feasible);
  EXPECT_EQ(schedule_best_order_oracle.makespan,
            schedule_partition_order_oracle.makespan);

  const ExhaustiveOracleResult joint_oracle = ExhaustiveJointOracle(model);
  ASSERT_TRUE(joint_oracle.feasible);
  ASSERT_EQ(joint_oracle.candidates_checked, 2940);
  ASSERT_EQ(joint_oracle.makespan, 24);

  EXPECT_TRUE(joint.partition_search_enabled);
  EXPECT_TRUE(joint.schedule_search_enabled);
  EXPECT_EQ(joint.makespan, joint_oracle.makespan);
  EXPECT_EQ(partition_only.makespan, fixed_order_oracle.makespan);
  EXPECT_EQ(schedule_best.makespan, schedule_best_order_oracle.makespan);

  EXPECT_TRUE(partition_only.partition_search_enabled);
  EXPECT_FALSE(partition_only.schedule_search_enabled);
  EXPECT_FALSE(partition_only.operation_order_changed);
  EXPECT_TRUE(partition_only.operation_order_fixed_validation_passed);

  for (const std::string name :
       {"schedule-only-uniform", "schedule-only-load-balanced",
        "schedule-only-partition-only", "schedule-only-best"}) {
    const auto &row = by_config.at(name);
    EXPECT_FALSE(row.partition_search_enabled);
    EXPECT_TRUE(row.schedule_search_enabled);
    EXPECT_FALSE(row.partition_changed);
    EXPECT_TRUE(row.partition_fixed_validation_passed);
    EXPECT_EQ(row.supplied_split, row.split);
  }
  EXPECT_EQ(schedule_best.fixed_partition_source,
            schedule_partition_row.fixed_partition_source);
  EXPECT_EQ(schedule_best.supplied_split,
            schedule_partition_row.supplied_split);
  EXPECT_EQ(schedule_best.split, schedule_partition_row.split);
  EXPECT_EQ(schedule_best.makespan, schedule_partition_row.makespan);

  EXPECT_EQ(schedule_partition_row.fixed_partition_source,
            std::string("partition_only_within_global_budget"));
  EXPECT_EQ(schedule_partition_row.canonical.semantics
                .partition_reference_source.value_or(""),
            std::string("partition_only_within_global_budget"));
  ASSERT_TRUE(schedule_partition_row.canonical.outcome.phase_budget
                  .reference_phase_limit_seconds);
  EXPECT_TRUE(*schedule_partition_row.canonical.outcome.phase_budget
                   .reference_phase_limit_seconds <=
              0.5 * config.timeout_seconds);
  EXPECT_FALSE(schedule_partition_row.joint_optimum_when_available);
  EXPECT_TRUE(schedule_best.canonical.outcome.result_validation);
  EXPECT_TRUE(schedule_best.canonical.outcome.result_validation->passed);
  EXPECT_EQ(schedule_best.canonical.outcome.result_validation
                ->reconstructed_makespan.value_or(0),
            schedule_best.makespan);
  EXPECT_TRUE(schedule_best.canonical.activation_analysis);
  EXPECT_TRUE(schedule_best.canonical.activation_analysis->passed);
  EXPECT_FALSE(schedule_best.canonical.semantics.partition_optimized);
  EXPECT_TRUE(schedule_best.canonical.semantics.schedule_optimized);
  EXPECT_TRUE(schedule_best.canonical.semantics.full_partition_fixed);
  EXPECT_FALSE(schedule_best.canonical.semantics
                   .predecessor_candidate_restriction_active);
  EXPECT_EQ(
      schedule_best.canonical.semantics.predecessor_candidate_rule.value_or(""),
      std::string("unrestricted_no_overlap"));
  EXPECT_EQ(schedule_best.canonical.budget_policy_version, 1);
  EXPECT_FALSE(schedule_best.fallback_used);
  EXPECT_TRUE(schedule_best.pipeline_utilization >
              partition_only.pipeline_utilization);
  EXPECT_TRUE(joint.makespan <= schedule_best.makespan);
  if (joint.makespan == schedule_best.makespan) {
    EXPECT_EQ(joint.pipeline_utilization, schedule_best.pipeline_utilization);
  } else {
    EXPECT_TRUE(joint.pipeline_utilization >
                schedule_best.pipeline_utilization);
  }
  ExpectUtilizationConsistent(model, joint);
  ExpectUtilizationConsistent(model, partition_only);
  ExpectUtilizationConsistent(model, schedule_best);
  EXPECT_TRUE(joint.utilization_ranking_validation_passed);
  EXPECT_TRUE(partition_only.utilization_ranking_validation_passed);
  EXPECT_TRUE(schedule_best.utilization_ranking_validation_passed);

  const std::string header = slackpipe::benchmark::ResultsCsvHeader();
  EXPECT_TRUE(header.find("simulated_iteration_time") != std::string::npos);
  EXPECT_TRUE(header.find("pipeline_utilization") != std::string::npos);
  EXPECT_TRUE(header.find("worker_local_operation_order") != std::string::npos);

  EXPECT_EQ(by_config.at("schedule-only-uniform").makespan, 26);
}

TEST(BenchmarkConfig, PrimaryAblationConfigDisablesScheduleOnlyBest) {
  const std::string json = TempPath("slackpipe-primary-ablation-config.json");
  {
    std::ofstream out(json);
    out << R"({
      "mode": "algorithm-comparison",
      "derive_schedule_only_best": false,
      "instances": [{"name":"a","B":2,"N":2,"J":2,"L":4,
        "ratio_num":2,"ratio_den":1,"min_layers":1}],
      "algorithms": ["joint", "partition-only", "schedule-only-uniform"]
    })";
  }

  const slackpipe::benchmark::BenchmarkConfig parsed =
      slackpipe::benchmark::ParseBenchmarkConfigFile(json);
  EXPECT_FALSE(parsed.derive_schedule_only_best);
  ASSERT_EQ(parsed.algorithms.size(), std::size_t{3});
  EXPECT_EQ(parsed.algorithms[2].configuration,
            std::string("schedule-only-uniform"));
  EXPECT_EQ(parsed.algorithms[2].fixed_partition_source,
            std::string("uniform"));
}

TEST(BenchmarkRun, HintLabelsValidateAgainstEffectivePolicy) {
  if (!slackpipe::IsJointOptimizerAvailable()) {
    return;
  }

  slackpipe::benchmark::BenchmarkConfig config;
  config.mode = slackpipe::benchmark::ExperimentMode::kAlgorithmComparison;
  config.timeout_seconds = 2.0;
  config.random_seed_base = 1;
  config.cp_sat_workers = {1};
  config.parallel_instances = 1;
  config.allow_oversubscription = true;
  config.derive_schedule_only_best = false;

  slackpipe::benchmark::InstanceSpec instance;
  instance.name = "hint-label";
  instance.instance.microbatches = 4;
  instance.instance.stages = 4;
  instance.instance.workers = 2;
  instance.instance.total_layers = 8;
  instance.instance.min_layers = 1;
  instance.instance.backward_ratio_num = 2;
  instance.instance.backward_ratio_den = 1;
  config.instances.push_back(instance);

  slackpipe::benchmark::AlgorithmSpec no_hint;
  no_hint.algorithm = slackpipe::benchmark::BenchmarkAlgorithm::kJointCpSat;
  no_hint.configuration = "joint-no-hint";
  no_hint.use_bfs_hints = false;
  config.algorithms.push_back(no_hint);

  slackpipe::benchmark::AlgorithmSpec with_hints;
  with_hints.algorithm = slackpipe::benchmark::BenchmarkAlgorithm::kJointCpSat;
  with_hints.configuration = "joint-with-hints";
  with_hints.use_bfs_hints = true;
  config.algorithms.push_back(with_hints);

  const std::string dir = TempPath("slackpipe-benchmark-hint-labels");
  std::filesystem::remove_all(dir);
  const std::vector<slackpipe::benchmark::BenchmarkRow> rows =
      slackpipe::benchmark::RunBenchmark(config, dir);
  ASSERT_EQ(rows.size(), std::size_t{2});

  std::map<std::string, slackpipe::benchmark::BenchmarkRow> by_config;
  for (const slackpipe::benchmark::BenchmarkRow &row : rows) {
    ASSERT_TRUE(row.status == "OPTIMAL" || row.status == "FEASIBLE");
    EXPECT_TRUE(row.label_validation_passed);
    by_config[row.configuration] = row;
  }
  ASSERT_TRUE(by_config.count("joint-no-hint") != 0);
  ASSERT_TRUE(by_config.count("joint-with-hints") != 0);
  EXPECT_FALSE(by_config.at("joint-no-hint").hints_requested);
  EXPECT_FALSE(by_config.at("joint-no-hint").hints_effective);
  EXPECT_TRUE(by_config.at("joint-with-hints").hints_requested);
  EXPECT_TRUE(by_config.at("joint-with-hints").hints_effective);

  std::ifstream jsonl(dir + "/results.jsonl");
  std::string contents((std::istreambuf_iterator<char>(jsonl)),
                       std::istreambuf_iterator<char>());
  EXPECT_TRUE(contents.find("\"configuration\":\"joint-no-hint\"") !=
              std::string::npos);
  EXPECT_TRUE(contents.find("\"hints_effective\":false") != std::string::npos);
}

TEST(BenchmarkConfig, SlackPipeLabelsValidateAgainstAdvertisedModes) {
  if (!slackpipe::IsSlackPipeSolverAvailable()) {
    return;
  }

  slackpipe::benchmark::BenchmarkConfig config;
  config.mode = slackpipe::benchmark::ExperimentMode::kLatency;
  config.timeout_seconds = 10.0;
  config.random_seed_base = 1;
  config.cp_sat_workers = {1};
  config.parallel_instances = 1;
  config.allow_oversubscription = true;

  slackpipe::benchmark::InstanceSpec instance;
  instance.name = "label-integrity";
  instance.instance.microbatches = 2;
  instance.instance.stages = 4;
  instance.instance.workers = 2;
  instance.instance.total_layers = 8;
  instance.instance.min_layers = 1;
  instance.instance.backward_ratio_num = 2;
  instance.instance.backward_ratio_den = 1;
  config.instances.push_back(instance);

  slackpipe::benchmark::AlgorithmSpec worker_fixed;
  worker_fixed.algorithm =
      slackpipe::benchmark::BenchmarkAlgorithm::kCanonicalSlackPipe;
  worker_fixed.configuration = "worker-fixed";
  worker_fixed.split_mode = slackpipe::SlackPipeSplitMode::kWorkerFixed;
  config.algorithms.push_back(worker_fixed);

  slackpipe::benchmark::AlgorithmSpec worker_local;
  worker_local.algorithm =
      slackpipe::benchmark::BenchmarkAlgorithm::kCanonicalSlackPipe;
  worker_local.configuration = "worker-local-1";
  worker_local.split_mode = slackpipe::SlackPipeSplitMode::kWorkerLocal;
  worker_local.worker_move_budget = 1;
  worker_local.worker_move_budget_provided = true;
  config.algorithms.push_back(worker_local);

  slackpipe::benchmark::AlgorithmSpec global;
  global.algorithm =
      slackpipe::benchmark::BenchmarkAlgorithm::kCanonicalSlackPipe;
  global.configuration = "global";
  global.split_mode = slackpipe::SlackPipeSplitMode::kGlobal;
  config.algorithms.push_back(global);

  const std::string dir = TempPath("slackpipe-benchmark-label-integrity");
  std::filesystem::remove_all(dir);
  const std::vector<slackpipe::benchmark::BenchmarkRow> rows =
      slackpipe::benchmark::RunBenchmark(config, dir);
  ASSERT_EQ(rows.size(), std::size_t{3});

  slackpipe::BfsSplitOptimizerOptions bfs_options;
  bfs_options.time_limit_seconds = config.timeout_seconds;
  bfs_options.num_workers = 1;
  bfs_options.random_seed = config.random_seed_base;
  bfs_options.require_optimal = false;
  const slackpipe::BfsSplitOptimizationResult reference =
      slackpipe::OptimizeBfsSplitAuto(instance.instance, bfs_options);
  ASSERT_TRUE(reference.status == "OPTIMAL" || reference.status == "FEASIBLE");

  for (const slackpipe::benchmark::BenchmarkRow &row : rows) {
    ASSERT_TRUE(row.status == "OPTIMAL" || row.status == "FEASIBLE");
    const std::vector<slackpipe::Tick> split = ParseSplitList(row.split);
    slackpipe::SlackPipeOptions options;
    if (row.configuration == "worker-fixed") {
      options.split_mode = slackpipe::SlackPipeSplitMode::kWorkerFixed;
    } else if (row.configuration == "worker-local-1") {
      options.split_mode = slackpipe::SlackPipeSplitMode::kWorkerLocal;
      options.worker_move_budget = 1;
      options.worker_move_budget_provided = true;
    } else if (row.configuration == "global") {
      options.split_mode = slackpipe::SlackPipeSplitMode::kGlobal;
    } else {
      EXPECT_TRUE(false);
      continue;
    }
    EXPECT_TRUE(slackpipe::SplitSatisfiesSlackPipeMode(
        instance.instance, split, reference.split, options));
  }
}

TEST(BenchmarkIds, DeterministicAndSensitive) {
  const slackpipe::benchmark::InstanceSpec instance = TinyInstance();
  slackpipe::benchmark::AlgorithmSpec algorithm;
  algorithm.algorithm = slackpipe::benchmark::BenchmarkAlgorithm::kJointCpSat;
  const std::string first = slackpipe::benchmark::DeterministicRunId(
      instance, algorithm, 0, 7, 1, 1.0);
  const std::string second = slackpipe::benchmark::DeterministicRunId(
      instance, algorithm, 0, 7, 1, 1.0);
  const std::string third = slackpipe::benchmark::DeterministicRunId(
      instance, algorithm, 1, 7, 1, 1.0);
  EXPECT_EQ(first, second);
  EXPECT_NE(first, third);
}

TEST(BenchmarkResume, CompleteRowsOnlyAndRecovery) {
  const std::string dir = TempPath("slackpipe-benchmark-recovery");
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::string jsonl = dir + "/results.jsonl";
  {
    std::ofstream out(jsonl);
    out << "{\"run_id\":\"a\",\"completed\":true}\n";
    out << "{\"run_id\":\"b\",\"completed\":false}\n";
    out << "{\"run_id\":\"truncated\",\"completed\":true";
  }
  slackpipe::benchmark::RecoverIncrementalOutputs(dir);
  const auto ids = slackpipe::benchmark::LoadCompleteRunIds(jsonl);
  EXPECT_TRUE(ids.count("a") != 0);
  EXPECT_TRUE(ids.count("b") == 0);
  std::ifstream in(jsonl);
  std::string contents((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
  EXPECT_EQ(contents.find("truncated"), std::string::npos);
}

TEST(BenchmarkCorrectness, GapAndFeasibleAccounting) {
  const slackpipe::benchmark::GapMetrics gap =
      slackpipe::benchmark::CalculateGap(100, 110);
  EXPECT_TRUE(gap.joint_optimum_when_available);
  EXPECT_TRUE(std::abs(gap.absolute_gap_to_joint - 10.0) < 1e-12);
  EXPECT_TRUE(std::abs(gap.relative_gap_to_joint - 0.1) < 1e-12);
  EXPECT_FALSE(gap.reaches_joint_objective);

  slackpipe::benchmark::BenchmarkRow feasible;
  feasible.status = "FEASIBLE";
  feasible.proven_optimal = false;
  feasible.total_planning_seconds = 1.0;
  const slackpipe::benchmark::SummaryStats stats =
      slackpipe::benchmark::ComputeSummaryStats({feasible});
  EXPECT_EQ(stats.optimal_count, 0);
  EXPECT_EQ(stats.timeout_count, 1);
}

TEST(BenchmarkCpuBudget, EnforcesWorkerBudget) {
  slackpipe::benchmark::BenchmarkConfig config;
  config.parallel_instances = 2;
  config.cp_sat_workers = {4};
  config.configurable_cpu_budget = 7;
  EXPECT_THROW(slackpipe::benchmark::ValidateCpuBudget(config, 16),
               slackpipe::Error);
  config.allow_oversubscription = true;
  slackpipe::benchmark::ValidateCpuBudget(config, 16);
  config.cp_sat_workers = {17};
  EXPECT_THROW(slackpipe::benchmark::ValidateCpuBudget(config, 16),
               slackpipe::Error);
}

TEST(BenchmarkSuites, DeterministicAndRejectInvalid) {
  const auto first = slackpipe::benchmark::GenerateNamedSuite("scaling-B");
  const auto second = slackpipe::benchmark::GenerateNamedSuite("scaling-B");
  ASSERT_EQ(first.size(), second.size());
  for (std::size_t i = 0; i < first.size(); ++i) {
    EXPECT_EQ(first[i].name, second[i].name);
    first[i].instance.Validate();
  }
  const std::string invalid = TempPath("slackpipe-benchmark-invalid.json");
  {
    std::ofstream out(invalid);
    out << R"({"instances":[{"name":"bad","B":1,"N":4,"J":2,"L":3,
      "ratio_num":2,"ratio_den":1,"min_layers":1}]})";
  }
  EXPECT_THROW(
      {
        const auto parsed =
            slackpipe::benchmark::ParseBenchmarkConfigFile(invalid);
        (void)parsed;
      },
      slackpipe::Error);
}

TEST(BenchmarkStats, SummaryAndPairedRatios) {
  std::vector<slackpipe::benchmark::BenchmarkRow> rows(4);
  rows[0].instance_name = "a";
  rows[0].algorithm = slackpipe::benchmark::BenchmarkAlgorithm::kJointCpSat;
  rows[0].repetition = 0;
  rows[0].seed = 1;
  rows[0].cp_sat_workers = 1;
  rows[0].status = "OPTIMAL";
  rows[0].proven_optimal = true;
  rows[0].total_planning_seconds = 2.0;
  rows[1] = rows[0];
  rows[1].algorithm =
      slackpipe::benchmark::BenchmarkAlgorithm::kCanonicalSlackPipe;
  rows[1].total_planning_seconds = 4.0;
  rows[2] = rows[0];
  rows[2].instance_name = "b";
  rows[2].total_planning_seconds = 8.0;
  rows[3] = rows[1];
  rows[3].instance_name = "b";
  rows[3].total_planning_seconds = 4.0;

  const slackpipe::benchmark::SummaryStats stats =
      slackpipe::benchmark::ComputeSummaryStats(rows);
  EXPECT_EQ(stats.count, 4);
  EXPECT_EQ(stats.optimal_count, 4);
  EXPECT_TRUE(std::abs(stats.minimum - 2.0) < 1e-12);
  EXPECT_TRUE(std::abs(stats.maximum - 8.0) < 1e-12);

  const auto ratios = slackpipe::benchmark::ComputePairedRuntimeRatios(
      rows, slackpipe::benchmark::BenchmarkAlgorithm::kCanonicalSlackPipe,
      slackpipe::benchmark::BenchmarkAlgorithm::kJointCpSat);
  EXPECT_EQ(ratios.size(), std::size_t{3});
  EXPECT_TRUE(std::abs(ratios.at("a:0:1:1") - 2.0) < 1e-12);
  EXPECT_TRUE(std::abs(ratios.at("b:0:1:1") - 0.5) < 1e-12);
}

TEST(BenchmarkRun, RecordsCrashRows) {
  slackpipe::benchmark::BenchmarkConfig config;
  slackpipe::benchmark::InstanceSpec invalid = TinyInstance();
  invalid.instance.total_layers = 1;
  config.instances.push_back(invalid);
  slackpipe::benchmark::AlgorithmSpec algorithm;
  algorithm.algorithm = slackpipe::benchmark::BenchmarkAlgorithm::kJointCpSat;
  config.algorithms.push_back(algorithm);
  config.mode = slackpipe::benchmark::ExperimentMode::kLatency;
  config.cp_sat_workers = {1};
  config.repetitions = 1;
  config.configurable_cpu_budget = 1;
  config.timeout_seconds = 0.01;
  const std::string dir = TempPath("slackpipe-benchmark-run");
  std::filesystem::remove_all(dir);
  const auto rows = slackpipe::benchmark::RunBenchmark(config, dir);
  ASSERT_EQ(rows.size(), std::size_t{1});
  EXPECT_TRUE(rows[0].completed);
  EXPECT_EQ(rows[0].status, std::string("CRASHED"));
  EXPECT_TRUE(std::filesystem::exists(dir + "/failures.jsonl"));
}
