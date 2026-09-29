#include <algorithm>
#include <cmath>
#include <filesystem>

#include <gtest/gtest.h>

#include "slackpipe/cost_profile.h"
#include "slackpipe/io.h"
#include "slackpipe/joint_solver.h"
#include "slackpipe/slackpipe_solver.h"
#include "slackpipe/workload_bound.h"

namespace {
using namespace slackpipe;

Instance Tiny() {
  Instance i;
  i.microbatches = 2; i.stages = 2; i.workers = 2; i.total_layers = 4;
  i.backward_ratio_num = 2;
  return i;
}

Instance Affine() {
  auto i = Tiny();
  i.profile_forward_slope_ticks = {2, 2};
  i.profile_backward_slope_ticks = {3, 3};
  i.profile_forward_bias_ticks = {1, 2};
  i.profile_backward_bias_ticks = {3, 4};
  return i;
}

Instance Range() {
  auto i = Tiny();
  i.profile_prefix_forward_ticks = {0, 1, 3, 6, 10};
  i.profile_prefix_backward_ticks = {0, 2, 6, 12, 20};
  i.profile_role_forward_bias_ticks = {1, 9, 2};
  i.profile_role_backward_bias_ticks = {1, 9, 3};
  return i;
}

Tick Work(const Instance& i, const std::vector<Tick>& split) {
  Tick total = 0;
  for (Index s = 0; s < i.stages; ++s) {
    total += i.microbatches * (i.Duration(s, false, split) + i.Duration(s, true, split));
  }
  return total;
}

TEST(WorkloadBound, InvariantCostsMatchEveryPartition) {
  for (const auto& i : {Tiny(), Affine(), Range()}) {
    const auto bound = ComputeWorkloadLowerBound(i);
    ASSERT_TRUE(bound.total_work_ticks.has_value());
    ASSERT_TRUE(bound.makespan_ticks.has_value());
    EXPECT_TRUE(bound.partition_invariant);
    for (Tick cut = 1; cut < i.total_layers; ++cut) {
      const Tick work = Work(i, {cut, i.total_layers - cut});
      EXPECT_EQ(*bound.total_work_ticks, work);
      EXPECT_EQ(*bound.makespan_ticks, work / i.workers + (work % i.workers != 0));
    }
  }
}

TEST(WorkloadBound, VariableAffineCostsUseGlobalMinimumNotIncumbentWork) {
  auto i = Affine();
  i.profile_forward_slope_ticks = {1, 10};
  i.profile_backward_slope_ticks = {2, 20};
  const auto bound = ComputeWorkloadLowerBound(i);
  ASSERT_TRUE(bound.total_work_ticks.has_value());
  EXPECT_FALSE(bound.partition_invariant);
  Tick minimum = kTickMax;
  for (Tick cut = 1; cut < i.total_layers; ++cut) {
    minimum = std::min(minimum, Work(i, {cut, i.total_layers - cut}));
  }
  EXPECT_EQ(*bound.total_work_ticks, minimum);
  EXPECT_TRUE(*bound.total_work_ticks < Work(i, {1, 3}));
  // Fixed combined slopes suffice even when each direction varies by stage.
  i.profile_backward_slope_ticks = {20, 11};
  EXPECT_TRUE(ComputeWorkloadLowerBound(i).partition_invariant);
}

TEST(WorkloadBound, PrefixRoundingRoleBiasAndZeroDurationSemantics) {
  auto i = Tiny();
  const auto path = std::filesystem::temp_directory_path() / "slackpipe_workload_rounding_test.json";
  WriteTextFile(path.string(), R"({"schema_version":"slackpipe.cost_profile.v2",
    "prefix_forward_us":[0,0.1,0.2,0.4,1.6],
    "prefix_backward_us":[0,0.2,0.3,0.4,2.5],
    "stage_role_bias_us":{"first":{"forward":0.1,"backward":0},
      "middle":{"forward":99,"backward":99},"last":{"forward":0,"backward":0.1}}})");
  ApplyCostProfileFile(i, path.string());
  std::filesystem::remove(path);
  EXPECT_TRUE(i.profile_prefix_forward_ticks == (std::vector<Tick>{0, 1, 1, 1, 2}));
  EXPECT_TRUE(i.profile_prefix_backward_ticks == (std::vector<Tick>{0, 1, 1, 1, 3}));
  EXPECT_EQ(i.BackwardDuration(0, 1, 2), 0);  // No post-difference min-one clamp.
  auto bound = ComputeWorkloadLowerBound(i);
  ASSERT_TRUE(bound.total_work_ticks.has_value());
  EXPECT_EQ(*bound.total_work_ticks, 14);
  i.stages = 1;
  bound = ComputeWorkloadLowerBound(i);
  EXPECT_EQ(*bound.total_work_ticks, 12);  // N=1 uses only the first role.
  i.microbatches = 1; i.workers = 4;
  bound = ComputeWorkloadLowerBound(i);
  EXPECT_EQ(*bound.makespan_ticks, 2);  // ceil(6/4), not floating rounding.
}

TEST(WorkloadBound, OverflowIsExplicitlyUnavailable) {
  auto i = Tiny();
  i.backward_ratio_num = kTickMax;
  const auto bound = ComputeWorkloadLowerBound(i);
  EXPECT_FALSE(bound.total_work_ticks.has_value());
  EXPECT_FALSE(bound.makespan_ticks.has_value());
  EXPECT_FALSE(bound.reason.empty());
}

TEST(WorkloadBound, ReportingPreservesRawBoundStatusAndLegacyGap) {
  const auto i = Range();
  JointOptimizationResult result;
  result.status = "FEASIBLE"; result.joint_status = "UNKNOWN";
  result.fallback_used = true; result.makespan_ticks = 54; result.best_bound_ticks = 10;
  const auto report = ComputeLowerBoundReport(i, result.best_bound_ticks, result.makespan_ticks);
  EXPECT_EQ(*report.raw_solver_bound_ticks, 10);
  EXPECT_EQ(*report.workload.makespan_ticks, 37);
  EXPECT_EQ(*report.effective_lower_bound_ticks, 37);
  EXPECT_TRUE(std::abs(*report.effective_relative_gap - 17.0 / 54) < 1e-12);
  const auto raw = OutcomeFromJointResult(result);
  EXPECT_TRUE(std::abs(*raw.relative_optimality_gap - 44.0 / 54) < 1e-12);
  const auto json = ToJson(i, result);
  EXPECT_TRUE(json.find("\"effective_lower_bound_ticks\":37") != std::string::npos);
  EXPECT_EQ(result.status, "FEASIBLE");
  EXPECT_EQ(result.joint_status, "UNKNOWN");
  EXPECT_FALSE(result.proven_optimal);
  auto canonical = BuildCanonicalResultMetadata(i, {}, {}, raw, std::nullopt, std::nullopt);
  canonical.outcome.feasible = false;  // Later result validation may reject it.
  EXPECT_TRUE(CanonicalResultToJson(canonical, "").find("\"effective_relative_optimality_gap\":null") !=
              std::string::npos);
  result.canonical = canonical;
  EXPECT_TRUE(ToJson(i, result).find("\n  \"lower_bound_report\":") != std::string::npos);
  EXPECT_TRUE(ToJson(i, result).find("\n  \"lower_bound_report\": " +
      LowerBoundReportJson(ComputeLowerBoundReport(i, raw.best_objective_bound,
                                                  std::nullopt, false))) != std::string::npos);
  // A deadline fallback before Solve() has no raw solver bound, not zero.
  canonical.lower_bound_report.raw_solver_bound_ticks = std::nullopt;
  result.canonical = canonical;
  EXPECT_TRUE(ToJson(i, result).find("\n  \"lower_bound_report\": {\"raw_solver_bound_ticks\":null") !=
              std::string::npos);
  EXPECT_EQ(*ComputeLowerBoundReport(i, 0, 54).effective_lower_bound_ticks, 37);
  EXPECT_FALSE(ComputeLowerBoundReport(i, 10, std::nullopt).effective_relative_gap.has_value());
  const auto inconsistent = ComputeLowerBoundReport(i, 100, 54);
  EXPECT_TRUE(inconsistent.inconsistent_with_incumbent);
  EXPECT_FALSE(inconsistent.effective_relative_gap.has_value());
}

TEST(WorkloadBound, JointOptimaUnchangedAndProfileCapacityConstraintsPresent) {
  if (!IsJointOptimizerAvailable()) return;
  Index index = 0;
  for (const auto& i : {Tiny(), Affine(), Range()}) {
    JointOptimizerOptions options;
    options.time_limit_seconds = 10;
    options.worker_balance_pruning = false;
    const auto result = OptimizeJointSplitAndScheduleCpSat(i, options);
    ASSERT_TRUE(result.proven_optimal);
    EXPECT_EQ(result.makespan_ticks, (std::vector<Tick>{18, 42, 54})[index++]);
    EXPECT_EQ(result.workload_constraint_count, i.workers + 1);
    const auto bound = ComputeWorkloadLowerBound(i);
    ASSERT_TRUE(bound.makespan_ticks.has_value());
    EXPECT_TRUE(*bound.makespan_ticks <= result.makespan_ticks);
    EXPECT_TRUE(AnalyticalGlobalLowerBound(i) <= result.makespan_ticks);
  }
}

TEST(WorkloadBound, ConditionalPhaseBoundsDoNotCertifyJointOptimality) {
  const auto instance = Range();
  CanonicalOutcome outcome;
  outcome.feasible = true;
  outcome.makespan = 63;
  outcome.best_objective_bound = 63;
  const auto metadata = BuildCanonicalResultMetadata(
      instance, {}, SemanticsForAlternatingPartitionSchedule(2), outcome,
      std::nullopt, std::nullopt);
  EXPECT_EQ(*metadata.lower_bound_report.raw_solver_bound_ticks, 63);
  EXPECT_FALSE(metadata.lower_bound_report.raw_bound_globally_valid);
  EXPECT_EQ(*metadata.lower_bound_report.effective_lower_bound_ticks, 37);
  EXPECT_TRUE(*metadata.lower_bound_report.effective_relative_gap > 0.4);
}
}  // namespace
