#pragma once

#include <optional>
#include <string>
#include <vector>

#include "slackpipe/bfs_solver.h"
#include "slackpipe/partition_restriction.h"

namespace slackpipe {

struct HybridSlackScores {
  std::vector<double> operation_scores;
  std::vector<double> forward_stage_scores;
  std::vector<double> backward_stage_scores;
  std::vector<double> stage_scores;
  double partition_min_slack = 0.0;
  std::vector<Index> bottleneck_stages;
  std::vector<Index> slack_rich_stages;
};

struct HybridSlackIncumbentOptions {
  double time_limit_seconds = 0.0;
  Index max_iterations = 128;
  Index partition_candidate_limit = 32;
  Index interleaving_candidate_limit = 64;
  std::optional<PartitionRestriction> partition_restriction;
};

struct HybridSlackCandidateSummary {
  Tick primary_objective = 0;
  double hybrid_min_slack = 0.0;
  std::vector<double> hybrid_stage_scores;
  std::vector<Tick> split;
  MachineOrders orders;
};

[[nodiscard]] HybridSlackScores ComputeHybridSlackScores(
    const Instance& instance, const std::vector<Tick>& split);

[[nodiscard]] bool HybridSlackCandidateBetter(
    const HybridSlackCandidateSummary& candidate,
    const HybridSlackCandidateSummary& incumbent);

[[nodiscard]] std::vector<std::vector<Tick>>
GenerateHybridSlackPartitionMoveCandidates(
    const Instance& instance, const std::vector<Tick>& split,
    const HybridSlackScores& scores,
    const std::optional<PartitionRestriction>& restriction = std::nullopt,
    Index limit = 32);

[[nodiscard]] std::vector<MachineOrders>
GenerateHybridSlackInterleavingMoveCandidates(const Instance& instance,
                                              const std::vector<Tick>& split,
                                              const MachineOrders& orders,
                                              const HybridSlackScores& scores,
                                              Index limit = 64);

[[nodiscard]] BfsSplitOptimizationResult BuildHybridSlackIncumbent(
    const Instance& instance, const HybridSlackIncumbentOptions& options = {});

}  // namespace slackpipe
