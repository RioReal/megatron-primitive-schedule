#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "slackpipe/instance.h"

namespace slackpipe {

struct PressurePruningOptions {
  bool enabled = false;
  Index partition_top_k = 0;
  double partition_epsilon = -1.0;
  double lambda = 1.0;
  double gamma = 2.0;
  double alpha = 0.7;
  Index predecessor_top_k = 0;
  Index beam_width = 512;
  Index branch_width = 16;
  Index generated_partitions = 0;
};

struct PressurePruningStats {
  bool enabled = false;
  std::string generation_mode;
  std::int64_t generated_partitions = 0;
  std::int64_t partitions_before = 0;
  std::int64_t partitions_after = 0;
  Index beam_width = 0;
  Index branch_width = 0;
  bool exhaustive_enumeration_skipped = false;
  std::int64_t anchor_splits_added = 0;
  bool incumbent_split_included = false;
  bool uniform_split_included = false;
  bool cost_balanced_split_included = false;
  std::int64_t selected_splits_total = 0;
  double best_pressure = 0.0;
  double pressure_cutoff = 0.0;
  bool pressure_cutoff_available = false;
  std::int64_t predecessor_candidates_before = 0;
  std::int64_t predecessor_candidates_after = 0;
  std::string predecessor_note;
};

struct PressurePartitionCandidate {
  std::vector<Tick> split;
  double pressure = 0.0;
};

struct PressurePruningAnchors {
  bool has_incumbent_split = false;
  std::vector<Tick> incumbent_split;
  bool has_uniform_split = false;
  std::vector<Tick> uniform_split;
  bool has_cost_balanced_split = false;
  std::vector<Tick> cost_balanced_split;
};

[[nodiscard]] std::vector<double> ComputeStagePressures(
    const Instance& instance, const std::vector<Tick>& split,
    const PressurePruningOptions& options);

[[nodiscard]] double ComputePartitionPressure(
    const Instance& instance, const std::vector<Tick>& split,
    const PressurePruningOptions& options);

[[nodiscard]] std::vector<PressurePartitionCandidate>
GeneratePressureBeamPartitions(const Instance& instance,
                               const PressurePruningOptions& options);

[[nodiscard]] std::vector<Tick> BuildUniformPressureSplit(
    const Instance& instance);

[[nodiscard]] std::vector<Tick> BuildCostBalancedPressureSplit(
    const Instance& instance);

[[nodiscard]] std::vector<PressurePartitionCandidate> SelectPressurePartitions(
    const Instance& instance, const PressurePruningOptions& options,
    PressurePruningStats* stats = nullptr,
    const PressurePruningAnchors& anchors = {});

}  // namespace slackpipe
