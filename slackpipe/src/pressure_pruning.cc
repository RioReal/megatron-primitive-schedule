#include "slackpipe/pressure_pruning.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <set>

#include "slackpipe/bfs_solver.h"

namespace slackpipe {

std::vector<Tick> BuildUniformPressureSplit(const Instance& instance) {
  instance.Validate();
  std::vector<Tick> split(static_cast<std::size_t>(instance.stages),
                          instance.min_layers);
  Tick remaining =
      instance.total_layers - instance.min_layers * instance.stages;
  for (Index s = 0; s < instance.stages && remaining > 0; ++s) {
    const Tick add = remaining / (instance.stages - s);
    split[static_cast<std::size_t>(s)] += add;
    remaining -= add;
  }
  for (Index s = 0; remaining > 0; s = (s + 1) % instance.stages) {
    ++split[static_cast<std::size_t>(s)];
    --remaining;
  }
  ValidateSplit(instance, split);
  return split;
}

std::vector<Tick> BuildCostBalancedPressureSplit(const Instance& instance) {
  // The current Instance has uniform per-layer stage costs, so cost-balanced
  // and layer-uniform anchors coincide. Keep this helper separate for future
  // heterogeneous layer profiles.
  return BuildUniformPressureSplit(instance);
}

namespace {

constexpr std::uint64_t kExactPressureEnumerationLimit = 100000;

[[nodiscard]] Index EffectiveBeamWidth(const PressurePruningOptions& options) {
  return std::max<Index>(1, options.beam_width);
}

[[nodiscard]] Index EffectiveBranchWidth(
    const PressurePruningOptions& options) {
  return std::max<Index>(1, options.branch_width);
}

[[nodiscard]] Index EffectiveGeneratedPartitionLimit(
    const PressurePruningOptions& options) {
  if (options.generated_partitions > 0) return options.generated_partitions;
  return std::max<Index>(512, 8 * std::max<Index>(1, options.partition_top_k));
}

[[nodiscard]] bool SameSplit(const std::vector<Tick>& a,
                             const std::vector<Tick>& b) {
  return a == b;
}

[[nodiscard]] std::vector<std::vector<Tick>> CollectAnchorSplits(
    const Instance& instance, const PressurePruningAnchors& anchors,
    PressurePruningStats* stats) {
  std::vector<std::vector<Tick>> splits;
  auto add_anchor = [&](bool enabled, const std::vector<Tick>& split) {
    if (!enabled) return;
    ValidateSplit(instance, split);
    splits.push_back(split);
  };
  add_anchor(anchors.has_incumbent_split, anchors.incumbent_split);
  add_anchor(anchors.has_uniform_split, anchors.uniform_split);
  add_anchor(anchors.has_cost_balanced_split, anchors.cost_balanced_split);
  std::sort(splits.begin(), splits.end());
  splits.erase(std::unique(splits.begin(), splits.end()), splits.end());
  if (stats != nullptr) {
    stats->anchor_splits_added = static_cast<std::int64_t>(splits.size());
  }
  return splits;
}

[[nodiscard]] bool IsProtectedAnchor(
    const std::vector<Tick>& split,
    const std::vector<std::vector<Tick>>& anchor_splits) {
  return std::binary_search(anchor_splits.begin(), anchor_splits.end(), split);
}

[[nodiscard]] std::vector<Tick> CompleteGreedySplit(
    const Instance& instance, const std::vector<Tick>& ends) {
  std::vector<Tick> full_ends = ends;
  Tick previous = full_ends.empty() ? 0 : full_ends.back();
  const Index placed = static_cast<Index>(full_ends.size());
  for (Index s = placed; s + 1 < instance.stages; ++s) {
    const Index remaining_stages = instance.stages - s;
    const Tick remaining_layers = instance.total_layers - previous;
    Tick size = remaining_layers / remaining_stages;
    size = std::max<Tick>(instance.min_layers, size);
    const Tick max_size =
        remaining_layers - instance.min_layers * (remaining_stages - 1);
    size = std::min(size, max_size);
    previous += size;
    full_ends.push_back(previous);
  }
  full_ends.push_back(instance.total_layers);

  std::vector<Tick> split;
  split.reserve(static_cast<std::size_t>(instance.stages));
  previous = 0;
  for (Tick end : full_ends) {
    split.push_back(end - previous);
    previous = end;
  }
  ValidateSplit(instance, split);
  return split;
}

[[nodiscard]] std::vector<Tick> NextBoundaryCandidates(
    const Instance& instance, const std::vector<Tick>& ends,
    Index branch_width) {
  const Index s = static_cast<Index>(ends.size());
  const Tick previous = ends.empty() ? 0 : ends.back();
  const Index remaining_after = instance.stages - s - 1;
  const Tick min_boundary = previous + instance.min_layers;
  const Tick max_boundary =
      instance.total_layers - instance.min_layers * remaining_after;
  if (min_boundary > max_boundary) return {};

  std::set<Tick> seeds;
  const double uniform_boundary = static_cast<double>(s + 1) *
                                  static_cast<double>(instance.total_layers) /
                                  static_cast<double>(instance.stages);
  seeds.insert(static_cast<Tick>(std::llround(uniform_boundary)));

  const Tick remaining_layers = instance.total_layers - previous;
  const Index remaining_stages = instance.stages - s;
  seeds.insert(previous + remaining_layers / remaining_stages);
  seeds.insert(previous + static_cast<Tick>(std::llround(
                              static_cast<double>(remaining_layers) /
                              static_cast<double>(remaining_stages))));

  const Tick target_stage_layers = std::max<Tick>(
      instance.min_layers, static_cast<Tick>(std::llround(
                               static_cast<double>(instance.total_layers) /
                               static_cast<double>(instance.stages))));
  seeds.insert(previous + target_stage_layers);

  std::vector<Tick> candidates;
  const Tick radius = std::max<Tick>(1, branch_width / 2);
  for (Tick seed : seeds) {
    for (Tick delta = -radius; delta <= radius; ++delta) {
      const Tick boundary = seed + delta;
      if (boundary >= min_boundary && boundary <= max_boundary) {
        candidates.push_back(boundary);
      }
    }
  }
  candidates.push_back(min_boundary);
  candidates.push_back(max_boundary);
  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());

  std::stable_sort(candidates.begin(), candidates.end(), [&](Tick a, Tick b) {
    const double da = std::abs(static_cast<double>(a) - uniform_boundary);
    const double db = std::abs(static_cast<double>(b) - uniform_boundary);
    if (da != db) return da < db;
    return a < b;
  });
  if (static_cast<Index>(candidates.size()) > branch_width) {
    candidates.resize(static_cast<std::size_t>(branch_width));
    std::sort(candidates.begin(), candidates.end());
  }
  return candidates;
}

struct BeamState {
  std::vector<Tick> ends;
  double score = 0.0;
};

[[nodiscard]] std::vector<PressurePartitionCandidate> ScoreAndDeduplicate(
    const Instance& instance, const PressurePruningOptions& options,
    std::vector<std::vector<Tick>> splits) {
  std::sort(splits.begin(), splits.end());
  splits.erase(std::unique(splits.begin(), splits.end()), splits.end());

  std::vector<PressurePartitionCandidate> candidates;
  candidates.reserve(splits.size());
  for (const std::vector<Tick>& split : splits) {
    ValidateSplit(instance, split);
    candidates.push_back(PressurePartitionCandidate{
        split, ComputePartitionPressure(instance, split, options)});
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const PressurePartitionCandidate& a,
               const PressurePartitionCandidate& b) {
              if (a.pressure != b.pressure) return a.pressure < b.pressure;
              return a.split < b.split;
            });
  return candidates;
}

[[nodiscard]] std::vector<PressurePartitionCandidate> ApplyPressureKeepPolicy(
    const std::vector<PressurePartitionCandidate>& candidates,
    const PressurePruningOptions& options,
    const std::vector<std::vector<Tick>>& anchor_splits,
    PressurePruningStats* stats) {
  std::vector<PressurePartitionCandidate> kept;
  if (candidates.empty()) return kept;

  const double best = candidates.front().pressure;
  std::optional<double> cutoff;
  if (options.partition_epsilon >= 0.0) {
    cutoff = (1.0 + options.partition_epsilon) * best;
  }

  std::set<std::vector<Tick>> kept_splits;
  for (const PressurePartitionCandidate& candidate : candidates) {
    if (IsProtectedAnchor(candidate.split, anchor_splits) &&
        kept_splits.insert(candidate.split).second) {
      kept.push_back(candidate);
    }
  }

  Index ranked_kept = 0;
  for (const PressurePartitionCandidate& candidate : candidates) {
    const bool is_anchor = IsProtectedAnchor(candidate.split, anchor_splits);
    bool keep = false;
    if (!is_anchor && options.partition_top_k > 0 &&
        ranked_kept < options.partition_top_k) {
      keep = true;
      ++ranked_kept;
    }
    if (!is_anchor && cutoff && candidate.pressure <= *cutoff) keep = true;
    if (!is_anchor && options.partition_top_k <= 0 && !cutoff) keep = true;
    if (keep && kept_splits.insert(candidate.split).second) {
      kept.push_back(candidate);
    }
  }
  if (kept.empty()) kept.push_back(candidates.front());
  std::sort(kept.begin(), kept.end(),
            [](const PressurePartitionCandidate& a,
               const PressurePartitionCandidate& b) {
              if (a.pressure != b.pressure) return a.pressure < b.pressure;
              return a.split < b.split;
            });

  if (stats != nullptr) {
    stats->best_pressure = best;
    if (cutoff) {
      stats->pressure_cutoff_available = true;
      stats->pressure_cutoff = *cutoff;
    }
  }
  return kept;
}

}  // namespace

std::vector<double> ComputeStagePressures(
    const Instance& instance, const std::vector<Tick>& split,
    const PressurePruningOptions& options) {
  instance.Validate();
  ValidateSplit(instance, split);
  const Tick ratio_sum =
      CheckedAdd(instance.backward_ratio_num, instance.backward_ratio_den,
                 "pressure ratio sum");
  std::vector<double> loads(static_cast<std::size_t>(instance.stages), 0.0);
  double total_load = 0.0;
  for (Index s = 0; s < instance.stages; ++s) {
    const double load =
        static_cast<double>(split[static_cast<std::size_t>(s)]) *
        static_cast<double>(ratio_sum);
    loads[static_cast<std::size_t>(s)] = load;
    total_load += load;
  }
  const double average_load =
      total_load / static_cast<double>(std::max<Index>(1, instance.stages));
  std::vector<double> pressure(static_cast<std::size_t>(instance.stages), 0.0);
  for (Index n = 0; n < instance.stages; ++n) {
    const double lhat_n = loads[static_cast<std::size_t>(n)] / average_load;
    double value = lhat_n;
    for (Index m = 0; m < instance.stages; ++m) {
      if (m == n) continue;
      if ((m % instance.workers) != (n % instance.workers)) continue;
      const double lhat_m = loads[static_cast<std::size_t>(m)] / average_load;
      const double distance = static_cast<double>(std::llabs(m - n) + 1);
      value +=
          options.lambda * lhat_n * lhat_m / std::pow(distance, options.gamma);
    }
    pressure[static_cast<std::size_t>(n)] = value;
  }
  return pressure;
}

double ComputePartitionPressure(const Instance& instance,
                                const std::vector<Tick>& split,
                                const PressurePruningOptions& options) {
  const std::vector<double> pressure =
      ComputeStagePressures(instance, split, options);
  double sum = 0.0;
  double max_value = 0.0;
  for (double value : pressure) {
    sum += value;
    max_value = std::max(max_value, value);
  }
  const double average = sum / static_cast<double>(std::max<std::size_t>(
                                   std::size_t{1}, pressure.size()));
  return options.alpha * max_value + (1.0 - options.alpha) * average;
}

std::vector<PressurePartitionCandidate> GeneratePressureBeamPartitions(
    const Instance& instance, const PressurePruningOptions& options) {
  instance.Validate();
  const Index beam_width = EffectiveBeamWidth(options);
  const Index branch_width = EffectiveBranchWidth(options);
  const Index generated_limit = EffectiveGeneratedPartitionLimit(options);

  std::vector<BeamState> beam(1);
  for (Index s = 0; s + 1 < instance.stages; ++s) {
    std::vector<BeamState> next;
    for (const BeamState& state : beam) {
      for (Tick boundary :
           NextBoundaryCandidates(instance, state.ends, branch_width)) {
        BeamState child;
        child.ends = state.ends;
        child.ends.push_back(boundary);
        child.score = ComputePartitionPressure(
            instance, CompleteGreedySplit(instance, child.ends), options);
        next.push_back(std::move(child));
      }
    }
    std::sort(next.begin(), next.end(),
              [](const BeamState& a, const BeamState& b) {
                if (a.score != b.score) return a.score < b.score;
                return a.ends < b.ends;
              });
    next.erase(std::unique(next.begin(), next.end(),
                           [](const BeamState& a, const BeamState& b) {
                             return a.ends == b.ends;
                           }),
               next.end());
    if (static_cast<Index>(next.size()) > beam_width) {
      next.resize(static_cast<std::size_t>(beam_width));
    }
    beam = std::move(next);
    if (beam.empty()) break;
  }

  std::vector<std::vector<Tick>> splits;
  splits.reserve(beam.size() + 1);
  splits.push_back(BuildUniformPressureSplit(instance));
  splits.push_back(BuildCostBalancedPressureSplit(instance));
  for (const BeamState& state : beam) {
    splits.push_back(CompleteGreedySplit(instance, state.ends));
  }

  std::vector<PressurePartitionCandidate> candidates =
      ScoreAndDeduplicate(instance, options, std::move(splits));
  if (static_cast<Index>(candidates.size()) > generated_limit) {
    candidates.resize(static_cast<std::size_t>(generated_limit));
  }
  return candidates;
}

std::vector<PressurePartitionCandidate> SelectPressurePartitions(
    const Instance& instance, const PressurePruningOptions& options,
    PressurePruningStats* stats, const PressurePruningAnchors& anchors) {
  instance.Validate();

  const std::uint64_t split_count =
      CountValidSplitsCapped(instance, kExactPressureEnumerationLimit + 1);
  const bool use_exhaustive = split_count <= kExactPressureEnumerationLimit;

  std::vector<std::vector<Tick>> anchor_splits =
      CollectAnchorSplits(instance, anchors, stats);
  std::vector<PressurePartitionCandidate> candidates;
  if (use_exhaustive) {
    std::vector<std::vector<Tick>> splits = anchor_splits;
    EnumerateValidSplits(instance, [&](const std::vector<Tick>& split) {
      splits.push_back(split);
    });
    candidates = ScoreAndDeduplicate(instance, options, std::move(splits));
  } else {
    candidates = GeneratePressureBeamPartitions(instance, options);
    std::vector<std::vector<Tick>> splits = anchor_splits;
    for (const PressurePartitionCandidate& candidate : candidates) {
      splits.push_back(candidate.split);
    }
    candidates = ScoreAndDeduplicate(instance, options, std::move(splits));
  }

  std::vector<PressurePartitionCandidate> kept =
      ApplyPressureKeepPolicy(candidates, options, anchor_splits, stats);
  if (kept.empty() && !candidates.empty()) kept.push_back(candidates.front());

  if (stats != nullptr) {
    stats->enabled = options.enabled;
    stats->generation_mode = use_exhaustive ? "exhaustive" : "beam";
    stats->generated_partitions = static_cast<std::int64_t>(candidates.size());
    stats->partitions_before = static_cast<std::int64_t>(candidates.size());
    stats->partitions_after = static_cast<std::int64_t>(kept.size());
    stats->selected_splits_total = static_cast<std::int64_t>(kept.size());
    stats->incumbent_split_included =
        anchors.has_incumbent_split &&
        std::any_of(kept.begin(), kept.end(), [&](const auto& candidate) {
          return SameSplit(candidate.split, anchors.incumbent_split);
        });
    stats->uniform_split_included =
        anchors.has_uniform_split &&
        std::any_of(kept.begin(), kept.end(), [&](const auto& candidate) {
          return SameSplit(candidate.split, anchors.uniform_split);
        });
    stats->cost_balanced_split_included =
        anchors.has_cost_balanced_split &&
        std::any_of(kept.begin(), kept.end(), [&](const auto& candidate) {
          return SameSplit(candidate.split, anchors.cost_balanced_split);
        });
    stats->beam_width = EffectiveBeamWidth(options);
    stats->branch_width = EffectiveBranchWidth(options);
    stats->exhaustive_enumeration_skipped = !use_exhaustive;
    stats->predecessor_candidates_before = 0;
    stats->predecessor_candidates_after = 0;
    if (options.predecessor_top_k > 0) {
      stats->predecessor_note =
          "Joint CP-SAT uses NoOverlap worker serialization, so there is no "
          "explicit predecessor candidate list to pressure-prune.";
    }
  }
  return kept;
}

}  // namespace slackpipe
