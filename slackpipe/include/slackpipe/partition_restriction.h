#pragma once

#include <optional>
#include <vector>

#include "slackpipe/types.h"

namespace slackpipe {

enum class SlackPipeSplitMode {
  kFixed,
  kLocal,
  kGlobal,
  kWorkerFixed,
  kWorkerLocal,
};

struct PartitionRestriction {
  SlackPipeSplitMode mode = SlackPipeSplitMode::kGlobal;
  std::vector<Tick> reference_split;
  std::optional<Index> move_budget;
  std::optional<Index> worker_move_budget;
  std::optional<Index> per_stage_delta;
  std::optional<Index> per_worker_delta;
};

}  // namespace slackpipe
