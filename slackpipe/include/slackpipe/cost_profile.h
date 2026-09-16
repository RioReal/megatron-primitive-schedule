#pragma once

#include <string>

#include "slackpipe/instance.h"

namespace slackpipe {

inline constexpr const char* kSlackPipeCostProfileSchemaVersion =
    "slackpipe.cost_profile.v1";
inline constexpr const char* kSlackPipeCostProfileSchemaVersionV2 =
    "slackpipe.cost_profile.v2";

void ApplyCostProfileFile(Instance& instance, const std::string& path);

}  // namespace slackpipe
