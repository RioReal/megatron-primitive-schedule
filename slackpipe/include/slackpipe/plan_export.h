#pragma once

#include <string>

#include "slackpipe/instance.h"
#include "slackpipe/schedule.h"

namespace slackpipe {

inline constexpr const char* kMegatronSlackPipePlanSchemaVersion =
    "slackpipe.plan.v1";
inline constexpr const char* kMegatronSlackPipePlanSchemaVersionV2 =
    "slackpipe.plan.v2";

void ValidateMegatronSlackPipePlanExport(const Instance& instance,
                                         const ScheduleSolution& schedule);

[[nodiscard]] std::string ToMegatronSlackPipePlanJson(
    const Instance& instance, const ScheduleSolution& schedule,
    const std::string& solver_status);

void WriteMegatronSlackPipePlanFile(const std::string& path,
                                    const Instance& instance,
                                    const ScheduleSolution& schedule,
                                    const std::string& solver_status);

}  // namespace slackpipe
