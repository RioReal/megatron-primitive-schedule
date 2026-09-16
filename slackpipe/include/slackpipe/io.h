#pragma once

#include <optional>
#include <string>

#include "slackpipe/alternating_solver.h"
#include "slackpipe/bfs_solver.h"
#include "slackpipe/joint_solver.h"
#include "slackpipe/result_schema.h"
#include "slackpipe/schedule.h"
#include "slackpipe/search_stats.h"
#include "slackpipe/slackpipe_solver.h"

namespace slackpipe {

[[nodiscard]] std::string ToJson(const Instance& instance,
                                 const ScheduleSolution& schedule);
[[nodiscard]] std::string ToJson(const Instance& instance,
                                 const ScheduleSolution& schedule,
                                 const SearchStats* search_stats);
[[nodiscard]] std::string ToJson(const Instance& instance,
                                 const ScheduleSolution& schedule,
                                 const CanonicalResultMetadata& canonical,
                                 const SearchStats* search_stats = nullptr);
[[nodiscard]] std::string ToJson(const Instance& instance,
                                 const SearchStats& search_stats);
[[nodiscard]] std::string SearchStatsSummary(const SearchStats& search_stats);
[[nodiscard]] std::string ToCsv(const Instance& instance,
                                const ScheduleSolution& schedule);
[[nodiscard]] std::string ToOrdersText(const Instance& instance,
                                       const MachineOrders& orders);
[[nodiscard]] std::string ToSvg(const Instance& instance,
                                const ScheduleSolution& schedule);
[[nodiscard]] std::string ToJson(const Instance& instance,
                                 const BfsSplitOptimizationResult& result);
[[nodiscard]] std::string ToJson(const Instance& instance,
                                 const JointOptimizationResult& result);
[[nodiscard]] std::string ToJson(const Instance& instance,
                                 const AlternatingOptimizationResult& result);
[[nodiscard]] std::string ToJson(const Instance& instance,
                                 const SlackPipeResult& result);

void WriteTextFile(const std::string& path, const std::string& contents);

}  // namespace slackpipe
