#pragma once

#include <optional>
#include <string>
#include <vector>

#include "slackpipe/instance.h"
#include "slackpipe/schedule.h"

namespace slackpipe {

struct InterleavingRunMetadata {
  std::string status;
  Tick makespan_ticks = 0;
  std::optional<double> objective_bound;
};

struct InterleavingEvent {
  Index worker = 0;
  std::string kind;
  bool is_direction_switch = false;
  char prev_dir = 'F';
  Index prev_b = 0;
  Index prev_n = 0;
  Index prev_stage_position = -1;
  Tick prev_start = 0;
  Tick prev_end = 0;
  Index prev_order_index = 0;
  char next_dir = 'F';
  Index next_b = 0;
  Index next_n = 0;
  Index next_stage_position = -1;
  Tick next_start = 0;
  Tick next_end = 0;
  Index next_order_index = 0;
  Index num_worker_ops = 0;
  double prev_order_fraction = 0.0;
  double next_order_fraction = 0.0;
  Tick gap_ticks = 0;
  std::optional<Index> forward_b;
  std::optional<Index> forward_n;
  std::optional<Index> forward_stage_position;
  std::optional<Index> backward_b;
  std::optional<Index> backward_n;
  std::optional<Index> backward_stage_position;
};

struct InterleavingStats {
  std::vector<InterleavingEvent> events;
};

[[nodiscard]] InterleavingStats CollectInterleavingStats(
    const Instance& instance, const ScheduleSolution& schedule);

[[nodiscard]] std::string ToInterleavingEventsCsv(
    const Instance& instance, const InterleavingRunMetadata& metadata,
    const InterleavingStats& stats);

[[nodiscard]] std::string ToInterleavingSummaryCsv(
    const Instance& instance, const InterleavingRunMetadata& metadata,
    const InterleavingStats& stats);

}  // namespace slackpipe
