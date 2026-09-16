#pragma once

#include <map>
#include <optional>

#include "slackpipe/schedule.h"

namespace slackpipe {

struct EvaluationResult {
  ScheduleSolution schedule;
};

struct ScheduleMetrics {
  Tick simulated_iteration_time = 0;
  Tick total_useful_work = 0;
  double pipeline_utilization = 0.0;
  std::vector<Tick> per_worker_busy_time;
  std::vector<Tick> per_worker_idle_time;
  Tick max_worker_load = 0;
  Tick pipeline_fill_time = 0;
  Tick pipeline_drain_time = 0;
  std::optional<Tick> communication_blocked_time;
};

[[nodiscard]] MachinePredecessors ExtractMachinePredecessors(
    const Instance& instance, const MachineOrders& orders,
    bool fifo_ordering = true);

[[nodiscard]] EvaluationResult EvaluateSchedule(const Instance& instance,
                                                const std::vector<Tick>& split,
                                                const MachineOrders& orders,
                                                bool fifo_ordering = true);
[[nodiscard]] EvaluationResult EvaluateScheduleWithPredecessors(
    const Instance& instance, const std::vector<Tick>& split,
    const MachinePredecessors& predecessors, bool fifo_ordering = true);
[[nodiscard]] Tick TotalUsefulWork(const Instance& instance);
[[nodiscard]] ScheduleMetrics ComputeScheduleMetrics(
    const Instance& instance, const ScheduleSolution& schedule,
    bool fifo_ordering = true);

}  // namespace slackpipe
