#pragma once

#include <optional>
#include <string>

#include "slackpipe/instance.h"

namespace slackpipe {

struct WorkloadLowerBound {
  std::optional<Tick> total_work_ticks;
  std::optional<Tick> makespan_ticks;
  bool partition_invariant = false;
  std::string reason;
};

// Independent of any incumbent. Variable affine slopes minimize total work over
// all nonempty contiguous partitions, relaxing any extra partition restrictions.
WorkloadLowerBound ComputeWorkloadLowerBound(const Instance& instance);

struct LowerBoundReport {
  WorkloadLowerBound workload;
  std::optional<double> raw_solver_bound_ticks;
  std::optional<double> effective_lower_bound_ticks;
  std::optional<double> effective_relative_gap;
  bool inconsistent_with_incumbent = false;
  bool raw_bound_globally_valid = true;
};

LowerBoundReport ComputeLowerBoundReport(const Instance& instance,
                                        std::optional<double> raw_solver_bound,
                                        std::optional<Tick> feasible_makespan,
                                        bool raw_bound_globally_valid = true);
std::string LowerBoundReportJson(const LowerBoundReport& report);

}  // namespace slackpipe
