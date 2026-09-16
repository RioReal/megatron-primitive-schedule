#pragma once

#include <optional>

#include "slackpipe/instance.h"

namespace slackpipe {

struct WorkerBalanceConstraintResult {
  bool enabled = false;
  double tolerance_percent = -1.0;
  Index tolerance_layers = -1;
  Tick lower_bound = 0;
  Tick upper_bound = 0;
};

[[nodiscard]] WorkerBalanceConstraintResult ComputeWorkerBalanceConstraint(
    const Instance& instance, double tolerance_percent,
    std::optional<Index> tolerance_layers = std::nullopt);

}  // namespace slackpipe
