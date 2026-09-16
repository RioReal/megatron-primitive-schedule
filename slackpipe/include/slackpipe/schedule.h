#pragma once

#include <map>
#include <string>
#include <vector>

#include "slackpipe/operation.h"

namespace slackpipe {

using MachineOrders = std::vector<std::vector<OperationId>>;

struct ScheduledOperation {
  OperationId id;
  Tick start = 0;
  Tick end = 0;
  Tick duration = 0;
  Index worker = 0;
};

struct ScheduleSolution {
  std::vector<Tick> split;
  MachineOrders orders;
  std::vector<ScheduledOperation> operations_by_id;
  Tick makespan = 0;
  std::vector<std::string> validation_errors;

  [[nodiscard]] bool ok() const { return validation_errors.empty(); }
};

using MachinePredecessors = std::map<Index, OperationId>;

}  // namespace slackpipe
