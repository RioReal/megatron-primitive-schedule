#pragma once

#include "slackpipe/schedule.h"

namespace slackpipe {

[[nodiscard]] MachineOrders BreadthFirstOrders(const Instance& instance);
[[nodiscard]] bool BreadthFirstLess(const Instance& instance, OperationId a,
                                    OperationId b);

}  // namespace slackpipe
