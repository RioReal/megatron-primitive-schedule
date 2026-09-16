#pragma once

#include "slackpipe/schedule.h"

namespace slackpipe {

[[nodiscard]] MachineOrders InterleavedOneFOneBOrders(const Instance& instance);

}  // namespace slackpipe
