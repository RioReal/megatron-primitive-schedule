#pragma once

#include "slackpipe/bfs_solver.h"
#include "slackpipe/schedule.h"

namespace slackpipe {

[[nodiscard]] BfsSplitOptimizationResult
OptimizePartitionForFixedOrderEnumerate(
    const Instance &instance, const MachineOrders &fixed_orders,
    const BfsSplitOptimizerOptions &options = {});
[[nodiscard]] BfsSplitOptimizationResult OptimizePartitionForFixedOrderCpSat(
    const Instance &instance, const MachineOrders &fixed_orders,
    const BfsSplitOptimizerOptions &options = {});
[[nodiscard]] BfsSplitOptimizationResult OptimizePartitionForFixedOrderAuto(
    const Instance &instance, const MachineOrders &fixed_orders,
    const BfsSplitOptimizerOptions &options = {});
[[nodiscard]] BfsSplitOptimizationResult OptimizePartitionForFixedOrder(
    const Instance &instance, const MachineOrders &fixed_orders,
    const BfsSplitOptimizerOptions &options = {});
[[nodiscard]] bool IsCpSatFixedOrderPartitionOptimizerAvailable();

}  // namespace slackpipe
