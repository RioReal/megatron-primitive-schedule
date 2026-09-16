#include "slackpipe/breadth_first.h"

#include <algorithm>
#include <tuple>

namespace slackpipe {

namespace {
auto BreadthFirstRank(const Instance& instance, OperationId id) {
  const OperationView view = DecodeOperation(instance, id);
  return std::tuple<Index, Index, Index>{view.microbatch + view.chain_index,
                                         -view.chain_index, view.microbatch};
}
}  // namespace

bool BreadthFirstLess(const Instance& instance, OperationId a, OperationId b) {
  return BreadthFirstRank(instance, a) < BreadthFirstRank(instance, b);
}

MachineOrders BreadthFirstOrders(const Instance& instance) {
  instance.Validate();
  MachineOrders orders(static_cast<std::size_t>(instance.workers));
  for (Index b = 0; b < instance.microbatches; ++b) {
    for (Index n = 0; n < 2 * instance.stages; ++n) {
      OperationId id = EncodeOperation(instance, b, n);
      const OperationView view = DecodeOperation(instance, id);
      orders[static_cast<std::size_t>(view.worker)].push_back(id);
    }
  }
  for (auto& worker_order : orders) {
    std::sort(worker_order.begin(), worker_order.end(),
              [&](OperationId a, OperationId b) {
                return BreadthFirstLess(instance, a, b);
              });
  }
  return orders;
}

}  // namespace slackpipe
