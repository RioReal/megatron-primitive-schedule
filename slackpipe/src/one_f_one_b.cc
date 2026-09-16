#include "slackpipe/one_f_one_b.h"

#include <algorithm>
#include <optional>
#include <tuple>
#include <vector>

#include "slackpipe/operation.h"

namespace slackpipe {

namespace {

std::vector<OperationId> StageLocalOneFOneBStream(const Instance& instance,
                                                  Index stage) {
  std::vector<OperationId> stream;
  const Index warmup =
      std::min(instance.microbatches, instance.stages - stage - 1);
  Index next_forward = 0;
  Index next_backward = 0;

  auto forward = [&]() {
    stream.push_back(EncodeOperation(instance, next_forward, stage));
    ++next_forward;
  };
  auto backward = [&]() {
    const Index backward_position = 2 * instance.stages - 1 - stage;
    stream.push_back(
        EncodeOperation(instance, next_backward, backward_position));
    ++next_backward;
  };

  while (next_forward < warmup) {
    forward();
  }
  while (next_forward < instance.microbatches) {
    forward();
    backward();
  }
  while (next_backward < instance.microbatches) {
    backward();
  }
  return stream;
}

bool BaseDependenciesEmitted(const Instance& instance, OperationId id,
                             const std::vector<bool>& emitted) {
  const std::optional<OperationId> data = DataPredecessor(instance, id);
  if (data && !emitted[static_cast<std::size_t>(data->value)]) {
    return false;
  }
  const std::optional<OperationId> fifo = FifoPredecessor(instance, id);
  if (fifo && !emitted[static_cast<std::size_t>(fifo->value)]) {
    return false;
  }
  return true;
}

auto OneFOneBMergeRank(const Instance& instance, OperationId id) {
  const OperationView view = DecodeOperation(instance, id);
  const Index phase_rank = view.backward ? 0 : 1;
  const Index stage_rank = -view.stage;
  return std::tuple<Index, Index, Index, Index, Index>{
      phase_rank, stage_rank, view.microbatch, view.worker, view.chain_index};
}

}  // namespace

MachineOrders InterleavedOneFOneBOrders(const Instance& instance) {
  instance.Validate();
  const Index op_count = instance.OperationCount();
  MachineOrders orders(static_cast<std::size_t>(instance.workers));
  std::vector<std::vector<OperationId>> streams(
      static_cast<std::size_t>(instance.stages));
  std::vector<std::size_t> cursors(static_cast<std::size_t>(instance.stages),
                                   0);
  std::vector<bool> emitted(static_cast<std::size_t>(op_count), false);

  for (Index stage = 0; stage < instance.stages; ++stage) {
    streams[static_cast<std::size_t>(stage)] =
        StageLocalOneFOneBStream(instance, stage);
  }

  for (Index emitted_count = 0; emitted_count < op_count; ++emitted_count) {
    std::vector<OperationId> candidates;
    for (Index stage = 0; stage < instance.stages; ++stage) {
      const std::vector<OperationId>& stream =
          streams[static_cast<std::size_t>(stage)];
      const std::size_t cursor = cursors[static_cast<std::size_t>(stage)];
      if (cursor >= stream.size()) {
        continue;
      }
      const OperationId candidate = stream[cursor];
      if (BaseDependenciesEmitted(instance, candidate, emitted)) {
        candidates.push_back(candidate);
      }
    }
    if (candidates.empty()) {
      throw Error("interleaved 1F1B construction reached no ready operation");
    }

    const OperationId chosen =
        *std::min_element(candidates.begin(), candidates.end(),
                          [&](OperationId a, OperationId b) {
                            return OneFOneBMergeRank(instance, a) <
                                   OneFOneBMergeRank(instance, b);
                          });
    const OperationView view = DecodeOperation(instance, chosen);
    orders[static_cast<std::size_t>(view.worker)].push_back(chosen);
    emitted[static_cast<std::size_t>(chosen.value)] = true;
    ++cursors[static_cast<std::size_t>(view.stage)];
  }
  return orders;
}

}  // namespace slackpipe
