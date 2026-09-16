#include "slackpipe/operation.h"

#include <algorithm>
#include <queue>
#include <sstream>

namespace slackpipe {

namespace {

[[nodiscard]] bool HasBaseDataOrFifoPath(const Instance& instance,
                                         OperationId from, OperationId to) {
  const Index count = instance.OperationCount();
  std::vector<bool> seen(static_cast<std::size_t>(count), false);
  std::queue<OperationId> ready;
  ready.push(from);
  seen[static_cast<std::size_t>(from.value)] = true;

  while (!ready.empty()) {
    const OperationId current = ready.front();
    ready.pop();
    if (current == to) return true;
    const OperationView view = DecodeOperation(instance, current);

    std::vector<OperationId> successors;
    if (view.chain_index + 1 < OperationPositionCount(instance)) {
      successors.push_back(
          EncodeOperation(instance, view.microbatch, view.chain_index + 1));
    }
    if (view.microbatch + 1 < instance.microbatches) {
      successors.push_back(
          EncodeOperation(instance, view.microbatch + 1, view.chain_index));
    }
    for (OperationId next : successors) {
      if (!seen[static_cast<std::size_t>(next.value)]) {
        seen[static_cast<std::size_t>(next.value)] = true;
        ready.push(next);
      }
    }
  }
  return false;
}

}  // namespace

OperationId EncodeOperation(const Instance& instance, Index microbatch,
                            Index chain_index) {
  instance.Validate();
  if (microbatch < 0 || microbatch >= instance.microbatches) {
    throw Error("microbatch out of range");
  }
  const Index chain_length = CheckedMul(2, instance.stages, "chain length");
  if (chain_index < 0 || chain_index >= chain_length) {
    throw Error("chain index out of range");
  }
  return OperationId{
      CheckedAdd(CheckedMul(microbatch, chain_length, "operation encoding"),
                 chain_index, "operation encoding")};
}

Index OperationPositionCount(const Instance& instance) {
  instance.Validate();
  return CheckedMul(2, instance.stages, "operation position count");
}

Index StageForOperationPosition(const Instance& instance, Index chain_index) {
  instance.Validate();
  const Index chain_length = OperationPositionCount(instance);
  if (chain_index < 0 || chain_index >= chain_length) {
    throw Error("chain index out of range");
  }
  return chain_index < instance.stages ? chain_index
                                       : chain_length - 1 - chain_index;
}

Index WorkerForOperationPosition(const Instance& instance, Index chain_index) {
  return StageForOperationPosition(instance, chain_index) % instance.workers;
}

OperationView DecodeOperation(const Instance& instance, OperationId id) {
  instance.Validate();
  const Index count = instance.OperationCount();
  if (id.value < 0 || id.value >= count) {
    throw Error("operation id out of range");
  }
  const Index chain_length = OperationPositionCount(instance);
  OperationView view;
  view.id = id;
  view.microbatch = id.value / chain_length;
  view.chain_index = id.value % chain_length;
  view.backward = view.chain_index >= instance.stages;
  view.stage = StageForOperationPosition(instance, view.chain_index);
  view.worker = WorkerForOperationPosition(instance, view.chain_index);
  return view;
}

std::string OperationName(const Instance& instance, OperationId id) {
  const OperationView view = DecodeOperation(instance, id);
  std::ostringstream out;
  out << (view.backward ? 'B' : 'F') << view.stage << "(b" << view.microbatch
      << ")";
  return out.str();
}

std::optional<OperationId> DataPredecessor(const Instance& instance,
                                           OperationId id) {
  const OperationView view = DecodeOperation(instance, id);
  if (view.chain_index == 0) return std::nullopt;
  return EncodeOperation(instance, view.microbatch, view.chain_index - 1);
}

std::optional<OperationId> FifoPredecessor(const Instance& instance,
                                           OperationId id) {
  const OperationView view = DecodeOperation(instance, id);
  if (view.microbatch == 0) return std::nullopt;
  return EncodeOperation(instance, view.microbatch - 1, view.chain_index);
}

bool IsDataPredecessor(const Instance& instance, OperationId predecessor,
                       OperationId id) {
  const std::optional<OperationId> data = DataPredecessor(instance, id);
  return data && *data == predecessor;
}

std::vector<Index> WorkerLocalNeighborPositions(const Instance& instance,
                                                Index chain_index) {
  instance.Validate();
  const Index chain_length = OperationPositionCount(instance);
  if (chain_index < 0 || chain_index >= chain_length) {
    throw Error("chain index out of range");
  }

  const Index worker = WorkerForOperationPosition(instance, chain_index);
  std::vector<Index> positions;
  for (Index n = 0; n < chain_length; ++n) {
    if (WorkerForOperationPosition(instance, n) == worker) {
      positions.push_back(n);
    }
  }

  const auto it = std::find(positions.begin(), positions.end(), chain_index);
  if (it == positions.end())
    throw Error("operation position missing on worker");

  std::vector<Index> neighbors;
  if (it != positions.begin()) neighbors.push_back(*(it - 1));
  if (std::next(it) != positions.end()) neighbors.push_back(*std::next(it));
  return neighbors;
}

std::vector<OperationId> PredNCandidates(const Instance& instance,
                                         OperationId id) {
  const OperationView view = DecodeOperation(instance, id);
  const std::vector<Index> neighbor_positions =
      WorkerLocalNeighborPositions(instance, view.chain_index);
  std::vector<OperationId> candidates;
  for (Index b = 0; b < instance.microbatches; ++b) {
    for (Index n : neighbor_positions) {
      OperationId candidate = EncodeOperation(instance, b, n);
      if (IsDataPredecessor(instance, candidate, id)) continue;
      if (HasBaseDataOrFifoPath(instance, id, candidate)) continue;
      candidates.push_back(candidate);
    }
  }
  return candidates;
}

}  // namespace slackpipe
