#pragma once

#include <optional>
#include <string>
#include <vector>

#include "slackpipe/instance.h"

namespace slackpipe {

struct OperationId {
  Index value = 0;

  friend bool operator==(OperationId a, OperationId b) {
    return a.value == b.value;
  }
  friend bool operator!=(OperationId a, OperationId b) { return !(a == b); }
  friend bool operator<(OperationId a, OperationId b) {
    return a.value < b.value;
  }
};

struct OperationView {
  OperationId id;
  Index microbatch = 0;
  Index chain_index = 0;
  Index stage = 0;
  bool backward = false;
  Index worker = 0;
};

[[nodiscard]] OperationId EncodeOperation(const Instance& instance,
                                          Index microbatch, Index chain_index);
[[nodiscard]] OperationView DecodeOperation(const Instance& instance,
                                            OperationId id);
[[nodiscard]] std::string OperationName(const Instance& instance,
                                        OperationId id);
[[nodiscard]] Index OperationPositionCount(const Instance& instance);
[[nodiscard]] Index StageForOperationPosition(const Instance& instance,
                                              Index chain_index);
[[nodiscard]] Index WorkerForOperationPosition(const Instance& instance,
                                               Index chain_index);
[[nodiscard]] std::optional<OperationId> DataPredecessor(
    const Instance& instance, OperationId id);
[[nodiscard]] std::optional<OperationId> FifoPredecessor(
    const Instance& instance, OperationId id);
[[nodiscard]] bool IsDataPredecessor(const Instance& instance,
                                     OperationId predecessor, OperationId id);
[[nodiscard]] std::vector<Index> WorkerLocalNeighborPositions(
    const Instance& instance, Index chain_index);
[[nodiscard]] std::vector<OperationId> PredNCandidates(const Instance& instance,
                                                       OperationId id);

}  // namespace slackpipe
