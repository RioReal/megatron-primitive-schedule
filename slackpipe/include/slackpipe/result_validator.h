#pragma once

#include <optional>
#include <string>
#include <vector>

#include "slackpipe/instance.h"
#include "slackpipe/schedule.h"

namespace slackpipe {

inline constexpr int kResultValidationVersion = 1;

struct StableOperationKey {
  OperationId id;
  Index microbatch = 0;
  Index operation_position = 0;
  std::string phase;
  Index stage = 0;
  Index worker = 0;
};

struct SerializedOperationRecord {
  OperationId id;
  std::optional<std::string> name;
  std::optional<Index> microbatch;
  std::optional<Index> operation_position;
  std::optional<Index> stage;
  std::optional<std::string> phase;
  std::optional<Index> worker;
  std::optional<Tick> start;
  std::optional<Tick> end;
  std::optional<Tick> duration;
};

struct SerializedWorkerPredecessorRecord {
  OperationId operation_id;
  std::optional<std::string> operation_name;
  OperationId predecessor_id;
  std::optional<std::string> predecessor_name;
  std::optional<Index> worker;
};

struct ResultValidationInput {
  Instance instance;
  std::optional<std::vector<Tick>> selected_partition;
  std::optional<std::vector<Tick>> fixed_partition_reference;
  std::optional<MachineOrders> worker_orders;
  std::optional<std::vector<std::vector<std::string>>> worker_order_names;
  std::optional<std::vector<SerializedWorkerPredecessorRecord>>
      derived_worker_predecessors;
  std::optional<std::vector<SerializedOperationRecord>> serialized_operations;
  bool require_serialized_worker_predecessors = false;
  bool require_serialized_operations = false;
  std::optional<Tick> reported_makespan;
  bool feasible_claimed = false;
  std::optional<std::string> solver_status_raw;
  std::optional<std::string> reported_status;
  std::optional<std::string> method_name;
  std::string communication_model = "constant_inter_worker_delay";
  std::optional<double> communication_alpha;
  std::optional<double> communication_beta;
  std::optional<std::string> communication_payload;
  bool fifo_ordering = true;
  std::optional<bool> fifo_ordering_requested;
  std::optional<bool> fifo_ordering_effective;
  std::optional<Index> fifo_constraint_count;
};

struct ResultValidationResult {
  int validation_version = kResultValidationVersion;
  bool passed = false;
  std::string error_code;
  std::string error_category;
  std::string message;
  std::vector<std::string> offending_operations;
  std::optional<Index> offending_worker;
  std::optional<std::string> offending_edge_type;
  std::optional<Tick> expected_tick;
  std::optional<Tick> actual_tick;
  std::optional<Index> expected_count;
  std::optional<Index> actual_count;
  std::optional<Tick> reconstructed_makespan;
  std::optional<Tick> serialized_makespan;
  std::optional<Tick> reported_makespan;
  Index expected_operation_count = 0;
  Index actual_operation_count = 0;
  Index data_edge_count = 0;
  Index fifo_edge_count = 0;
  Index worker_edge_count = 0;
  Index total_edge_count = 0;
  Index intervals_checked = 0;
  std::vector<std::string> cycle_witness;
  std::vector<std::string> warnings;
  double validation_runtime_seconds = 0.0;
};

[[nodiscard]] StableOperationKey ExpectedOperationKey(const Instance& instance,
                                                      OperationId id);
[[nodiscard]] std::string StableOperationName(const StableOperationKey& key);
[[nodiscard]] ResultValidationInput ValidationInputFromSchedule(
    const Instance& instance, const ScheduleSolution& schedule,
    const std::string& status, std::optional<std::string> method_name,
    bool fifo_ordering = true);
[[nodiscard]] ResultValidationResult ValidateResult(
    const ResultValidationInput& input);
[[nodiscard]] ResultValidationResult ValidateScheduleSolutionIndependent(
    const Instance& instance, const ScheduleSolution& schedule,
    const std::string& status = "FEASIBLE",
    std::optional<std::string> method_name = std::nullopt,
    bool fifo_ordering = true);
[[nodiscard]] ResultValidationInput ValidationInputFromResultJsonText(
    const std::string& json_text);
[[nodiscard]] ScheduleSolution ScheduleSolutionFromValidationInput(
    const ResultValidationInput& input);
[[nodiscard]] ResultValidationResult ValidateResultJsonText(
    const std::string& json_text);
[[nodiscard]] std::string ResultValidationToJson(
    const ResultValidationResult& result, const std::string& indent);

}  // namespace slackpipe
