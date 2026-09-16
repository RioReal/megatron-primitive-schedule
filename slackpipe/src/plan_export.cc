#include "slackpipe/plan_export.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

#include "slackpipe/operation.h"
#include "slackpipe/types.h"

namespace slackpipe {

namespace {

std::string JsonEscape(const std::string& text) {
  std::ostringstream out;
  for (char ch : text) {
    if (ch == '"' || ch == '\\') out << '\\';
    if (ch == '\n') {
      out << "\\n";
    } else {
      out << ch;
    }
  }
  return out.str();
}

void WriteTickArray(std::ostringstream& out, const std::vector<Tick>& values) {
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ", ";
    out << values[i];
  }
  out << "]";
}

void WriteStageLayerRanges(std::ostringstream& out, const Instance& instance,
                           const std::vector<Tick>& split) {
  out << "[";
  for (Index stage = 0; stage < instance.stages; ++stage) {
    if (stage != 0) out << ", ";
    out << "{\"begin\": " << instance.StageBeginLayer(stage, split)
        << ", \"end\": " << instance.StageEndLayer(stage, split) << "}";
  }
  out << "]";
}

void WriteCostModelMetadata(std::ostringstream& out, const Instance& instance) {
  out << "  \"cost_model\": {\n";
  if (instance.HasCostProfile()) {
    out << "    \"kind\": \"cost_profile\",\n";
    out << "    \"schema_version\": \""
        << JsonEscape(instance.cost_profile_schema_version) << "\",\n";
    out << "    \"path\": \"" << JsonEscape(instance.cost_profile_path)
        << "\",\n";
    out << "    \"units\": \"" << JsonEscape(instance.cost_profile_units)
        << "\"";
    if (!instance.cost_profile_hash.empty()) {
      out << ",\n    \"cost_profile_hash\": \""
          << JsonEscape(instance.cost_profile_hash) << "\"";
    }
    if (!instance.model_manifest_hash.empty()) {
      out << ",\n    \"model_manifest_hash\": \""
          << JsonEscape(instance.model_manifest_hash) << "\"";
    }
    out << ",\n";
    out << "    \"tick_scale\": \"microseconds\"\n";
  } else {
    out << "    \"kind\": \"ratio\",\n";
    out << "    \"ratio_num\": " << instance.backward_ratio_num << ",\n";
    out << "    \"ratio_den\": " << instance.backward_ratio_den << ",\n";
    out << "    \"units\": \"ticks\"\n";
  }
  out << "  }";
}

std::vector<Tick> StageToWorker(const Instance& instance) {
  std::vector<Tick> stage_to_worker;
  stage_to_worker.reserve(static_cast<std::size_t>(instance.stages));
  for (Index stage = 0; stage < instance.stages; ++stage) {
    stage_to_worker.push_back(stage % instance.workers);
  }
  return stage_to_worker;
}

std::vector<Tick> StageCosts(const Instance& instance,
                             const std::vector<Tick>& split, bool backward) {
  std::vector<Tick> costs;
  costs.reserve(static_cast<std::size_t>(instance.stages));
  for (Index stage = 0; stage < instance.stages; ++stage) {
    costs.push_back(instance.Duration(stage, backward, split));
  }
  return costs;
}

}  // namespace

void ValidateMegatronSlackPipePlanExport(const Instance& instance,
                                         const ScheduleSolution& schedule) {
  instance.Validate();
  ValidateSplit(instance, schedule.split);
  if (!schedule.ok()) {
    std::ostringstream msg;
    msg << "cannot export invalid SlackPipe schedule";
    if (!schedule.validation_errors.empty())
      msg << ": " << schedule.validation_errors.front();
    throw Error(msg.str());
  }
  if (schedule.orders.size() != static_cast<std::size_t>(instance.workers)) {
    throw Error("cannot export SlackPipe plan: worker order count mismatch");
  }

  const Index expected_count = instance.OperationCount();
  std::vector<Index> seen(static_cast<std::size_t>(expected_count), 0);
  Index actual_count = 0;
  for (Index worker = 0; worker < instance.workers; ++worker) {
    for (OperationId id : schedule.orders[static_cast<std::size_t>(worker)]) {
      if (id.value < 0 || id.value >= expected_count) {
        throw Error("cannot export SlackPipe plan: operation id out of range");
      }
      const OperationView view = DecodeOperation(instance, id);
      if (view.worker != worker) {
        std::ostringstream msg;
        msg << "cannot export SlackPipe plan: " << OperationName(instance, id)
            << " assigned to worker " << worker << " but expected "
            << view.worker;
        throw Error(msg.str());
      }
      ++seen[static_cast<std::size_t>(id.value)];
      ++actual_count;
    }
  }
  if (actual_count != expected_count) {
    std::ostringstream msg;
    msg << "cannot export SlackPipe plan: expected " << expected_count
        << " operations, found " << actual_count;
    throw Error(msg.str());
  }
  for (Index id_value = 0; id_value < expected_count; ++id_value) {
    const Index count = seen[static_cast<std::size_t>(id_value)];
    if (count != 1) {
      std::ostringstream msg;
      msg << "cannot export SlackPipe plan: operation "
          << OperationName(instance, OperationId{id_value}) << " appears "
          << count << " times";
      throw Error(msg.str());
    }
  }
}

std::string ToMegatronSlackPipePlanJson(const Instance& instance,
                                        const ScheduleSolution& schedule,
                                        const std::string& solver_status) {
  ValidateMegatronSlackPipePlanExport(instance, schedule);

  const bool export_v2 = instance.HasRangeCostProfile();
  std::ostringstream out;
  out << "{\n";
  out << "  \"schema_version\": \""
      << (export_v2 ? kMegatronSlackPipePlanSchemaVersionV2
                    : kMegatronSlackPipePlanSchemaVersion)
      << "\",\n";
  out << "  \"num_microbatches\": " << instance.microbatches << ",\n";
  out << "  \"num_stages\": " << instance.stages << ",\n";
  out << "  \"num_workers\": " << instance.workers << ",\n";
  out << "  \"num_layers\": " << instance.total_layers << ",\n";
  out << "  \"layer_split\": ";
  WriteTickArray(out, schedule.split);
  out << ",\n";
  if (export_v2) {
    out << "  \"stage_layer_ranges\": ";
    WriteStageLayerRanges(out, instance, schedule.split);
    out << ",\n";
    if (!instance.model_manifest_hash.empty()) {
      out << "  \"model_manifest_hash\": \""
          << JsonEscape(instance.model_manifest_hash) << "\",\n";
    }
    if (!instance.cost_profile_hash.empty()) {
      out << "  \"cost_profile_hash\": \""
          << JsonEscape(instance.cost_profile_hash) << "\",\n";
    }
    out << "  \"cost_profile_version\": \""
        << JsonEscape(instance.cost_profile_schema_version) << "\",\n";
  }
  out << "  \"stage_to_worker\": ";
  WriteTickArray(out, StageToWorker(instance));
  out << ",\n";
  out << "  \"operations\": [\n";
  for (Index worker = 0; worker < instance.workers; ++worker) {
    const auto& worker_ops = schedule.orders[static_cast<std::size_t>(worker)];
    out << "    [\n";
    for (std::size_t i = 0; i < worker_ops.size(); ++i) {
      const OperationView view = DecodeOperation(instance, worker_ops[i]);
      out << "      {\"kind\": \"" << (view.backward ? "B" : "F")
          << "\", \"microbatch\": " << view.microbatch
          << ", \"stage\": " << view.stage << "}";
      if (i + 1 != worker_ops.size()) out << ",";
      out << "\n";
    }
    out << "    ]";
    if (worker + 1 != instance.workers) out << ",";
    out << "\n";
  }
  out << "  ],\n";
  out << "  \"solver_status\": \"" << JsonEscape(solver_status) << "\",\n";
  out << "  \"predicted_makespan\": " << schedule.makespan << ",\n";
  WriteCostModelMetadata(out, instance);
  out << ",\n";
  out << "  \"forward_costs\": ";
  WriteTickArray(out, StageCosts(instance, schedule.split, false));
  out << ",\n";
  out << "  \"backward_costs\": ";
  WriteTickArray(out, StageCosts(instance, schedule.split, true));
  out << "\n";
  out << "}\n";
  return out.str();
}

void WriteMegatronSlackPipePlanFile(const std::string& path,
                                    const Instance& instance,
                                    const ScheduleSolution& schedule,
                                    const std::string& solver_status) {
  const std::string contents =
      ToMegatronSlackPipePlanJson(instance, schedule, solver_status);
  const std::filesystem::path final_path(path);
  const std::filesystem::path tmp_path = final_path.string() + ".tmp";
  std::error_code ignored;
  std::filesystem::remove(tmp_path, ignored);

  try {
    std::ofstream out(tmp_path);
    if (!out) throw Error("failed to open output file: " + tmp_path.string());
    out << contents;
    out.close();
    if (!out) throw Error("failed to write output file: " + tmp_path.string());
    std::filesystem::rename(tmp_path, final_path);
  } catch (...) {
    std::filesystem::remove(tmp_path, ignored);
    throw;
  }
}

}  // namespace slackpipe
