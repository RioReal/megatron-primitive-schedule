#include "slackpipe/interleaving_stats.h"

#include <algorithm>
#include <map>
#include <optional>
#include <sstream>
#include <tuple>

namespace slackpipe {
namespace {

struct OrderedOperation {
  OperationId id;
  OperationView view;
  Tick start = 0;
  Tick end = 0;
};

struct SummaryKey {
  Index worker = 0;
  std::string kind;
  Index prev_stage_position = -1;
  Index next_stage_position = -1;

  friend bool operator<(const SummaryKey& a, const SummaryKey& b) {
    return std::tuple<Index, std::string, Index, Index>{
               a.worker, a.kind, a.prev_stage_position, a.next_stage_position} <
           std::tuple<Index, std::string, Index, Index>{
               b.worker, b.kind, b.prev_stage_position, b.next_stage_position};
  }
};

struct SummaryValue {
  Index count = 0;
  Tick sum_gap = 0;
  Tick min_gap = 0;
  Tick max_gap = 0;
};

[[nodiscard]] std::vector<Index> WorkerLocalStagePositions(
    const Instance& instance) {
  std::vector<Index> stage_position(static_cast<std::size_t>(instance.stages),
                                    -1);
  std::vector<Index> next_pos(static_cast<std::size_t>(instance.workers), 0);
  for (Index stage = 0; stage < instance.stages; ++stage) {
    const Index worker = stage % instance.workers;
    stage_position[static_cast<std::size_t>(stage)] =
        next_pos[static_cast<std::size_t>(worker)]++;
  }
  return stage_position;
}

[[nodiscard]] char Direction(const OperationView& view) {
  return view.backward ? 'B' : 'F';
}

[[nodiscard]] int DirectionSortKey(const OperationView& view) {
  return view.backward ? 1 : 0;
}

[[nodiscard]] std::string Kind(char prev_dir, char next_dir) {
  std::string kind;
  kind.push_back(prev_dir);
  kind += "_to_";
  kind.push_back(next_dir);
  return kind;
}

void WriteRunColumns(std::ostringstream& out, const Instance& instance,
                     const InterleavingRunMetadata& metadata) {
  out << instance.microbatches << ',' << instance.stages << ','
      << instance.workers << ',' << instance.total_layers << ','
      << metadata.makespan_ticks << ',' << metadata.status << ',';
  if (metadata.objective_bound) out << *metadata.objective_bound;
  out << ',';
}

void WriteOptionalIndex(std::ostringstream& out,
                        const std::optional<Index>& value) {
  if (value) out << *value;
}

}  // namespace

InterleavingStats CollectInterleavingStats(const Instance& instance,
                                           const ScheduleSolution& schedule) {
  instance.Validate();
  if (!schedule.ok()) {
    throw Error("cannot collect interleaving stats from invalid schedule");
  }
  if (schedule.operations_by_id.size() !=
      static_cast<std::size_t>(instance.OperationCount())) {
    throw Error(
        "cannot collect interleaving stats without complete operations");
  }

  const std::vector<Index> stage_position = WorkerLocalStagePositions(instance);
  std::vector<std::vector<OrderedOperation>> by_worker(
      static_cast<std::size_t>(instance.workers));
  for (const ScheduledOperation& op : schedule.operations_by_id) {
    const OperationView view = DecodeOperation(instance, op.id);
    by_worker[static_cast<std::size_t>(view.worker)].push_back(
        OrderedOperation{op.id, view, op.start, op.end});
  }

  InterleavingStats stats;
  for (Index worker = 0; worker < instance.workers; ++worker) {
    auto& ops = by_worker[static_cast<std::size_t>(worker)];
    std::sort(ops.begin(), ops.end(),
              [](const OrderedOperation& a, const OrderedOperation& b) {
                return std::tuple<Tick, Tick, int, Index, Index>{
                           a.start, a.end, DirectionSortKey(a.view),
                           a.view.microbatch, a.view.stage} <
                       std::tuple<Tick, Tick, int, Index, Index>{
                           b.start, b.end, DirectionSortKey(b.view),
                           b.view.microbatch, b.view.stage};
              });

    const Index num_worker_ops = static_cast<Index>(ops.size());
    const double order_denominator =
        static_cast<double>(std::max<Index>(1, num_worker_ops - 1));
    for (std::size_t i = 1; i < ops.size(); ++i) {
      const OrderedOperation& prev = ops[i - 1];
      const OrderedOperation& next = ops[i];
      const Index prev_order_index = static_cast<Index>(i - 1);
      const Index next_order_index = static_cast<Index>(i);
      const char prev_dir = Direction(prev.view);
      const char next_dir = Direction(next.view);

      InterleavingEvent event;
      event.worker = worker;
      event.kind = Kind(prev_dir, next_dir);
      event.is_direction_switch = prev_dir != next_dir;
      event.prev_dir = prev_dir;
      event.prev_b = prev.view.microbatch;
      event.prev_n = prev.view.stage;
      event.prev_stage_position =
          stage_position[static_cast<std::size_t>(prev.view.stage)];
      event.prev_start = prev.start;
      event.prev_end = prev.end;
      event.prev_order_index = prev_order_index;
      event.next_dir = next_dir;
      event.next_b = next.view.microbatch;
      event.next_n = next.view.stage;
      event.next_stage_position =
          stage_position[static_cast<std::size_t>(next.view.stage)];
      event.next_start = next.start;
      event.next_end = next.end;
      event.next_order_index = next_order_index;
      event.num_worker_ops = num_worker_ops;
      event.prev_order_fraction =
          static_cast<double>(prev_order_index) / order_denominator;
      event.next_order_fraction =
          static_cast<double>(next_order_index) / order_denominator;
      event.gap_ticks = next.start - prev.end;

      if (event.is_direction_switch) {
        const OrderedOperation& forward = prev.view.backward ? next : prev;
        const OrderedOperation& backward = prev.view.backward ? prev : next;
        event.forward_b = forward.view.microbatch;
        event.forward_n = forward.view.stage;
        event.forward_stage_position =
            stage_position[static_cast<std::size_t>(forward.view.stage)];
        event.backward_b = backward.view.microbatch;
        event.backward_n = backward.view.stage;
        event.backward_stage_position =
            stage_position[static_cast<std::size_t>(backward.view.stage)];
      }

      stats.events.push_back(event);
    }
  }
  return stats;
}

std::string ToInterleavingEventsCsv(const Instance& instance,
                                    const InterleavingRunMetadata& metadata,
                                    const InterleavingStats& stats) {
  std::ostringstream out;
  out << "B,N,W,L,makespan_ticks,status,objective_bound,worker,kind,"
         "is_direction_switch,prev_dir,prev_b,prev_n,prev_stage_position,"
         "prev_start,prev_end,next_dir,next_b,next_n,next_stage_position,"
         "next_start,next_end,prev_order_index,next_order_index,"
         "num_worker_ops,prev_order_fraction,next_order_fraction,gap_ticks,"
         "forward_b,forward_n,"
         "forward_stage_position,backward_b,backward_n,"
         "backward_stage_position\n";
  for (const InterleavingEvent& event : stats.events) {
    WriteRunColumns(out, instance, metadata);
    out << event.worker << ',' << event.kind << ','
        << (event.is_direction_switch ? "true" : "false") << ','
        << event.prev_dir << ',' << event.prev_b << ',' << event.prev_n << ','
        << event.prev_stage_position << ',' << event.prev_start << ','
        << event.prev_end << ',' << event.next_dir << ',' << event.next_b << ','
        << event.next_n << ',' << event.next_stage_position << ','
        << event.next_start << ',' << event.next_end << ','
        << event.prev_order_index << ',' << event.next_order_index << ','
        << event.num_worker_ops << ',' << event.prev_order_fraction << ','
        << event.next_order_fraction << ',' << event.gap_ticks << ',';
    WriteOptionalIndex(out, event.forward_b);
    out << ',';
    WriteOptionalIndex(out, event.forward_n);
    out << ',';
    WriteOptionalIndex(out, event.forward_stage_position);
    out << ',';
    WriteOptionalIndex(out, event.backward_b);
    out << ',';
    WriteOptionalIndex(out, event.backward_n);
    out << ',';
    WriteOptionalIndex(out, event.backward_stage_position);
    out << '\n';
  }
  return out.str();
}

std::string ToInterleavingSummaryCsv(const Instance& instance,
                                     const InterleavingRunMetadata&,
                                     const InterleavingStats& stats) {
  std::map<SummaryKey, SummaryValue> summary;
  for (const InterleavingEvent& event : stats.events) {
    SummaryKey key{event.worker, event.kind, event.prev_stage_position,
                   event.next_stage_position};
    SummaryValue& value = summary[key];
    if (value.count == 0) {
      value.min_gap = event.gap_ticks;
      value.max_gap = event.gap_ticks;
    } else {
      value.min_gap = std::min(value.min_gap, event.gap_ticks);
      value.max_gap = std::max(value.max_gap, event.gap_ticks);
    }
    ++value.count;
    value.sum_gap += event.gap_ticks;
  }

  std::ostringstream out;
  out << "B,N,W,L,worker,kind,prev_stage_position,next_stage_position,"
         "count,avg_gap_ticks,min_gap_ticks,max_gap_ticks\n";
  for (const auto& [key, value] : summary) {
    out << instance.microbatches << ',' << instance.stages << ','
        << instance.workers << ',' << instance.total_layers << ',' << key.worker
        << ',' << key.kind << ',' << key.prev_stage_position << ','
        << key.next_stage_position << ',' << value.count << ',';
    if (value.count > 0) {
      out << static_cast<double>(value.sum_gap) /
                 static_cast<double>(value.count);
    }
    out << ',' << value.min_gap << ',' << value.max_gap << '\n';
  }
  return out.str();
}

}  // namespace slackpipe
