#pragma once

#include <cstdint>
#include <string>

namespace slackpipe {

struct SearchStats {
  std::string algorithm;
  bool enumerative_search = false;

  bool worker_partitions_enabled = false;
  std::string worker_partitions_reason =
      "Stage-to-worker mapping is fixed by StageOfOp(n,N)%J.";
  bool worker_partitions_theoretical_available = false;
  std::int64_t worker_partitions_theoretical = 0;
  bool worker_partitions_enumerated = false;
  std::int64_t worker_partitions_visited = 0;
  std::int64_t worker_partitions_pruned = 0;
  std::int64_t worker_partitions_kept = 0;

  bool stage_partitions_theoretical_available = false;
  std::int64_t stage_partitions_theoretical = 0;
  std::string stage_partitions_note =
      "Stage partition visited/pruned/kept counts are populated only when "
      "stage partitions are explicitly enumerated by an outer loop.";
  bool stage_partitions_enumerated = false;
  std::int64_t stage_partitions_visited = 0;
  std::int64_t stage_partitions_pruned = 0;
  std::int64_t stage_partitions_kept = 0;

  bool interleave_orders_theoretical_available = false;
  std::int64_t interleave_orders_theoretical = 0;
  std::string interleave_orders_note =
      "Interleave visited/deduplicated/evaluated/kept counts are populated "
      "only when worker-local orders are explicitly enumerated by an outer "
      "loop.";
  bool interleave_orders_enumerated = false;
  std::int64_t interleave_orders_visited = 0;
  std::int64_t interleave_orders_deduplicated = 0;
  std::int64_t interleave_orders_evaluated = 0;
  std::int64_t interleave_orders_kept = 0;

  std::int64_t candidate_schedules_extracted = 0;
  std::int64_t candidate_schedules_deterministically_evaluated = 0;
  std::int64_t candidate_schedules_accepted = 0;
  std::int64_t candidate_schedules_rejected = 0;

  bool cp_sat_available = false;
  std::string cp_sat_status;
  double cp_sat_objective = 0.0;
  double cp_sat_best_bound = 0.0;
  std::int64_t cp_sat_branches = 0;
  std::int64_t cp_sat_conflicts = 0;
  double cp_sat_wall_time_seconds = 0.0;
  double cp_sat_deterministic_time = 0.0;
};

}  // namespace slackpipe
