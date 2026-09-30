#pragma once

#include <functional>
#include <optional>

#include "slackpipe/dag_evaluator.h"

namespace slackpipe {

// Algorithm 1 under SlackPipe's fixed-stage, unsplit-backward representation.
// No stage dispersion/refinement, B/W split, W scheduling, or second simulator.
enum class OctoPipePhase { kPartition, kFixedStagePlacement, kScheduleFBOnly };
const char* ToString(OctoPipePhase phase);

struct OctoPipeBubbleMetrics {
  std::vector<Tick> leading, trailing, boundary, residual, bubble;
  Tick delta_b = 0;
  Tick boundary_sum = 0;
  Tick residual_sum = 0;
};

OctoPipeBubbleMetrics ComputeOctoPipeBubbleMetrics(const Instance& instance,
                                                  const ScheduleSolution& schedule);
Tick MinimumOctoPipeLayerComputeCost(const Instance& instance);
OctoPipePhase SelectOctoPipePhase(const OctoPipeBubbleMetrics& metrics, Tick t_layer);

struct OctoPipeState {
  Instance instance;
  ScheduleSolution schedule;
};

// Proposals use native orders; accepted states always pass both common validators
// and EvaluateScheduleWithPredecessors. Invalid edits return nullopt.
std::optional<OctoPipeState> EvaluateOctoPipeCandidate(
    const Instance& instance, const std::vector<Tick>& split, const MachineOrders& orders);

struct OctoPipeIteration {
  Index iteration = 0;
  double elapsed_seconds = 0;
  Tick input_makespan = 0;
  OctoPipeBubbleMetrics metrics;
  Tick t_layer = 0;
  OctoPipePhase phase = OctoPipePhase::kPartition;
  Index candidates = 0;
  Index valid_candidates = 0;
  bool accepted = false;
  Tick best_makespan = 0;
};

struct OctoPipeOptions {
  double time_limit_seconds = 0;  // Zero disables the wall-clock limit.
  Index max_iterations = 0;      // Zero disables this limit; at least one is required.
  Index candidates_per_iteration = 64;  // Scheduling only; at most 4 slots earlier.
  bool tune_placement = true;
  std::function<void(const OctoPipeIteration&, const OctoPipeState&)> progress;
};

struct OctoPipeResult {
  OctoPipeState best;
  Tick initial_makespan = 0;
  Index iterations = 0;
  double tuning_seconds = 0;
};

// Exposed for deterministic tests. These are proposals, not feasible schedules.
// Partition and placement enumerate their full local neighborhoods; scheduling
// is bounded. The main loop validates each through EvaluateOctoPipeCandidate.
std::vector<OctoPipeState> OctoPipeNeighborProposals(
    const OctoPipeState& state, OctoPipePhase phase, const OctoPipeOptions& options,
    const std::function<bool()>& stop_requested = {});

OctoPipeResult TuneOctoPipeAlgorithm1(const Instance& instance,
                                     const std::vector<Tick>& initial_split,
                                     const MachineOrders& initial_orders,
                                     const OctoPipeOptions& options);

// Diagnostic only: one complete production neighborhood, without phase selection,
// state updates, deadline truncation, or convergence search. Not called by the tuner.
struct OctoPipePhaseDiagnostic {
  OctoPipePhase phase = OctoPipePhase::kPartition;
  Tick initial_makespan = 0;
  Index generated = 0;
  Index valid = 0;
  Index improving = 0;
  std::optional<OctoPipeState> best_valid;
};

OctoPipePhaseDiagnostic DiagnoseOctoPipePhase(
    const OctoPipeState& initial, OctoPipePhase phase,
    const OctoPipeOptions& options = {});

}  // namespace slackpipe
