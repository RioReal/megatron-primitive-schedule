#include "slackpipe/bfs_solver.h"

#if SLACKPIPE_HAVE_ORTOOLS

#include <chrono>
#include <limits>
#include <string>
#include <vector>

#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/sat/sat_parameters.pb.h"
#include "slackpipe/breadth_first.h"
#include "slackpipe/dag_evaluator.h"

namespace slackpipe {

namespace {

using Clock = std::chrono::steady_clock;
using operations_research::Domain;
using operations_research::sat::CpModelBuilder;
using operations_research::sat::CpSolverResponse;
using operations_research::sat::CpSolverStatus;
using operations_research::sat::IntVar;
using operations_research::sat::LinearExpr;
using operations_research::sat::Model;
using operations_research::sat::NewSatParameters;
using operations_research::sat::SatParameters;
using operations_research::sat::SolutionIntegerValue;
using operations_research::sat::SolveCpModel;

[[nodiscard]] std::string StatusName(CpSolverStatus status) {
  switch (status) {
    case operations_research::sat::OPTIMAL:
      return "OPTIMAL";
    case operations_research::sat::FEASIBLE:
      return "FEASIBLE";
    case operations_research::sat::INFEASIBLE:
      return "INFEASIBLE";
    case operations_research::sat::MODEL_INVALID:
      return "MODEL_INVALID";
    case operations_research::sat::UNKNOWN:
      return "UNKNOWN";
    default:
      return "UNRECOGNIZED";
  }
}

[[nodiscard]] Tick SafeHorizon(const Instance &instance) {
  if (instance.HasCostProfile()) {
    Tick max_slope = 0;
    Tick max_bias = 0;
    for (Index stage = 0; stage < instance.stages; ++stage) {
      const auto index = static_cast<std::size_t>(stage);
      max_slope =
          std::max(max_slope, instance.profile_forward_slope_ticks[index]);
      max_slope =
          std::max(max_slope, instance.profile_backward_slope_ticks[index]);
      max_bias = std::max(max_bias, instance.profile_forward_bias_ticks[index]);
      max_bias =
          std::max(max_bias, instance.profile_backward_bias_ticks[index]);
    }
    const Tick max_duration =
        CheckedAdd(CheckedMul(instance.total_layers, max_slope,
                              "breadth-first profile horizon duration"),
                   max_bias, "breadth-first profile horizon duration");
    return CheckedMul(instance.OperationCount(), max_duration,
                      "breadth-first profile horizon");
  }
  const Tick ratio_sum =
      CheckedAdd(instance.backward_ratio_num, instance.backward_ratio_den,
                 "breadth-first horizon ratio sum");
  return CheckedMul(CheckedMul(instance.microbatches, instance.total_layers,
                               "breadth-first horizon"),
                    ratio_sum, "breadth-first horizon");
}

double Since(Clock::time_point started) {
  return std::chrono::duration<double>(Clock::now() - started).count();
}

void EmitLifecycle(const BfsSplitOptimizerOptions &options,
                   Clock::time_point started, const std::string &phase,
                   const std::string &status = "",
                   double effective_limit_seconds = 0.0,
                   const std::string &detail = "") {
  if (!options.lifecycle) return;
  LifecycleEvent event;
  event.phase = phase;
  event.algorithm = "optimize-bfs";
  event.solver_status = status;
  event.elapsed_seconds = Since(started);
  event.configured_solver_limit_seconds = options.time_limit_seconds;
  event.effective_solver_limit_seconds = effective_limit_seconds;
  event.requested_solver_limit_seconds = options.time_limit_seconds;
  event.phase_specific_cap_seconds = options.time_limit_seconds;
  event.solver_threads = options.num_workers;
  event.detail = detail;
  options.lifecycle(event);
}

}  // namespace

BfsSplitOptimizationResult OptimizeBfsSplitCpSat(
    const Instance &instance, const BfsSplitOptimizerOptions &options) {
  const auto started = Clock::now();
  instance.Validate();
  SearchStats *stats = options.search_stats;
  if (stats != nullptr) {
    constexpr std::uint64_t kMaxCount =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    const std::uint64_t count = CountValidSplitsCapped(instance, kMaxCount);
    if (count < kMaxCount) {
      stats->stage_partitions_theoretical_available = true;
      stats->stage_partitions_theoretical = static_cast<std::int64_t>(count);
    }
  }
  const Tick horizon = SafeHorizon(instance);
  const MachineOrders orders = BreadthFirstOrders(instance);

  const auto model_build_started = Clock::now();
  EmitLifecycle(options, started, "MODEL_BUILD_START", "", 0.0,
                "bfs_incumbent");
  CpModelBuilder builder;
  std::vector<IntVar> layers;
  layers.reserve(static_cast<std::size_t>(instance.stages));
  for (Index s = 0; s < instance.stages; ++s) {
    layers.push_back(
        builder.NewIntVar(Domain(instance.min_layers, instance.total_layers))
            .WithName("layers_" + std::to_string(s)));
  }
  builder.AddEquality(LinearExpr::Sum(layers), instance.total_layers);

  const Index op_count = instance.OperationCount();
  std::vector<IntVar> starts;
  std::vector<IntVar> ends;
  starts.reserve(static_cast<std::size_t>(op_count));
  ends.reserve(static_cast<std::size_t>(op_count));
  for (Index id = 0; id < op_count; ++id) {
    starts.push_back(builder.NewIntVar(Domain(0, horizon))
                         .WithName("start_" + std::to_string(id)));
    ends.push_back(builder.NewIntVar(Domain(0, horizon))
                       .WithName("end_" + std::to_string(id)));
  }

  for (Index id = 0; id < op_count; ++id) {
    const OperationView view = DecodeOperation(instance, OperationId{id});
    const Tick coefficient =
        instance.HasCostProfile()
            ? (view.backward
                   ? instance
                         .profile_backward_slope_ticks[static_cast<std::size_t>(
                             view.stage)]
                   : instance
                         .profile_forward_slope_ticks[static_cast<std::size_t>(
                             view.stage)])
            : (view.backward ? instance.backward_ratio_num
                             : instance.backward_ratio_den);
    const Tick bias =
        instance.HasCostProfile()
            ? (view.backward
                   ? instance
                         .profile_backward_bias_ticks[static_cast<std::size_t>(
                             view.stage)]
                   : instance
                         .profile_forward_bias_ticks[static_cast<std::size_t>(
                             view.stage)])
            : 0;
    builder.AddEquality(
        ends[static_cast<std::size_t>(id)],
        starts[static_cast<std::size_t>(id)] +
            coefficient * layers[static_cast<std::size_t>(view.stage)] + bias);

    if (view.chain_index + 1 < 2 * instance.stages) {
      const OperationId next =
          EncodeOperation(instance, view.microbatch, view.chain_index + 1);
      const OperationView next_view = DecodeOperation(instance, next);
      builder.AddGreaterOrEqual(
          starts[static_cast<std::size_t>(next.value)],
          ends[static_cast<std::size_t>(id)] +
              instance.EdgeDelay(view.worker, next_view.worker));
    }
  }

  for (const auto &worker_order : orders) {
    for (std::size_t i = 1; i < worker_order.size(); ++i) {
      builder.AddGreaterOrEqual(
          starts[static_cast<std::size_t>(worker_order[i].value)],
          ends[static_cast<std::size_t>(worker_order[i - 1].value)]);
    }
  }

  IntVar makespan = builder.NewIntVar(Domain(0, horizon)).WithName("makespan");
  for (const IntVar &end : ends) builder.AddGreaterOrEqual(makespan, end);

  if (!instance.HasCostProfile()) {
    const Tick ratio_sum =
        CheckedAdd(instance.backward_ratio_num, instance.backward_ratio_den,
                   "critical-chain lower bound");
    builder.AddGreaterOrEqual(makespan,
                              CheckedMul(instance.total_layers, ratio_sum,
                                         "critical-chain lower bound"));
    for (Index w = 0; w < instance.workers; ++w) {
      LinearExpr worker_layers;
      for (Index s = 0; s < instance.stages; ++s) {
        if (s % instance.workers == w) {
          worker_layers += layers[static_cast<std::size_t>(s)];
        }
      }
      builder.AddGreaterOrEqual(makespan,
                                CheckedMul(instance.microbatches, ratio_sum,
                                           "worker-load lower bound") *
                                    worker_layers);
    }
  }
  builder.Minimize(makespan);

  Model model;
  SatParameters parameters;
  parameters.set_num_workers(options.num_workers);
  parameters.set_random_seed(options.random_seed);
  parameters.set_log_search_progress(options.log_search_progress);
  if (options.time_limit_seconds > 0.0) {
    parameters.set_max_time_in_seconds(options.time_limit_seconds);
  }
  model.Add(NewSatParameters(parameters));
  const auto cp_model = builder.Build();
  const auto solve_started = Clock::now();
  const double model_build_seconds =
      std::chrono::duration<double>(solve_started - model_build_started)
          .count();
  EmitLifecycle(options, started, "MODEL_BUILD_END", "", 0.0, "bfs_incumbent");
  EmitLifecycle(options, started, "SOLVE_START", "",
                parameters.max_time_in_seconds(), "bfs_incumbent");
  const CpSolverResponse response = SolveCpModel(cp_model, &model);
  const double solver_seconds =
      std::chrono::duration<double>(Clock::now() - solve_started).count();
  EmitLifecycle(options, started, "SOLVE_END", StatusName(response.status()),
                parameters.max_time_in_seconds(), "bfs_incumbent");

  BfsSplitOptimizationResult result;
  result.method = "cpsat";
  result.status = StatusName(response.status());
  result.solver_status_raw = result.status;
  result.best_bound_ticks = static_cast<Tick>(response.best_objective_bound());
  result.solver_objective_ticks = response.objective_value();
  result.wall_time_seconds =
      std::chrono::duration<double>(Clock::now() - started).count();
  result.timing.model_build_seconds = model_build_seconds;
  result.timing.solver_seconds = solver_seconds;
  result.timing.ortools_wall_time_seconds = response.wall_time();
  result.branches = response.num_branches();
  result.conflicts = response.num_conflicts();
  result.deterministic_time = response.deterministic_time();
  result.proven_optimal =
      response.status() == operations_research::sat::OPTIMAL;
  result.machine_orders = orders;
  result.cp_sat_launched = true;
  result.cp_sat_models_solved = 1;
  result.solution_source = "none";
  result.returned_solution_source = "none";
  result.search_stats_enabled = stats != nullptr;
  if (stats != nullptr) {
    if (stats->algorithm.empty()) stats->algorithm = "optimize-bfs";
    stats->enumerative_search = false;
    stats->stage_partitions_note =
        "Stage partitions are encoded as CP-SAT "
        "variables, not explicitly enumerated.";
    stats->interleave_orders_note =
        "Worker-local order is fixed for BFS CP-SAT split optimization, not "
        "explicitly enumerated.";
    stats->cp_sat_available = true;
    stats->cp_sat_status = result.status;
    stats->cp_sat_objective = result.solver_objective_ticks;
    stats->cp_sat_best_bound = result.best_bound_ticks;
    stats->cp_sat_branches = result.branches;
    stats->cp_sat_conflicts = result.conflicts;
    stats->cp_sat_wall_time_seconds = result.timing.ortools_wall_time_seconds;
    stats->cp_sat_deterministic_time = response.deterministic_time();
  }

  if (response.status() == operations_research::sat::OPTIMAL ||
      response.status() == operations_research::sat::FEASIBLE) {
    const auto extraction_started = Clock::now();
    EmitLifecycle(options, started, "SOLUTION_EXTRACTION_START", result.status,
                  parameters.max_time_in_seconds(), "bfs_incumbent");
    result.makespan_ticks = SolutionIntegerValue(response, makespan);
    result.solution_source = "bfs_cpsat_solution";
    result.returned_solution_source = result.solution_source;
    result.split.reserve(static_cast<std::size_t>(instance.stages));
    for (const IntVar &layer : layers) {
      result.split.push_back(SolutionIntegerValue(response, layer));
    }
    result.timing.extraction_seconds =
        std::chrono::duration<double>(Clock::now() - extraction_started)
            .count();
    result.time_to_first_feasible_seconds =
        std::chrono::duration<double>(extraction_started - started).count();
    result.time_to_best_incumbent_seconds =
        result.time_to_first_feasible_seconds;
    result.first_feasible_objective = result.makespan_ticks;
    result.incumbent_improvement_count = 1;
    EmitLifecycle(options, started, "SOLUTION_EXTRACTION_END", result.status,
                  parameters.max_time_in_seconds(), "bfs_incumbent");
    if (stats != nullptr) {
      ++stats->candidate_schedules_extracted;
    }
    const auto canonical_started = Clock::now();
    EmitLifecycle(options, started, "REPLAY_START", result.status,
                  parameters.max_time_in_seconds(), "bfs_incumbent");
    ApplyBfsCanonicalEvaluationPolicy(instance, result, result.makespan_ticks,
                                      stats);
    result.timing.canonicalization_seconds =
        std::chrono::duration<double>(Clock::now() - canonical_started).count();
    EmitLifecycle(options, started, "REPLAY_END", result.status,
                  parameters.max_time_in_seconds(), "bfs_incumbent");
  }

  if (options.require_optimal && !result.proven_optimal) {
    result.status += "_REJECTED_REQUIRE_OPTIMAL";
  }
  result.timing.total_seconds =
      std::chrono::duration<double>(Clock::now() - started).count();
  result.wall_time_seconds = result.timing.total_seconds;
  if (stats != nullptr) result.search_stats = *stats;
  return result;
}

bool IsCpSatBfsSplitOptimizerAvailable() { return true; }

}  // namespace slackpipe

#endif
