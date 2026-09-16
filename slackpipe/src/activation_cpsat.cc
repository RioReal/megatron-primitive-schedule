#include "slackpipe/activation_cpsat.h"

#if SLACKPIPE_HAVE_ORTOOLS

#ifndef SLACKPIPE_HAVE_CUMULATIVE
#define SLACKPIPE_HAVE_CUMULATIVE SLACKPIPE_CUMULATIVE_CONSTRAINT_SUPPORTED
#endif

#ifndef SLACKPIPE_HAVE_VARIABLE_CUMULATIVE_DEMAND
#define SLACKPIPE_HAVE_VARIABLE_CUMULATIVE_DEMAND \
  SLACKPIPE_VARIABLE_CUMULATIVE_DEMAND_SUPPORTED
#endif

#include <chrono>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "slackpipe/operation.h"

namespace slackpipe {
namespace {

using Clock = std::chrono::steady_clock;
using operations_research::Domain;
using operations_research::sat::CpModelBuilder;
using operations_research::sat::IntervalVar;
using operations_research::sat::IntVar;
using operations_research::sat::LinearExpr;

[[nodiscard]] double Since(Clock::time_point started) {
  return std::chrono::duration<double>(Clock::now() - started).count();
}

[[nodiscard]] OperationId ForwardOperationId(const Instance &instance,
                                             Index microbatch, Index stage) {
  return EncodeOperation(instance, microbatch, stage);
}

[[nodiscard]] OperationId BackwardOperationId(const Instance &instance,
                                              Index microbatch, Index stage) {
  return EncodeOperation(instance, microbatch, 2 * instance.stages - 1 - stage);
}

[[nodiscard]] Tick FixedDemandForStage(const ActivationAnalysisOptions &options,
                                       const std::vector<Tick> &split,
                                       Index stage) {
  switch (options.model) {
    case ActivationModel::kCount:
      return 1;
    case ActivationModel::kLinearInStageLayers:
      return CheckedMul(options.activation_units_per_layer,
                        split[static_cast<std::size_t>(stage)],
                        "activation CP-SAT demand");
    case ActivationModel::kExplicitStageUnits:
      return options
          .explicit_stage_activation_units[static_cast<std::size_t>(stage)];
  }
  throw Error("unsupported activation model");
}

[[nodiscard]] std::string RetainedIntervalName(Index microbatch, Index stage) {
  std::ostringstream out;
  out << "activation_retained_b" << microbatch << "_s" << stage;
  return out.str();
}

[[nodiscard]] std::string RetainedSizeName(Index microbatch, Index stage) {
  std::ostringstream out;
  out << "activation_retained_size_b" << microbatch << "_s" << stage;
  return out.str();
}

void RecordActivationCapTerm(ActivationCapModelDebugDump *debug_dump,
                             Index microbatch, Index stage, Index worker,
                             const IntVar &start, const IntVar &end,
                             const IntVar &size, const std::string &demand_type,
                             const std::string &demand_source,
                             Tick capacity_units) {
  if (debug_dump == nullptr) return;
  ActivationCapModelTerm term;
  term.microbatch = microbatch;
  term.stage = stage;
  term.worker = worker;
  term.start_variable_name = start.Name();
  term.end_variable_name = end.Name();
  term.size_variable_name = size.Name();
  term.demand_type = demand_type;
  term.demand_source = demand_source;
  term.capacity_units = capacity_units;
  term.included_in_cumulative = true;
  debug_dump->terms.push_back(std::move(term));
}

}  // namespace

ActivationCapConstraintMetadata AddActivationCapacityConstraints(
    CpModelBuilder &builder, const Instance &instance,
    const ActivationAnalysisOptions &options, const std::vector<IntVar> &starts,
    const std::vector<IntVar> &ends, const std::vector<IntVar> *layers,
    const std::vector<Tick> *fixed_split, Tick horizon,
    bool partition_optimized, ActivationCapModelDebugDump *debug_dump) {
  const auto started = Clock::now();
  if (debug_dump != nullptr) {
    debug_dump->terms.clear();
  }
  ActivationCapConstraintMetadata metadata;
  metadata.model_support_level = ToString(ActivationCapSolverSupport());
  metadata.solver_supported =
      ActivationCapSolverCanEnforce(options, partition_optimized);
  if (!options.enforce_activation_cap ||
      options.cap_mode == ActivationCapMode::kNone) {
    metadata.build_runtime_seconds = Since(started);
    return metadata;
  }
  if (!metadata.solver_supported) {
    metadata.unsupported_reason =
        ActivationCapSolverUnsupportedReason(options, partition_optimized);
    metadata.build_runtime_seconds = Since(started);
    return metadata;
  }
  if (options.cap_mode == ActivationCapMode::kUniformBaseline &&
      !options.uniform_baseline) {
    metadata.solver_supported = false;
    metadata.unsupported_reason =
        "uniform-baseline activation cap solver enforcement requires a "
        "pre-resolved baseline cap";
    metadata.build_runtime_seconds = Since(started);
    return metadata;
  }

#if !SLACKPIPE_HAVE_CUMULATIVE
  metadata.solver_supported = false;
  metadata.unsupported_reason =
      "activation cap solver enforcement requires OR-Tools cumulative "
      "constraint support";
  metadata.build_runtime_seconds = Since(started);
  return metadata;
#else
  ValidateActivationOptions(instance, options);
  const std::vector<Tick> cap_units =
      ResolveActivationCapUnits(instance, options);
  if (cap_units.size() != static_cast<std::size_t>(instance.workers)) {
    metadata.solver_supported = false;
    metadata.unsupported_reason =
        "activation cap resolution did not produce a W-length cap vector";
    metadata.build_runtime_seconds = Since(started);
    return metadata;
  }
  if (starts.size() != static_cast<std::size_t>(instance.OperationCount()) ||
      ends.size() != static_cast<std::size_t>(instance.OperationCount())) {
    throw Error("activation CP-SAT constraints received incomplete timings");
  }
  if (partition_optimized &&
      options.model == ActivationModel::kLinearInStageLayers &&
      layers == nullptr) {
    throw Error("activation CP-SAT variable demand requires layer variables");
  }
  if ((!partition_optimized ||
       options.model != ActivationModel::kLinearInStageLayers) &&
      fixed_split == nullptr &&
      options.model == ActivationModel::kLinearInStageLayers) {
    throw Error("activation CP-SAT fixed demand requires a fixed split");
  }
  if (fixed_split != nullptr) {
    ValidateSplit(instance, *fixed_split);
  }

  struct RetainedInterval {
    IntervalVar interval;
    Index microbatch = 0;
    Index stage = 0;
    Index worker = 0;
    IntVar start;
    IntVar end;
    IntVar size;
  };
  std::vector<std::vector<RetainedInterval> > by_worker(
      static_cast<std::size_t>(instance.workers));
  for (Index b = 0; b < instance.microbatches; ++b) {
    for (Index s = 0; s < instance.stages; ++s) {
      const OperationId forward = ForwardOperationId(instance, b, s);
      const OperationId backward = BackwardOperationId(instance, b, s);
      IntVar size = builder.NewIntVar(Domain(0, horizon))
                        .WithName(RetainedSizeName(b, s));
      builder.AddEquality(size,
                          starts[static_cast<std::size_t>(backward.value)] -
                              ends[static_cast<std::size_t>(forward.value)]);
      builder.AddGreaterOrEqual(size, 0);
      IntervalVar retained =
          builder
              .NewIntervalVar(ends[static_cast<std::size_t>(forward.value)],
                              size,
                              starts[static_cast<std::size_t>(backward.value)])
              .WithName(RetainedIntervalName(b, s));
      const Index worker = s % instance.workers;
      by_worker[static_cast<std::size_t>(worker)].push_back(RetainedInterval{
          retained,
          b,
          s,
          worker,
          ends[static_cast<std::size_t>(forward.value)],
          starts[static_cast<std::size_t>(backward.value)],
          size,
      });
      ++metadata.retained_interval_count;
    }
  }

  for (Index w = 0; w < instance.workers; ++w) {
    const std::vector<RetainedInterval> &intervals =
        by_worker[static_cast<std::size_t>(w)];
    if (intervals.empty()) {
      continue;
    }
    const Tick capacity = cap_units[static_cast<std::size_t>(w)];
    operations_research::sat::CumulativeConstraint cumulative =
        builder.AddCumulative(capacity);
    ++metadata.cumulative_constraint_count;
    metadata.workers_with_constraints.push_back(w);
    for (const RetainedInterval &retained : intervals) {
      if (partition_optimized &&
          options.model == ActivationModel::kLinearInStageLayers) {
#if SLACKPIPE_HAVE_VARIABLE_CUMULATIVE_DEMAND
        LinearExpr demand = options.activation_units_per_layer *
                            (*layers)[static_cast<std::size_t>(retained.stage)];
        cumulative.AddDemand(retained.interval, demand);
        RecordActivationCapTerm(
            debug_dump, retained.microbatch, retained.stage, retained.worker,
            retained.start, retained.end, retained.size, "variable",
            "partition_variable_stage_" + std::to_string(retained.stage) + ":" +
                (*layers)[static_cast<std::size_t>(retained.stage)].Name(),
            capacity);
        ++metadata.variable_demand_count;
#else
        metadata.solver_supported = false;
        metadata.unsupported_reason =
            "optimized linear activation demand requires variable cumulative "
            "demand support";
        metadata.build_runtime_seconds = Since(started);
        return metadata;
#endif
      } else {
        Tick demand = 1;
        if (options.model == ActivationModel::kExplicitStageUnits) {
          demand =
              options.explicit_stage_activation_units[static_cast<std::size_t>(
                  retained.stage)];
        } else if (options.model == ActivationModel::kLinearInStageLayers) {
          demand = FixedDemandForStage(options, *fixed_split, retained.stage);
        }
        if (demand < 0) {
          metadata.solver_supported = false;
          metadata.unsupported_reason =
              "activation cap solver enforcement requires non-negative "
              "demands";
          metadata.build_runtime_seconds = Since(started);
          return metadata;
        }
        cumulative.AddDemand(retained.interval, demand);
        RecordActivationCapTerm(debug_dump, retained.microbatch, retained.stage,
                                retained.worker, retained.start, retained.end,
                                retained.size, "fixed", std::to_string(demand),
                                capacity);
        ++metadata.fixed_demand_count;
      }
    }
  }
  metadata.constraints_added = metadata.cumulative_constraint_count > 0;
  metadata.solver_supported = metadata.constraints_added;
  metadata.build_runtime_seconds = Since(started);
  return metadata;
#endif
}

}  // namespace slackpipe

#endif
