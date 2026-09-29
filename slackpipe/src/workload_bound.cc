#include "slackpipe/workload_bound.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace slackpipe {

WorkloadLowerBound ComputeWorkloadLowerBound(const Instance& instance) {
  WorkloadLowerBound result;
  try {
    instance.Validate();
    Tick per_microbatch = 0;
    if (instance.HasRangeCostProfile()) {
      // Prefix entries and role biases have already been rounded/clamped by the
      // loader. Differences telescope in these integer arrays, not in floats.
      per_microbatch = CheckedAdd(instance.profile_prefix_forward_ticks.back(),
                                 instance.profile_prefix_backward_ticks.back(), "range workload");
      for (Index s = 0; s < instance.stages; ++s) {
        const Index role = s == 0 ? 0 : (s == instance.stages - 1 ? 2 : 1);
        per_microbatch = CheckedAdd(per_microbatch,
            CheckedAdd(instance.profile_role_forward_bias_ticks[role],
                       instance.profile_role_backward_bias_ticks[role], "role workload"),
            "range workload");
      }
      result.partition_invariant = true;
      result.reason = "integer_prefix_telescoping_plus_stage_role_biases";
    } else if (instance.HasAffineCostProfile()) {
      Tick minimum_slope = kTickMax, maximum_slope = 0;
      for (Index s = 0; s < instance.stages; ++s) {
        const Tick slope = CheckedAdd(instance.profile_forward_slope_ticks[s],
                                      instance.profile_backward_slope_ticks[s], "affine slope");
        minimum_slope = std::min(minimum_slope, slope);
        maximum_slope = std::max(maximum_slope, slope);
        const Tick bias = CheckedAdd(instance.profile_forward_bias_ticks[s],
                                    instance.profile_backward_bias_ticks[s], "affine bias");
        per_microbatch = CheckedAdd(per_microbatch,
            CheckedAdd(CheckedMul(instance.min_layers, slope, "minimum stage work"), bias,
                       "minimum stage work"), "affine workload");
      }
      const Tick remaining = instance.total_layers -
          CheckedMul(instance.stages, instance.min_layers, "minimum partition layers");
      per_microbatch = CheckedAdd(per_microbatch,
          CheckedMul(remaining, minimum_slope, "remaining layer work"), "affine workload");
      result.partition_invariant = minimum_slope == maximum_slope || remaining == 0;
      result.reason = result.partition_invariant ? "constant_affine_total_work"
                                                : "minimum_affine_work_over_all_partitions";
    } else {
      per_microbatch = CheckedMul(instance.total_layers,
          CheckedAdd(instance.backward_ratio_den, instance.backward_ratio_num, "uniform slope"),
          "uniform workload");
      result.partition_invariant = true;
      result.reason = "constant_uniform_total_work";
    }
    const Tick work = CheckedMul(instance.microbatches, per_microbatch, "total workload");
    result.total_work_ticks = work;
    result.makespan_ticks = work / instance.workers + (work % instance.workers != 0);
  } catch (const Error&) {
    result = {};
    result.reason = "unavailable_invalid_instance_or_tick_overflow";
  }
  return result;
}

LowerBoundReport ComputeLowerBoundReport(const Instance& instance,
                                        std::optional<double> raw_solver_bound,
                                        std::optional<Tick> feasible_makespan,
                                        bool raw_bound_globally_valid) {
  LowerBoundReport result;
  result.workload = ComputeWorkloadLowerBound(instance);
  result.raw_solver_bound_ticks = raw_solver_bound;
  result.raw_bound_globally_valid = raw_bound_globally_valid;
  if (raw_bound_globally_valid && raw_solver_bound && std::isfinite(*raw_solver_bound) && *raw_solver_bound > 0) {
    result.effective_lower_bound_ticks = raw_solver_bound;
  }
  if (result.workload.makespan_ticks) {
    result.effective_lower_bound_ticks = std::max(
        result.effective_lower_bound_ticks.value_or(0),
        static_cast<double>(*result.workload.makespan_ticks));
  }
  if (feasible_makespan && *feasible_makespan > 0 && result.effective_lower_bound_ticks) {
    result.inconsistent_with_incumbent = *result.effective_lower_bound_ticks > *feasible_makespan;
    if (!result.inconsistent_with_incumbent) {
      result.effective_relative_gap =
          (static_cast<double>(*feasible_makespan) - *result.effective_lower_bound_ticks) /
          static_cast<double>(*feasible_makespan);
    }
  }
  return result;
}

std::string LowerBoundReportJson(const LowerBoundReport& report) {
  std::ostringstream out;
  out << std::setprecision(17) << std::boolalpha;
  auto number = [&](const auto& value) {
    if (value && std::isfinite(static_cast<double>(*value))) out << *value;
    else out << "null";
  };
  out << "{\"raw_solver_bound_ticks\":"; number(report.raw_solver_bound_ticks);
  out << ",\"raw_bound_globally_valid\":" << report.raw_bound_globally_valid;
  out << ",\"effective_bound_scope\":\"joint_all_partitions_and_orders\"";
  out << ",\"independent_workload_lower_bound_ticks\":"; number(report.workload.makespan_ticks);
  out << ",\"total_work_lower_bound_ticks\":"; number(report.workload.total_work_ticks);
  out << ",\"total_work_partition_invariant\":" << report.workload.partition_invariant;
  out << ",\"workload_bound_reason\":\"" << report.workload.reason << "\"";
  out << ",\"effective_lower_bound_ticks\":"; number(report.effective_lower_bound_ticks);
  out << ",\"effective_relative_optimality_gap\":"; number(report.effective_relative_gap);
  out << ",\"gap_formula\":\"(feasible_makespan_ticks - effective_lower_bound_ticks) / feasible_makespan_ticks\"";
  out << ",\"inconsistent_with_incumbent\":" << report.inconsistent_with_incumbent << '}';
  return out.str();
}

}  // namespace slackpipe
