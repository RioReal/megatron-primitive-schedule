#pragma once

#include <algorithm>

namespace slackpipe {

struct SolverPhaseTiming {
  double incumbent_seconds = 0.0;
  double model_build_seconds = 0.0;
  double solver_seconds = 0.0;
  double ortools_wall_time_seconds = 0.0;
  double extraction_seconds = 0.0;
  double canonicalization_seconds = 0.0;
  double policy_seconds = 0.0;
  double serialization_seconds = 0.0;
  double total_seconds = 0.0;

  [[nodiscard]] double ExplicitNonOverlappingSeconds() const {
    return incumbent_seconds + model_build_seconds + solver_seconds +
           extraction_seconds + canonicalization_seconds + policy_seconds +
           serialization_seconds;
  }

  [[nodiscard]] double OrchestrationSeconds() const {
    return std::max(0.0, total_seconds - ExplicitNonOverlappingSeconds());
  }
};

inline void AddTiming(SolverPhaseTiming& aggregate,
                      const SolverPhaseTiming& part) {
  aggregate.incumbent_seconds += part.incumbent_seconds;
  aggregate.model_build_seconds += part.model_build_seconds;
  aggregate.solver_seconds += part.solver_seconds;
  aggregate.ortools_wall_time_seconds += part.ortools_wall_time_seconds;
  aggregate.extraction_seconds += part.extraction_seconds;
  aggregate.canonicalization_seconds += part.canonicalization_seconds;
  aggregate.policy_seconds += part.policy_seconds;
  aggregate.serialization_seconds += part.serialization_seconds;
}

}  // namespace slackpipe
