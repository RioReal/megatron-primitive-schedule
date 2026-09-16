#pragma once

#include <functional>
#include <string>

namespace slackpipe {

struct LifecycleEvent {
  std::string phase;
  std::string algorithm;
  std::string solver_status;
  double elapsed_seconds = 0.0;
  double configured_solver_limit_seconds = 0.0;
  double effective_solver_limit_seconds = 0.0;
  double requested_solver_limit_seconds = 0.0;
  double remaining_global_time_seconds = 0.0;
  double phase_specific_cap_seconds = 0.0;
  int solver_threads = 1;
  std::string detail;
};

using LifecycleCallback = std::function<void(const LifecycleEvent&)>;

}  // namespace slackpipe
