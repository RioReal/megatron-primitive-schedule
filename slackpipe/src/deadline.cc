#include "slackpipe/deadline.h"

#include <algorithm>
#include <utility>

namespace slackpipe {

Deadline::Deadline(double requested_seconds)
    : started_(Clock::now()),
      requested_seconds_(requested_seconds),
      bounded_(requested_seconds > 0.0 && std::isfinite(requested_seconds)) {}

double Deadline::elapsed_seconds() const {
  return std::chrono::duration<double>(Clock::now() - started_).count();
}

double Deadline::remaining_seconds() const {
  if (!bounded_) return std::numeric_limits<double>::infinity();
  return std::max(0.0, requested_seconds_ - elapsed_seconds());
}

double Deadline::remaining_seconds_for_reporting() const {
  if (!bounded_) return 0.0;
  return remaining_seconds();
}

bool Deadline::expired() const {
  return bounded_ && remaining_seconds() <= 0.0;
}

double Deadline::clamp_solver_limit(double requested_phase_limit) const {
  if (!bounded_) {
    return requested_phase_limit > 0.0 ? requested_phase_limit : 0.0;
  }
  const double remaining = remaining_seconds();
  if (remaining <= 0.0) return 0.0;
  if (requested_phase_limit <= 0.0 || !std::isfinite(requested_phase_limit)) {
    return remaining;
  }
  return std::min(remaining, requested_phase_limit);
}

DeadlinePhaseTimer::DeadlinePhaseTimer(
    const Deadline& deadline, std::string phase,
    std::optional<double> requested_limit_seconds)
    : deadline_(deadline),
      phase_(std::move(phase)),
      requested_limit_seconds_(requested_limit_seconds),
      started_(Deadline::Clock::now()) {
  remaining_before_seconds_ =
      deadline_.bounded() ? std::optional<double>(deadline_.remaining_seconds())
                          : std::nullopt;
  expired_before_start_ = deadline_.expired();
  if (requested_limit_seconds_) {
    effective_limit_seconds_ =
        deadline_.clamp_solver_limit(*requested_limit_seconds_);
  } else if (deadline_.bounded()) {
    effective_limit_seconds_ = deadline_.remaining_seconds();
  }
}

DeadlinePhaseSnapshot DeadlinePhaseTimer::Finish(
    const std::string& status) const {
  DeadlinePhaseSnapshot snapshot;
  snapshot.phase = phase_;
  snapshot.requested_limit_seconds = requested_limit_seconds_;
  snapshot.effective_limit_seconds = effective_limit_seconds_;
  snapshot.remaining_before_seconds = remaining_before_seconds_;
  snapshot.remaining_after_seconds =
      deadline_.bounded() ? std::optional<double>(deadline_.remaining_seconds())
                          : std::nullopt;
  snapshot.runtime_seconds =
      std::chrono::duration<double>(Deadline::Clock::now() - started_).count();
  snapshot.expired_before_start = expired_before_start_;
  snapshot.status = status;
  return snapshot;
}

}  // namespace slackpipe
