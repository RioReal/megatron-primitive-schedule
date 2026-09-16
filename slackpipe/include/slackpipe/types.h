#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace slackpipe {

using Tick = std::int64_t;
using Index = std::int64_t;

constexpr Tick kTickMax = std::numeric_limits<Tick>::max();

class Error : public std::runtime_error {
 public:
  explicit Error(const std::string& message) : std::runtime_error(message) {}
};

[[nodiscard]] Tick CheckedAdd(Tick a, Tick b, const char* context);
[[nodiscard]] Tick CheckedMul(Tick a, Tick b, const char* context);
[[nodiscard]] Tick CheckedDivExact(Tick numerator, Tick denominator,
                                   const char* context);

}  // namespace slackpipe
