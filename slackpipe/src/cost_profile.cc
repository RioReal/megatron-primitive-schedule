#include "slackpipe/cost_profile.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace slackpipe {

namespace {

std::string ReadTextFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw Error("failed to open cost profile: " + path);
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

std::string ExtractString(const std::string& text, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  const std::size_t key_pos = text.find(needle);
  if (key_pos == std::string::npos) {
    throw Error("cost profile missing string field: " + key);
  }
  const std::size_t colon = text.find(':', key_pos + needle.size());
  if (colon == std::string::npos) {
    throw Error("cost profile malformed string field: " + key);
  }
  const std::size_t start = text.find('"', colon + 1);
  if (start == std::string::npos) {
    throw Error("cost profile malformed string field: " + key);
  }
  const std::size_t end = text.find('"', start + 1);
  if (end == std::string::npos) {
    throw Error("cost profile malformed string field: " + key);
  }
  return text.substr(start + 1, end - start - 1);
}

std::string ExtractStringOptional(const std::string& text,
                                  const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  if (text.find(needle) == std::string::npos) return "";
  return ExtractString(text, key);
}

std::string ExtractObjectText(const std::string& text, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  const std::size_t key_pos = text.find(needle);
  if (key_pos == std::string::npos) {
    throw Error("cost profile missing object field: " + key);
  }
  const std::size_t open = text.find('{', key_pos + needle.size());
  if (open == std::string::npos) {
    throw Error("cost profile malformed object field: " + key);
  }
  int depth = 0;
  bool in_string = false;
  bool escaped = false;
  for (std::size_t pos = open; pos < text.size(); ++pos) {
    const char ch = text[pos];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (ch == '\\') {
      escaped = in_string;
      continue;
    }
    if (ch == '"') {
      in_string = !in_string;
      continue;
    }
    if (in_string) continue;
    if (ch == '{') {
      ++depth;
    } else if (ch == '}') {
      --depth;
      if (depth == 0) return text.substr(open, pos - open + 1);
    }
  }
  throw Error("cost profile malformed object field: " + key);
}

double ExtractNumber(const std::string& text, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  const std::size_t key_pos = text.find(needle);
  if (key_pos == std::string::npos) {
    throw Error("cost profile missing numeric field: " + key);
  }
  const std::size_t colon = text.find(':', key_pos + needle.size());
  if (colon == std::string::npos) {
    throw Error("cost profile malformed numeric field: " + key);
  }
  const std::size_t start = text.find_first_of("-0123456789", colon + 1);
  if (start == std::string::npos) {
    throw Error("cost profile malformed numeric field: " + key);
  }
  std::size_t parsed = 0;
  const double value = std::stod(text.substr(start), &parsed);
  if (parsed == 0 || !std::isfinite(value)) {
    throw Error("cost profile has invalid numeric field: " + key);
  }
  return value;
}

std::vector<double> ExtractNumberArray(const std::string& text,
                                       const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  const std::size_t key_pos = text.find(needle);
  if (key_pos == std::string::npos) {
    throw Error("cost profile missing array field: " + key);
  }
  const std::size_t open = text.find('[', key_pos + needle.size());
  const std::size_t close = text.find(']', open + 1);
  if (open == std::string::npos || close == std::string::npos) {
    throw Error("cost profile malformed array field: " + key);
  }
  std::vector<double> values;
  std::size_t pos = open + 1;
  while (pos < close) {
    pos = text.find_first_of("-0123456789", pos);
    if (pos == std::string::npos || pos >= close) break;
    std::size_t parsed = 0;
    const double value = std::stod(text.substr(pos, close - pos), &parsed);
    if (parsed == 0 || !std::isfinite(value)) {
      throw Error("cost profile has invalid array value: " + key);
    }
    values.push_back(value);
    pos += parsed;
  }
  return values;
}

Tick MillisecondsToMicrosecondTicks(double milliseconds,
                                    const std::string& field) {
  if (!std::isfinite(milliseconds) || milliseconds < 0.0) {
    throw Error("cost profile field must be a non-negative finite number: " +
                field);
  }
  const double ticks_double = std::round(milliseconds * 1000.0);
  if (ticks_double > static_cast<double>(kTickMax)) {
    throw Error("cost profile field overflows tick range: " + field);
  }
  Tick ticks = static_cast<Tick>(ticks_double);
  if (milliseconds > 0.0 && ticks <= 0) ticks = 1;
  return ticks;
}

std::vector<Tick> BroadcastSlope(double milliseconds, Index stages,
                                 const std::string& field) {
  const Tick ticks = MillisecondsToMicrosecondTicks(milliseconds, field);
  return std::vector<Tick>(static_cast<std::size_t>(stages), ticks);
}

std::vector<Tick> BiasTicks(const std::vector<double>& milliseconds,
                            Index stages, const std::string& field) {
  if (milliseconds.size() != static_cast<std::size_t>(stages)) {
    throw Error("cost profile " + field + " length must equal N");
  }
  std::vector<Tick> ticks;
  ticks.reserve(milliseconds.size());
  for (double value : milliseconds) {
    ticks.push_back(MillisecondsToMicrosecondTicks(value, field));
  }
  return ticks;
}

Tick MicrosecondsToTicks(double microseconds, const std::string& field) {
  if (!std::isfinite(microseconds) || microseconds < 0.0) {
    throw Error("cost profile field must be a non-negative finite number: " +
                field);
  }
  const double ticks_double = std::round(microseconds);
  if (ticks_double > static_cast<double>(kTickMax)) {
    throw Error("cost profile field overflows tick range: " + field);
  }
  Tick ticks = static_cast<Tick>(ticks_double);
  if (microseconds > 0.0 && ticks <= 0) ticks = 1;
  return ticks;
}

std::vector<Tick> MicrosecondArrayTicks(const std::vector<double>& values,
                                        const std::string& field) {
  std::vector<Tick> ticks;
  ticks.reserve(values.size());
  for (double value : values)
    ticks.push_back(MicrosecondsToTicks(value, field));
  return ticks;
}

std::vector<Tick> RoleBiasTicks(const std::string& text,
                                const std::string& direction) {
  const std::string biases = ExtractObjectText(text, "stage_role_bias_us");
  std::vector<Tick> ticks;
  ticks.reserve(3);
  for (const char* role_name : {"first", "middle", "last"}) {
    const std::string role(role_name);
    const std::string role_object = ExtractObjectText(biases, role);
    ticks.push_back(
        MicrosecondsToTicks(ExtractNumber(role_object, direction),
                            "stage_role_bias_us." + role + "." + direction));
  }
  return ticks;
}

}  // namespace

void ApplyCostProfileFile(Instance& instance, const std::string& path) {
  const std::string text = ReadTextFile(path);
  const std::string schema = ExtractString(text, "schema_version");
  if (schema == kSlackPipeCostProfileSchemaVersion) {
    const std::vector<double> calibration_partition =
        ExtractNumberArray(text, "calibration_partition");
    if (calibration_partition.size() !=
        static_cast<std::size_t>(instance.stages)) {
      throw Error("cost profile calibration_partition length must equal N");
    }
    instance.profile_forward_slope_ticks =
        BroadcastSlope(ExtractNumber(text, "a_fwd"), instance.stages, "a_fwd");
    instance.profile_backward_slope_ticks =
        BroadcastSlope(ExtractNumber(text, "a_bwd"), instance.stages, "a_bwd");
    instance.profile_forward_bias_ticks = BiasTicks(
        ExtractNumberArray(text, "bias_fwd"), instance.stages, "bias_fwd");
    instance.profile_backward_bias_ticks = BiasTicks(
        ExtractNumberArray(text, "bias_bwd"), instance.stages, "bias_bwd");
  } else if (schema == kSlackPipeCostProfileSchemaVersionV2) {
    instance.profile_prefix_forward_ticks = MicrosecondArrayTicks(
        ExtractNumberArray(text, "prefix_forward_us"), "prefix_forward_us");
    instance.profile_prefix_backward_ticks = MicrosecondArrayTicks(
        ExtractNumberArray(text, "prefix_backward_us"), "prefix_backward_us");
    instance.profile_role_forward_bias_ticks = RoleBiasTicks(text, "forward");
    instance.profile_role_backward_bias_ticks = RoleBiasTicks(text, "backward");
    instance.cost_profile_hash =
        ExtractStringOptional(text, "cost_profile_hash");
    instance.model_manifest_hash =
        ExtractStringOptional(text, "model_manifest_hash");
  } else {
    throw Error("unsupported SlackPipe cost profile schema_version: " + schema);
  }
  instance.cost_profile_path = std::filesystem::absolute(path).string();
  instance.cost_profile_schema_version = schema;
  instance.cost_profile_units = "microseconds";
  instance.Validate();
}

}  // namespace slackpipe
