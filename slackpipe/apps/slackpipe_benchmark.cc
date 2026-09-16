#include <exception>
#include <iostream>
#include <string>

#include "slackpipe/benchmark.h"
#include "slackpipe/result_schema.h"

namespace {

std::string ValueAfter(int& i, int argc, char** argv) {
  if (i + 1 >= argc) throw slackpipe::Error("missing value for flag");
  ++i;
  return argv[i];
}

bool ParseBool(const std::string& text) {
  if (text == "true" || text == "1") return true;
  if (text == "false" || text == "0") return false;
  throw slackpipe::Error("expected true or false");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string config_path;
    std::string output_dir = "benchmark-results";
    slackpipe::benchmark::BenchmarkConfig overrides;
    bool have_repetitions = false;
    bool have_warmups = false;
    bool have_parallel = false;
    bool have_workers = false;
    bool have_timeout = false;
    bool have_seed = false;
    bool have_resume = false;
    for (int i = 1; i < argc; ++i) {
      const std::string flag = argv[i];
      if (flag == "--config") {
        config_path = ValueAfter(i, argc, argv);
      } else if (flag == "--output-dir") {
        output_dir = ValueAfter(i, argc, argv);
      } else if (flag == "--resume") {
        overrides.resume = ParseBool(ValueAfter(i, argc, argv));
        have_resume = true;
      } else if (flag == "--repetitions") {
        overrides.repetitions = std::stoi(ValueAfter(i, argc, argv));
        have_repetitions = true;
      } else if (flag == "--warmups") {
        overrides.warmups = std::stoi(ValueAfter(i, argc, argv));
        have_warmups = true;
      } else if (flag == "--parallel-instances") {
        overrides.parallel_instances = std::stoi(ValueAfter(i, argc, argv));
        have_parallel = true;
      } else if (flag == "--cp-sat-workers") {
        overrides.cp_sat_workers = {std::stoi(ValueAfter(i, argc, argv))};
        have_workers = true;
      } else if (flag == "--timeout-seconds") {
        overrides.timeout_seconds = std::stod(ValueAfter(i, argc, argv));
        have_timeout = true;
      } else if (flag == "--random-seed-base") {
        overrides.random_seed_base = std::stoi(ValueAfter(i, argc, argv));
        have_seed = true;
      } else if (flag == "--allow-oversubscription") {
        overrides.allow_oversubscription = ParseBool(ValueAfter(i, argc, argv));
      } else {
        throw slackpipe::Error("unknown flag: " + flag);
      }
    }
    if (config_path.empty()) throw slackpipe::Error("--config is required");
    slackpipe::benchmark::BenchmarkConfig config =
        slackpipe::benchmark::ParseBenchmarkConfigFile(config_path);
    config.requested_command = slackpipe::CommandLineFromArgv(argc, argv);
    config.executable_name = argc > 0 && argv[0] != nullptr
                                 ? slackpipe::ExecutableNameFromArgv0(argv[0])
                                 : std::string("slackpipe_benchmark");
    if (have_repetitions) config.repetitions = overrides.repetitions;
    if (have_warmups) config.warmups = overrides.warmups;
    if (have_parallel) config.parallel_instances = overrides.parallel_instances;
    if (have_workers) config.cp_sat_workers = overrides.cp_sat_workers;
    if (have_timeout) config.timeout_seconds = overrides.timeout_seconds;
    if (have_seed) config.random_seed_base = overrides.random_seed_base;
    if (have_resume) config.resume = overrides.resume;
    if (overrides.allow_oversubscription) config.allow_oversubscription = true;
    const auto rows = slackpipe::benchmark::RunBenchmark(config, output_dir);
    std::cout << "measured_rows=" << rows.size() << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
