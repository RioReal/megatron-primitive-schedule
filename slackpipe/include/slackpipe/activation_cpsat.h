#pragma once

#include <optional>
#include <vector>

#include "slackpipe/activation_analyzer.h"
#include "slackpipe/instance.h"

#if SLACKPIPE_HAVE_ORTOOLS
#include "ortools/sat/cp_model.h"
#endif

namespace slackpipe {

#if SLACKPIPE_HAVE_ORTOOLS
[[nodiscard]] ActivationCapConstraintMetadata AddActivationCapacityConstraints(
    operations_research::sat::CpModelBuilder &builder, const Instance &instance,
    const ActivationAnalysisOptions &options,
    const std::vector<operations_research::sat::IntVar> &starts,
    const std::vector<operations_research::sat::IntVar> &ends,
    const std::vector<operations_research::sat::IntVar> *layers,
    const std::vector<Tick> *fixed_split, Tick horizon,
    bool partition_optimized,
    ActivationCapModelDebugDump *debug_dump = nullptr);
#endif

}  // namespace slackpipe
