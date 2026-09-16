#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build/ortools}"
ORTOOLS_PREFIX="${ORTOOLS_PREFIX:-/opt/or-tools}"
ORTOOLS_DIR="${ORTOOLS_DIR:-${ORTOOLS_PREFIX}/lib/cmake/ortools}"
ABSL_DIR="${ABSL_DIR:-${ORTOOLS_PREFIX}/lib/cmake/absl}"
PROTOBUF_DIR="${PROTOBUF_DIR:-${ORTOOLS_PREFIX}/lib/cmake/protobuf}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc 2>/dev/null || printf '2')}"
TEST_FILTER="${SLACKPIPE_OR_TEST_FILTER:-^slackpipe_core_ortools_smoke_tests$}"

if [[ ! -f "${ORTOOLS_DIR}/ortoolsConfig.cmake" ]]; then
  cat >&2 <<EOF
OR-Tools CMake package was not found.

Expected:
  ${ORTOOLS_DIR}/ortoolsConfig.cmake

Set either:
  ORTOOLS_PREFIX=/path/to/ortools
or:
  ORTOOLS_DIR=/path/to/ortools/lib/cmake/ortools

No OR-disabled SlackPipe binary will be produced by this script.
EOF
  exit 1
fi

cmake_args=(
  -S "${ROOT_DIR}"
  -B "${BUILD_DIR}"
  -DSLACKPIPE_ENABLE_ORTOOLS=ON
  -DSLACKPIPE_BUILD_TESTS=ON
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_PREFIX_PATH="${ORTOOLS_PREFIX}"
  -Dortools_DIR="${ORTOOLS_DIR}"
)
if [[ -f "${ABSL_DIR}/abslConfig.cmake" ]]; then
  cmake_args+=(-Dabsl_DIR="${ABSL_DIR}")
fi
if [[ -f "${PROTOBUF_DIR}/protobuf-config.cmake" ]]; then
  cmake_args+=(-DProtobuf_DIR="${PROTOBUF_DIR}")
fi

cmake "${cmake_args[@]}"
cmake --build "${BUILD_DIR}" \
  --target slackpipe_core_tests slackpipe_cli slackpipe_benchmark \
  -j"${BUILD_JOBS}"

build_info="$("${BUILD_DIR}/slackpipe_cli" build-info)"
python3 - "${build_info}" <<'PY'
import json
import sys

info = json.loads(sys.argv[1])
required = [
    "ortools_enabled",
    "ortools_version",
    "cumulative_constraint_supported",
    "variable_cumulative_demand_supported",
    "activation_cap_solver_support",
]
missing = [name for name in required if name not in info]
if missing:
    raise SystemExit(f"build-info missing fields: {missing}")
if info["ortools_enabled"] is not True:
    raise SystemExit("build-info reports ortools_enabled != true")
if info["cumulative_constraint_supported"] is not True:
    raise SystemExit("OR build lacks cumulative constraint support")
if info["activation_cap_solver_support"] not in (
    "fixed_demands_only",
    "variable_demands",
):
    raise SystemExit(
        "unexpected activation_cap_solver_support="
        f"{info['activation_cap_solver_support']!r}"
    )
print(
    "build-info ok: "
    f"ortools_version={info.get('ortools_version')!r}, "
    f"activation_cap_solver_support={info['activation_cap_solver_support']}"
)
PY

ctest --test-dir "${BUILD_DIR}" --output-on-failure -R "${TEST_FILTER}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

common=(
  --B 2
  --N 2
  --J 2
  --L 2
  --min-layers 1
  --ratio-num 2
  --ratio-den 1
  --communication 0
  --time-limit-seconds 5
  --num-workers 1
  --random-seed 123
  --require-optimal false
)

run_cli() {
  local prefix="$1"
  shift
  "${BUILD_DIR}/slackpipe_cli" "${common[@]}" "$@" \
    --output-prefix "${TMP_DIR}/${prefix}" >/dev/null
  "${BUILD_DIR}/slackpipe_cli" validate-result \
    --input "${TMP_DIR}/${prefix}.json" --activation-summary >/dev/null
}

methods=(
  uniform-breadth-first
  uniform-interleaved-1f1b
  partition-only-fixed-order
  schedule-only-uniform
  sequential-partition-then-schedule
  alternating-partition-schedule
  joint-unrestricted-no-overlap
)

for method in "${methods[@]}"; do
  extra=()
  case "${method}" in
    partition-only-fixed-order)
      extra+=(--fixed-order-partition-backend cpsat)
      ;;
    sequential-partition-then-schedule|alternating-partition-schedule)
      extra+=(--fixed-order-partition-backend cpsat)
      ;;
  esac
  if [[ "${method}" == "alternating-partition-schedule" ]]; then
    extra+=(--alternating-max-rounds 1)
  fi
  run_cli "uncapped_${method}" --algorithm "${method}" "${extra[@]}"
  run_cli "capped_${method}" --algorithm "${method}" "${extra[@]}" \
    --activation-model count \
    --activation-cap-mode uniform-baseline \
    --activation-cap-enforcement solver
done

"${BUILD_DIR}/slackpipe_cli" describe-method \
  joint-unrestricted-no-overlap >/dev/null

python3 - "${TMP_DIR}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
solver_backed = {
    "partition-only-fixed-order",
    "schedule-only-uniform",
    "sequential-partition-then-schedule",
    "alternating-partition-schedule",
    "joint-unrestricted-no-overlap",
}

def load(prefix):
    with (root / f"{prefix}.json").open() as handle:
        return json.load(handle)

def value(payload, key):
    if key in payload:
        return payload[key]
    canonical = payload.get("canonical_result") or payload.get("canonical") or {}
    return canonical.get(key)

for path in sorted(root.glob("*.json")):
    payload = json.loads(path.read_text())
    method = value(payload, "canonical_method")
    if value(payload, "budget_policy_version") != 1:
        raise SystemExit(f"{path.name}: unexpected budget policy version")
    if value(payload, "evaluation_method_version") != 1:
        raise SystemExit(f"{path.name}: unexpected evaluation method version")
    if value(payload, "result_validation_passed") is not True:
        raise SystemExit(f"{path.name}: validation did not pass")
    if value(payload, "predecessor_candidate_restriction_active") is not False:
        raise SystemExit(f"{path.name}: claimed predecessor restriction")

for method in [
    "uniform-breadth-first",
    "uniform-interleaved-1f1b",
    "partition-only-fixed-order",
    "schedule-only-uniform",
    "sequential-partition-then-schedule",
    "alternating-partition-schedule",
    "joint-unrestricted-no-overlap",
]:
    uncapped = load(f"uncapped_{method}")
    capped = load(f"capped_{method}")
    for payload in (uncapped, capped):
        if value(payload, "micro_batches") != 2:
            raise SystemExit(f"{method}: B changed")
        if value(payload, "logical_stages") != 2:
            raise SystemExit(f"{method}: N changed")
        if value(payload, "physical_workers") != 2:
            raise SystemExit(f"{method}: W changed")
        if value(payload, "total_layers") != 2:
            raise SystemExit(f"{method}: L changed")
        if value(payload, "communication_ticks") != 0:
            raise SystemExit(f"{method}: communication changed")
    if value(capped, "activation_cap_satisfied") is not True:
        raise SystemExit(f"{method}: capped row is not cap-satisfied")
    mode = value(capped, "activation_cap_enforcement_mode")
    if method in solver_backed:
        if mode == "posthoc_only":
            raise SystemExit(f"{method}: capped solver row is posthoc_only")
        if value(capped, "activation_cap_enforced_in_solver") is not True:
            raise SystemExit(f"{method}: cap not enforced in solver")
        if value(capped, "activation_cap_constraints_added") is not True:
            raise SystemExit(f"{method}: cap constraints missing")
    elif method in {"uniform-breadth-first", "uniform-interleaved-1f1b"}:
        if mode != "deterministic_postconstruction_check":
            raise SystemExit(
                f"{method}: expected deterministic cap check"
            )

print(f"tiny canonical smoke ok: {len(list(root.glob('*.json')))} JSON files")
PY

echo "OR-Tools evaluation validation completed successfully."
