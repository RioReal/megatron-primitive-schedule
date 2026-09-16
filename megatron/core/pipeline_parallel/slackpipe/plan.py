# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""JSON plan parsing and validation for SlackPipe schedules."""

import json
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

SLACKPIPE_PLAN_SCHEMA_VERSION = "slackpipe.plan.v1"
SLACKPIPE_PLAN_SCHEMA_VERSION_V2 = "slackpipe.plan.v2"


@dataclass(frozen=True)
class SlackPipeOperation:
    """A single SlackPipe forward or backward operation."""

    kind: str
    microbatch: int
    stage: int


@dataclass(frozen=True)
class SlackPipePlan:
    """A validated SlackPipe execution plan."""

    num_microbatches: int
    num_stages: int
    num_workers: int
    num_layers: int
    layer_split: Tuple[int, ...]
    stage_layer_ranges: Tuple[Tuple[int, int], ...]
    stage_to_worker: Tuple[int, ...]
    operations: Tuple[Tuple[SlackPipeOperation, ...], ...]
    schema_version: str = SLACKPIPE_PLAN_SCHEMA_VERSION
    solver_status: Optional[str] = None
    predicted_makespan: Optional[float] = None
    forward_costs: Tuple[float, ...] = ()
    backward_costs: Tuple[float, ...] = ()
    model_manifest_hash: Optional[str] = None
    cost_profile_hash: Optional[str] = None
    cost_profile_version: Optional[str] = None
    cost_profile_path: Optional[str] = None

    @property
    def local_operations(self) -> Tuple[SlackPipeOperation, ...]:
        """Operations assigned to worker 0 for the PP=1 prototype."""

        return self.operations[0]

    def worker_operations(self, worker: int) -> Tuple[SlackPipeOperation, ...]:
        """Operations assigned to a physical pipeline worker."""

        return self.operations[worker]

    def stage_begin_layer(self, stage: int) -> int:
        return self.stage_layer_ranges[stage][0]

    def stage_end_layer(self, stage: int) -> int:
        return self.stage_layer_ranges[stage][1]

    def stage_layer_ids(self, stage: int) -> Tuple[int, ...]:
        begin, end = self.stage_layer_ranges[stage]
        return tuple(range(begin, end))


def load_slackpipe_plan(
    path: str | Path, pipeline_model_parallel_size: Optional[int] = None
) -> SlackPipePlan:
    """Load and validate a SlackPipe plan from a JSON file."""

    with open(path, "r", encoding="utf-8") as plan_file:
        payload = json.load(plan_file)
    plan = parse_slackpipe_plan(payload, pipeline_model_parallel_size=pipeline_model_parallel_size)
    if plan.cost_profile_path and not Path(plan.cost_profile_path).is_absolute():
        plan = replace(
            plan, cost_profile_path=str(Path(path).resolve().parent / plan.cost_profile_path)
        )
    return plan


def parse_slackpipe_plan(
    payload: Mapping[str, object], pipeline_model_parallel_size: Optional[int] = None
) -> SlackPipePlan:
    """Parse and validate a SlackPipe plan payload."""

    schema_version = _parse_schema_version(payload)

    required_fields = [
        "num_microbatches",
        "num_stages",
        "num_workers",
        "num_layers",
        "stage_to_worker",
    ]
    if schema_version == SLACKPIPE_PLAN_SCHEMA_VERSION:
        required_fields.append("layer_split")
    for field in required_fields:
        if field not in payload:
            raise ValueError(f"SlackPipe plan is missing required field '{field}'")

    num_microbatches = _require_non_negative_int(payload["num_microbatches"], "num_microbatches")
    num_stages = _require_positive_int(payload["num_stages"], "num_stages")
    num_workers = _require_positive_int(payload["num_workers"], "num_workers")
    num_layers = _require_non_negative_int(payload["num_layers"], "num_layers")

    if pipeline_model_parallel_size is not None and num_workers != pipeline_model_parallel_size:
        raise ValueError(
            "SlackPipe plan num_workers must equal pipeline model parallel size "
            f"({num_workers} != {pipeline_model_parallel_size})"
        )

    solver_status = _parse_optional_str(payload, "solver_status")
    predicted_makespan = _parse_optional_non_negative_float(payload, "predicted_makespan")
    forward_costs = tuple(_parse_optional_costs(payload, "forward_costs", num_stages))
    backward_costs = tuple(_parse_optional_costs(payload, "backward_costs", num_stages))
    model_manifest_hash = _parse_optional_str(payload, "model_manifest_hash")
    cost_profile_hash = _parse_optional_str(payload, "cost_profile_hash")
    cost_profile_version = _parse_optional_str(payload, "cost_profile_version")
    cost_profile_path = None
    if schema_version == SLACKPIPE_PLAN_SCHEMA_VERSION_V2:
        metadata = payload.get("cost_model", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("SlackPipe cost_model must be an object")
        cost_profile_path = _parse_optional_str(metadata, "path")

    if schema_version == SLACKPIPE_PLAN_SCHEMA_VERSION_V2:
        stage_layer_ranges = _parse_stage_layer_ranges(payload, num_stages, num_layers)
        derived_layer_split = tuple(end - begin for begin, end in stage_layer_ranges)
        if "layer_split" in payload:
            layer_split = tuple(_require_int_sequence(payload["layer_split"], "layer_split"))
            if layer_split != derived_layer_split:
                raise ValueError(
                    "SlackPipe plan.v2 layer_split must match stage_layer_ranges "
                    f"({layer_split} != {derived_layer_split})"
                )
        else:
            layer_split = derived_layer_split
    else:
        layer_split = tuple(_require_int_sequence(payload["layer_split"], "layer_split"))
        stage_layer_ranges = _ranges_from_split(layer_split)
    _validate_layer_split(layer_split, num_stages, num_layers)

    stage_to_worker = tuple(_require_int_sequence(payload["stage_to_worker"], "stage_to_worker"))
    if len(stage_to_worker) != num_stages:
        raise ValueError(
            "SlackPipe plan stage_to_worker must have num_stages entries "
            f"({len(stage_to_worker)} != {num_stages})"
        )
    invalid_workers = [worker for worker in stage_to_worker if worker < 0 or worker >= num_workers]
    if invalid_workers:
        raise ValueError(f"SlackPipe plan has invalid worker indices: {invalid_workers}")

    operations = _parse_worker_operations(payload, num_workers)
    _validate_operations(
        operations,
        num_microbatches=num_microbatches,
        num_stages=num_stages,
        stage_to_worker=stage_to_worker,
    )

    return SlackPipePlan(
        num_microbatches=num_microbatches,
        num_stages=num_stages,
        num_workers=num_workers,
        num_layers=num_layers,
        layer_split=layer_split,
        stage_layer_ranges=stage_layer_ranges,
        stage_to_worker=stage_to_worker,
        operations=operations,
        schema_version=schema_version,
        solver_status=solver_status,
        predicted_makespan=predicted_makespan,
        forward_costs=forward_costs,
        backward_costs=backward_costs,
        model_manifest_hash=model_manifest_hash,
        cost_profile_hash=cost_profile_hash,
        cost_profile_version=cost_profile_version,
        cost_profile_path=cost_profile_path,
    )


def validate_cyclic_placement(plan: SlackPipePlan) -> None:
    """Validate cyclic stage placement."""

    expected = tuple(stage % plan.num_workers for stage in range(plan.num_stages))
    if plan.stage_to_worker != expected:
        raise ValueError(
            f"SlackPipe prototype requires cyclic placement ({plan.stage_to_worker} != {expected})"
        )


def derive_pipeline_model_parallel_layout(plan: SlackPipePlan) -> str:
    """Derive Megatron's custom pipeline layout string from a SlackPipe layer split."""

    stage_specs = []
    for stage, layer_count in enumerate(plan.layer_split):
        spec = ""
        if stage == 0:
            spec += "E"
        if layer_count == 1:
            spec += "t"
        elif layer_count > 1:
            spec += f"t*{layer_count}"
        if stage == plan.num_stages - 1:
            spec += "L"
        stage_specs.append(spec)
    return "|".join(stage_specs)


def validate_plan_parallel_layout(plan: SlackPipePlan, pipeline_model_parallel_size: int) -> int:
    """Validate plan placement against physical PP size and return VP size."""

    if plan.num_workers != pipeline_model_parallel_size:
        raise ValueError(
            "SlackPipe plan num_workers must equal pipeline model parallel size "
            f"({plan.num_workers} != {pipeline_model_parallel_size})"
        )
    validate_cyclic_placement(plan)
    if plan.num_stages % plan.num_workers != 0:
        raise ValueError(
            "SlackPipe plan num_stages must be divisible by num_workers "
            f"({plan.num_stages} % {plan.num_workers} != 0)"
        )
    return plan.num_stages // plan.num_workers


def _parse_worker_operations(
    payload: Mapping[str, object], num_workers: int
) -> Tuple[Tuple[SlackPipeOperation, ...], ...]:
    operations_payload = payload.get("operations", payload.get("worker_operations"))
    if operations_payload is None:
        raise ValueError("SlackPipe plan is missing required field 'operations'")

    if isinstance(operations_payload, Mapping):
        worker_payloads = [operations_payload.get(str(worker), []) for worker in range(num_workers)]
    elif isinstance(operations_payload, Sequence) and not isinstance(
        operations_payload, (str, bytes)
    ):
        if len(operations_payload) != num_workers:
            raise ValueError(
                "SlackPipe plan operations must have num_workers entries "
                f"({len(operations_payload)} != {num_workers})"
            )
        worker_payloads = list(operations_payload)
    else:
        raise ValueError("SlackPipe plan operations must be a list or object keyed by worker")

    parsed_workers: List[Tuple[SlackPipeOperation, ...]] = []
    for worker, worker_ops in enumerate(worker_payloads):
        if not isinstance(worker_ops, Sequence) or isinstance(worker_ops, (str, bytes)):
            raise ValueError(f"SlackPipe plan operations for worker {worker} must be a list")
        parsed_workers.append(tuple(_parse_operation(op, worker) for op in worker_ops))
    return tuple(parsed_workers)


def _parse_operation(payload: object, worker: int) -> SlackPipeOperation:
    if not isinstance(payload, Mapping):
        raise ValueError(f"SlackPipe operation on worker {worker} must be an object")
    for field in ("kind", "microbatch", "stage"):
        if field not in payload:
            raise ValueError(f"SlackPipe operation on worker {worker} missing '{field}'")

    kind = payload["kind"]
    if kind not in ("F", "B"):
        raise ValueError(f"SlackPipe operation kind must be 'F' or 'B', got {kind!r}")
    return SlackPipeOperation(
        kind=kind,
        microbatch=_require_non_negative_int(payload["microbatch"], "microbatch"),
        stage=_require_non_negative_int(payload["stage"], "stage"),
    )


def _parse_schema_version(payload: Mapping[str, object]) -> str:
    schema_version = payload.get("schema_version", payload.get("version"))
    if schema_version is None:
        return SLACKPIPE_PLAN_SCHEMA_VERSION
    if schema_version not in (SLACKPIPE_PLAN_SCHEMA_VERSION, SLACKPIPE_PLAN_SCHEMA_VERSION_V2):
        raise ValueError(
            "Unsupported SlackPipe plan schema_version "
            f"{schema_version!r}; expected {SLACKPIPE_PLAN_SCHEMA_VERSION!r} "
            f"or {SLACKPIPE_PLAN_SCHEMA_VERSION_V2!r}"
        )
    return schema_version


def _ranges_from_split(layer_split: Tuple[int, ...]) -> Tuple[Tuple[int, int], ...]:
    ranges = []
    begin = 0
    for layer_count in layer_split:
        end = begin + layer_count
        ranges.append((begin, end))
        begin = end
    return tuple(ranges)


def _parse_stage_layer_ranges(
    payload: Mapping[str, object], num_stages: int, num_layers: int
) -> Tuple[Tuple[int, int], ...]:
    if "stage_layer_ranges" in payload:
        value = payload["stage_layer_ranges"]
        field = "stage_layer_ranges"
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"SlackPipe plan field '{field}' must be a list")
        ranges = []
        for stage, item in enumerate(value):
            if isinstance(item, Mapping):
                if "begin" in item and "end" in item:
                    begin = _require_non_negative_int(item["begin"], f"{field}[{stage}].begin")
                    end = _require_non_negative_int(item["end"], f"{field}[{stage}].end")
                elif "start" in item and "stop" in item:
                    begin = _require_non_negative_int(item["start"], f"{field}[{stage}].start")
                    end = _require_non_negative_int(item["stop"], f"{field}[{stage}].stop")
                else:
                    raise ValueError(
                        f"SlackPipe plan field '{field}[{stage}]' must contain begin/end"
                    )
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                if len(item) != 2:
                    raise ValueError(
                        f"SlackPipe plan field '{field}[{stage}]' must have two entries"
                    )
                begin = _require_non_negative_int(item[0], f"{field}[{stage}][0]")
                end = _require_non_negative_int(item[1], f"{field}[{stage}][1]")
            else:
                raise ValueError(
                    f"SlackPipe plan field '{field}[{stage}]' must be a pair or object"
                )
            ranges.append((begin, end))
    elif "layer_cuts" in payload:
        cuts = tuple(_require_int_sequence(payload["layer_cuts"], "layer_cuts"))
        if len(cuts) != num_stages + 1:
            raise ValueError(
                "SlackPipe plan layer_cuts must have num_stages + 1 entries "
                f"({len(cuts)} != {num_stages + 1})"
            )
        ranges = list(zip(cuts, cuts[1:]))
    elif "cuts" in payload:
        cuts = tuple(_require_int_sequence(payload["cuts"], "cuts"))
        if len(cuts) != num_stages + 1:
            raise ValueError(
                "SlackPipe plan cuts must have num_stages + 1 entries "
                f"({len(cuts)} != {num_stages + 1})"
            )
        ranges = list(zip(cuts, cuts[1:]))
    else:
        raise ValueError(
            "SlackPipe plan.v2 is missing required field 'stage_layer_ranges' " "or 'layer_cuts'"
        )

    ranges = tuple(ranges)
    _validate_stage_layer_ranges(ranges, num_stages, num_layers)
    return ranges


def _validate_stage_layer_ranges(
    stage_layer_ranges: Tuple[Tuple[int, int], ...], num_stages: int, num_layers: int
) -> None:
    if len(stage_layer_ranges) != num_stages:
        raise ValueError(
            "SlackPipe plan stage_layer_ranges must have num_stages entries "
            f"({len(stage_layer_ranges)} != {num_stages})"
        )
    expected_begin = 0
    for stage, (begin, end) in enumerate(stage_layer_ranges):
        if begin != expected_begin:
            raise ValueError(
                "SlackPipe plan stage_layer_ranges must be contiguous "
                f"(stage {stage} begins at {begin}, expected {expected_begin})"
            )
        if end <= begin:
            raise ValueError(
                "SlackPipe plan stage_layer_ranges must be non-empty "
                f"(stage {stage} has range [{begin}, {end}))"
            )
        expected_begin = end
    if expected_begin != num_layers:
        raise ValueError(
            "SlackPipe plan stage_layer_ranges must cover num_layers "
            f"({expected_begin} != {num_layers})"
        )


def _validate_layer_split(layer_split: Tuple[int, ...], num_stages: int, num_layers: int) -> None:
    if len(layer_split) != num_stages:
        raise ValueError(
            "SlackPipe plan layer_split must have num_stages entries "
            f"({len(layer_split)} != {num_stages})"
        )
    if any(layer_count <= 0 for layer_count in layer_split):
        raise ValueError(f"SlackPipe plan layer_split entries must be positive: {layer_split}")
    if sum(layer_split) != num_layers:
        raise ValueError(
            "SlackPipe plan layer_split must sum to num_layers "
            f"({sum(layer_split)} != {num_layers})"
        )


def _parse_optional_str(payload: Mapping[str, object], field: str) -> Optional[str]:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"SlackPipe plan field '{field}' must be a string")
    return value


def _parse_optional_non_negative_float(
    payload: Mapping[str, object], field: str
) -> Optional[float]:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"SlackPipe plan field '{field}' must be numeric")
    value = float(value)
    if value < 0.0:
        raise ValueError(f"SlackPipe plan field '{field}' must be non-negative")
    return value


def _parse_optional_costs(
    payload: Mapping[str, object], field: str, num_stages: int
) -> Iterable[float]:
    value = payload.get(field)
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"SlackPipe plan field '{field}' must be a list")
    if len(value) != num_stages:
        raise ValueError(
            f"SlackPipe plan field '{field}' must have num_stages entries "
            f"({len(value)} != {num_stages})"
        )
    costs = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError(f"SlackPipe plan field '{field}' entries must be numeric")
        item = float(item)
        if item < 0.0:
            raise ValueError(f"SlackPipe plan field '{field}' entries must be non-negative")
        costs.append(item)
    return costs


def _validate_operations(
    operations: Tuple[Tuple[SlackPipeOperation, ...], ...],
    *,
    num_microbatches: int,
    num_stages: int,
    stage_to_worker: Tuple[int, ...],
) -> None:
    seen: Dict[Tuple[str, int, int], int] = {}
    for worker, worker_ops in enumerate(operations):
        for op in worker_ops:
            if op.microbatch >= num_microbatches:
                raise ValueError(
                    f"SlackPipe operation has invalid microbatch index {op.microbatch}"
                )
            if op.stage >= num_stages:
                raise ValueError(f"SlackPipe operation has invalid stage index {op.stage}")
            expected_worker = stage_to_worker[op.stage]
            if worker != expected_worker:
                raise ValueError(
                    "SlackPipe operation appears on the wrong worker: "
                    f"{op.kind}({op.microbatch},{op.stage}) is on {worker}, "
                    f"expected {expected_worker}"
                )
            key = (op.kind, op.microbatch, op.stage)
            seen[key] = seen.get(key, 0) + 1

    expected_keys = {
        (kind, microbatch, stage)
        for kind in ("F", "B")
        for microbatch in range(num_microbatches)
        for stage in range(num_stages)
    }
    duplicates = sorted(key for key, count in seen.items() if count != 1)
    if duplicates:
        raise ValueError(f"SlackPipe plan has duplicate operations: {duplicates}")

    missing = sorted(expected_keys - set(seen))
    if missing:
        raise ValueError(f"SlackPipe plan is missing operations: {missing}")

    _validate_fifo_microbatch_order(
        operations,
        num_microbatches=num_microbatches,
        num_stages=num_stages,
        stage_to_worker=stage_to_worker,
    )
    _validate_operation_dag(operations, num_microbatches=num_microbatches, num_stages=num_stages)


def _validate_fifo_microbatch_order(
    operations: Tuple[Tuple[SlackPipeOperation, ...], ...],
    *,
    num_microbatches: int,
    num_stages: int,
    stage_to_worker: Tuple[int, ...],
) -> None:
    for stage in range(num_stages):
        worker = stage_to_worker[stage]
        for kind in ("F", "B"):
            microbatches = [
                op.microbatch for op in operations[worker] if op.kind == kind and op.stage == stage
            ]
            expected = list(range(num_microbatches))
            if microbatches != expected:
                raise ValueError(
                    "SlackPipe plan violates FIFO microbatch order for "
                    f"{kind} stage {stage}: {microbatches} != {expected}"
                )


def _validate_operation_dag(
    operations: Tuple[Tuple[SlackPipeOperation, ...], ...],
    *,
    num_microbatches: int,
    num_stages: int,
) -> None:
    nodes: Set[Tuple[str, int, int]] = {
        (kind, microbatch, stage)
        for kind in ("F", "B")
        for microbatch in range(num_microbatches)
        for stage in range(num_stages)
    }
    edges: Dict[Tuple[str, int, int], Set[Tuple[str, int, int]]] = {node: set() for node in nodes}

    def add_edge(src: Tuple[str, int, int], dst: Tuple[str, int, int]) -> None:
        edges[src].add(dst)

    for microbatch in range(num_microbatches):
        for stage in range(num_stages):
            add_edge(("F", microbatch, stage), ("B", microbatch, stage))
            if stage > 0:
                add_edge(("F", microbatch, stage - 1), ("F", microbatch, stage))
            if stage < num_stages - 1:
                add_edge(("B", microbatch, stage + 1), ("B", microbatch, stage))

    for microbatch in range(num_microbatches - 1):
        for stage in range(num_stages):
            add_edge(("F", microbatch, stage), ("F", microbatch + 1, stage))
            add_edge(("B", microbatch, stage), ("B", microbatch + 1, stage))

    for worker_ops in operations:
        for prev_op, next_op in zip(worker_ops, worker_ops[1:]):
            add_edge(
                (prev_op.kind, prev_op.microbatch, prev_op.stage),
                (next_op.kind, next_op.microbatch, next_op.stage),
            )

    in_degree = {node: 0 for node in nodes}
    for successors in edges.values():
        for successor in successors:
            in_degree[successor] += 1

    ready = deque(node for node, degree in in_degree.items() if degree == 0)
    visited = 0
    while ready:
        node = ready.popleft()
        visited += 1
        for successor in edges[node]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                ready.append(successor)

    if visited != len(nodes):
        cyclic_nodes = sorted(node for node, degree in in_degree.items() if degree > 0)
        raise ValueError(f"SlackPipe plan dependency graph contains a cycle: {cyclic_nodes}")


def _require_int_sequence(value: object, field: str) -> Iterable[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"SlackPipe plan field '{field}' must be a list")
    for item in value:
        yield _require_non_negative_int(item, field)


def _require_positive_int(value: object, field: str) -> int:
    integer = _require_non_negative_int(value, field)
    if integer <= 0:
        raise ValueError(f"SlackPipe plan field '{field}' must be positive")
    return integer


def _require_non_negative_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"SlackPipe plan field '{field}' must be an integer")
    if value < 0:
        raise ValueError(f"SlackPipe plan field '{field}' must be non-negative")
    return value
