"""Frozen evaluation contracts shared by Baseline, SFT, and GRPO."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

EVALUATION_SCHEMA = "travel-userbench-evaluation-v1"
STAGES = ("baseline", "sft", "grpo")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class EvaluationContract:
    task_ids: tuple[str, ...]
    compositions: tuple[str, ...]
    actor_temperature: float = 0.0
    actor_do_sample: bool = False
    simulator_model: str = "deepseek-v4-flash"
    simulator_temperature: float = 0.0
    simulator_endpoint_fingerprint: str = "unbound"
    seed: int = 42
    max_steps: int = 20
    tool_name: str = "interact_with_env"
    tool_parser: str = "qwen3_coder"
    context_length: int = 32768
    schema_version: str = EVALUATION_SCHEMA

    @property
    def contract_hash(self) -> str:
        return canonical_hash(asdict(self))

    def to_dict(self, *, stage: str, model: str) -> dict[str, Any]:
        if stage not in STAGES:
            raise ValueError(f"unknown evaluation stage {stage!r}")
        return {"stage": stage, "model": model, "contract_hash": self.contract_hash, **asdict(self)}


def build_contract(
    records: Sequence[Mapping[str, Any]], *, simulator_endpoint: str | None = None
) -> EvaluationContract:
    if len(records) != 471:
        raise ValueError(f"frozen evaluation requires 471 tasks, found {len(records)}")
    task_ids = tuple(str(row["task_id"]) for row in records)
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("frozen evaluation task IDs must be unique")
    if any(row.get("source_split") != "test" for row in records):
        raise ValueError("frozen evaluation may contain only official test rows")
    fingerprint = (
        hashlib.sha256(simulator_endpoint.rstrip("/").encode()).hexdigest()
        if simulator_endpoint
        else "unbound"
    )
    return EvaluationContract(
        task_ids,
        tuple(str(row["composition"]) for row in records),
        simulator_endpoint_fingerprint=fingerprint,
    )
