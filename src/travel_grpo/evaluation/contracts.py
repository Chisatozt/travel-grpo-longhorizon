"""Frozen evaluation contracts shared by Baseline, SFT, and GRPO."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

EVALUATION_SCHEMA = "travel-userbench-evaluation-v1"
STAGES = ("baseline", "sft", "grpo")
FORMAL_EVALUATION_TASK_COUNT = 471


# [项目注释] 功能：`canonical_hash`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：hexdigest, sha256, encode, dumps。
# [项目注释] 输入：`value`: Any。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
# [项目注释] 类型：`EvaluationContract` 封装相关状态、协议或数据结构。类属性和方法共同维护其不变量。
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
    # [项目注释] 功能：`contract_hash`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：canonical_hash, asdict。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
    def contract_hash(self) -> str:
        return canonical_hash(asdict(self))

    # [项目注释] 功能：`to_dict`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：ValueError, asdict。
    # [项目注释] 输入：`stage`: str；`model`: str。
    # [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
    def to_dict(self, *, stage: str, model: str) -> dict[str, Any]:
        if stage not in STAGES:
            raise ValueError(f"unknown evaluation stage {stage!r}")
        return {"stage": stage, "model": model, "contract_hash": self.contract_hash, **asdict(self)}


# [项目注释] 功能：`_build_contract`：根据输入配置和中间状态构建或生成新的项目产物。 主要协作调用：tuple, any, EvaluationContract, ValueError。
# [项目注释] 输入：`records`: Sequence[Mapping[str, Any]]；`simulator_endpoint`: str | None。
# [项目注释] 输出：标注返回 `EvaluationContract`；具体值由各分支决定。
def _build_contract(
    records: Sequence[Mapping[str, Any]], *, simulator_endpoint: str | None = None
) -> EvaluationContract:
    if not records:
        raise ValueError("evaluation requires at least one task")
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


def build_contract(
    records: Sequence[Mapping[str, Any]], *, simulator_endpoint: str | None = None
) -> EvaluationContract:
    """Build the unchanged formal 471-task evaluation contract."""

    if len(records) != FORMAL_EVALUATION_TASK_COUNT:
        raise ValueError(
            "frozen evaluation requires 471 tasks, "
            f"found {len(records)}"
        )
    return _build_contract(records, simulator_endpoint=simulator_endpoint)


def build_subset_contract(
    records: Sequence[Mapping[str, Any]], *, simulator_endpoint: str | None = None
) -> EvaluationContract:
    """Build an explicit diagnostic contract for a non-empty test subset.

    The formal contract remains strict at 471 tasks. This separate entry point
    is used only when the caller has validated a reproducible subset manifest.
    """

    return _build_contract(records, simulator_endpoint=simulator_endpoint)
