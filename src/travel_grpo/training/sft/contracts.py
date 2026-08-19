"""Serializable SFT collection contracts and checkpoint schemas."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from travel_grpo.prompts.actor_policy import ACTOR_RUNTIME_POLICY_VERSION
from travel_grpo.training.sft.errors import TeacherCollectionError
from travel_grpo.training.teacher_policy import POLICY_VERSION, AttemptStrategy


TRAJECTORY_SCHEMA_VERSION = "userbench-teacher-trajectory-v4"
COLLECTION_DIAGNOSTIC_SCHEMA_VERSION = "userbench-teacher-diagnostic-v4"


@dataclass(frozen=True)
# [项目注释] 类型：`TeacherTrajectory` 封装相关状态、协议或数据结构。类属性和方法共同维护其不变量。
class TeacherTrajectory:
    task_id: str
    composition: str
    difficulty: str
    source_split: str
    teacher_model: str
    simulator_model: str
    messages: tuple[dict[str, Any], ...]
    step_rewards: tuple[float, ...]
    terminated: bool
    truncated: bool
    expected_aspects: tuple[str, ...] = ()
    answered_aspects: tuple[str, ...] = ()
    simulator_fallbacks: int = 0
    simulator_judgment_fallbacks: int = 0
    simulator_search_fallbacks: int = 0
    generation_diagnostics: tuple[dict[str, Any], ...] = ()
    trajectory_attempt: int = 1
    reward_breakdown: Mapping[str, Any] | None = None
    policy_version: str = POLICY_VERSION
    actor_policy_version: str = ACTOR_RUNTIME_POLICY_VERSION
    attempt_strategy: str = AttemptStrategy.NATURAL.value
    teacher_request_count: int = 0
    teacher_usage: Mapping[str, int] | None = None
    quality_tier: str = "gold"

    @property
    # [项目注释] 功能：`total_reward`：计算奖励、指标或聚合统计，供训练、评测或报告使用。 主要协作调用：float, sum。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：标注返回 `float`；具体值由各分支决定。
    def total_reward(self) -> float:
        return float(sum(self.step_rewards))

    # [项目注释] 功能：`to_record`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：dict, list, len。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
    def to_record(self) -> dict[str, Any]:
        reward = dict(self.reward_breakdown or {})
        teacher_usage = dict(self.teacher_usage or {})
        return {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "task_id": self.task_id,
            "composition": self.composition,
            "difficulty": self.difficulty,
            "source_split": self.source_split,
            "teacher_model": self.teacher_model,
            "simulator_model": self.simulator_model,
            "messages": list(self.messages),
            "step_rewards": list(self.step_rewards),
            "total_reward": self.total_reward,
            "num_steps": len(self.step_rewards),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "expected_aspects": list(self.expected_aspects),
            "answered_aspects": list(self.answered_aspects),
            "simulator_fallbacks": self.simulator_fallbacks,
            "simulator_judgment_fallbacks": self.simulator_judgment_fallbacks,
            "simulator_search_fallbacks": self.simulator_search_fallbacks,
            "trajectory_attempt": self.trajectory_attempt,
            "policy_version": self.policy_version,
            "actor_policy_version": self.actor_policy_version,
            "attempt_strategy": self.attempt_strategy,
            "teacher_request_count": self.teacher_request_count,
            "teacher_usage": teacher_usage,
            "quality_tier": self.quality_tier,
            "reward_version": reward.get("reward_version"),
            "reward_valid": reward.get("reward_valid"),
            "terminal_reward": reward.get("terminal_reward"),
            "reward_breakdown": reward or None,
            "completion_rate": reward.get("completion_rate"),
            "correct_answer_rate": reward.get(
                "correct_answer_rate", reward.get("completion_rate")
            ),
            "answer_submission_rate": reward.get("answer_submission_rate"),
            "correct_itinerary": reward.get("correct_itinerary"),
            "gold_itinerary": reward.get("gold_itinerary"),
            "fully_grounded": reward.get("fully_grounded"),
            "active_preference_coverage": reward.get(
                "active_preference_coverage"
            ),
            "passive_preference_coverage": reward.get(
                "passive_preference_coverage"
            ),
            "policy_penalty": reward.get("policy_penalty"),
            "invalid_actions": reward.get("invalid_actions", 0),
            "exact_repeats": reward.get("exact_repeats", 0),
            "semantic_repeats": reward.get("semantic_repeats", 0),
            "ambiguous_actions": reward.get("ambiguous_actions", 0),
            "unsearched_answers": reward.get("unsearched_answers", 0),
            "wrong_answers": reward.get("wrong_answers", 0),
            "infrastructure_errors": reward.get("infrastructure_errors", []),
        }

    @classmethod
    # [项目注释] 功能：`from_record`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：cls, TeacherCollectionError, str,
    # [项目注释]    tuple。
    # [项目注释] 输入：`record`: Mapping[str, Any]。
    # [项目注释] 输出：标注返回 `'TeacherTrajectory'`；具体值由各分支决定。
    def from_record(cls, record: Mapping[str, Any]) -> "TeacherTrajectory":
        if record.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
            raise TeacherCollectionError("checkpoint contains an unsupported trajectory")
        return cls(
            task_id=str(record["task_id"]),
            composition=str(record["composition"]),
            difficulty=str(record["difficulty"]),
            source_split=str(record["source_split"]),
            teacher_model=str(record["teacher_model"]),
            simulator_model=str(record["simulator_model"]),
            messages=tuple(copy.deepcopy(record["messages"])),
            step_rewards=tuple(float(value) for value in record["step_rewards"]),
            terminated=record.get("terminated") is True,
            truncated=record.get("truncated") is True,
            expected_aspects=tuple(str(value) for value in record["expected_aspects"]),
            answered_aspects=tuple(str(value) for value in record["answered_aspects"]),
            simulator_fallbacks=int(record.get("simulator_fallbacks", 0)),
            simulator_judgment_fallbacks=int(
                record.get("simulator_judgment_fallbacks", 0)
            ),
            simulator_search_fallbacks=int(record.get("simulator_search_fallbacks", 0)),
            generation_diagnostics=tuple(
                copy.deepcopy(record.get("generation_diagnostics", ()))
            ),
            trajectory_attempt=int(record.get("trajectory_attempt", 1)),
            reward_breakdown=copy.deepcopy(record.get("reward_breakdown")),
            policy_version=str(record.get("policy_version", "")),
            actor_policy_version=str(record.get("actor_policy_version", "legacy-unknown")),
            attempt_strategy=str(record.get("attempt_strategy", "")),
            teacher_request_count=int(record.get("teacher_request_count", 0)),
            teacher_usage=copy.deepcopy(record.get("teacher_usage", {})),
            quality_tier=str(record.get("quality_tier", "gold")),
        )


@dataclass(frozen=True)
# [项目注释] 类型：`TeacherAttemptDiagnostic` 封装相关状态、协议或数据结构。类属性和方法共同维护其不变量。
class TeacherAttemptDiagnostic:
    task_id: str
    attempt: int
    accepted: bool
    rejection_reasons: tuple[str, ...]
    generation_diagnostics: tuple[dict[str, Any], ...] = ()
    error_type: str | None = None
    error_message: str | None = None
    partial_trajectory: Mapping[str, Any] | None = None
    policy_version: str = POLICY_VERSION
    actor_policy_version: str = ACTOR_RUNTIME_POLICY_VERSION
    attempt_strategy: str = AttemptStrategy.NATURAL.value
    quality_tier: str = "rejected"

    # [项目注释] 功能：`to_record`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：list, dict。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": COLLECTION_DIAGNOSTIC_SCHEMA_VERSION,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "accepted": self.accepted,
            "policy_version": self.policy_version,
            "actor_policy_version": self.actor_policy_version,
            "attempt_strategy": self.attempt_strategy,
            "quality_tier": self.quality_tier,
            "rejection_reasons": list(self.rejection_reasons),
            "generation_diagnostics": list(self.generation_diagnostics),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "partial_trajectory": (
                None
                if self.partial_trajectory is None
                else dict(self.partial_trajectory)
            ),
        }

    @classmethod
    # [项目注释] 功能：`from_record`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：cls, TeacherCollectionError, str, int。
    # [项目注释] 输入：`record`: Mapping[str, Any]。
    # [项目注释] 输出：标注返回 `'TeacherAttemptDiagnostic'`；具体值由各分支决定。
    def from_record(cls, record: Mapping[str, Any]) -> "TeacherAttemptDiagnostic":
        if record.get("schema_version") != COLLECTION_DIAGNOSTIC_SCHEMA_VERSION:
            raise TeacherCollectionError("checkpoint contains unsupported diagnostics")
        return cls(
            task_id=str(record["task_id"]),
            attempt=int(record["attempt"]),
            accepted=record.get("accepted") is True,
            rejection_reasons=tuple(str(value) for value in record["rejection_reasons"]),
            generation_diagnostics=tuple(
                copy.deepcopy(record.get("generation_diagnostics", ()))
            ),
            error_type=(
                None if record.get("error_type") is None else str(record["error_type"])
            ),
            error_message=(
                None
                if record.get("error_message") is None
                else str(record["error_message"])
            ),
            partial_trajectory=copy.deepcopy(record.get("partial_trajectory")),
            policy_version=str(record.get("policy_version", "")),
            actor_policy_version=str(record.get("actor_policy_version", "legacy-unknown")),
            attempt_strategy=str(record.get("attempt_strategy", "")),
            quality_tier=str(record.get("quality_tier", "rejected")),
        )


@dataclass(frozen=True)
# [项目注释] 类型：`TeacherTaskOutcome` 封装相关状态、协议或数据结构。类属性和方法共同维护其不变量。
class TeacherTaskOutcome:
    task_id: str
    trajectory: TeacherTrajectory | None
    attempts: tuple[TeacherAttemptDiagnostic, ...]
    quality_tier: str = "rejected"

    @property
    # [项目注释] 功能：`accepted`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：标注返回 `bool`；具体值由各分支决定。
    def accepted(self) -> bool:
        return self.trajectory is not None

    @property
    # [项目注释] 功能：`gold`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：标注返回 `bool`；具体值由各分支决定。
    def gold(self) -> bool:
        return self.quality_tier == "gold"

    # [项目注释] 功能：`rejected_record`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：len, list。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
    def rejected_record(self) -> dict[str, Any]:
        final = self.attempts[-1]
        return {
            "schema_version": COLLECTION_DIAGNOSTIC_SCHEMA_VERSION,
            "task_id": self.task_id,
            "attempts": len(self.attempts),
            "rejection_reasons": list(final.rejection_reasons),
            "error_type": final.error_type,
            "error_message": final.error_message,
            "policy_version": final.policy_version,
            "actor_policy_version": final.actor_policy_version,
            "attempt_strategy": final.attempt_strategy,
            "quality_tier": self.quality_tier,
        }

    # [项目注释] 功能：`to_checkpoint_record`：执行输入、状态或产物校验，并在不满足约束时返回诊断或抛出异常。 主要协作调用：to_record。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
    def to_checkpoint_record(self) -> dict[str, Any]:
        return {
            "schema_version": "userbench-teacher-task-checkpoint-v1",
            "policy_version": POLICY_VERSION,
            "actor_policy_version": ACTOR_RUNTIME_POLICY_VERSION,
            "task_id": self.task_id,
            "trajectory": None if self.trajectory is None else self.trajectory.to_record(),
            "attempts": [value.to_record() for value in self.attempts],
            "quality_tier": self.quality_tier,
        }

    @classmethod
    # [项目注释] 功能：`from_checkpoint_record`：执行输入、状态或产物校验，并在不满足约束时返回诊断或抛出异常。 主要协作调用：cls, TeacherCollectionError,
    # [项目注释]    str, tuple。
    # [项目注释] 输入：`record`: Mapping[str, Any]。
    # [项目注释] 输出：标注返回 `'TeacherTaskOutcome'`；具体值由各分支决定。
    def from_checkpoint_record(cls, record: Mapping[str, Any]) -> "TeacherTaskOutcome":
        if record.get("schema_version") != "userbench-teacher-task-checkpoint-v1":
            raise TeacherCollectionError("unsupported teacher task checkpoint")
        if record.get("policy_version") != POLICY_VERSION:
            raise TeacherCollectionError("teacher task checkpoint policy mismatch")
        trajectory_record = record.get("trajectory")
        return cls(
            task_id=str(record["task_id"]),
            trajectory=(
                None
                if trajectory_record is None
                else TeacherTrajectory.from_record(trajectory_record)
            ),
            attempts=tuple(
                TeacherAttemptDiagnostic.from_record(value)
                for value in record.get("attempts", ())
            ),
            quality_tier=str(
                record.get(
                    "quality_tier",
                    "rejected" if trajectory_record is None else "gold",
                )
            ),
        )
