"""Pinned-source validation and per-trajectory UserBench context."""

from __future__ import annotations

import contextvars
import json
from collections.abc import Mapping as ABCMapping, Set as AbstractSet
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from travel_grpo.envs.observation import UserBenchStepResult
from travel_grpo.envs.reward import (
    RawRewardTrace,
    TravelRewardTask,
    UserBenchRewardSnapshot,
    compute_travel_reward,
)
from travel_grpo.envs.userbench_tools import (
    ActionChoice,
    UserBenchAction,
    action_query_issue,
    aspect_from_option_id,
    normalized_action_signature,
    semantic_action_signature,
)

if TYPE_CHECKING:
    from travel_grpo.envs.userbench_wrapper import UserBenchWrapper


PINNED_USERBENCH_COMMIT = "80506d2ab484cab843e60a2401ff3e0290d05b87"
DEFAULT_USERBENCH_ROOT = (
    Path(__file__).resolve().parents[3] / "environments" / "UserBench"
)


class UserBenchSourceError(RuntimeError):
    """Raised when the embedded UserBench source is missing or untrusted."""


class UserBenchSessionError(RuntimeError):
    """Raised when a rollout violates its UserBench session lifecycle."""


_SIMULATOR_FALLBACK_NAMES = (
    "userbench_judgment_fallbacks",
    "userbench_response_fallbacks",
    "userbench_search_fallbacks",
)


def _is_complete_reward_task(task: TravelRewardTask | None) -> bool:
    if not isinstance(task, TravelRewardTask) or not task.task_id or not task.aspects:
        return False
    return all(
        aspect in task.best_ids
        and isinstance(task.best_ids[aspect], str)
        and bool(task.best_ids[aspect])
        and aspect in task.correct_ids
        and isinstance(task.correct_ids[aspect], AbstractSet)
        and aspect in task.preference_ids_by_aspect
        and isinstance(task.preference_ids_by_aspect[aspect], AbstractSet)
        for aspect in task.aspects
    )


def _is_complete_reward_snapshot(
    snapshot: UserBenchRewardSnapshot | None,
) -> bool:
    if not isinstance(snapshot, UserBenchRewardSnapshot):
        return False
    if (
        not isinstance(snapshot.active_elicited_count, int)
        or isinstance(snapshot.active_elicited_count, bool)
        or snapshot.active_elicited_count < 0
        or not isinstance(snapshot.passive_elicited_count, int)
        or isinstance(snapshot.passive_elicited_count, bool)
        or snapshot.passive_elicited_count < 0
    ):
        return False
    return all(
        isinstance(value, AbstractSet)
        and all(isinstance(item, str) and item for item in value)
        for value in (
            snapshot.remaining_preference_ids,
            snapshot.remaining_search_aspects,
            snapshot.choice_initials,
        )
    )


def _snapshot_matches_task(
    task: TravelRewardTask,
    snapshot: UserBenchRewardSnapshot,
) -> bool:
    known_preferences = set().union(
        *(set(task.preference_ids_by_aspect[aspect]) for aspect in task.aspects)
    )
    return (
        set(snapshot.remaining_preference_ids) <= known_preferences
        and set(snapshot.remaining_search_aspects) <= set(task.aspects)
        and snapshot.active_elicited_count + snapshot.passive_elicited_count
        <= len(known_preferences)
    )


def _evidence_transition_is_valid(
    task: TravelRewardTask,
    before: UserBenchRewardSnapshot,
    after: UserBenchRewardSnapshot,
) -> bool:
    if not _snapshot_matches_task(task, before) or not _snapshot_matches_task(
        task, after
    ):
        return False
    newly_elicited = before.remaining_preference_ids - after.remaining_preference_ids
    return (
        set(after.remaining_preference_ids)
        <= set(before.remaining_preference_ids)
        and set(after.remaining_search_aspects)
        <= set(before.remaining_search_aspects)
        and before.choice_initials <= after.choice_initials
        and after.active_elicited_count >= before.active_elicited_count
        and after.passive_elicited_count >= before.passive_elicited_count
        and (
            after.active_elicited_count
            - before.active_elicited_count
            + after.passive_elicited_count
            - before.passive_elicited_count
            <= len(newly_elicited)
        )
    )


def _fallback_counts(diagnostics: object) -> dict[str, int]:
    if not isinstance(diagnostics, ABCMapping):
        return {}
    return {
        name: int(value)
        for name in _SIMULATOR_FALLBACK_NAMES
        if isinstance(value := diagnostics.get(name), int)
        and not isinstance(value, bool)
        and value > 0
    }


@dataclass(frozen=True)
class EmbeddedUserBench:
    root: Path
    upstream_commit: str
    license: str
    license_file: Path


def validate_embedded_userbench(root: str | Path | None = None) -> EmbeddedUserBench:
    """Validate provenance and licensing for the project-pinned snapshot."""

    source_root = Path(root) if root is not None else DEFAULT_USERBENCH_ROOT
    source_root = source_root.resolve()
    return _validate_embedded_userbench_cached(source_root)


@cache
def _validate_embedded_userbench_cached(source_root: Path) -> EmbeddedUserBench:
    manifest_path = source_root / "EMBEDDED_SOURCE.json"
    if not manifest_path.is_file():
        raise UserBenchSourceError(
            f"missing UserBench provenance manifest: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UserBenchSourceError(
            f"invalid UserBench provenance manifest: {manifest_path}"
        ) from exc
    commit = manifest.get("upstream_commit")
    if commit != PINNED_USERBENCH_COMMIT:
        raise UserBenchSourceError(
            f"UserBench commit is {commit!r}, expected {PINNED_USERBENCH_COMMIT!r}"
        )
    if manifest.get("license") != "Apache-2.0":
        raise UserBenchSourceError("embedded UserBench license must be Apache-2.0")
    license_name = manifest.get("license_file")
    if not isinstance(license_name, str) or not license_name:
        raise UserBenchSourceError("UserBench provenance is missing license_file")
    license_file = source_root / license_name
    if not license_file.is_file():
        raise UserBenchSourceError(
            f"missing embedded UserBench license: {license_file}"
        )
    try:
        license_text = license_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise UserBenchSourceError(
            f"cannot read embedded UserBench license: {license_file}"
        ) from exc
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise UserBenchSourceError("embedded UserBench license text is not Apache-2.0")
    if not (source_root / "travelgym" / "env" / "travel_env.py").is_file():
        raise UserBenchSourceError(f"missing TravelGym source under {source_root}")
    if any(source_root.rglob(".git")):
        raise UserBenchSourceError(
            "embedded UserBench snapshot must not contain a nested .git"
        )
    return EmbeddedUserBench(
        root=source_root,
        upstream_commit=commit,
        license="Apache-2.0",
        license_file=license_file,
    )


@dataclass
class UserBenchSessionState:
    """Mutable state shared by one AgentLoop task and its tool child tasks."""

    request_id: str
    task_id: str
    wrapper: UserBenchWrapper
    rewards: RawRewardTrace = field(default_factory=RawRewardTrace)
    num_tool_calls: int = 0
    actor_attempts: int = 0
    terminated: bool = False
    truncated: bool = False
    protocol_error: str | None = None
    invalid_actions: int = 0
    termination_reason: str | None = None
    reward_task: TravelRewardTask | None = None
    reward_snapshot: UserBenchRewardSnapshot | None = None
    active_preference_ids: set[str] = field(default_factory=set)
    passive_preference_ids: set[str] = field(default_factory=set)
    searched_aspects: set[str] = field(default_factory=set)
    answers: dict[str, str] = field(default_factory=dict)
    exact_repeats: int = 0
    semantic_repeats: int = 0
    ambiguous_actions: int = 0
    unsearched_answers: int = 0
    wrong_answers: int = 0
    parallel_tool_calls: bool = False
    infrastructure_errors: list[str] = field(default_factory=list)
    reward_degraded: bool = False
    simulator_fallback_counts: dict[str, int] = field(default_factory=dict)
    _action_signatures: set[str] = field(default_factory=set, repr=False)
    _semantic_signatures: set[tuple[str, str]] = field(default_factory=set, repr=False)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def record_step(
        self,
        result: UserBenchStepResult,
        action: UserBenchAction | None = None,
        snapshot: UserBenchRewardSnapshot | None = None,
        *,
        count_action_repetition: bool = True,
    ) -> None:
        if result.task_id != self.task_id:
            raise UserBenchSessionError(
                f"step task ID {result.task_id!r} does not match session {self.task_id!r}"
            )
        self.rewards.append(result.reward)
        self.num_tool_calls += 1
        self.terminated = result.terminated
        self.truncated = result.truncated
        if result.terminated and self.termination_reason is None:
            self.termination_reason = "environment_terminated"
        if result.truncated:
            self.termination_reason = "max_steps"

        repeated_exactly = False
        semantic: tuple[str, str] | None = None
        if action is not None and count_action_repetition:
            signature = normalized_action_signature(action)
            repeated_exactly = signature in self._action_signatures
            if repeated_exactly:
                self.exact_repeats += 1
            self._action_signatures.add(signature)
        if self.reward_task is not None and action is not None:
            semantic = semantic_action_signature(action, self.reward_task.aspects)
            if (
                count_action_repetition
                and semantic is not None
                and not repeated_exactly
            ):
                if semantic in self._semantic_signatures:
                    self.semantic_repeats += 1
                self._semantic_signatures.add(semantic)
            if action_query_issue(action, self.reward_task.aspects) is not None:
                self.ambiguous_actions += 1

        fallback_counts = _fallback_counts(result.diagnostics)
        for name, count in fallback_counts.items():
            self.simulator_fallback_counts[name] = (
                self.simulator_fallback_counts.get(name, 0) + count
            )

        before = self.reward_snapshot
        if (
            not _is_complete_reward_task(self.reward_task)
            or not _is_complete_reward_snapshot(before)
            or not _is_complete_reward_snapshot(snapshot)
            or not _snapshot_matches_task(self.reward_task, before)
            or not _snapshot_matches_task(self.reward_task, snapshot)
            or not _evidence_transition_is_valid(
                self.reward_task, before, snapshot
            )
        ):
            if "missing_reward_evidence" not in self.infrastructure_errors:
                self.infrastructure_errors.append("missing_reward_evidence")
            self.reward_snapshot = snapshot
            return

        newly_elicited = sorted(
            before.remaining_preference_ids - snapshot.remaining_preference_ids,
            key=lambda value: int(value[1:]) if value[1:].isdigit() else value,
        )
        active_delta = max(
            0, snapshot.active_elicited_count - before.active_elicited_count
        )
        passive_delta = max(
            0, snapshot.passive_elicited_count - before.passive_elicited_count
        )
        active_candidates = list(newly_elicited)
        if semantic is not None:
            target_ids = self.reward_task.preference_ids_by_aspect.get(
                semantic[0], frozenset()
            )
            active_candidates.sort(key=lambda value: value not in target_ids)
        active_values = active_candidates[:active_delta]
        passive_candidates = [
            value for value in newly_elicited if value not in set(active_values)
        ]
        self.active_preference_ids.update(active_values)
        self.passive_preference_ids.update(
            passive_candidates[:passive_delta]
        )
        self.searched_aspects.update(
            before.remaining_search_aspects - snapshot.remaining_search_aspects
        )

        if action is not None and action.choice is ActionChoice.ANSWER:
            newly_chosen = snapshot.choice_initials - before.choice_initials
            feedback = result.observation.feedback
            if (
                "Invalid option ID format" in feedback
                or "already recommended an option" in feedback
            ):
                self.invalid_actions += 1
            for option_id in (value.strip() for value in action.content.split(",")):
                aspect = aspect_from_option_id(option_id)
                if aspect is None or option_id[:1] not in newly_chosen:
                    continue
                self.answers.setdefault(aspect, option_id)
                if aspect not in self.searched_aspects:
                    self.unsearched_answers += 1
                if option_id not in self.reward_task.correct_ids.get(aspect, frozenset()):
                    self.wrong_answers += 1

        if fallback_counts:
            # A pinned simulator fallback is soft degradation only when both
            # snapshots and the evidence-ledger deltas above were valid.
            self.reward_degraded = True
        self.reward_snapshot = snapshot

    def reward_report(self) -> dict[str, Any]:
        if not _is_complete_reward_task(self.reward_task):
            return {
                "reward_version": "userbench-travel-reward-v2",
                "reward_valid": False,
                "terminal_reward": 0.0,
                "termination_reason": self.termination_reason,
                "infrastructure_invalid": True,
                "infrastructure_errors": [
                    *self.infrastructure_errors,
                    "missing_reward_task",
                ],
                "reward_degraded": bool(self.reward_degraded),
                "simulator_fallback_counts": dict(self.simulator_fallback_counts),
            }
        if not _is_complete_reward_snapshot(self.reward_snapshot) or not _snapshot_matches_task(
            self.reward_task, self.reward_snapshot
        ):
            return {
                "reward_version": "userbench-travel-reward-v2",
                "reward_valid": False,
                "terminal_reward": 0.0,
                "termination_reason": self.termination_reason,
                "infrastructure_invalid": True,
                "infrastructure_errors": [
                    *self.infrastructure_errors,
                    "missing_reward_evidence",
                ],
                "reward_degraded": bool(self.reward_degraded),
                "simulator_fallback_counts": dict(self.simulator_fallback_counts),
            }
        no_tool_output = self.termination_reason == "no_tool_output"
        report = compute_travel_reward(
            task=self.reward_task,
            answers=self.answers,
            active_preference_ids=self.active_preference_ids,
            passive_preference_ids=self.passive_preference_ids,
            searched_aspects=self.searched_aspects,
            steps=self.num_tool_calls,
            actor_attempts=self.actor_attempts,
            max_steps=20,
            invalid_actions=self.invalid_actions,
            exact_repeats=self.exact_repeats,
            semantic_repeats=self.semantic_repeats,
            ambiguous_actions=self.ambiguous_actions,
            unsearched_answers=self.unsearched_answers,
            wrong_answers=self.wrong_answers,
            parallel_tool_calls=self.parallel_tool_calls,
            no_tool_output=no_tool_output,
            max_steps_reached=self.truncated,
            reward_valid=not self.infrastructure_errors,
            termination_reason=self.termination_reason,
        )
        report["infrastructure_errors"] = list(self.infrastructure_errors)
        report["reward_degraded"] = bool(self.reward_degraded)
        report["simulator_fallback_counts"] = dict(self.simulator_fallback_counts)
        report.update(
            {
                "invalid_actions": self.invalid_actions,
                "exact_repeats": self.exact_repeats,
                "semantic_repeats": self.semantic_repeats,
                "ambiguous_actions": self.ambiguous_actions,
                "unsearched_answers": self.unsearched_answers,
                "wrong_answers": self.wrong_answers,
            }
        )
        return report

    def metrics(self) -> dict[str, Any]:
        report = self.reward_report()
        return {
            "task_id": self.task_id,
            "reward_version": report["reward_version"],
            "reward_valid": report["reward_valid"],
            "terminal_reward": report["terminal_reward"],
            "reward_breakdown": report,
            "step_rewards": list(self.rewards.values),
            "raw_cumulative_reward": self.rewards.total,
            "cumulative_reward": self.rewards.total,
            "num_tool_calls": self.num_tool_calls,
            "actor_attempts": self.actor_attempts,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "protocol_error": self.protocol_error,
            "invalid_actions": self.invalid_actions,
            "exact_repeats": self.exact_repeats,
            "semantic_repeats": self.semantic_repeats,
            "ambiguous_actions": self.ambiguous_actions,
            "termination_reason": self.termination_reason,
            "reward_degraded": report["reward_degraded"],
            "simulator_fallback_counts": report["simulator_fallback_counts"],
        }


CURRENT_USERBENCH_SESSION: contextvars.ContextVar[UserBenchSessionState | None] = (
    contextvars.ContextVar("current_userbench_session", default=None)
)


def set_current_session(session: UserBenchSessionState) -> None:
    CURRENT_USERBENCH_SESSION.set(session)


def get_current_session() -> UserBenchSessionState | None:
    return CURRENT_USERBENCH_SESSION.get()


def require_current_session() -> UserBenchSessionState:
    session = get_current_session()
    if session is None:
        raise UserBenchSessionError(
            "UserBench session is missing from the current asyncio context"
        )
    return session


def clear_current_session(*, close: bool = True) -> None:
    session = get_current_session()
    CURRENT_USERBENCH_SESSION.set(None)
    if close and session is not None:
        session.wrapper.close()
