"""Pinned-source validation and per-trajectory UserBench context."""

from __future__ import annotations

import contextvars
import json
from collections.abc import Mapping as ABCMapping, Set as AbstractSet
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from travel_grpo.envs.observation import UserBenchStepResult
from travel_grpo.envs.public_control import (
    PublicAspectStatus,
    PublicControlState,
    PublicControlEvent,
    RecoveryMode,
    advance_public_aspect,
    classify_public_observation,
    new_public_control_state,
    note_public_non_progress,
    reduce_public_control_state,
    render_actor_control_info,
    validate_public_action,
)
from travel_grpo.envs.reward import (
    REWARD_VERSION,
    RawRewardTrace,
    TravelRewardTask,
    UserBenchRewardSnapshot,
    compute_travel_reward,
)
from travel_grpo.envs.userbench_tools import (
    ActionChoice,
    OPTION_ID,
    UserBenchAction,
    action_query_issue,
    aspect_from_option_id,
    extract_visible_option_ids,
    normalized_action_signature,
    semantic_action_signature,
)
from travel_grpo.training.grpo.turn_credit import (
    TurnCreditConfig,
    TurnCreditError,
    TurnCreditTrace,
    TurnEvent,
    build_turn_credit_trace,
    validate_turn_credit_mode,
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


_EXPLICIT_NO_PREFERENCE_MARKERS = (
    "no preference",
    "no specific preference",
    "do not have a preference",
    "don't have a preference",
    "does not matter",
    "doesn't matter",
    "anything is fine",
    "any is fine",
    "no requirement",
    "no requirements",
)


@dataclass(frozen=True)
class _TurnLedgerSnapshot:
    """Private aggregate snapshot used only to derive one turn event."""

    public_aspect: str | None
    public_phase: str
    public_fallback_count: int
    visible_option_ids: frozenset[str]
    preference_count: int
    answer_count: int
    exact_repeats: int
    semantic_repeats: int
    wrong_answers: int
    infrastructure_error_count: int


def _contains_explicit_no_preference(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    return any(marker in normalized for marker in _EXPLICIT_NO_PREFERENCE_MARKERS)


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
    # Public phase/guard accounting. These counters use only the actor-visible
    # control ledger and are safe to expose as diagnostics; hidden reward
    # snapshots never participate in their updates.
    guard_rejections: int = 0
    guard_rejection_reasons: dict[str, int] = field(default_factory=dict)
    valid_search_required_transitions: int = 0
    search_required_opportunities: int = 0
    valid_candidate_answer_transitions: int = 0
    candidate_answer_opportunities: int = 0
    valid_retry_search_transitions: int = 0
    retry_search_opportunities: int = 0
    valid_aspect_switch_transitions: int = 0
    aspect_switch_opportunities: int = 0
    # Runtime-only GRPO training control.  These fields are intentionally not
    # part of any teacher/SFT/Parquet contract.
    stall_recovery_enabled: bool = False
    stall_no_progress_threshold: int = 4
    consecutive_no_progress: int = 0
    max_consecutive_no_progress: int = 0
    stall_recovery_triggered: bool = False
    stall_recovery_used: bool = False
    answer_only_pending: bool = False
    answer_only_generation_started: bool = False
    stall_hard_truncated: bool = False
    visible_option_ids_by_aspect: dict[str, set[str]] = field(default_factory=dict)
    _action_signatures: set[str] = field(default_factory=set, repr=False)
    _semantic_signatures: set[tuple[str, str]] = field(default_factory=set, repr=False)
    # Public control is an independent, actor-visible ledger. It is optional
    # for legacy/offline callers that construct a session without reset text;
    # runtime AgentLoop sessions always initialize it from the reset feedback.
    public_control_state: PublicControlState | None = None
    public_initial_message: str | None = None
    # Optional trainer-only causal turn ledger. Hidden reward information is
    # reduced to counts/booleans and is never used by public control methods.
    turn_credit_mode: str = "off"
    turn_credit_config: TurnCreditConfig = field(default_factory=TurnCreditConfig)
    turn_events: list[TurnEvent] = field(default_factory=list, repr=False)
    _active_turn: TurnEvent | None = field(default=None, repr=False)
    _turn_ledger_snapshot: _TurnLedgerSnapshot | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.stall_no_progress_threshold, bool)
            or not isinstance(self.stall_no_progress_threshold, int)
            or self.stall_no_progress_threshold < 1
        ):
            raise ValueError("stall_no_progress_threshold must be an integer >= 1")
        for name in (
            "guard_rejections",
            "valid_search_required_transitions",
            "search_required_opportunities",
            "valid_candidate_answer_transitions",
            "candidate_answer_opportunities",
            "valid_retry_search_transitions",
            "retry_search_opportunities",
            "valid_aspect_switch_transitions",
            "aspect_switch_opportunities",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.guard_rejection_reasons.values()
        ):
            raise ValueError("guard_rejection_reasons counts must be non-negative integers")
        if self.public_control_state is not None and not isinstance(
            self.public_control_state, PublicControlState
        ):
            raise TypeError("public_control_state must be PublicControlState or None")
        if self.public_control_state is None and self.public_initial_message is not None:
            if not isinstance(self.public_initial_message, str):
                raise TypeError("public_initial_message must be a string or None")
            self.public_control_state = new_public_control_state(
                self.public_initial_message,
                no_progress_threshold=self.stall_no_progress_threshold,
            )
        self.turn_credit_mode = validate_turn_credit_mode(self.turn_credit_mode)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def configure_turn_credit(
        self,
        *,
        mode: str,
        config: TurnCreditConfig | None = None,
    ) -> None:
        """Configure the trainer-only ledger without changing public control."""

        normalized = validate_turn_credit_mode(mode)
        if self.turn_events or self._active_turn is not None:
            if normalized != self.turn_credit_mode:
                raise TurnCreditError("turn-credit mode cannot change during a rollout")
        self.turn_credit_mode = normalized
        if config is not None:
            self.turn_credit_config = config

    def _capture_turn_ledger_snapshot(self) -> _TurnLedgerSnapshot:
        state = self.public_control_state
        aspect = state.current_aspect if state is not None else None
        aspect_state = state.current if state is not None else None
        public_ids = (
            aspect_state.visible_option_ids
            if aspect_state is not None
            else frozenset(self.visible_option_ids_by_aspect.get(aspect or "", set()))
        )
        return _TurnLedgerSnapshot(
            public_aspect=aspect,
            public_phase=state.phase.value if state is not None else RecoveryMode.ELICITING.value,
            public_fallback_count=(
                aspect_state.search_fallbacks if aspect_state is not None else 0
            ),
            visible_option_ids=frozenset(public_ids),
            preference_count=len(self.active_preference_ids | self.passive_preference_ids),
            answer_count=len(self.answers),
            exact_repeats=self.exact_repeats,
            semantic_repeats=self.semantic_repeats,
            wrong_answers=self.wrong_answers,
            infrastructure_error_count=len(self.infrastructure_errors),
        )

    def begin_actor_turn(self) -> None:
        """Open one ledger event immediately before assistant generation."""

        if self.turn_credit_mode == "off":
            return
        if self._active_turn is not None:
            raise TurnCreditError("previous actor turn was not finalized")
        snapshot = self._capture_turn_ledger_snapshot()
        self._turn_ledger_snapshot = snapshot
        self._active_turn = TurnEvent(
            turn_index=len(self.turn_events),
            public_aspect=snapshot.public_aspect,
            phase_before=snapshot.public_phase,
            phase_after=snapshot.public_phase,
        )

    def _ensure_actor_turn(self) -> tuple[TurnEvent, _TurnLedgerSnapshot] | None:
        if self.turn_credit_mode == "off":
            return None
        if self._active_turn is None:
            # Provider-neutral tool tests and offline callers do not pass
            # through AgentLoop._handle_generating_state.
            self.begin_actor_turn()
        if self._active_turn is None or self._turn_ledger_snapshot is None:
            raise TurnCreditError("actor turn ledger is unavailable")
        return self._active_turn, self._turn_ledger_snapshot

    def _finish_actor_turn(self) -> None:
        if self.turn_credit_mode == "off":
            return
        if self._active_turn is None:
            raise TurnCreditError("cannot finish an unopened actor turn")
        self.turn_events.append(self._active_turn)
        self._active_turn = None
        self._turn_ledger_snapshot = None

    def _turn_aspect_for_action(
        self, action: UserBenchAction, fallback: str | None
    ) -> str | None:
        if action.choice is ActionChoice.ANSWER:
            submitted = [value.strip() for value in action.content.split(",") if value.strip()]
            if len(submitted) == 1:
                aspect = aspect_from_option_id(submitted[0])
                if aspect is not None:
                    return aspect
        state = self.public_control_state
        if state is not None:
            semantic = semantic_action_signature(action, state.public_aspects)
            if semantic is not None:
                return semantic[0]
        return fallback

    def complete_actor_turn_from_step(
        self, action: UserBenchAction, result: UserBenchStepResult
    ) -> None:
        """Close a successful environment dispatch after ``record_step``."""

        active = self._ensure_actor_turn()
        if active is None:
            return
        event, before = active
        public = self.public_control_state
        observation = classify_public_observation(
            result.observation.feedback, choice=action.choice
        )
        aspect = self._turn_aspect_for_action(action, before.public_aspect)
        after_aspect = None
        if public is not None and aspect is not None:
            after_aspect = next(
                (item for item in public.aspects if item.aspect == aspect), None
            )
        feedback = result.observation.feedback
        preference_delta = max(
            0,
            len(self.active_preference_ids | self.passive_preference_ids)
            - before.preference_count,
        )
        explicit_none = (
            action.choice is ActionChoice.ACTION
            and _contains_explicit_no_preference(feedback)
        )
        submitted = [value.strip() for value in action.content.split(",") if value.strip()]
        answer_visible = (
            action.choice is ActionChoice.ANSWER
            and len(submitted) == 1
            and submitted[0] in before.visible_option_ids
        )
        answer_added = len(self.answers) > before.answer_count
        correct_answer = False
        if (
            answer_added
            and len(submitted) == 1
            and self.reward_task is not None
            and aspect is not None
        ):
            correct_answer = submitted[0] in self.reward_task.correct_ids.get(
                aspect, frozenset()
            )

        event.choice = action.choice.value
        event.tool_name = "interact_with_env"
        event.public_aspect = aspect
        event.phase_after = public.phase.value if public is not None else event.phase_before
        event.field_resolved = preference_delta > 0 or explicit_none
        event.explicit_no_preference = explicit_none
        event.normal_candidates_observed = observation.is_normal_search and bool(
            observation.visible_option_ids
        )
        event.answer_id_visible = answer_visible
        fallback_after = (
            after_aspect.search_fallbacks
            if after_aspect is not None
            else before.public_fallback_count
        )
        fallback_delta = max(0, fallback_after - before.public_fallback_count)
        event.first_fallback = fallback_delta > 0 and fallback_after == 1
        event.second_fallback = fallback_delta > 0 and fallback_after >= 2
        event.successful_rewritten_retry = (
            before.public_phase == RecoveryMode.SEARCH_RETRY_REQUIRED.value
            and event.normal_candidates_observed
        )
        event.switched_aspect = (
            public is not None
            and before.public_aspect is not None
            and public.current_aspect != before.public_aspect
        )
        event.no_progress_action = (
            action.choice is ActionChoice.ACTION and not event.field_resolved
        )
        event.exact_query_repeat = self.exact_repeats > before.exact_repeats
        event.semantic_repeat = self.semantic_repeats > before.semantic_repeats
        event.infrastructure_failure = (
            len(self.infrastructure_errors) > before.infrastructure_error_count
        )
        event.reward_new_preference_count = preference_delta
        event.reward_correct_answer = correct_answer
        event.reward_wrong_answer = self.wrong_answers > before.wrong_answers
        self._finish_actor_turn()

    def reject_actor_turn(
        self,
        *,
        reason: str,
        action: UserBenchAction | None = None,
        category: str = "guard_rejection",
    ) -> None:
        """Close a rejected call using only its public reason and actor action."""

        active = self._ensure_actor_turn()
        if active is None:
            return
        event, before = active
        normalized = " ".join(reason.casefold().split())
        if action is not None:
            event.choice = action.choice.value
            event.public_aspect = self._turn_aspect_for_action(
                action, before.public_aspect
            )
        event.tool_name = "interact_with_env"
        state = self.public_control_state
        event.phase_after = state.phase.value if state is not None else event.phase_before
        event.guard_rejected = category in {
            "guard_rejection",
            "public_phase_guard",
            "answer_only_violation",
        }
        event.malformed_tool_call = category in {
            "invalid_tool_call",
            "malformed_tool_call",
            "unknown_tool",
        }
        event.no_tool_output = category in {"no_tool_output", "parallel_tool_calls"}
        event.exact_query_repeat = (
            action is not None
            and action.choice is ActionChoice.SEARCH
            and (
                "already attempted" in normalized
                or "materially change" in normalized
                or "repeat" in normalized and "query" in normalized
            )
        )
        event.wrong_aspect = (
            action is not None
            and action.choice in {ActionChoice.SEARCH, ActionChoice.ANSWER}
            and any(
                marker in normalized
                for marker in (
                    "different public aspect",
                    "current public aspect only",
                    "must target the current public aspect",
                )
            )
        )
        event.invisible_answer_id = (
            action is not None
            and action.choice is ActionChoice.ANSWER
            and (
                "answer id is not visible" in normalized
                or "unseen or already answered option" in normalized
            )
        )
        self._finish_actor_turn()

    def record_turn_infrastructure_failure(
        self, action: UserBenchAction | None = None
    ) -> None:
        active = self._ensure_actor_turn()
        if active is None:
            return
        event, before = active
        if action is not None:
            event.choice = action.choice.value
            event.public_aspect = self._turn_aspect_for_action(action, before.public_aspect)
        event.tool_name = "interact_with_env"
        event.infrastructure_failure = True
        self._finish_actor_turn()

    def finalize_pending_actor_turn(self, *, reason: str = "no_tool_output") -> None:
        if self.turn_credit_mode == "off" or self._active_turn is None:
            return
        self.reject_actor_turn(reason=reason, category=reason)

    def finalize_turn_credit(
        self, reward_report: ABCMapping[str, Any]
    ) -> TurnCreditTrace:
        """Build the hidden-ID-free trainer trace after terminal reward."""

        if self._active_turn is not None:
            self.finalize_pending_actor_turn()
        public = self.public_control_state
        aspects = public.public_aspects if public is not None else ()
        blocked = public.blocked_aspects if public is not None else ()
        events = self.turn_events if self.turn_credit_mode != "off" else ()
        return build_turn_credit_trace(
            events,
            aspects,
            blocked_aspects=blocked,
            reward_valid=bool(reward_report.get("reward_valid", False)),
            config=self.turn_credit_config,
        )

    def record_public_guard_rejection(self, reason: str) -> None:
        """Record one pre-environment public guard rejection."""

        if not isinstance(reason, str) or not reason.strip():
            raise TypeError("guard rejection reason must be a non-empty string")
        self.guard_rejections += 1
        key = reason.strip()
        self.guard_rejection_reasons[key] = (
            self.guard_rejection_reasons.get(key, 0) + 1
        )

    def _record_public_phase_attempt(
        self, action: UserBenchAction, reason: str | None
    ) -> None:
        """Count public phase opportunities and accepted transitions."""

        state = self.public_control_state
        if state is None:
            return
        phase = state.phase
        if phase is RecoveryMode.SEARCH_REQUIRED:
            self.search_required_opportunities += 1
            if reason is None and action.choice is ActionChoice.SEARCH:
                self.valid_search_required_transitions += 1
        elif phase is RecoveryMode.SEARCH_RETRY_REQUIRED:
            self.retry_search_opportunities += 1
            if reason is None and action.choice is ActionChoice.SEARCH:
                self.valid_retry_search_transitions += 1
        elif phase is RecoveryMode.ANSWER_REQUIRED:
            self.candidate_answer_opportunities += 1
            if reason is None and action.choice is ActionChoice.ANSWER:
                self.valid_candidate_answer_transitions += 1

    def _sync_public_control_metrics(self) -> None:
        state = self.public_control_state
        if state is None:
            return
        self.consecutive_no_progress = state.consecutive_no_progress
        self.max_consecutive_no_progress = state.max_consecutive_no_progress
        self.stall_recovery_triggered = state.phase is RecoveryMode.SEARCH_REQUIRED
        if state.all_aspects_terminal:
            self.termination_reason = (
                self.termination_reason or "public_control_complete"
            )
            self.terminated = True

    def _advance_public_control_if_needed(self) -> None:
        state = self.public_control_state
        if state is None:
            return
        if state.phase is RecoveryMode.SWITCH_ASPECT_REQUIRED and state.current is not None:
            if state.current.status in {
                PublicAspectStatus.ANSWERED,
                PublicAspectStatus.BLOCKED,
            }:
                self.aspect_switch_opportunities += 1
                self.public_control_state = advance_public_aspect(state)
                advanced = self.public_control_state
                if (
                    advanced.current_aspect != state.current_aspect
                    or advanced.episode_done
                ):
                    self.valid_aspect_switch_transitions += 1
        self._sync_public_control_metrics()

    def prepare_public_action(self) -> None:
        """Advance a terminal public aspect before the next actor call."""

        self._advance_public_control_if_needed()

    def validate_public_action(self, action: UserBenchAction) -> str | None:
        """Validate an actor call without consulting reward state."""

        self.prepare_public_action()
        if self.public_control_state is None:
            return None
        reason = validate_public_action(self.public_control_state, action)
        self._record_public_phase_attempt(action, reason)
        return reason

    def record_public_non_progress(self, reason: str | None = None) -> None:
        del reason
        if self.public_control_state is None:
            return
        self.public_control_state = note_public_non_progress(
            self.public_control_state
        )
        self._sync_public_control_metrics()

    def _record_public_step(
        self,
        result: UserBenchStepResult,
        action: UserBenchAction | None,
    ) -> None:
        state = self.public_control_state
        if state is None or action is None:
            return
        observation = classify_public_observation(
            result.observation.feedback,
            choice=action.choice,
        )
        self.public_control_state = reduce_public_control_state(
            state,
            PublicControlEvent(action=action, observation=observation),
        )
        self._sync_public_control_metrics()

    def configure_stall_recovery(self, *, enabled: bool, threshold: int) -> None:
        """Set effective per-rollout control after sampling mode is known."""

        if not isinstance(enabled, bool):
            raise TypeError("stall recovery enabled must be a bool")
        if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 1:
            raise ValueError("stall recovery threshold must be an integer >= 1")
        self.stall_recovery_enabled = enabled
        self.stall_no_progress_threshold = threshold
        if self.public_control_state is not None:
            self.public_control_state = replace(
                self.public_control_state,
                no_progress_threshold=threshold,
            )

    @property
    def visible_answer_options(self) -> set[str]:
        """Return actor-visible options for aspects not answered yet."""

        if self.public_control_state is not None:
            return {
                option_id
                for aspect_state in self.public_control_state.aspects
                if aspect_state.status is PublicAspectStatus.OPEN
                for option_id in aspect_state.visible_option_ids
            }
        return {
            option_id
            for aspect, option_ids in self.visible_option_ids_by_aspect.items()
            if aspect not in self.answers
            for option_id in option_ids
        }

    def render_actor_feedback(self, observation_text: str) -> str:
        """Render one actor-visible observation plus public control guidance.

        Public sessions always receive the same compact control summary,
        including ordinary elicitation turns.  The renderer only consumes the
        independent public ledger; reward snapshots and hidden preference
        fields are never consulted or serialized here.  The operation is
        idempotent so an adapter boundary cannot append the same summary
        twice.
        """

        if not isinstance(observation_text, str):
            raise TypeError("observation_text must be a string")
        if self.public_control_state is not None:
            control = render_actor_control_info(self.public_control_state)
            if observation_text.endswith(control):
                return observation_text
            if not observation_text:
                return control
            return f"{observation_text}\n\n{control}"
        if not self.answer_only_pending:
            return observation_text
        instruction = self.recovery_instruction()
        if observation_text.endswith(instruction):
            return observation_text
        return f"{observation_text}\n\n{instruction}"

    def append_recovery_instruction(self, feedback: str) -> str:
        """Compatibility alias for :meth:`render_actor_feedback`."""

        return self.render_actor_feedback(feedback)


    def validate_answer_only_action(self, action: UserBenchAction) -> str | None:
        """Return a legacy recovery rejection for sessions without public state."""

        if self.public_control_state is not None:
            return None
        if not self.answer_only_pending:
            return None
        if action.choice is not ActionChoice.ANSWER:
            return "answer-only recovery accepts choice=answer only"
        submitted = [value.strip() for value in action.content.split(",")]
        if not submitted or any(not OPTION_ID.fullmatch(value) for value in submitted):
            return "answer-only recovery requires official option IDs"
        unseen = set(submitted) - self.visible_answer_options
        if unseen:
            return "answer-only recovery received an unseen or already answered option"
        return None

    @staticmethod
    def recovery_instruction() -> str:
        return (
            "Recovery instruction:\n"
            "You have made no verifiable progress for several consecutive turns.\n"
            'Your next interact_with_env call must use choice="answer".\n'
            "Only submit option IDs that were explicitly shown in previous successful\n"
            "search results for unanswered travel aspects.\n"
            "Do not search or ask another question."
        )

    def begin_answer_only_generation(self) -> None:
        """Consume the one recovery generation opportunity."""

        if not self.answer_only_pending or self.answer_only_generation_started:
            self.hard_stop_stalled()
            return
        self.answer_only_generation_started = True

    def hard_stop_stalled(self) -> None:
        """End a valid-but-stalled trajectory without marking max steps."""

        self.terminated = True
        self.truncated = False
        self.termination_reason = "stalled_no_progress"
        self.stall_hard_truncated = True
        self.answer_only_pending = False

    def _stall_evidence_is_valid(self) -> bool:
        return (
            _is_complete_reward_task(self.reward_task)
            and _is_complete_reward_snapshot(self.reward_snapshot)
            and _snapshot_matches_task(self.reward_task, self.reward_snapshot)
        )

    def _maybe_trigger_stall(self) -> None:
        # Runtime sessions with a public ledger use RecoveryMode instead. The
        # hidden reward evidence below remains only as a legacy compatibility
        # path for offline callers that have no public initial message.
        if self.public_control_state is not None:
            return
        if (
            not self.stall_recovery_enabled
            or self.infrastructure_errors
            or not self._stall_evidence_is_valid()
            or self.done
            or self.consecutive_no_progress < self.stall_no_progress_threshold
        ):
            return
        if self.stall_recovery_used or self.stall_recovery_triggered:
            self.stall_recovery_triggered = True
            self.hard_stop_stalled()
            return
        self.stall_recovery_triggered = True
        if self.visible_answer_options:
            self.answer_only_pending = True
            self.answer_only_generation_started = False
        else:
            self.hard_stop_stalled()

    def _record_no_progress(self) -> None:
        if self.public_control_state is not None:
            return
        if (
            not self.stall_recovery_enabled
            or self.infrastructure_errors
            or not self._stall_evidence_is_valid()
            or self.done
        ):
            return
        self.consecutive_no_progress += 1
        self.max_consecutive_no_progress = max(
            self.max_consecutive_no_progress,
            self.consecutive_no_progress,
        )
        self._maybe_trigger_stall()

    def record_non_progress(self, reason: str | None = None) -> None:
        """Account for a protocol event without consulting hidden state."""

        if self.public_control_state is not None:
            self.record_public_non_progress(reason)
            return
        del reason  # The existing protocol/invalid-action diagnostics remain authoritative.
        if self.infrastructure_errors or not self._stall_evidence_is_valid():
            return
        if self.answer_only_pending:
            self.consecutive_no_progress += 1
            self.max_consecutive_no_progress = max(
                self.max_consecutive_no_progress,
                self.consecutive_no_progress,
            )
            self.hard_stop_stalled()
            return
        self._record_no_progress()

    def _complete_answer_only_recovery(self) -> None:
        self.consecutive_no_progress = 0
        self.answer_only_pending = False
        self.answer_only_generation_started = False
        self.stall_recovery_used = True

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
        recovery_pending = self.answer_only_pending
        previous_answers = set(self.answers)
        self.rewards.append(result.reward)
        self.num_tool_calls += 1
        self.terminated = result.terminated
        self.truncated = result.truncated
        # Public phase transitions happen before reward-evidence validation and
        # therefore cannot be gated by hidden snapshots.
        self._record_public_step(result, action)
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
            self._sync_public_control_metrics()
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
        preference_progress = active_delta > 0 or passive_delta > 0
        newly_searched_aspects = (
            before.remaining_search_aspects - snapshot.remaining_search_aspects
        )
        search_progress = bool(newly_searched_aspects)
        self.active_preference_ids.update(active_values)
        self.passive_preference_ids.update(
            passive_candidates[:passive_delta]
        )
        self.searched_aspects.update(newly_searched_aspects)

        newly_chosen: frozenset[str] = frozenset()
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

        if (
            self.stall_recovery_enabled
            and action is not None
            and action.choice is ActionChoice.SEARCH
        ):
            for option_id in extract_visible_option_ids(result.observation.feedback):
                aspect = aspect_from_option_id(option_id)
                if aspect is not None:
                    self.visible_option_ids_by_aspect.setdefault(aspect, set()).add(
                        option_id
                    )

        if fallback_counts:
            # A pinned simulator fallback is soft degradation only when both
            # snapshots and the evidence-ledger deltas above were valid.
            self.reward_degraded = True
        self.reward_snapshot = snapshot

        answer_progress = bool(newly_chosen) and bool(
            set(self.answers) - previous_answers
        )
        made_progress = preference_progress or search_progress or answer_progress
        if self.public_control_state is not None:
            self._sync_public_control_metrics()
            return
        if not self.stall_recovery_enabled or self.infrastructure_errors:
            return
        if recovery_pending:
            if action is not None and action.choice is ActionChoice.ANSWER and answer_progress:
                self._complete_answer_only_recovery()
            else:
                self.hard_stop_stalled()
            return
        if self.done:
            return
        if made_progress:
            self.consecutive_no_progress = 0
        else:
            self._record_no_progress()

    def reward_report(self) -> dict[str, Any]:
        if not _is_complete_reward_task(self.reward_task):
            return {
                "reward_version": REWARD_VERSION,
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
                "reward_version": REWARD_VERSION,
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
            guard_rejections=self.guard_rejections,
            blocked_aspects=(
                self.public_control_state.blocked_count
                if self.public_control_state is not None
                else 0
            ),
            valid_search_required_transitions=self.valid_search_required_transitions,
            search_required_opportunities=self.search_required_opportunities,
            valid_candidate_answer_transitions=self.valid_candidate_answer_transitions,
            candidate_answer_opportunities=self.candidate_answer_opportunities,
            valid_retry_search_transitions=self.valid_retry_search_transitions,
            retry_search_opportunities=self.retry_search_opportunities,
            valid_aspect_switch_transitions=self.valid_aspect_switch_transitions,
            aspect_switch_opportunities=self.aspect_switch_opportunities,
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
                "guard_rejection_reasons": dict(self.guard_rejection_reasons),
            }
        )
        return report

    def metrics(self) -> dict[str, Any]:
        report = self.reward_report()
        public_state = self.public_control_state
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
            "stall_recovery_enabled": self.stall_recovery_enabled,
            "stall_recovery_triggered": self.stall_recovery_triggered,
            "stall_recovery_used": self.stall_recovery_used,
            "stall_hard_truncated": self.stall_hard_truncated,
            "consecutive_no_progress": self.consecutive_no_progress,
            "max_consecutive_no_progress": self.max_consecutive_no_progress,
            "answer_only_generation_started": self.answer_only_generation_started,
            "visible_answer_option_count": len(self.visible_answer_options),
            "public_control_phase": (
                public_state.phase.value if public_state is not None else None
            ),
            "public_control_recovery_mode": (
                public_state.recovery_mode.value if public_state is not None else None
            ),
            "public_control_episode_done": (
                bool(public_state.episode_done) if public_state is not None else False
            ),
            "public_answered_aspect_count": (
                public_state.answered_count if public_state is not None else 0
            ),
            "public_blocked_aspect_count": (
                public_state.blocked_count if public_state is not None else 0
            ),
            "public_open_aspect_count": (
                len(public_state.open_aspects) if public_state is not None else 0
            ),
            "guard_rejections": self.guard_rejections,
            "guard_rejection_rate": report.get("guard_rejection_rate", 0.0),
            "blocked_aspects": report.get("blocked_aspects", 0),
            "preference_coverage": report.get("preference_coverage", 0.0),
            "phase_transition_score": report.get("phase_transition_score", 0.0),
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
