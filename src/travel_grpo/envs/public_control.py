"""Public-only control state for a UserBench trajectory.

This module deliberately sits beside, rather than inside, the reward/session
ledger.  Every input accepted by the reducer is actor-visible: the initial
user message, an actor-owned :class:`UserBenchAction`, and the text returned
to the actor.  Reward snapshots, task labels, diagnostics mappings, and
correctness values are intentionally not part of this API.

The reducer tracks public evidence and exposes the finite recovery phase
guard. It never reads reward snapshots or hidden task labels; the adapter
can call validate_public_action before invoking UserBench and then feed
visible feedback back through the reducer.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from travel_grpo.envs.userbench_tools import (
    ASPECT_QUERY_HINTS,
    ActionChoice,
    OPTION_ID,
    UserBenchAction,
    aspect_from_option_id,
    extract_visible_option_ids,
    normalized_action_signature,
    semantic_action_signature,
)


class PublicControlError(ValueError):
    """Raised when a public control event violates this module's contract."""


class RecoveryMode(str, Enum):
    """Public phase/recovery signal used by the finite control guard.

    NONE/ELICITING and BLOCKED/SWITCH_ASPECT_REQUIRED are identity-compatible
    aliases. This keeps old rollout callers working while the canonical phase
    guard has one transition path.
    """

    NONE = "none"
    # Compatibility alias: idle open-aspect state is the canonical
    # elicitation phase.
    ELICITING = "none"
    SEARCH_REQUIRED = "search_required"
    SEARCH_RETRY_REQUIRED = "search_retry_required"
    ANSWER_REQUIRED = "answer_required"
    SWITCH_ASPECT_REQUIRED = "switch_aspect_required"
    # A second fallback marks the aspect BLOCKED and immediately enters the
    # switch phase; retain the old symbol as an identity-compatible alias.
    BLOCKED = "switch_aspect_required"
    ANSWERED = "answered"


class PublicAspectStatus(str, Enum):
    """Terminal status derived only from public evidence."""

    OPEN = "open"
    ANSWERED = "answered"
    BLOCKED = "blocked"


class PublicObservationKind(str, Enum):
    """Classification of text returned to the actor."""

    TEXT = "text"
    SEARCH_NORMAL = "search_normal"
    SEARCH_EMPTY = "search_empty"
    SEARCH_FALLBACK = "search_fallback"
    FALLBACK = "fallback"


# These are public phrases, not wrapper diagnostics.  A hidden diagnostics
# mapping is intentionally not accepted by ``classify_public_observation``.
PUBLIC_SEARCH_FALLBACK_MARKERS = (
    "searching backend is experiencing some issues",
    "search backend is experiencing some issues",
    "by default will return n/a",
    "search backend is unavailable",
    "search is temporarily unavailable",
)


def _normalise_public_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _hint_pattern(hint: str) -> re.Pattern[str]:
    parts = [re.escape(part) for part in hint.casefold().split()]
    return re.compile(
        rf"(?<![a-z0-9]){'\\s+'.join(parts)}(?![a-z0-9])",
        re.IGNORECASE,
    )


def extract_public_aspects(initial_user_message: str) -> tuple[str, ...]:
    """Return only travel aspects literally named in the public user message.

    The function never accepts task dimensions or a reward task.  A known
    static hint is usable only when it occurs in ``initial_user_message``;
    this prevents a hidden task aspect from being inferred just because the
    option prefix is known to the project.
    """

    if not isinstance(initial_user_message, str):
        raise TypeError("initial_user_message must be a string")
    mentions: list[tuple[int, str]] = []
    for aspect, hints in ASPECT_QUERY_HINTS.items():
        for hint in hints:
            match = _hint_pattern(hint).search(initial_user_message)
            if match is not None:
                mentions.append((match.start(), aspect))
                break
    mentions.sort(key=lambda item: (item[0], item[1]))
    result: list[str] = []
    for _, aspect in mentions:
        if aspect not in result:
            result.append(aspect)
    return tuple(result)


@dataclass(frozen=True)
class PublicObservation:
    """Actor-visible observation and its public-only classification."""

    text: str
    kind: PublicObservationKind
    visible_option_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("public observation text must be a string")
        if not isinstance(self.kind, PublicObservationKind):
            raise TypeError("public observation kind must be PublicObservationKind")
        if not isinstance(self.visible_option_ids, frozenset):
            raise TypeError("visible_option_ids must be a frozenset")
        if any(
            not isinstance(option_id, str) or OPTION_ID.fullmatch(option_id) is None
            for option_id in self.visible_option_ids
        ):
            raise PublicControlError("visible_option_ids must contain official option IDs")

    @property
    def is_fallback(self) -> bool:
        return self.kind in {
            PublicObservationKind.SEARCH_FALLBACK,
            PublicObservationKind.FALLBACK,
        }

    @property
    def is_normal_search(self) -> bool:
        return self.kind is PublicObservationKind.SEARCH_NORMAL

    @property
    def signature(self) -> str:
        """Stable transcript fingerprint used only for public progress."""

        ids = ",".join(sorted(self.visible_option_ids))
        return f"{self.kind.value}:{_normalise_public_text(self.text)}:{ids}"


def _coerce_choice(choice: ActionChoice | str | None) -> ActionChoice | None:
    if choice is None:
        return None
    if isinstance(choice, ActionChoice):
        return choice
    if isinstance(choice, str):
        try:
            return ActionChoice(choice.strip().casefold())
        except ValueError as exc:
            raise PublicControlError(f"unknown public action choice: {choice!r}") from exc
    raise TypeError("choice must be ActionChoice, string, or None")


def classify_public_observation(
    feedback: str,
    *,
    choice: ActionChoice | str | None = None,
) -> PublicObservation:
    """Classify only the text delivered to the actor.

    ``diagnostics`` and ``UserBenchRewardSnapshot`` are intentionally absent
    from the signature.  Fallback markers take precedence over visible IDs so
    a degraded response containing an incidental ID cannot become a normal
    candidate list.
    """

    if not isinstance(feedback, str):
        raise TypeError("feedback must be a string")
    normalized = _normalise_public_text(feedback)
    selected_choice = _coerce_choice(choice)
    visible_ids = frozenset(extract_visible_option_ids(feedback))
    fallback = any(marker in normalized for marker in PUBLIC_SEARCH_FALLBACK_MARKERS)
    if fallback:
        kind = (
            PublicObservationKind.SEARCH_FALLBACK
            if selected_choice is ActionChoice.SEARCH
            else PublicObservationKind.FALLBACK
        )
    elif selected_choice is ActionChoice.SEARCH:
        kind = (
            PublicObservationKind.SEARCH_NORMAL
            if visible_ids
            else PublicObservationKind.SEARCH_EMPTY
        )
    else:
        kind = PublicObservationKind.TEXT
    return PublicObservation(feedback, kind, visible_ids)


def public_action_signature(action: UserBenchAction) -> str:
    """Return a thought-independent signature of an actor-owned action."""

    if not isinstance(action, UserBenchAction):
        raise TypeError("action must be a UserBenchAction")
    return normalized_action_signature(action)


def public_search_signature(action: UserBenchAction) -> str | None:
    """Return a public query signature, or ``None`` for non-search actions."""

    if not isinstance(action, UserBenchAction):
        raise TypeError("action must be a UserBenchAction")
    if action.choice is not ActionChoice.SEARCH:
        return None
    return public_action_signature(action)


def public_semantic_signature(
    action: UserBenchAction,
    public_aspects: Sequence[str],
) -> tuple[str, str] | None:
    """Infer an action field only from aspect names in the public transcript."""

    if not isinstance(action, UserBenchAction):
        raise TypeError("action must be a UserBenchAction")
    aspects = _validate_public_aspects(public_aspects)
    semantic = semantic_action_signature(action, aspects)
    if semantic is None or semantic[0] not in aspects:
        return None
    return semantic


def _validate_public_aspects(aspects: Sequence[str]) -> tuple[str, ...]:
    if isinstance(aspects, (str, bytes)):
        raise TypeError("public_aspects must be a sequence of aspect names")
    result: list[str] = []
    for aspect in aspects:
        if not isinstance(aspect, str) or not aspect.strip():
            raise PublicControlError("public aspect names must be non-empty strings")
        normalized = aspect.strip()
        if normalized not in ASPECT_QUERY_HINTS:
            raise PublicControlError(
                f"public aspect {normalized!r} is not in the known public vocabulary"
            )
        if normalized not in result:
            result.append(normalized)
    return tuple(result)


def _aspect_mentioned_in_content(content: str, public_aspects: Sequence[str]) -> str | None:
    matches: list[str] = []
    for aspect in public_aspects:
        if any(
            _hint_pattern(hint).search(content) is not None
            for hint in ASPECT_QUERY_HINTS[aspect]
        ):
            matches.append(aspect)
    return matches[0] if len(matches) == 1 else None


def _option_aspect(option_id: str, public_aspects: Sequence[str]) -> str | None:
    aspect = aspect_from_option_id(option_id)
    return aspect if aspect in public_aspects else None


@dataclass(frozen=True)
class PublicAspectState:
    """Public evidence ledger for one explicitly named aspect."""

    aspect: str
    status: PublicAspectStatus = PublicAspectStatus.OPEN
    visible_option_ids: frozenset[str] = frozenset()
    normal_search_seen: bool = False
    search_attempts: int = 0
    search_fallbacks: int = 0
    search_signatures: tuple[str, ...] = ()
    last_search_signature: str | None = None
    submitted_answer: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.aspect, str) or not self.aspect.strip():
            raise PublicControlError("aspect must be a non-empty string")
        if self.aspect not in ASPECT_QUERY_HINTS:
            raise PublicControlError(f"unknown public aspect: {self.aspect!r}")
        if not isinstance(self.status, PublicAspectStatus):
            raise TypeError("status must be PublicAspectStatus")
        if not isinstance(self.visible_option_ids, frozenset):
            raise TypeError("visible_option_ids must be a frozenset")
        if any(
            not isinstance(option_id, str) or OPTION_ID.fullmatch(option_id) is None
            for option_id in self.visible_option_ids
        ):
            raise PublicControlError("visible_option_ids must contain official option IDs")
        if (
            isinstance(self.search_attempts, bool)
            or not isinstance(self.search_attempts, int)
            or self.search_attempts < 0
        ):
            raise PublicControlError("search_attempts must be a non-negative integer")
        if (
            isinstance(self.search_fallbacks, bool)
            or not isinstance(self.search_fallbacks, int)
            or self.search_fallbacks < 0
        ):
            raise PublicControlError("search_fallbacks must be a non-negative integer")
        if any(not isinstance(signature, str) for signature in self.search_signatures):
            raise PublicControlError("search_signatures must contain strings")


@dataclass(frozen=True)
class PublicControlState:
    """Public-only control state.

    No field here represents hidden preferences, reward snapshots, correct
    answers, or task labels not named in the initial public message.
    """

    aspects: tuple[PublicAspectState, ...]
    current_aspect: str | None = None
    recovery_mode: RecoveryMode = RecoveryMode.NONE
    no_progress_threshold: int = 4
    consecutive_no_progress: int = 0
    max_consecutive_no_progress: int = 0
    last_action_signature: str | None = None
    last_observation_signature: str | None = None
    last_observation_kind: PublicObservationKind | None = None
    seen_observation_signatures: frozenset[str] = frozenset()
    submitted_answer_ids: tuple[str, ...] = ()
    episode_done: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.aspects, tuple):
            raise TypeError("aspects must be a tuple")
        if any(not isinstance(item, PublicAspectState) for item in self.aspects):
            raise TypeError("aspects must contain PublicAspectState values")
        names = [item.aspect for item in self.aspects]
        if len(names) != len(set(names)):
            raise PublicControlError("public aspects must be unique")
        if self.current_aspect is not None and self.current_aspect not in names:
            raise PublicControlError("current_aspect must be one of the public aspects")
        if not isinstance(self.recovery_mode, RecoveryMode):
            raise TypeError("recovery_mode must be RecoveryMode")
        if (
            isinstance(self.no_progress_threshold, bool)
            or not isinstance(self.no_progress_threshold, int)
            or self.no_progress_threshold < 1
        ):
            raise PublicControlError("no_progress_threshold must be an integer >= 1")
        for name, value in (
            ("consecutive_no_progress", self.consecutive_no_progress),
            ("max_consecutive_no_progress", self.max_consecutive_no_progress),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PublicControlError(f"{name} must be a non-negative integer")
        if any(
            not isinstance(option_id, str) or OPTION_ID.fullmatch(option_id) is None
            for option_id in self.submitted_answer_ids
        ):
            raise PublicControlError("submitted_answer_ids must contain official option IDs")

    @property
    def public_aspects(self) -> tuple[str, ...]:
        return tuple(item.aspect for item in self.aspects)

    @property
    def open_aspects(self) -> tuple[str, ...]:
        return tuple(
            item.aspect
            for item in self.aspects
            if item.status is PublicAspectStatus.OPEN
        )

    @property
    def current(self) -> PublicAspectState | None:
        if self.current_aspect is None:
            return None
        return next(
            (item for item in self.aspects if item.aspect == self.current_aspect),
            None,
        )


    @property
    def phase(self) -> RecoveryMode:
        """Return the canonical finite-state-machine phase.

        recovery_mode remains the compatibility field. In particular, the old
        reducer exposed NONE before the first recovery event and BLOCKED
        immediately after the second fallback. The phase guard needs the more
        precise interpretation of those values without making old rollout
        metrics or callers change shape.
        """

        if self.episode_done:
            return RecoveryMode.SWITCH_ASPECT_REQUIRED
        if self.recovery_mode is RecoveryMode.NONE and self.current is not None:
            return RecoveryMode.ELICITING
        if (
            self.recovery_mode is RecoveryMode.BLOCKED
            and self.current is not None
            and self.current.status is PublicAspectStatus.BLOCKED
        ):
            return RecoveryMode.SWITCH_ASPECT_REQUIRED
        return self.recovery_mode

    @property
    def answered_aspects(self) -> tuple[str, ...]:
        """Publicly answered aspects, kept separate from blocked aspects."""

        return tuple(
            item.aspect
            for item in self.aspects
            if item.status is PublicAspectStatus.ANSWERED
        )

    @property
    def blocked_aspects(self) -> tuple[str, ...]:
        """Publicly blocked aspects, never counted as answered."""

        return tuple(
            item.aspect
            for item in self.aspects
            if item.status is PublicAspectStatus.BLOCKED
        )

    @property
    def answered_count(self) -> int:
        return len(self.answered_aspects)

    @property
    def blocked_count(self) -> int:
        return len(self.blocked_aspects)

    @property
    def all_aspects_terminal(self) -> bool:
        return bool(self.aspects) and not self.open_aspects


@dataclass(frozen=True)
class PublicControlEvent:
    """One reducer input composed only of public action and public feedback."""

    action: UserBenchAction | None = None
    observation: PublicObservation | None = None

    def __post_init__(self) -> None:
        if self.action is not None and not isinstance(self.action, UserBenchAction):
            raise TypeError("event action must be a UserBenchAction or None")
        if self.observation is not None and not isinstance(
            self.observation, PublicObservation
        ):
            raise TypeError("event observation must be a PublicObservation or None")


def new_public_control_state(
    initial_user_message: str,
    *,
    no_progress_threshold: int = 4,
) -> PublicControlState:
    """Create a state from the public initial user message only."""

    public_aspects = extract_public_aspects(initial_user_message)
    validated_threshold = _validate_threshold(no_progress_threshold)
    return PublicControlState(
        aspects=tuple(PublicAspectState(aspect) for aspect in public_aspects),
        current_aspect=public_aspects[0] if public_aspects else None,
        no_progress_threshold=validated_threshold,
    )


def _validate_threshold(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PublicControlError("no_progress_threshold must be an integer >= 1")
    return value


def _replace_aspect(
    state: PublicControlState,
    updated: PublicAspectState,
) -> PublicControlState:
    return replace(
        state,
        aspects=tuple(
            updated if item.aspect == updated.aspect else item for item in state.aspects
        ),
    )


def _progress_reset(state: PublicControlState) -> PublicControlState:
    return replace(state, consecutive_no_progress=0)


def _progress_increment(state: PublicControlState) -> PublicControlState:
    current = state.consecutive_no_progress + 1
    # Only elicitation/no-op turns can trigger the forced-search phase. Once a
    # normal list or a fallback has established a stricter phase, invalid
    # calls must remain in that phase instead of silently bypassing it.
    may_force_search = state.phase in {RecoveryMode.NONE, RecoveryMode.ELICITING}
    return replace(
        state,
        consecutive_no_progress=current,
        max_consecutive_no_progress=max(state.max_consecutive_no_progress, current),
        recovery_mode=(
            RecoveryMode.SEARCH_REQUIRED
            if (
                may_force_search
                and current >= state.no_progress_threshold
                and state.current_aspect is not None
            )
            else state.recovery_mode
        ),
    )


def _target_aspect(
    state: PublicControlState,
    action: UserBenchAction,
) -> str | None:
    public_aspects = state.public_aspects
    if action.choice is ActionChoice.ANSWER:
        submitted = [item.strip() for item in action.content.split(",")]
        answer_aspects = {
            _option_aspect(item, public_aspects)
            for item in submitted
            if item
        }
        answer_aspects.discard(None)
        if len(answer_aspects) == 1:
            return next(iter(answer_aspects))
    mentioned = _aspect_mentioned_in_content(action.content, public_aspects)
    if mentioned is not None:
        return mentioned
    return state.current_aspect


def normalize_public_query(value: str) -> str:
    """Normalize a search query using an explainable public-only rule.

    Case, whitespace, punctuation, and token order are intentionally ignored
    for retry comparison. No embedding, model, task label, or hidden state is
    consulted.
    """

    if not isinstance(value, str):
        raise TypeError("query must be a string")
    tokens = re.findall(r"[a-z0-9]+", value.casefold())
    # Pre-step versions stored the full action signature ("search:...").
    # Strip that compatibility prefix before comparing against new query-only
    # signatures.
    if tokens and tokens[0] == "search":
        tokens = tokens[1:]
    return " ".join(tokens)


def public_query_signature(action: UserBenchAction) -> str | None:
    """Return the normalized query text for a search action."""

    if not isinstance(action, UserBenchAction):
        raise TypeError("action must be a UserBenchAction")
    if action.choice is not ActionChoice.SEARCH:
        return None
    return normalize_public_query(action.content)


def is_substantive_query_change(previous_query: str, candidate_query: str) -> bool:
    """Return whether a retry changes public query content materially.

    A changed token multiset is the first, deliberately conservative version of
    the policy. Thus punctuation/case/whitespace and mere word reordering are
    rejected, while adding or replacing a meaningful token is accepted.
    """

    previous = normalize_public_query(previous_query)
    candidate = normalize_public_query(candidate_query)
    if not previous or not candidate or previous == candidate:
        return False
    return Counter(previous.split()) != Counter(candidate.split())


def _answer_option_ids(action: UserBenchAction) -> list[str]:
    return [item.strip() for item in action.content.split(",") if item.strip()]


def validate_public_action(
    state: PublicControlState,
    action: UserBenchAction,
) -> str | None:
    """Return a public-only rejection reason, or None when allowed.

    This is the pre-environment phase guard. It intentionally knows only
    public aspect status, visible option IDs, public query signatures, and the
    recovery phase.
    """

    if not isinstance(state, PublicControlState):
        raise TypeError("state must be PublicControlState")
    if not isinstance(action, UserBenchAction):
        raise TypeError("action must be a UserBenchAction")
    current = state.current
    target = _target_aspect(state, action)
    target_state = next(
        (item for item in state.aspects if item.aspect == target),
        None,
    )
    if target_state is not None and target_state.status is not PublicAspectStatus.OPEN:
        return f"public aspect {target_state.aspect!r} is terminal"
    if state.episode_done:
        return "public control episode is complete"

    phase = state.phase
    if phase is RecoveryMode.SWITCH_ASPECT_REQUIRED:
        return "advance to the next public aspect before another tool call"
    if phase is RecoveryMode.ANSWERED:
        return "the public aspect is already answered"
    if current is None:
        if action.choice in {ActionChoice.SEARCH, ActionChoice.ANSWER}:
            return "no publicly named aspect is available for this call"
        return None

    if (
        action.choice in {ActionChoice.SEARCH, ActionChoice.ANSWER}
        and target != current.aspect
    ):
        return "search or answer must target the current public aspect"

    if phase is RecoveryMode.SEARCH_REQUIRED:
        if action.choice is not ActionChoice.SEARCH:
            return "SEARCH_REQUIRED accepts choice=search only"
        if target != current.aspect:
            return "SEARCH_REQUIRED accepts search for the current public aspect only"
        return None

    if phase is RecoveryMode.SEARCH_RETRY_REQUIRED:
        if action.choice is not ActionChoice.SEARCH:
            return "SEARCH_RETRY_REQUIRED accepts one revised search only"
        if target != current.aspect:
            return "SEARCH_RETRY_REQUIRED accepts the current public aspect only"
        candidate = public_query_signature(action)
        previous = current.last_search_signature
        if candidate is None or previous is None:
            return "SEARCH_RETRY_REQUIRED is missing the previous public query"
        if candidate in current.search_signatures:
            return "search query was already attempted for this public aspect"
        if not is_substantive_query_change(previous, candidate):
            return "retry query must materially change the previous public query"
        return None

    if phase is RecoveryMode.ANSWER_REQUIRED:
        if action.choice is not ActionChoice.ANSWER:
            return "ANSWER_REQUIRED accepts choice=answer only"
        submitted = _answer_option_ids(action)
        if len(submitted) != 1 or OPTION_ID.fullmatch(submitted[0]) is None:
            return "answer must contain exactly one official option ID"
        if submitted[0] not in current.visible_option_ids:
            return "answer ID is not visible in the current candidate list"
        if _option_aspect(submitted[0], state.public_aspects) != current.aspect:
            return "answer ID belongs to a different public aspect"
        return None

    if action.choice is ActionChoice.ANSWER:
        submitted = _answer_option_ids(action)
        if len(submitted) != 1 or OPTION_ID.fullmatch(submitted[0]) is None:
            return "answer must contain exactly one official option ID"
        if submitted[0] not in current.visible_option_ids:
            return "answer ID is not visible in the current candidate list"
        if _option_aspect(submitted[0], state.public_aspects) != current.aspect:
            return "answer ID belongs to a different public aspect"
    return None


def note_public_non_progress(state: PublicControlState) -> PublicControlState:
    """Record a rejected/no-progress public event without touching the env."""

    if not isinstance(state, PublicControlState):
        raise TypeError("state must be PublicControlState")
    if state.episode_done:
        return state
    # A rejected call in answer/search recovery cannot turn into a different
    # phase. Elicitation is the only phase where the configurable streak can
    # force the next action to be a search.
    if state.phase not in {RecoveryMode.NONE, RecoveryMode.ELICITING}:
        return state
    return _progress_increment(state)


def _record_submitted_answers(
    state: PublicControlState,
    action: UserBenchAction,
) -> PublicControlState:
    if action.choice is not ActionChoice.ANSWER:
        return state
    ids = [item.strip() for item in action.content.split(",") if item.strip()]
    public_ids = [item for item in ids if OPTION_ID.fullmatch(item) is not None]
    if not public_ids:
        return state
    merged = list(state.submitted_answer_ids)
    for option_id in public_ids:
        if option_id not in merged:
            merged.append(option_id)
    return replace(state, submitted_answer_ids=tuple(merged))


def _all_public_aspects_terminal(state: PublicControlState) -> bool:
    return bool(state.aspects) and not state.open_aspects


def _apply_search_action(
    state: PublicControlState,
    action: UserBenchAction,
    observation: PublicObservation | None,
) -> PublicControlState:
    target = _target_aspect(state, action)
    if target is None:
        return _progress_increment(state) if observation is None else state
    aspect_state = state.current
    if aspect_state is None or aspect_state.aspect != target:
        aspect_state = next(
            (item for item in state.aspects if item.aspect == target),
            None,
        )
    if aspect_state is None:
        return state
    if aspect_state.status is not PublicAspectStatus.OPEN:
        return state
    signature = public_query_signature(action)
    if signature is None:
        return state

    # The retry phase is deliberately one-shot. A direct reducer caller gets
    # the same protection as the adapter guard: punctuation/case/word-order
    # variants cannot consume a second fallback budget.
    if state.phase is RecoveryMode.SEARCH_RETRY_REQUIRED:
        previous = aspect_state.last_search_signature
        if (
            target != state.current_aspect
            or previous is None
            or signature in aspect_state.search_signatures
            or not is_substantive_query_change(previous, signature)
        ):
            return state

    signatures = list(aspect_state.search_signatures)
    if signature not in signatures:
        signatures.append(signature)
    updated = replace(
        aspect_state,
        search_attempts=min(2, aspect_state.search_attempts + 1),
        search_signatures=tuple(signatures),
        last_search_signature=signature,
    )
    if observation is not None:
        public_ids = frozenset(
            option_id
            for option_id in observation.visible_option_ids
            if _option_aspect(option_id, state.public_aspects) == target
        )
        if public_ids:
            updated = replace(
                updated,
                visible_option_ids=updated.visible_option_ids | public_ids,
            )
        if observation.kind is PublicObservationKind.SEARCH_NORMAL and public_ids:
            updated = replace(
                updated,
                normal_search_seen=True,
                visible_option_ids=public_ids,
            )
            state = _replace_aspect(state, updated)
            return replace(
                _progress_reset(state),
                recovery_mode=RecoveryMode.ANSWER_REQUIRED,
            )
        if (
            observation.kind
            in {
                PublicObservationKind.SEARCH_FALLBACK,
                PublicObservationKind.SEARCH_EMPTY,
            }
            or (
                observation.kind is PublicObservationKind.SEARCH_NORMAL
                and not public_ids
            )
        ):
            # Empty results are treated as a public retryable search failure,
            # even when the backend did not emit its fallback phrase.
            fallback_count = updated.search_fallbacks + 1
            if fallback_count >= 2:
                updated = replace(
                    updated,
                    search_fallbacks=fallback_count,
                    status=PublicAspectStatus.BLOCKED,
                )
                state = _replace_aspect(state, updated)
                # BLOCKED is retained in recovery_mode for old metrics. The
                # phase property exposes the required SWITCH_ASPECT_REQUIRED.
                return replace(
                    state,
                    recovery_mode=RecoveryMode.BLOCKED,
                    episode_done=_all_public_aspects_terminal(state),
                )
            updated = replace(
                updated,
                search_fallbacks=fallback_count,
            )
            state = _replace_aspect(state, updated)
            return replace(
                state,
                recovery_mode=RecoveryMode.SEARCH_RETRY_REQUIRED,
            )
    state = _replace_aspect(state, updated)
    return _progress_increment(state) if observation is None else state


def _apply_answer_action(
    state: PublicControlState,
    action: UserBenchAction,
) -> PublicControlState:
    state = _record_submitted_answers(state, action)
    target = _target_aspect(state, action)
    if target is None:
        return _progress_increment(state)
    aspect_state = next(
        (item for item in state.aspects if item.aspect == target),
        None,
    )
    if aspect_state is None:
        return _progress_increment(state)
    if aspect_state.status is not PublicAspectStatus.OPEN:
        return _progress_increment(state)
    submitted = [item.strip() for item in action.content.split(",") if item.strip()]
    if len(submitted) != 1 or submitted[0] not in aspect_state.visible_option_ids:
        return _progress_increment(state)
    updated = replace(
        aspect_state,
        status=PublicAspectStatus.ANSWERED,
        submitted_answer=submitted[0],
    )
    next_state = replace(
        _replace_aspect(_progress_reset(state), updated),
        current_aspect=target,
        recovery_mode=RecoveryMode.SWITCH_ASPECT_REQUIRED,
    )
    if next_state.all_aspects_terminal:
        next_state = replace(next_state, episode_done=True)
    return next_state


def reduce_public_control_state(
    state: PublicControlState,
    event: PublicControlEvent,
) -> PublicControlState:
    """Apply one public event without consulting hidden reward state."""

    if not isinstance(state, PublicControlState):
        raise TypeError("state must be PublicControlState")
    if not isinstance(event, PublicControlEvent):
        raise TypeError("event must be PublicControlEvent")
    if state.episode_done or (event.action is None and event.observation is None):
        return state

    next_state = state
    action = event.action
    observation = event.observation
    repeated_action = False
    if action is not None:
        action_signature = public_action_signature(action)
        repeated_action = action_signature == state.last_action_signature
        guard_reason = validate_public_action(next_state, action)
        if (
            guard_reason is not None
            and next_state.phase
            not in {RecoveryMode.NONE, RecoveryMode.ELICITING}
        ):
            # Direct reducer callers get the same fail-closed behavior as the
            # adapter. The actor-owned submitted ID is still retained for
            # audit, but no environment transition or terminal status changes.
            return _record_submitted_answers(
                replace(
                    next_state,
                    last_action_signature=public_action_signature(action),
                ),
                action,
            )
        next_state = replace(
            next_state,
            last_action_signature=public_action_signature(action),
        )
        next_state = _record_submitted_answers(next_state, action)
        if action.choice is ActionChoice.SEARCH:
            next_state = _apply_search_action(next_state, action, observation)
        elif action.choice is ActionChoice.ANSWER:
            next_state = _apply_answer_action(next_state, action)
        elif observation is None:
            next_state = _progress_increment(next_state)

    if observation is not None:
        next_state = replace(
            next_state,
            last_observation_signature=observation.signature,
            last_observation_kind=observation.kind,
            seen_observation_signatures=(
                next_state.seen_observation_signatures | {observation.signature}
            ),
        )
        if action is None and observation.is_normal_search:
            next_state = _progress_reset(next_state)
        elif (
            action is not None
            and action.choice is ActionChoice.ACTION
            and observation.kind is PublicObservationKind.TEXT
        ):
            if (
                repeated_action
                or observation.signature in state.seen_observation_signatures
            ):
                next_state = _progress_increment(next_state)
            else:
                next_state = _progress_reset(next_state)
        elif observation.is_fallback or observation.kind is PublicObservationKind.SEARCH_EMPTY:
            if action is None:
                next_state = _progress_increment(next_state)
    return next_state


def reduce_public_feedback(
    state: PublicControlState,
    action: UserBenchAction,
    feedback: str,
) -> PublicControlState:
    """Convenience reducer entry point for one actor call and public feedback."""

    if not isinstance(action, UserBenchAction):
        raise TypeError("action must be a UserBenchAction")
    observation = classify_public_observation(feedback, choice=action.choice)
    return reduce_public_control_state(
        state,
        PublicControlEvent(action=action, observation=observation),
    )


def advance_public_aspect(state: PublicControlState) -> PublicControlState:
    """Advance after a public terminal signal; no hidden status is consulted."""

    if not isinstance(state, PublicControlState):
        raise TypeError("state must be PublicControlState")
    if state.episode_done:
        if state.current is not None and state.current.status in {
            PublicAspectStatus.ANSWERED,
            PublicAspectStatus.BLOCKED,
        }:
            return replace(state, current_aspect=None)
        return state
    current = state.current
    if current is None or current.status not in {
        PublicAspectStatus.ANSWERED,
        PublicAspectStatus.BLOCKED,
    }:
        raise PublicControlError(
            "public aspect can advance only after an answered or blocked status"
        )
    current_index = next(
        index
        for index, item in enumerate(state.aspects)
        if item.aspect == current.aspect
    )
    next_aspect = next(
        (
            item.aspect
            for item in state.aspects[current_index + 1 :]
            if item.status is PublicAspectStatus.OPEN
        ),
        None,
    )
    if next_aspect is None:
        return replace(
            state,
            current_aspect=None,
            recovery_mode=RecoveryMode.NONE,
            episode_done=True,
        )
    return replace(
        state,
        current_aspect=next_aspect,
        recovery_mode=RecoveryMode.NONE,
        consecutive_no_progress=0,
    )


def _render_phase_label(state: PublicControlState) -> str:
    """Return the stable public name for the current finite-state phase."""

    phase = state.phase
    # ``ELICITING`` is an identity-compatible alias of ``NONE``.  The
    # renderer must use the state-machine name rather than Enum.name (which
    # would expose the legacy ``NONE`` spelling to the actor).
    if phase is RecoveryMode.NONE and not state.episode_done:
        return "ELICITING"
    return phase.value.upper()


def _render_allowed_tool_calls(state: PublicControlState) -> str:
    """Return a public-only, human-readable allow-list for the next call."""

    if state.episode_done:
        return "none"
    current = state.current
    if current is None:
        # With no explicitly named aspect, only a public clarification/action
        # is safe; search and answer require a concrete public target.
        return "action"
    phase = state.phase
    if phase is RecoveryMode.SEARCH_REQUIRED:
        return "search"
    if phase is RecoveryMode.SEARCH_RETRY_REQUIRED:
        return "search (revised query)"
    if phase is RecoveryMode.ANSWER_REQUIRED:
        return "answer (one visible option ID)"
    if phase is RecoveryMode.SWITCH_ASPECT_REQUIRED:
        return "none for this aspect; next public aspect only"
    if phase is RecoveryMode.ANSWERED:
        return "none for this aspect"
    return "action, search, answer (visible ID only)"


def _render_constraint(state: PublicControlState) -> str | None:
    """Return the phase-specific public constraint, if one is needed."""

    if state.episode_done:
        return "Do not issue another tool call or fabricate completion."
    current = state.current
    phase = state.phase
    if phase is RecoveryMode.SEARCH_REQUIRED:
        return "Do not call action; search using information already visible in the conversation."
    if phase is RecoveryMode.SEARCH_RETRY_REQUIRED:
        return "Rewrite the query materially once; do not repeat the previous query."
    if phase is RecoveryMode.ANSWER_REQUIRED:
        return "Answer with exactly one ID from the visible option list; do not search or call action."
    if phase is RecoveryMode.SWITCH_ASPECT_REQUIRED:
        if current is not None and current.status is PublicAspectStatus.BLOCKED:
            return "The current aspect is blocked; do not search or answer it, and do not fabricate an answer."
        return "The current aspect is terminal; do not call it again; continue with the next public aspect."
    if phase is RecoveryMode.ANSWERED:
        return "This aspect is answered; do not call a tool for it again."
    if phase is RecoveryMode.NONE and current is not None:
        return (
            "Ask only for a missing preference visible in the conversation; "
            "do not repeat an answered or declined preference."
        )
    return None


def render_actor_control_info(state: PublicControlState) -> str:
    """Render stable actor feedback using public evidence only.

    The first line intentionally has a fixed field order so rollout logs and
    snapshot tests can compare it byte-for-byte.  The optional constraint line
    is phase-specific and never contains reward, hidden preference, or
    correctness fields.
    """

    if not isinstance(state, PublicControlState):
        raise TypeError("state must be PublicControlState")
    current = state.current
    aspect = current.aspect if current is not None else "none"
    fallback_count = current.search_fallbacks if current is not None else 0
    visible_ids = (
        ",".join(sorted(current.visible_option_ids))
        if current is not None and current.visible_option_ids
        else "none"
    )
    summary = (
        "Public control | "
        f"Current public aspect: {aspect} | "
        f"Current control state: {_render_phase_label(state)} | "
        f"Current aspect fallback count: {fallback_count} | "
        f"Current visible option IDs: {visible_ids} | "
        f"Allowed next tool calls: {_render_allowed_tool_calls(state)}"
    )
    constraint = _render_constraint(state)
    return summary if constraint is None else f"{summary}\nConstraint: {constraint}"


__all__ = [
    "PUBLIC_SEARCH_FALLBACK_MARKERS",
    "PublicAspectState",
    "PublicAspectStatus",
    "PublicControlError",
    "PublicControlEvent",
    "PublicControlState",
    "PublicObservation",
    "PublicObservationKind",
    "RecoveryMode",
    "advance_public_aspect",
    "classify_public_observation",
    "extract_public_aspects",
    "new_public_control_state",
    "normalize_public_query",
    "public_action_signature",
    "public_query_signature",
    "public_search_signature",
    "public_semantic_signature",
    "is_substantive_query_change",
    "note_public_non_progress",
    "reduce_public_control_state",
    "reduce_public_feedback",
    "render_actor_control_info",
    "validate_public_action",
]
