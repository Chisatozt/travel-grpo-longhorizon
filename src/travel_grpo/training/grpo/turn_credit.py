"""Causal turn evidence and bounded GRPO advantage reshaping.

The terminal UserBench reward remains the sole trajectory objective.  This
module assigns an independent, bounded *relative* importance signal to the
assistant turns in one trajectory.  It never changes ``rm_scores`` and it
never renders hidden reward evidence to the Actor.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any


TURN_CREDIT_VERSION = "causal-turn-credit-v1"
TURN_CREDIT_MODES = frozenset({"off", "shadow", "train"})


class TurnCreditError(ValueError):
    """Raised when a turn-credit trace or token alignment is invalid."""


@dataclass
class TurnEvent:
    """One assistant generation plus its public tool result.

    Fields whose names start with ``reward_`` are reward-only aggregates.  No
    hidden ID or value is retained, so the record can be safely transported to
    the trainer.  It must still never be included in Actor-visible feedback.
    """

    turn_index: int
    public_aspect: str | None
    phase_before: str
    phase_after: str
    choice: str | None = None
    tool_name: str | None = None

    field_resolved: bool = False
    explicit_no_preference: bool = False
    normal_candidates_observed: bool = False
    answer_id_visible: bool = False
    first_fallback: bool = False
    second_fallback: bool = False
    successful_rewritten_retry: bool = False
    switched_aspect: bool = False

    guard_rejected: bool = False
    no_progress_action: bool = False
    semantic_repeat: bool = False
    exact_query_repeat: bool = False
    wrong_aspect: bool = False
    invisible_answer_id: bool = False
    malformed_tool_call: bool = False
    no_tool_output: bool = False
    infrastructure_failure: bool = False

    reward_new_preference_count: int = 0
    reward_correct_answer: bool = False
    reward_wrong_answer: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.turn_index, bool) or self.turn_index < 0:
            raise TurnCreditError("turn_index must be a non-negative integer")
        if self.reward_new_preference_count < 0:
            raise TurnCreditError("reward_new_preference_count must be non-negative")

    def to_training_record(self) -> dict[str, Any]:
        """Return a hidden-ID-free trainer record."""

        return asdict(self)


@dataclass(frozen=True)
class AspectCausalTrace:
    aspect: str
    preference_turn_indices: tuple[int, ...] = ()
    successful_search_turn_index: int | None = None
    accepted_answer_turn_index: int | None = None
    correctly_completed: bool = False
    blocked: bool = False


@dataclass(frozen=True)
class TurnCreditConfig:
    preference_chain: float = 0.20
    successful_search: float = 0.35
    correct_answer: float = 0.45
    partial_preference_field: float = 0.05
    partial_preference_cap: float = 0.10
    partial_normal_search: float = 0.20
    partial_visible_answer: float = 0.05
    no_progress_action: float = -0.10
    semantic_repeat: float = -0.10
    guard_rejection_generic: float = -0.15
    exact_query_repeat: float = -0.20
    wrong_aspect: float = -0.20
    invisible_answer_id: float = -0.20
    wrong_answer: float = -0.15
    malformed_tool_call: float = -0.15
    no_tool_output: float = -0.15
    evidence_clip: float = 0.50
    mix_lambda: float = 0.50
    multiplier_band: float = 0.20
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        for name in (
            "preference_chain",
            "successful_search",
            "correct_answer",
            "partial_preference_field",
            "partial_preference_cap",
            "partial_normal_search",
            "partial_visible_answer",
            "evidence_clip",
            "epsilon",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise TurnCreditError(f"{name} must be non-negative and finite")
        if not 0.0 <= self.mix_lambda <= 1.0:
            raise TurnCreditError("mix_lambda must be in [0, 1]")
        if not 0.0 <= self.multiplier_band < 1.0:
            raise TurnCreditError("multiplier_band must be in [0, 1)")
        if self.epsilon <= 0.0:
            raise TurnCreditError("epsilon must be positive")
        for name in (
            "no_progress_action",
            "semantic_repeat",
            "guard_rejection_generic",
            "exact_query_repeat",
            "wrong_aspect",
            "invisible_answer_id",
            "wrong_answer",
            "malformed_tool_call",
            "no_tool_output",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value > 0:
                raise TurnCreditError(f"{name} must be non-positive and finite")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "TurnCreditConfig":
        if value is None:
            return cls()
        completion = value.get("completion_allocation", {})
        partial = value.get("partial_progress", {})
        errors = value.get("error_adjustment", {})
        if not all(isinstance(item, Mapping) for item in (completion, partial, errors)):
            raise TurnCreditError("turn-credit component configs must be mappings")
        return cls(
            preference_chain=float(completion.get("preference_chain", 0.20)),
            successful_search=float(completion.get("successful_search", 0.35)),
            correct_answer=float(completion.get("correct_answer", 0.45)),
            partial_preference_field=float(partial.get("preference_field_resolved", 0.05)),
            partial_preference_cap=float(partial.get("preference_chain_cap", 0.10)),
            partial_normal_search=float(partial.get("normal_candidates_observed", 0.20)),
            partial_visible_answer=float(partial.get("visible_answer_call", 0.05)),
            no_progress_action=float(errors.get("no_progress_action", -0.10)),
            semantic_repeat=float(errors.get("semantic_repeat", -0.10)),
            guard_rejection_generic=float(errors.get("guard_rejection_generic", -0.15)),
            exact_query_repeat=float(errors.get("exact_query_repeat", -0.20)),
            wrong_aspect=float(errors.get("wrong_aspect", -0.20)),
            invisible_answer_id=float(errors.get("invisible_answer_id", -0.20)),
            wrong_answer=float(errors.get("wrong_answer", -0.15)),
            malformed_tool_call=float(errors.get("malformed_tool_call", -0.15)),
            no_tool_output=float(errors.get("no_tool_output", -0.15)),
            evidence_clip=float(value.get("evidence_clip", 0.50)),
            mix_lambda=float(value.get("mix_lambda", 0.50)),
            multiplier_band=float(value.get("multiplier_band", 0.20)),
            epsilon=float(value.get("epsilon", 1.0e-6)),
        )


@dataclass(frozen=True)
class TurnCreditTrace:
    version: str
    reward_valid: bool
    events: tuple[TurnEvent, ...]
    aspects: tuple[AspectCausalTrace, ...]
    evidence: tuple[float, ...]
    _config: TurnCreditConfig = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.version != TURN_CREDIT_VERSION:
            raise TurnCreditError(f"unsupported turn-credit version: {self.version!r}")
        if len(self.events) != len(self.evidence):
            raise TurnCreditError("turn events and evidence must have equal length")
        if any(not math.isfinite(value) for value in self.evidence):
            raise TurnCreditError("turn evidence must be finite")

    def to_extra_field(self, *, mode: str) -> dict[str, Any]:
        normalized_mode = validate_turn_credit_mode(mode)
        return {
            "version": self.version,
            "mode": normalized_mode,
            "reward_valid": self.reward_valid,
            "turn_count": len(self.events),
            "evidence": list(self.evidence),
        }

    def metrics(self, *, mode: str) -> dict[str, Any]:
        values = self.evidence
        return {
            "turn_credit_mode": validate_turn_credit_mode(mode),
            "turn_credit_version": self.version,
            "turn_credit_turn_count": len(values),
            "turn_credit_positive_count": sum(value > 0 for value in values),
            "turn_credit_negative_count": sum(value < 0 for value in values),
            "turn_credit_zero_count": sum(value == 0 for value in values),
            "turn_credit_min": min(values, default=0.0),
            "turn_credit_max": max(values, default=0.0),
            "turn_credit_mean": sum(values) / len(values) if values else 0.0,
        }


def validate_turn_credit_mode(value: Any) -> str:
    if not isinstance(value, str):
        raise TurnCreditError("turn-credit mode must be a string")
    mode = value.strip().casefold()
    if mode not in TURN_CREDIT_MODES:
        raise TurnCreditError(
            f"turn-credit mode must be one of {sorted(TURN_CREDIT_MODES)}"
        )
    return mode


def build_aspect_causal_traces(
    events: Sequence[TurnEvent],
    aspects: Sequence[str],
    *,
    blocked_aspects: Sequence[str] = (),
) -> tuple[AspectCausalTrace, ...]:
    """Link each correct answer to its visible search and preference chain."""

    blocked = set(blocked_aspects)
    traces: list[AspectCausalTrace] = []
    for aspect in aspects:
        answer = next(
            (
                event
                for event in reversed(events)
                if event.public_aspect == aspect
                and event.choice == "answer"
                and event.answer_id_visible
                and event.reward_correct_answer
            ),
            None,
        )
        answer_index = answer.turn_index if answer is not None else None
        search = next(
            (
                event
                for event in reversed(events)
                if event.public_aspect == aspect
                and event.choice == "search"
                and event.normal_candidates_observed
                and (answer_index is None or event.turn_index < answer_index)
            ),
            None,
        )
        search_index = search.turn_index if search is not None else None
        cutoff = search_index if search_index is not None else answer_index
        preference_indices = tuple(
            event.turn_index
            for event in events
            if event.public_aspect == aspect
            and event.choice == "action"
            and event.field_resolved
            and (cutoff is None or event.turn_index < cutoff)
        )
        traces.append(
            AspectCausalTrace(
                aspect=aspect,
                preference_turn_indices=preference_indices,
                successful_search_turn_index=search_index,
                accepted_answer_turn_index=answer_index,
                correctly_completed=answer is not None and search is not None,
                blocked=aspect in blocked,
            )
        )
    return tuple(traces)


def primary_violation_penalty(event: TurnEvent, config: TurnCreditConfig) -> float:
    """Return one root-cause penalty, avoiding guard/repeat double counting."""

    specific: list[float] = []
    if event.exact_query_repeat:
        specific.append(config.exact_query_repeat)
    if event.wrong_aspect:
        specific.append(config.wrong_aspect)
    if event.invisible_answer_id:
        specific.append(config.invisible_answer_id)
    if event.reward_wrong_answer:
        specific.append(config.wrong_answer)
    if event.malformed_tool_call:
        specific.append(config.malformed_tool_call)
    if event.no_tool_output:
        specific.append(config.no_tool_output)
    if event.semantic_repeat:
        specific.append(config.semantic_repeat)
    if event.no_progress_action:
        specific.append(config.no_progress_action)
    if specific:
        return min(specific)
    if event.guard_rejected:
        return config.guard_rejection_generic
    return 0.0


def allocate_turn_evidence(
    events: Sequence[TurnEvent],
    traces: Sequence[AspectCausalTrace],
    *,
    reward_valid: bool,
    config: TurnCreditConfig | None = None,
) -> tuple[float, ...]:
    """Allocate completion chains, partial progress, and root-cause errors."""

    selected = config or TurnCreditConfig()
    if not reward_valid:
        return tuple(0.0 for _ in events)
    for expected, event in enumerate(events):
        if event.turn_index != expected:
            raise TurnCreditError("turn indices must be contiguous and ordered")
    evidence = [0.0 for _ in events]
    event_by_index = {event.turn_index: event for event in events}

    for trace in traces:
        if trace.correctly_completed:
            preferences = tuple(
                index for index in trace.preference_turn_indices if index in event_by_index
            )
            if preferences:
                share = selected.preference_chain / len(preferences)
                for index in preferences:
                    evidence[index] += share
            if trace.successful_search_turn_index is not None:
                evidence[trace.successful_search_turn_index] += selected.successful_search
            if trace.accepted_answer_turn_index is not None:
                evidence[trace.accepted_answer_turn_index] += selected.correct_answer
            continue

        preference_total = 0.0
        for event in events:
            if event.public_aspect != trace.aspect:
                continue
            if event.choice == "action" and event.field_resolved:
                increment = min(
                    selected.partial_preference_field,
                    max(0.0, selected.partial_preference_cap - preference_total),
                )
                evidence[event.turn_index] += increment
                preference_total += increment
            if event.normal_candidates_observed:
                # A recovered retry is the same successful search milestone,
                # not an additional reward source.
                evidence[event.turn_index] = max(
                    evidence[event.turn_index], selected.partial_normal_search
                )
            if (
                event.choice == "answer"
                and event.answer_id_visible
                and not event.reward_correct_answer
            ):
                evidence[event.turn_index] += selected.partial_visible_answer

    for event in events:
        if event.infrastructure_failure:
            evidence[event.turn_index] = 0.0
            continue
        evidence[event.turn_index] += primary_violation_penalty(event, selected)
        evidence[event.turn_index] = max(
            -selected.evidence_clip,
            min(selected.evidence_clip, evidence[event.turn_index]),
        )
    return tuple(float(value) for value in evidence)


def build_turn_credit_trace(
    events: Sequence[TurnEvent],
    aspects: Sequence[str],
    *,
    blocked_aspects: Sequence[str] = (),
    reward_valid: bool,
    config: TurnCreditConfig | None = None,
) -> TurnCreditTrace:
    selected = config or TurnCreditConfig()
    traces = build_aspect_causal_traces(
        events, aspects, blocked_aspects=blocked_aspects
    )
    evidence = allocate_turn_evidence(
        events, traces, reward_valid=reward_valid, config=selected
    )
    return TurnCreditTrace(
        version=TURN_CREDIT_VERSION,
        reward_valid=reward_valid,
        events=tuple(events),
        aspects=traces,
        evidence=evidence,
        _config=selected,
    )


def normalized_turn_evidence(
    evidence: Sequence[float], *, epsilon: float = 1.0e-6
) -> tuple[float, ...]:
    if not evidence:
        return ()
    values = [float(value) for value in evidence]
    if any(not math.isfinite(value) for value in values):
        raise TurnCreditError("turn evidence must be finite")
    if len(values) <= 1:
        return tuple(0.0 for _ in values)
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    std_value = math.sqrt(variance)
    if std_value < epsilon:
        return tuple(0.0 for _ in values)
    return tuple((value - mean_value) / (std_value + epsilon) for value in values)


def reshape_turn_advantages(
    sequence_advantage: float,
    evidence: Sequence[float],
    *,
    config: TurnCreditConfig | None = None,
) -> tuple[float, ...]:
    selected = config or TurnCreditConfig()
    advantage = float(sequence_advantage)
    if not math.isfinite(advantage):
        raise TurnCreditError("sequence advantage must be finite")
    if abs(advantage) < selected.epsilon:
        return tuple(0.0 for _ in evidence)
    z_scores = normalized_turn_evidence(evidence, epsilon=selected.epsilon)
    if not z_scores:
        return ()
    direction = 1.0 if advantage > 0 else -1.0
    result: list[float] = []
    for z_score in z_scores:
        aligned = direction * z_score
        multiplier = max(
            1.0 - selected.multiplier_band,
            min(1.0 + selected.multiplier_band, 1.0 + selected.multiplier_band * aligned),
        )
        factor = (1.0 - selected.mix_lambda) + selected.mix_lambda * multiplier
        reshaped = advantage * factor
        if reshaped * advantage <= 0:
            raise TurnCreditError("turn reshaping reversed sequence advantage")
        result.append(reshaped)
    return tuple(result)


def assistant_turn_spans(mask: Sequence[int | bool | float]) -> tuple[tuple[int, int], ...]:
    """Return half-open spans for contiguous assistant-token mask runs."""

    spans: list[tuple[int, int]] = []
    start: int | None = None
    for position, raw in enumerate(mask):
        active = bool(raw)
        if active and start is None:
            start = position
        elif not active and start is not None:
            spans.append((start, position))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return tuple(spans)


def reshape_batch_advantages(data: Any, algorithm_config: Any) -> Any:
    """Replace standard GRPO's broadcast advantage with bounded turn values.

    This entry point is called by the project's hash-checked veRL connection
    patch immediately after standard GRPO advantage computation.  The
    trajectory reward tensor is untouched.
    """

    turn_cfg_raw = algorithm_config.get("travel_turn_credit", {})
    mode = validate_turn_credit_mode(turn_cfg_raw.get("mode", "off"))
    if mode != "train":
        return data
    config = TurnCreditConfig.from_mapping(turn_cfg_raw)
    records = data.non_tensor_batch.get("turn_credit")
    if records is None:
        raise TurnCreditError("training batch is missing turn_credit records")

    import torch

    response_mask = data.batch["response_mask"]
    original = data.batch["advantages"]
    reshaped = torch.zeros_like(original)
    for row in range(response_mask.shape[0]):
        record = records[row]
        if not isinstance(record, Mapping):
            raise TurnCreditError(f"row {row} has no turn-credit mapping")
        if record.get("version") != TURN_CREDIT_VERSION:
            raise TurnCreditError(f"row {row} has an incompatible turn-credit version")
        evidence = record.get("evidence")
        if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
            raise TurnCreditError(f"row {row} has invalid turn evidence")
        mask_values = response_mask[row].detach().cpu().tolist()
        spans = assistant_turn_spans(mask_values)
        if len(spans) != len(evidence):
            raise TurnCreditError(
                f"row {row} turn/span mismatch: turns={len(evidence)}, spans={len(spans)}"
            )
        active = response_mask[row].bool()
        if bool(active.any()):
            sequence_advantage = float(original[row][active][0].item())
        else:
            sequence_advantage = 0.0
        turn_values = reshape_turn_advantages(
            sequence_advantage, [float(value) for value in evidence], config=config
        )
        for (start, end), value in zip(spans, turn_values, strict=True):
            reshaped[row, start:end] = value
    reshaped = reshaped * response_mask
    data.batch["advantages"] = reshaped
    data.batch["returns"] = reshaped.clone()
    return data


__all__ = [
    "TURN_CREDIT_MODES",
    "TURN_CREDIT_VERSION",
    "AspectCausalTrace",
    "TurnCreditConfig",
    "TurnCreditError",
    "TurnCreditTrace",
    "TurnEvent",
    "allocate_turn_evidence",
    "assistant_turn_spans",
    "build_aspect_causal_traces",
    "build_turn_credit_trace",
    "normalized_turn_evidence",
    "primary_violation_penalty",
    "reshape_batch_advantages",
    "reshape_turn_advantages",
    "validate_turn_credit_mode",
]
