"""Deterministic phase control for strict UserBench teacher trajectories."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from travel_grpo.envs.reward import TravelRewardTask
from travel_grpo.envs.userbench_tools import (
    ASPECT_BY_OPTION_PREFIX,
    FIELD_QUERY_HINTS,
    ActionChoice,
    UserBenchAction,
    action_field_matches,
    action_mentions_aspect,
)

if TYPE_CHECKING:
    from travel_grpo.envs.userbench_context import UserBenchSessionState


POLICY_VERSION = "teacher-state-machine-v2"
_OPTION_ID = re.compile(r"(?<![A-Z0-9])([ACFHR]\d+)(?![A-Z0-9])")
_PREFIX_BY_ASPECT = {value: key for key, value in ASPECT_BY_OPTION_PREFIX.items()}


class TeacherPhase(str, Enum):
    ELICIT = "elicit"
    SEARCH = "search"
    ANSWER = "answer"


class AttemptStrategy(str, Enum):
    NATURAL = "natural"
    STRICT = "strict"
    CANONICAL = "canonical"

    @classmethod
    def for_attempt(cls, attempt: int) -> "AttemptStrategy":
        return (cls.NATURAL, cls.STRICT, cls.CANONICAL)[min(max(attempt, 1), 3) - 1]


_ASPECT_LABEL = {
    "flight": "flight",
    "hotel": "hotel",
    "apartment": "apartment",
    "rental_car": "rental car",
    "restaurant": "restaurant",
}

_CANONICAL_QUESTIONS = {
    "flight": {
        "company": "Which airline or carrier do you prefer for the flight?",
        "path": "Do you prefer a direct flight or are connections acceptable?",
        "time": "What departure or arrival time do you prefer for the flight?",
        "amenities": "Do you need flight Wi-Fi, meals, or power outlets?",
        "service": "Which baggage or flight services are important to you?",
    },
    "hotel": {
        "name": "Do you prefer a specific hotel name or property?",
        "room": "What hotel room or bed configuration do you need?",
        "amenities": "Which hotel amenities are important to you?",
        "service": "Do you need hotel cleaning or early check-in service?",
        "rating": "What minimum hotel rating do you prefer?",
    },
    "apartment": {
        "name": "Do you prefer a specific apartment name or property?",
        "room": "What apartment room or bedroom configuration do you need?",
        "amenities": "Which apartment amenities are important to you?",
        "service": "Do you need apartment cleaning or early check-in service?",
        "rating": "What minimum apartment rating do you prefer?",
    },
    "rental_car": {
        "brand": "Which rental car brand or company do you prefer?",
        "model": "Which rental car model or vehicle type do you prefer?",
        "seats": "How many seats do you need in the rental car?",
        "insurance": "Which rental car insurance coverage do you prefer?",
        "service": "Do you need an additional driver service for the rental car?",
    },
    "restaurant": {
        "cuisine": "Which restaurant cuisine do you prefer?",
        "tags": "Which restaurant atmosphere or seating features do you prefer?",
        "rating": "What minimum restaurant rating do you prefer?",
        "expectation": "What restaurant price range or budget do you prefer?",
    },
}


@dataclass(frozen=True)
class TeacherTurnPlan:
    phase: TeacherPhase
    aspect: str
    field: str | None = None
    available_option_ids: tuple[str, ...] = ()
    strategy: AttemptStrategy = AttemptStrategy.NATURAL

    @property
    def choice(self) -> ActionChoice:
        return ActionChoice(self.phase.value if self.phase is not TeacherPhase.ELICIT else "action")

    @property
    def canonical_content(self) -> str | None:
        if self.phase is TeacherPhase.ELICIT:
            assert self.field is not None
            return _CANONICAL_QUESTIONS[self.aspect][self.field]
        if self.phase is TeacherPhase.SEARCH:
            return f"Search for {_ASPECT_LABEL[self.aspect]} options matching the trip request."
        return None

    def instruction(self, generation_attempt: int) -> str:
        label = _ASPECT_LABEL[self.aspect]
        if self.phase is TeacherPhase.ELICIT:
            assert self.field is not None
            example = _CANONICAL_QUESTIONS[self.aspect][self.field]
            other = ", ".join(
                value for value in FIELD_QUERY_HINTS[self.aspect] if value != self.field
            )
            if generation_attempt == 1 and self.strategy is AttemptStrategy.NATURAL:
                return (
                    f"Current phase: preference elicitation for {label}. Ask exactly one "
                    f"focused question about `{self.field}` only. Do not search or answer."
                )
            return (
                f"Current phase: preference elicitation for {label}. Ask only about "
                f"`{self.field}`. Do not mention these other fields: {other}. Do not ask "
                f"a general preference question. A valid single-field example is: {example!r}. "
                "Emit exactly one interact_with_env call with choice=`action`."
            )
        if self.phase is TeacherPhase.SEARCH:
            return (
                f"Current phase: search for {label}. Emit one search query for this aspect "
                "using the trip request and elicited user preferences. Do not ask another "
                "question and do not search any other aspect."
            )
        options = ", ".join(self.available_option_ids)
        return (
            f"Current phase: answer for {label}. Select exactly one option ID from the "
            f"visible search results. Allowed visible IDs: {options}. Emit choice=`answer`; "
            "do not ask or search."
        )

    def validate(self, action: UserBenchAction) -> str | None:
        if action.choice is not self.choice:
            return "wrong_phase_choice"
        if self.phase is TeacherPhase.ELICIT:
            assert self.field is not None
            if not action_mentions_aspect(action.content, self.aspect):
                return "wrong_preference_aspect"
            matches = action_field_matches(action.content, (self.aspect,))
            if not matches:
                return "vague_action"
            if len(matches) > 1:
                return "bundled_action"
            if matches != {(self.aspect, self.field)}:
                return "wrong_preference_field"
            return None
        if self.phase is TeacherPhase.SEARCH:
            label_tokens = set(_ASPECT_LABEL[self.aspect].split("_"))
            normalized = action.content.casefold().replace("_", " ")
            if not any(token in normalized for token in label_tokens):
                return "wrong_search_aspect"
            return None
        content = action.content.strip()
        if not re.fullmatch(r"[ACFHR]\d+", content):
            return "invalid_answer_option"
        if content not in self.available_option_ids:
            return "answer_not_in_visible_results"
        return None


@dataclass
class TeacherPolicyState:
    task: TravelRewardTask
    strategy: AttemptStrategy
    asked_fields: dict[str, set[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.asked_fields = {aspect: set() for aspect in self.task.aspects}

    def _active_complete(self, aspect: str, session: "UserBenchSessionState") -> bool:
        expected = set(self.task.preference_ids_by_aspect[aspect])
        return expected <= session.active_preference_ids

    def _field_order(self, aspect: str) -> tuple[str, ...]:
        values = tuple(FIELD_QUERY_HINTS[aspect])
        if self.strategy is AttemptStrategy.STRICT and len(values) > 1:
            return (*values[1:], values[0])
        if self.strategy is AttemptStrategy.CANONICAL:
            return tuple(reversed(values))
        return values

    def _visible_option_ids(self, aspect: str, messages: list[dict]) -> tuple[str, ...]:
        prefix = _PREFIX_BY_ASPECT[aspect]
        found: list[str] = []
        for message in messages:
            if message.get("role") != "tool":
                continue
            for value in _OPTION_ID.findall(str(message.get("content") or "")):
                if value.startswith(prefix) and value not in found:
                    found.append(value)
        return tuple(found)

    def next_plan(
        self, session: "UserBenchSessionState", messages: list[dict]
    ) -> TeacherTurnPlan:
        remaining = [aspect for aspect in self.task.aspects if aspect not in session.answers]
        if not remaining:
            raise RuntimeError("environment_not_terminated_after_answers")
        aspect = remaining[0]
        if not self._active_complete(aspect, session):
            available = [
                field
                for field in self._field_order(aspect)
                if field not in self.asked_fields[aspect]
            ]
            if not available:
                raise RuntimeError(f"preference_coverage_unreachable.{aspect}")
            return TeacherTurnPlan(
                TeacherPhase.ELICIT, aspect, available[0], strategy=self.strategy
            )
        if aspect not in session.searched_aspects:
            return TeacherTurnPlan(TeacherPhase.SEARCH, aspect, strategy=self.strategy)
        options = self._visible_option_ids(aspect, messages)
        if not options:
            raise RuntimeError(f"search_no_visible_options.{aspect}")
        return TeacherTurnPlan(
            TeacherPhase.ANSWER,
            aspect,
            available_option_ids=options,
            strategy=self.strategy,
        )

    def record_committed(self, plan: TeacherTurnPlan) -> None:
        if plan.phase is TeacherPhase.ELICIT:
            assert plan.field is not None
            self.asked_fields[plan.aspect].add(plan.field)


def canonical_content_for(plan: TeacherTurnPlan) -> tuple[str, ...]:
    """Return a request-only content enum for the strongest retry."""

    if plan.phase is TeacherPhase.ANSWER:
        return plan.available_option_ids
    value = plan.canonical_content
    return (value,) if value is not None else ()
