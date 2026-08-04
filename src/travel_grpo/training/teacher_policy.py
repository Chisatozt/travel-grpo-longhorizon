"""Deterministic phase control for strict UserBench teacher trajectories."""

from __future__ import annotations

import re
import json
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
_GENERIC_FIELD_QUESTION = re.compile(
    r"\b(?:what|which)\b.*\b(?:amenities|atmosphere|seating features|settings?|services?)\b"
    r".*\b(?:important|prefer|preferred)\b",
    re.IGNORECASE,
)
_PUBLIC_SEARCH_REQUIREMENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("air conditioning", ("air conditioning",)),
    ("late checkout", ("cost_late_checkout_fee",)),
    ("early check-in", ("cost_early_checkin_fee",)),
    ("parking", ("parking",)),
    ("elevator", ("elevator",)),
    ("pets allowed", ("pets allowed", "pet-friendly")),
    ("washer and dryer", ("washer and dryer", "washer", "dryer")),
    ("delivery", ("delivery",)),
    ("vegetarian", ("vegetarian",)),
    ("electric", ("electric",)),
    ("economy", ("economy",)),
    ("damage waiver", ("cost_damage_waiver",)),
    ("personal accident protection", ("cost_personal_accident_waiver",)),
    ("personal belongings coverage", ("cost_belonging_waiver",)),
)


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
        "company": "Which specific airline brand or carrier do you prefer for the flight?",
        "path": "Do you prefer a direct flight or are connections acceptable?",
        "time": "Do you prefer a shorter flight duration, a longer layover duration, or a particular departure or arrival time?",
        "amenities": "Do you need a flight carry-on baggage allowance, Wi-Fi, meal service, entertainment, or power outlets?",
        "service": "Do you need checked baggage, carry-on baggage, business class, priority boarding, or another specific flight service?",
    },
    "hotel": {
        "name": "Do you prefer a specific hotel name, brand, platform, or property?",
        "room": "What hotel room, suite, bedroom, bathroom, bed, or guest-capacity configuration do you need?",
        "amenities": "Do you need hotel Wi-Fi, air conditioning, parking, a kitchen, a pool, a gym, an elevator, business workspace, pet-friendly access, or another specific amenity?",
        "service": "Do you need hotel breakfast, parking service, cleaning, early check-in, late checkout, barrier-free access, or another specific service?",
        "rating": "What minimum hotel rating do you prefer?",
    },
    "apartment": {
        "name": "Do you prefer a specific apartment name, brand, platform, or property?",
        "room": "How many bedrooms or bathrooms, and what guest capacity, do you need for the apartment?",
        "amenities": "Do you need apartment Wi-Fi, air conditioning, parking, a kitchen, a washer, a dryer, pet access, an elevator, or another specific amenity?",
        "service": "Do you need apartment breakfast, parking service, cleaning, early check-in, late checkout, barrier-free access, or another specific service?",
        "rating": "What minimum apartment rating do you prefer?",
    },
    "rental_car": {
        "brand": "Which rental car brand or company do you prefer?",
        "model": "Which rental car model or vehicle type do you prefer, such as an electric, hybrid, gasoline, compact, sedan, or SUV vehicle?",
        "seats": "How many seats do you need in the rental car?",
        "insurance": "Which rental car insurance coverage do you prefer?",
        "service": "Do you need a child seat, an additional driver, GPS navigation, an underage-driver service, or another specific rental-car service?",
    },
    "restaurant": {
        "cuisine": "Which restaurant cuisine do you prefer?",
        "tags": "Do you prefer restaurant delivery, outdoor seating, a quiet atmosphere, family-friendly dining, or another specific restaurant setting?",
        "rating": "Do you prefer a minimum restaurant rating or avoiding restaurants with any one-star reviews?",
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
    visible_option_details: tuple[str, ...] = ()
    public_requirements: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @property
    def choice(self) -> ActionChoice:
        return ActionChoice(self.phase.value if self.phase is not TeacherPhase.ELICIT else "action")

    @property
    def canonical_content(self) -> str | None:
        if self.phase is TeacherPhase.ELICIT:
            assert self.field is not None
            return _CANONICAL_QUESTIONS[self.aspect][self.field]
        if self.phase is TeacherPhase.SEARCH:
            # A correct UserBench search must contain every public argument for the
            # aspect.  A static sentence cannot safely reconstruct dates, places,
            # or preferences and must therefore never be installed as a content
            # enum.  SEARCH remains choice-constrained while the teacher composes
            # its query from the visible conversation.
            return None
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
                "that explicitly restates every relevant public trip argument (such as "
                "locations and dates) and every preference the user has disclosed for this "
                "aspect. Do not say only `matching the trip request`, do not invent missing "
                "arguments, do not ask another question, and do not search any other aspect."
            )
        options = ", ".join(self.available_option_ids)
        details = ""
        if self.visible_option_details:
            details = (
                " Public candidate facts copied only from the visible search output:\n"
                + "\n".join(f"- {value}" for value in self.visible_option_details)
                + "\n"
            )
        return (
            f"Current phase: answer for {label}. Select exactly one option ID from the "
            f"visible search results. Allowed visible IDs: {options}. Emit choice=`answer`; "
            f"{details}"
            "review every visible option field and every preference disclosed by the user; "
            "choose an option that satisfies all disclosed requirements, and when the "
            "budget is limited choose the cheapest satisfying option. Do not choose by "
            "option order or ID, and do not ask or search."
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
            if self.field in {"amenities", "tags", "service"} and _GENERIC_FIELD_QUESTION.search(
                action.content
            ):
                return "vague_action"
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
        if self.visible_option_details and self.public_requirements:
            selected = next(
                (
                    value
                    for value in self.visible_option_details
                    if f'"id":"{content}"' in value
                ),
                None,
            )
            if selected is not None:
                for name, aliases in self.public_requirements:
                    matching = [
                        value
                        for value in self.visible_option_details
                        if _public_candidate_satisfies(value, aliases)
                    ]
                    if matching and not _public_candidate_satisfies(selected, aliases):
                        return (
                            "answer_not_matching_public_requirement."
                            + name.replace(" ", "_")
                        )
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
        configured = tuple(self.task.preference_fields_by_aspect.get(aspect, ()))
        known = tuple(value for value in configured if value in FIELD_QUERY_HINTS[aspect])
        # Fake/test tasks and older task adapters may not expose field metadata;
        # retain the previous conservative fallback in that case. For real
        # UserBench tasks, querying only fields that can contain a hidden
        # preference prevents deterministic coverage dead-ends.
        values = known or tuple(FIELD_QUERY_HINTS[aspect])
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

    def _visible_option_details(
        self, aspect: str, messages: list[dict], option_ids: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Extract a compact, public-only copy of visible search candidates."""

        allowed = set(option_ids)
        if not allowed:
            return ()
        decoder = json.JSONDecoder()
        details: list[str] = []
        seen: set[str] = set()
        prefix = _PREFIX_BY_ASPECT[aspect]
        for message in messages:
            if message.get("role") != "tool":
                continue
            content = str(message.get("content") or "")
            for index, character in enumerate(content):
                if character != "{":
                    continue
                try:
                    value, _ = decoder.raw_decode(content[index:])
                except json.JSONDecodeError:
                    continue
                if not isinstance(value, dict):
                    continue
                option_id = value.get("id")
                if (
                    not isinstance(option_id, str)
                    or option_id not in allowed
                    or not option_id.startswith(prefix)
                    or option_id in seen
                ):
                    continue
                seen.add(option_id)
                details.append(
                    json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))[:1800]
                )
        return tuple(details)

    def _public_search_requirements(
        self, messages: list[dict]
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Infer lexical requirements from the Teacher's public search query."""

        query = ""
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue
            calls = message.get("tool_calls")
            if not isinstance(calls, list):
                continue
            for call in calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if not isinstance(function, dict):
                    continue
                try:
                    arguments = json.loads(str(function.get("arguments") or "{}"))
                except json.JSONDecodeError:
                    continue
                if arguments.get("choice") == "search":
                    query = str(arguments.get("content") or "").casefold()
                    break
            if query:
                break
        if not query:
            return ()
        return tuple(
            (name, aliases)
            for name, aliases in _PUBLIC_SEARCH_REQUIREMENTS
            if name in query
            or any(
                alias in query
                for alias in aliases
                if not alias.startswith("cost_")
            )
        )

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
            visible_option_details=self._visible_option_details(aspect, messages, options),
            public_requirements=self._public_search_requirements(messages),
        )

    def record_committed(self, plan: TeacherTurnPlan) -> None:
        if plan.phase is TeacherPhase.ELICIT:
            assert plan.field is not None
            self.asked_fields[plan.aspect].add(plan.field)


def _public_candidate_satisfies(detail: str, aliases: tuple[str, ...]) -> bool:
    """Check a public option JSON object without consulting hidden task labels."""

    try:
        value = json.loads(detail)
    except json.JSONDecodeError:
        return False

    def non_null_key(node: object, key: str) -> bool:
        if isinstance(node, dict):
            if key in node and node[key] is not None:
                return True
            return any(non_null_key(child, key) for child in node.values())
        if isinstance(node, list):
            return any(non_null_key(child, key) for child in node)
        return False

    text = json.dumps(value, ensure_ascii=False).casefold()
    for alias in aliases:
        if alias.startswith("cost_"):
            if non_null_key(value, alias):
                return True
        elif alias.casefold() in text:
            return True
    return False


def canonical_content_for(plan: TeacherTurnPlan) -> tuple[str, ...]:
    """Return a request-only content enum for the strongest retry."""

    if plan.phase is TeacherPhase.ANSWER:
        return plan.available_option_ids
    value = plan.canonical_content
    return (value,) if value is not None else ()
