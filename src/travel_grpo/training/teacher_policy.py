"""Deterministic phase control for strict UserBench teacher trajectories."""

from __future__ import annotations

import re
import json
import unicodedata
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
    semantic_action_signature,
)

if TYPE_CHECKING:
    from travel_grpo.envs.userbench_context import UserBenchSessionState


POLICY_VERSION = "teacher-state-machine-v5"
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


# [项目注释] 类型：`TeacherPhase` 封装相关状态、协议或数据结构。类属性和方法共同维护其不变量。
class TeacherPhase(str, Enum):
    ELICIT = "elicit"
    SEARCH = "search"
    ANSWER = "answer"


# [项目注释] 类型：`AttemptStrategy` 封装相关状态、协议或数据结构。类属性和方法共同维护其不变量。
class AttemptStrategy(str, Enum):
    NATURAL = "natural"
    STRICT = "strict"
    CANONICAL = "canonical"

    @classmethod
    # [项目注释] 功能：`for_attempt`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：min, max。
    # [项目注释] 输入：`attempt`: int。
    # [项目注释] 输出：标注返回 `'AttemptStrategy'`；具体值由各分支决定。
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
        "company": "Do you prefer a specific airline, the same carrier across flight legs, or different carriers across legs?",
        "path": "Do you prefer a direct flight, exactly one connection, or a particular layover city?",
        "time": "Do you prefer shorter individual flight legs, minimum total travel time, or a longer or reasonable layover duration?",
        "amenities": "Do you need flight Wi-Fi, meal service, lounge access, or a carry-on baggage allowance?",
        "service": "Do you need checked baggage or business class for the flight?",
    },
    "hotel": {
        "name": "Do you prefer a specific hotel name, brand, platform, or property?",
        "room": "What hotel room, suite, bedroom, bathroom, bed, or guest-capacity configuration do you need?",
        "amenities": "Do you need hotel Wi-Fi, air conditioning, a gym, business workspace, a city, ocean, or mountain view, pet-friendly access, or a washer and dryer?",
        "service": "Do you need hotel breakfast, parking service, or barrier-free accessibility?",
        "rating": "What hotel rating or rating range do you prefer?",
    },
    "apartment": {
        "name": "Do you prefer a specific apartment name, brand, platform, or property?",
        "room": "How many bedrooms or bathrooms, and what guest capacity, do you need for the apartment?",
        "amenities": "Do you need apartment Wi-Fi, air conditioning, parking, a kitchen, a washer and dryer, pet access, an elevator, or a garden?",
        "service": "Do you need apartment daily cleaning, early check-in, or late checkout?",
        "rating": "What apartment rating or rating range do you prefer?",
    },
    "rental_car": {
        "brand": "Which rental car brand or company do you prefer?",
        "model": "Do you prefer a specific rental car model, or an economy, electric, gasoline, or SUV vehicle?",
        "seats": "How many seats do you need in the rental car?",
        "insurance": "Do you need a damage waiver, liability coverage, personal accident protection, or personal belongings insurance for the rental car?",
        "service": "Do you need a child seat, an additional driver, or an underage-driver service for the rental car?",
    },
    "restaurant": {
        "cuisine": "Do you prefer a country-specific cuisine, vegetarian food, fast food, seafood or steakhouse dining, or a dessert, bakery, or cafe?",
        "tags": "Do you need restaurant delivery, outdoor seating, business dining, late-night service, parking, pet-friendly access, reservations, or walk-in availability?",
        "rating": "Do you prefer a restaurant rating range, zero one-star reviews, all reviews at least three-star, or more than half five-star reviews?",
        "expectation": "What restaurant price range or budget do you prefer?",
    },
}

# A failed elicitation turn remains visible in the environment transcript, so its
# one permitted repair must be materially different from the original question.
# Keep these templates deterministic and subject to the same single-field
# validation contract as the primary canonical questions.
_ELICITATION_REPAIR_QUESTIONS = {
    "flight": {
        "company": "For the flight, should the airline be a particular brand, stay consistent across legs, or differ between legs?",
        "path": "For the flight route, should it be nonstop, have exactly one stop, or connect through a city you choose?",
        "time": "For the flight, should individual legs or total travel time be minimized, or should the layover duration be longer or reasonably timed?",
        "amenities": "For the flight, do you require Wi-Fi, meals, airport lounge access, or carry-on baggage allowance?",
        "service": "For the flight, do you require a checked bag or a business-class seat?",
    },
    "hotel": {
        "name": "For the hotel, is there a particular hotel name or property name you want?",
        "room": "For the hotel, what room, bed, bathroom, or guest capacity would suit you?",
        "amenities": "For the hotel, do you require Wi-Fi, air conditioning, a gym, workspace, a city, ocean, or mountain view, pets allowed, or laundry facilities?",
        "service": "For the hotel, do you require breakfast, a parking service, or accessible barrier-free service?",
        "rating": "For the hotel, what rating interval or review score would be acceptable?",
    },
    "apartment": {
        "name": "For the apartment, is there a particular apartment name or property name you want?",
        "room": "For the apartment, how many bedrooms, bathrooms, or guests must it accommodate?",
        "amenities": "For the apartment, do you require Wi-Fi, air conditioning, parking, a kitchen, laundry, pets allowed, an elevator, or a garden?",
        "service": "For the apartment, do you require daily cleaning, an early check-in, or a late checkout?",
        "rating": "For the apartment, what rating interval or review score would be acceptable?",
    },
    "rental_car": {
        "brand": "For the rental car, is there a vehicle brand or rental company you prefer?",
        "model": "For the rental car, do you want a named model, economy, electric, gasoline, or SUV vehicle?",
        "seats": "For the rental car, how many passenger seats do you require?",
        "insurance": "For the rental car, do you require damage waiver, liability, personal accident, or belongings insurance?",
        "service": "For the rental car, do you require a child seat, an additional driver, or permission for an underage driver?",
    },
    "restaurant": {
        "cuisine": "For the restaurant, would you like a named national cuisine, vegetarian, fast food, seafood or steakhouse, or dessert, bakery, or cafe food?",
        "tags": "For the restaurant, do you require business dining, outdoor seating, delivery, late-night opening, parking, pet-friendly access, reservations, or walk-ins?",
        "rating": "For the restaurant, do you care about its rating range, one-star reviews, three-star minimum reviews, or share of five-star reviews?",
        "expectation": "For the restaurant, what price range, cost, or budget should it fit?",
    },
}

# Concrete option properties that may be copied into a search only after they
# appear in the public trip request or a simulator response. This is a global
# schema vocabulary, never task-specific hidden reward data.
_SEARCH_PREFERENCE_CONCEPTS: dict[str, dict[str, tuple[str, ...]]] = {
    "flight": {
        "direct": ("direct flight", "nonstop", "non-stop"),
        "one_stop": ("one-stop", "one stop", "single connection", "one connection"),
        "layover": ("layover", "connection", "connecting flight"),
        "short_duration": (
            "shortest flight",
            "shorter flight",
            "shorter travel time",
            "minimum travel time",
            "quick flight",
        ),
        "long_layover": ("longer layover", "long layover", "extended layover"),
        "wifi": ("wi-fi", "wifi", "internet", "online access"),
        "meal": ("meal service", "meals", "meal", "food service"),
        "lounge_access": ("lounge access", "airport lounge", "lounge"),
        "carry_on": (
            "carry-on baggage",
            "carry on baggage",
            "carry-on allowance",
            "carry-on luggage",
            "cabin baggage",
            "hand luggage",
        ),
        "checked_bag": (
            "checked baggage",
            "checked bag",
            "checked luggage",
            "checked suitcase",
        ),
        "business_class": ("business class", "business-class"),
    },
    "hotel": {
        "double_room": (
            "double room",
            "double bed",
            "one big bed",
            "one large bed",
            "one shared bed",
        ),
        "king_room": ("king room", "king bed", "king-sized bed", "king size bed"),
        "suite": ("suite",),
        "guest_capacity": ("guest capacity", "guests", "people"),
        "wifi": ("wi-fi", "wifi", "internet"),
        "air_conditioning": ("air conditioning",),
        "workspace": ("business workspace", "workspace"),
        "view": ("city view", "ocean view", "mountain view"),
        "gym": ("gym",),
        "pets": ("pets allowed", "pet-friendly", "pet friendly", "dog"),
        "laundry": ("washer and dryer", "washer", "dryer", "laundry"),
        "breakfast": ("breakfast",),
        "parking_service": ("parking service", "parking", "place to park"),
        "accessibility": ("barrier-free", "barrier free", "accessible"),
        "rating": (
            "rating",
            "review score",
            "score",
            "star",
            "stars",
            "star rating",
            "rated",
        ),
    },
    "apartment": {
        "bedrooms": ("bedrooms", "bedroom"),
        "bathrooms": ("bathrooms", "bathroom"),
        "guest_capacity": ("guest capacity", "guests", "people"),
        "wifi": ("wi-fi", "wifi", "internet"),
        "air_conditioning": ("air conditioning",),
        "parking": ("parking", "place to park"),
        "kitchen": ("kitchen",),
        "laundry": ("washer and dryer", "washer", "dryer", "laundry"),
        "pets": ("pets allowed", "pet-friendly", "pet friendly", "dog"),
        "elevator": ("elevator",),
        "garden": ("garden", "greenery"),
        "cleaning": ("daily cleaning", "cleaning"),
        "early_checkin": ("early check-in", "early checkin"),
        "late_checkout": ("late checkout", "late check-out"),
        "rating": (
            "rating",
            "review score",
            "score",
            "star",
            "stars",
            "star rating",
            "rated",
        ),
    },
    "rental_car": {
        "economy": ("economy car", "economy vehicle"),
        "electric": ("electric car", "electric vehicle"),
        "gasoline": ("gasoline car", "gasoline vehicle"),
        "suv": ("suv",),
        "seats": ("seats", "passengers"),
        "damage_waiver": (
            "damage waiver",
            "collision damage waiver",
            "damage protection",
            "coverage against damage",
            "protection against damage",
            "rental damage coverage",
        ),
        "liability": (
            "liability insurance",
            "liability coverage",
            "liability waiver",
            "claims from other drivers",
            "other cars involved",
            "protection against claims",
        ),
        "accident": (
            "personal accident",
            "accident protection",
            "protection if anyone is injured",
        ),
        "belongings": (
            "belongings insurance",
            "personal belongings",
            "luggage protection",
            "personal property coverage",
        ),
        "child_seat": ("child seat", "baby seat"),
        "additional_driver": ("additional driver", "more than one driver"),
        "underage_driver": ("underage driver", "underage-driver"),
    },
    "restaurant": {
        "vegetarian": ("vegetarian",),
        "fast_food": ("fast food",),
        "seafood": ("seafood", "fish", "fresh from the sea", "from the sea"),
        "steakhouse": ("steakhouse", "steak", "steaks"),
        "dessert": ("dessert", "bakery", "cafe"),
        "business_dining": ("business dining",),
        "outdoor_seating": ("outdoor seating",),
        "delivery": ("delivery",),
        "late_night": (
            "late-night",
            "late night",
            "doesn't close early",
            "does not close early",
            "closing early",
            "open late",
            "open later",
            "closes late",
            "closing late",
            "late hours",
            "after hours",
            "kitchen open later",
        ),
        "parking": ("parking",),
        "pets": ("pet-friendly", "pet friendly", "pets allowed", "dog"),
        "reservations": ("reservations", "reservation"),
        "walk_ins": ("walk-ins", "walk ins", "walk-in"),
        "cheap": (
            "cheap",
            "cheapest",
            "inexpensive",
            "low price",
            "limited budget",
            "budget is limited",
        ),
        "expensive": ("expensive", "high-end", "high end"),
        "rating": (
            "rating",
            "review score",
            "score",
            "star",
            "stars",
            "star rating",
            "rated",
        ),
        "one_star": ("one-star", "1-star"),
        "three_star": ("three-star", "3-star"),
        "five_star": ("five-star", "5-star"),
    },
}
_SPECULATIVE_SEARCH_PHRASES = (
    "nice-to-have",
    "nice to have",
    "ideally",
)
_SEARCH_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


_SEARCH_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def _search_tokens(text: str) -> tuple[str, ...]:
    """Normalize public search evidence for conservative phrase matching."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = normalized.replace("’", "'").replace("–", "-").replace("—", "-")
    normalized = normalized.replace("-", " ")
    return tuple(_SEARCH_TOKEN.findall(normalized))


# [项目注释] 功能：`_search_token_matches`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：endswith, len。
# [项目注释] 输入：`actual`: str；`expected`: str。
# [项目注释] 输出：标注返回 `bool`；具体值由各分支决定。
def _search_token_matches(actual: str, expected: str) -> bool:
    if actual == expected:
        return True
    # Accept common English inflections without allowing arbitrary substrings.
    if len(expected) > 3 and actual in {expected + "s", expected + "es"}:
        return True
    if len(actual) > 3 and expected in {actual + "s", actual + "es"}:
        return True
    if expected.endswith("y") and len(expected) > 3 and actual == expected[:-1] + "ies":
        return True
    if actual.endswith("y") and len(actual) > 3 and expected == actual[:-1] + "ies":
        return True
    return False


def _contains_public_alias(text: str, alias: str) -> bool:
    """Match an alias against public evidence with punctuation/inflection tolerance."""

    actual = _search_tokens(text)
    expected = _search_tokens(alias)
    if not actual or not expected or len(expected) > len(actual):
        return False
    for start in range(len(actual) - len(expected) + 1):
        if all(
            _search_token_matches(actual[start + offset], expected[offset])
            for offset in range(len(expected))
        ):
            return True
    return False


# [项目注释] 功能：`_search_query_issue`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：join, items, set, findall。
# [项目注释] 输入：`content`: str；`aspect`: str；`public_context`: tuple[str, ...]。
# [项目注释] 输出：标注返回 `str | None`；具体值由各分支决定。
def _search_query_issue(
    content: str, aspect: str, public_context: tuple[str, ...]
) -> str | None:
    if not public_context:
        return None
    reference = "\n".join(public_context)
    for phrase in _SPECULATIVE_SEARCH_PHRASES:
        if _contains_public_alias(content, phrase) and not _contains_public_alias(
            reference, phrase
        ):
            return "search_adds_speculative_requirement"
    for concept, aliases in _SEARCH_PREFERENCE_CONCEPTS[aspect].items():
        if any(_contains_public_alias(content, alias) for alias in aliases) and not any(
            _contains_public_alias(reference, alias) for alias in aliases
        ):
            return f"search_invents_preference.{concept}"
    public_years = set(_SEARCH_YEAR.findall(reference))
    if set(_SEARCH_YEAR.findall(content)) - public_years:
        return "search_invents_year"
    return None


@dataclass(frozen=True)
# [项目注释] 类型：`TeacherTurnPlan` 封装相关状态、协议或数据结构。类属性和方法共同维护其不变量。
class TeacherTurnPlan:
    phase: TeacherPhase
    aspect: str
    field: str | None = None
    available_option_ids: tuple[str, ...] = ()
    strategy: AttemptStrategy = AttemptStrategy.NATURAL
    visible_option_details: tuple[str, ...] = ()
    public_requirements: tuple[tuple[str, tuple[str, ...]], ...] = ()
    public_search_context: tuple[str, ...] = ()

    @property
    # [项目注释] 功能：`choice`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：ActionChoice。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：标注返回 `ActionChoice`；具体值由各分支决定。
    def choice(self) -> ActionChoice:
        return ActionChoice(self.phase.value if self.phase is not TeacherPhase.ELICIT else "action")

    @property
    # [项目注释] 功能：`canonical_content`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：标注返回 `str | None`；具体值由各分支决定。
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

    @property
    # [项目注释] 功能：`elicitation_repair_content`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：标注返回 `str | None`；具体值由各分支决定。
    def elicitation_repair_content(self) -> str | None:
        if self.phase is not TeacherPhase.ELICIT:
            return None
        assert self.field is not None
        return _ELICITATION_REPAIR_QUESTIONS[self.aspect][self.field]

    # [项目注释] 功能：`instruction`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：join。
    # [项目注释] 输入：`generation_attempt`: int。
    # [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
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
            evidence = ""
            if self.public_search_context:
                evidence = (
                    " Public evidence allowed for this search:\n"
                    + "\n".join(f"- {value}" for value in self.public_search_context)
                    + "\n"
                )
            return (
                f"Current phase: search for {label}. Emit one search query for this aspect "
                "that explicitly restates every relevant public trip argument (such as "
                "locations and dates) and every preference the user has disclosed for this "
                "aspect. Copy constraints only from the public evidence below: do not add "
                "nice-to-have, ideal, inferred, or default requirements and do not invent a "
                "year, party size, amenity, service, rating, or vehicle property. Do not say "
                "only `matching the trip request`, do not ask another question, and do not "
                f"search any other aspect.{evidence}"
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

    # [项目注释] 功能：`validate`：执行输入、状态或产物校验，并在不满足约束时返回诊断或抛出异常。 主要协作调用：strip, action_field_matches, set, replace。
    # [项目注释] 输入：`action`: UserBenchAction。
    # [项目注释] 输出：标注返回 `str | None`；具体值由各分支决定。
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
            return _search_query_issue(
                action.content, self.aspect, self.public_search_context
            )
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
# [项目注释] 类型：`TeacherPolicyState` 封装相关状态、协议或数据结构。类属性和方法共同维护其不变量。
class TeacherPolicyState:
    task: TravelRewardTask
    strategy: AttemptStrategy
    asked_fields: dict[str, set[str]] = field(default_factory=dict)

    # [项目注释] 功能：`__post_init__`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：set。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
    def __post_init__(self) -> None:
        self.asked_fields = {aspect: set() for aspect in self.task.aspects}

    # [项目注释] 功能：`_active_complete`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：set。
    # [项目注释] 输入：`aspect`: str；`session`: 'UserBenchSessionState'。
    # [项目注释] 输出：标注返回 `bool`；具体值由各分支决定。
    def _active_complete(self, aspect: str, session: "UserBenchSessionState") -> bool:
        expected = set(self.task.preference_ids_by_aspect[aspect])
        return expected <= session.active_preference_ids

    # [项目注释] 功能：`_field_order`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：tuple, len, reversed。
    # [项目注释] 输入：`aspect`: str。
    # [项目注释] 输出：标注返回 `tuple[str, ...]`；具体值由各分支决定。
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

    # [项目注释] 功能：`_visible_option_ids`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：tuple, findall, str,
    # [项目注释]    startswith。
    # [项目注释] 输入：`aspect`: str；`messages`: list[dict]。
    # [项目注释] 输出：标注返回 `tuple[str, ...]`；具体值由各分支决定。
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

    def _public_search_context(
        self, aspect: str, messages: list[dict]
    ) -> tuple[str, ...]:
        """Collect only initial user facts and simulator disclosures for an aspect."""

        values = [
            str(message.get("content") or "")[:1600]
            for message in messages
            if message.get("role") == "user" and message.get("content")
        ]
        for index, message in enumerate(messages[:-1]):
            if message.get("role") != "assistant":
                continue
            calls = message.get("tool_calls")
            if not isinstance(calls, list) or len(calls) != 1:
                continue
            function = calls[0].get("function") if isinstance(calls[0], dict) else None
            if not isinstance(function, dict):
                continue
            try:
                arguments = json.loads(str(function.get("arguments") or "{}"))
                action = UserBenchAction.from_parameters(arguments)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            semantic = semantic_action_signature(action, self.task.aspects)
            following = messages[index + 1]
            if (
                semantic is None
                or semantic[0] != aspect
                or following.get("role") != "tool"
                or not following.get("content")
            ):
                continue
            values.append(str(following["content"])[:1600])
        return tuple(values)

    # [项目注释] 功能：`next_plan`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：_visible_option_ids, TeacherTurnPlan,
    # [项目注释]    RuntimeError, _active_complete。
    # [项目注释] 输入：`session`: 'UserBenchSessionState'；`messages`: list[dict]。
    # [项目注释] 输出：标注返回 `TeacherTurnPlan`；具体值由各分支决定。
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
            return TeacherTurnPlan(
                TeacherPhase.SEARCH,
                aspect,
                strategy=self.strategy,
                public_search_context=self._public_search_context(aspect, messages),
            )
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

    # [项目注释] 功能：`record_committed`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：add。
    # [项目注释] 输入：`plan`: TeacherTurnPlan。
    # [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
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

    # [项目注释] 功能：`non_null_key`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：isinstance, any, non_null_key, values。
    # [项目注释] 输入：`node`: object；`key`: str。
    # [项目注释] 输出：标注返回 `bool`；具体值由各分支决定。
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
