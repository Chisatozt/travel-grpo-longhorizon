"""Provider-neutral contract for UserBench's single interaction tool."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

TOOL_NAME = "interact_with_env"
_REQUIRED_PARAMETERS = frozenset({"thought", "choice", "content"})
_ACTION_PREFIX = re.compile(r"^\[(search|action|answer|finish)\]\s*", re.IGNORECASE)
OPTION_ID = re.compile(r"^[ACFHR]\d+$")
ASPECT_BY_OPTION_PREFIX = {
    "F": "flight",
    "H": "hotel",
    "A": "apartment",
    "C": "rental_car",
    "R": "restaurant",
}
ASPECT_QUERY_HINTS = {
    "flight": ("flight", "airline", "carrier"),
    "hotel": ("hotel",),
    "apartment": ("apartment",),
    "rental_car": ("rental car", "car rental", "rental vehicle"),
    "restaurant": ("restaurant", "dining"),
}
FIELD_QUERY_HINTS = {
    "flight": {
        "company": ("airline", "carrier", "flight company"),
        "path": ("direct", "nonstop", "non-stop", "layover", "connection", "route"),
        "time": (
            "departure time",
            "arrival time",
            "depart",
            "arrive",
            "morning",
            "evening",
        ),
        "amenities": ("wifi", "wi-fi", "meal", "entertainment", "power outlet"),
        "service": ("baggage", "luggage", "carry-on", "checked bag", "flight service"),
    },
    "hotel": {
        "name": ("hotel name", "property name", "specific hotel", "name"),
        "room": ("room", "bedroom", "bathroom", "suite", "guest", "people", "capacity"),
        "amenities": (
            "amenity",
            "amenities",
            "air conditioning",
            "wifi",
            "wi-fi",
            "parking",
            "kitchen",
            "pool",
            "gym",
            "elevator",
        ),
        "service": ("cleaning", "early check-in", "early checkin", "late checkout"),
        "rating": ("rating", "star", "review score"),
    },
    "apartment": {
        "name": ("apartment name", "property name", "specific apartment", "name"),
        "room": ("room", "bedroom", "bathroom", "guest", "people", "capacity"),
        "amenities": (
            "amenity",
            "amenities",
            "air conditioning",
            "wifi",
            "wi-fi",
            "parking",
            "kitchen",
            "washer",
            "dryer",
            "pet",
            "elevator",
        ),
        "service": ("cleaning", "early check-in", "early checkin", "late checkout"),
        "rating": ("rating", "review score"),
    },
    "rental_car": {
        "brand": ("brand", "rental company", "car company"),
        "model": ("model", "vehicle type", "car type", "economy", "suv", "gasoline"),
        "seats": ("seat", "seats", "passenger"),
        "insurance": ("insurance", "waiver", "liability", "accident", "protection"),
        "service": ("child seat", "additional driver", "underage", "rental service"),
    },
    "restaurant": {
        "cuisine": ("cuisine", "food", "dish"),
        "tags": ("outdoor seating", "reservation", "walk-in", "occasion", "atmosphere"),
        "rating": ("rating", "review score"),
        "expectation": ("price", "cost", "budget", "cheap", "expensive", "affordable"),
    },
}


class UserBenchActionError(ValueError):
    """Raised when an actor emits an invalid UserBench tool call."""


class ActionChoice(str, Enum):
    """Actions accepted by the pinned TravelGym environment."""

    SEARCH = "search"
    ACTION = "action"
    ANSWER = "answer"


@dataclass(frozen=True)
class UserBenchAction:
    """A validated ``interact_with_env`` call."""

    thought: str
    choice: ActionChoice
    content: str

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any]) -> UserBenchAction:
        if not isinstance(parameters, Mapping):
            raise UserBenchActionError("tool parameters must be a mapping")
        keys = set(parameters)
        missing = _REQUIRED_PARAMETERS - keys
        extra = keys - _REQUIRED_PARAMETERS
        if missing:
            raise UserBenchActionError(
                "missing required tool parameters: " + ", ".join(sorted(missing))
            )
        if extra:
            raise UserBenchActionError(
                "unexpected tool parameters: " + ", ".join(sorted(extra))
            )

        thought = parameters["thought"]
        content = parameters["content"]
        raw_choice = parameters["choice"]
        if not isinstance(thought, str) or not thought.strip():
            raise UserBenchActionError("thought must be a non-empty string")
        if not isinstance(content, str) or not content.strip():
            raise UserBenchActionError("content must be a non-empty string")
        if not isinstance(raw_choice, str):
            raise UserBenchActionError("choice must be a string")
        try:
            choice = ActionChoice(raw_choice.strip().lower())
        except ValueError as exc:
            allowed = ", ".join(choice.value for choice in ActionChoice)
            raise UserBenchActionError(f"choice must be one of: {allowed}") from exc

        normalized_content = content.strip()
        prefix = _ACTION_PREFIX.match(normalized_content)
        if prefix:
            prefixed_choice = prefix.group(1).lower()
            if prefixed_choice == "finish":
                raise UserBenchActionError(
                    "[finish] is not exposed by interact_with_env"
                )
            if prefixed_choice != choice.value:
                raise UserBenchActionError(
                    f"content prefix [{prefixed_choice}] conflicts with choice {choice.value!r}"
                )
            normalized_content = normalized_content[prefix.end() :].strip()
            if not normalized_content:
                raise UserBenchActionError(
                    "content must not be empty after its action prefix"
                )

        return cls(
            thought=thought.strip(),
            choice=choice,
            content=normalized_content,
        )

    def to_environment_action(self) -> str:
        """Render the exact text protocol consumed by ``TravelEnv.step``."""

        return f"[{self.choice.value}] {self.content}"


_INTERACT_WITH_ENV_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "A tool for interact with a target environment. The detailed environment "
            "description and action space is provided in the system prompt, so please "
            "follow the system prompt. You can use this tool to analyze and interact "
            "with the environment step by step through three actions including "
            "`search`, `action` or `answer`."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": (
                        "Your thought of what to do next, including your reason or "
                        "analysis of your choice and why."
                    ),
                },
                "choice": {
                    "type": "string",
                    "enum": ["action", "answer", "search"],
                    "description": (
                        "Your choice of what to do next, must be one of `action`, "
                        "`answer` or `search`."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": (
                        "The content of your choice, must be a string. If you choose "
                        "`action`, you should provide the action you want to take. If "
                        "you choose `answer`, you should provide the answer that you "
                        "want to submit. If you choose `search`, you should provide the "
                        "search query. The specific format of the content is determined "
                        "by the environment description, which should be provided in the "
                        "system prompt. Please follow the format strictly in order to "
                        "invocate this tool."
                    ),
                },
            },
            "required": ["thought", "choice", "content"],
        },
    },
}


def get_interact_with_env_schema() -> dict[str, Any]:
    """Return a defensive copy of the official UserBench function schema."""

    return copy.deepcopy(_INTERACT_WITH_ENV_SCHEMA)


def normalized_action_signature(action: UserBenchAction) -> str:
    """Return the canonical identity used to reject exact action repeats."""

    content = " ".join(action.content.split()).casefold()
    return f"{action.choice.value}:{content}"


def _contains_hint(query: str, hint: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(hint)}(?![a-z0-9])", query) is not None


def semantic_action_signature(
    action: UserBenchAction, task_dimensions: Sequence[str]
) -> tuple[str, str] | None:
    """Infer an unambiguous ``(aspect, preference field)`` question identity."""

    if action.choice is not ActionChoice.ACTION:
        return None
    dimensions = {value for value in task_dimensions if value in FIELD_QUERY_HINTS}
    query = " ".join(action.content.casefold().replace("_", " ").split())
    explicit = {
        aspect
        for aspect in dimensions
        if any(_contains_hint(query, hint) for hint in ASPECT_QUERY_HINTS[aspect])
    }
    if len(explicit) > 1:
        return None
    candidates = explicit or dimensions
    matches = {
        (aspect, field)
        for aspect in candidates
        for field, hints in FIELD_QUERY_HINTS[aspect].items()
        if any(_contains_hint(query, hint) for hint in hints)
    }
    return next(iter(matches)) if len(matches) == 1 else None


def aspect_from_option_id(option_id: object) -> str | None:
    """Map an official answer option such as ``H3`` to its travel aspect."""

    if not isinstance(option_id, str) or not OPTION_ID.fullmatch(option_id.strip()):
        return None
    return ASPECT_BY_OPTION_PREFIX.get(option_id.strip()[0])
