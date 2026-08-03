"""Provider-neutral contract for UserBench's single interaction tool."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

TOOL_NAME = "interact_with_env"
_REQUIRED_PARAMETERS = frozenset({"thought", "choice", "content"})
_ACTION_PREFIX = re.compile(r"^\[(search|action|answer|finish)\]\s*", re.IGNORECASE)


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
