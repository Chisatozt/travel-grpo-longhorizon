"""Actor-safe projections of TravelGym observations and step results."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class UserBenchObservationError(ValueError):
    """Raised when TravelGym returns an invalid observation contract."""


@dataclass(frozen=True)
class UserBenchObservation:
    """The subset of an upstream observation that is safe to expose to the actor."""

    feedback: str
    step_count: int
    episode_complete: bool
    last_reward: float
    diagnostics: Mapping[str, Any] = field(repr=False, compare=False)

    @classmethod
    def from_upstream(cls, observation: Mapping[str, Any]) -> UserBenchObservation:
        if not isinstance(observation, Mapping):
            raise UserBenchObservationError("TravelGym observation must be a mapping")
        feedback = observation.get("feedback")
        step_count = observation.get("step_count")
        episode_complete = observation.get("episode_complete")
        last_reward = observation.get("last_reward")
        if not isinstance(feedback, str):
            raise UserBenchObservationError("observation.feedback must be a string")
        if (
            not isinstance(step_count, int)
            or isinstance(step_count, bool)
            or step_count < 0
        ):
            raise UserBenchObservationError(
                "observation.step_count must be non-negative"
            )
        if not isinstance(episode_complete, bool):
            raise UserBenchObservationError(
                "observation.episode_complete must be boolean"
            )
        if not isinstance(last_reward, (int, float)) or isinstance(last_reward, bool):
            raise UserBenchObservationError("observation.last_reward must be numeric")
        return cls(
            feedback=feedback,
            step_count=step_count,
            episode_complete=episode_complete,
            last_reward=float(last_reward),
            diagnostics=copy.deepcopy(
                {key: value for key, value in observation.items() if key != "feedback"}
            ),
        )

    def to_tool_text(self) -> str:
        return self.feedback


@dataclass(frozen=True)
class UserBenchStepResult:
    """One validated TravelGym transition without hidden preference state."""

    task_id: str
    observation: UserBenchObservation
    reward: float
    terminated: bool
    truncated: bool
    diagnostics: Mapping[str, Any] = field(repr=False, compare=False)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated
