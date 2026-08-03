"""Project-owned lifecycle wrapper around the pinned ``travelgym.TravelEnv``."""

from __future__ import annotations

import asyncio
import copy
import importlib
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self, cast

from travel_grpo.envs.observation import UserBenchObservation, UserBenchStepResult
from travel_grpo.envs.userbench_context import (
    EmbeddedUserBench,
    validate_embedded_userbench,
)
from travel_grpo.envs.userbench_interaction import (
    UserSimulatorRuntime,
    bind_user_simulator_process,
)
from travel_grpo.envs.userbench_tools import UserBenchAction


class UserBenchEnvironmentError(RuntimeError):
    """Raised when TravelGym cannot be created or violates its runtime contract."""


class UserBenchLifecycleError(RuntimeError):
    """Raised when reset/step/close are called in an invalid order."""


@dataclass(frozen=True)
class UserBenchEnvironmentConfig:
    """Pinned one-choice environment defaults used by training and evaluation."""

    max_steps: int = 20
    wrong_choice_number: int = 10
    noise_choice_number: int = 5
    one_choice_per_aspect: bool = True
    seed: int = 42
    reward_scale: float = 1.0
    step_penalty: float = 0.0
    search_correct_reward: float = 0.2
    preference_correct_reward: float = 0.2
    choice_best_reward: float = 1.0
    choice_correct_reward: float = 0.8
    wrong_choice_penalty: float = 0.0
    normalize_rewards: bool = False
    verbose: bool = False

    def __post_init__(self) -> None:
        fixed_values = {
            "max_steps": 20,
            "wrong_choice_number": 10,
            "noise_choice_number": 5,
            "seed": 42,
            "reward_scale": 1.0,
            "step_penalty": 0.0,
            "search_correct_reward": 0.2,
            "preference_correct_reward": 0.2,
            "choice_best_reward": 1.0,
            "choice_correct_reward": 0.8,
            "wrong_choice_penalty": 0.0,
        }
        for name, expected in fixed_values.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} is pinned to {expected!r}")
        if not self.one_choice_per_aspect:
            raise ValueError("this project only supports UserBench one-choice mode")
        if self.normalize_rewards:
            raise ValueError("raw UserBench rewards must not be normalized")


class TravelEnvProtocol(Protocol):
    def reset(
        self, *, seed: int | None = None, options: Any = None
    ) -> tuple[Any, Any]: ...

    def step(self, action_input: str) -> tuple[Any, float, bool, bool, Any]: ...

    def step_async(
        self, action_input: str
    ) -> Awaitable[tuple[Any, float, bool, bool, Any]]: ...

    def close(self) -> None: ...


EnvironmentFactory = Callable[
    [str, UserBenchEnvironmentConfig, UserSimulatorRuntime, EmbeddedUserBench],
    TravelEnvProtocol,
]


def _default_environment_factory(
    task_id: str,
    config: UserBenchEnvironmentConfig,
    runtime: UserSimulatorRuntime,
    source: EmbeddedUserBench,
) -> TravelEnvProtocol:
    try:
        travelgym = importlib.import_module("travelgym")
    except ImportError as exc:
        raise UserBenchEnvironmentError(
            "travelgym is not installed; run `pip install -e environments/UserBench`"
        ) from exc
    module_file = getattr(travelgym, "__file__", None)
    if not module_file:
        raise UserBenchEnvironmentError("installed travelgym module has no source path")
    imported_path = Path(module_file).resolve()
    try:
        imported_path.relative_to(source.root)
    except ValueError as exc:
        raise UserBenchEnvironmentError(
            f"travelgym was imported from {imported_path}, expected pinned source {source.root}"
        ) from exc

    upstream = travelgym.get_default_config()
    upstream.data_mode = "single"
    upstream.data_source = task_id
    upstream.api_key = runtime.api_key
    upstream.model_name = runtime.model
    upstream.temperature = runtime.temperature
    upstream.max_tokens = runtime.max_tokens
    upstream.timeout = runtime.timeout
    upstream.max_steps = config.max_steps
    upstream.wrong_choice_number = config.wrong_choice_number
    upstream.noise_choice_number = config.noise_choice_number
    upstream.one_choice_per_aspect = config.one_choice_per_aspect
    upstream.seed = config.seed
    upstream.reward_scale = config.reward_scale
    upstream.step_penalty = config.step_penalty
    upstream.search_correct_reward = config.search_correct_reward
    upstream.preference_correct_reward = config.preference_correct_reward
    upstream.choice_best_reward = config.choice_best_reward
    upstream.choice_correct_reward = config.choice_correct_reward
    upstream.wrong_choice_penalty = config.wrong_choice_penalty
    upstream.normalize_rewards = config.normalize_rewards
    upstream.verbose = config.verbose
    return cast(TravelEnvProtocol, travelgym.TravelEnv(config=upstream))


class UserBenchWrapper:
    """One-task TravelGym wrapper with actor-safe observations and strict lifecycle."""

    def __init__(
        self,
        task_id: str,
        runtime: UserSimulatorRuntime,
        config: UserBenchEnvironmentConfig | None = None,
        *,
        source_root: str | Path | None = None,
        environment_factory: EnvironmentFactory | None = None,
    ) -> None:
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        self.task_id = task_id.strip()
        self.runtime = runtime
        self.config = config or UserBenchEnvironmentConfig()
        self.source = validate_embedded_userbench(source_root)
        bind_user_simulator_process(runtime)
        factory = environment_factory or _default_environment_factory
        try:
            self._environment = factory(self.task_id, self.config, runtime, self.source)
        except (UserBenchEnvironmentError, ValueError):
            raise
        except Exception as exc:
            raise UserBenchEnvironmentError(
                f"failed to create TravelGym environment for task {self.task_id!r}"
            ) from exc
        self._reset = False
        self._done = False
        self._closed = False

    def reset(self) -> UserBenchObservation:
        self._require_open()
        try:
            observation, info = self._environment.reset(seed=self.config.seed)
        except Exception as exc:
            raise UserBenchEnvironmentError(
                f"TravelGym reset failed for task {self.task_id!r}"
            ) from exc
        self._validate_info_task_id(info)
        projected = UserBenchObservation.from_upstream(observation)
        self._reset = True
        self._done = projected.episode_complete
        return projected

    def step(self, action: UserBenchAction | Mapping[str, Any]) -> UserBenchStepResult:
        normalized = self._prepare_step(action)
        try:
            transition = self._environment.step(normalized.to_environment_action())
        except Exception as exc:
            raise UserBenchEnvironmentError(
                f"TravelGym step failed for task {self.task_id!r}"
            ) from exc
        return self._project_transition(transition)

    async def astep(
        self, action: UserBenchAction | Mapping[str, Any]
    ) -> UserBenchStepResult:
        normalized = self._prepare_step(action)
        rendered = normalized.to_environment_action()
        try:
            async_step = getattr(self._environment, "step_async", None)
            if callable(async_step):
                transition = await async_step(rendered)
            else:
                transition = await asyncio.to_thread(self._environment.step, rendered)
        except Exception as exc:
            raise UserBenchEnvironmentError(
                f"TravelGym async step failed for task {self.task_id!r}"
            ) from exc
        return self._project_transition(transition)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._environment.close()
        except Exception as exc:
            raise UserBenchEnvironmentError(
                f"TravelGym close failed for task {self.task_id!r}"
            ) from exc

    def _prepare_step(
        self, action: UserBenchAction | Mapping[str, Any]
    ) -> UserBenchAction:
        self._require_open()
        if not self._reset:
            raise UserBenchLifecycleError("reset() must be called before step()")
        if self._done:
            raise UserBenchLifecycleError("cannot step a completed UserBench episode")
        if isinstance(action, UserBenchAction):
            return action
        return UserBenchAction.from_parameters(action)

    def _project_transition(self, transition: Any) -> UserBenchStepResult:
        if not isinstance(transition, tuple) or len(transition) != 5:
            raise UserBenchEnvironmentError(
                "TravelGym step must return observation, reward, terminated, truncated, info"
            )
        observation, reward, terminated, truncated, info = transition
        if not isinstance(terminated, bool) or not isinstance(truncated, bool):
            raise UserBenchEnvironmentError(
                "TravelGym termination flags must be boolean"
            )
        if (
            not isinstance(reward, (int, float))
            or isinstance(reward, bool)
            or not math.isfinite(float(reward))
        ):
            raise UserBenchEnvironmentError("TravelGym reward must be a finite number")
        self._validate_info_task_id(info)
        projected = UserBenchObservation.from_upstream(observation)
        self._done = terminated or truncated
        return UserBenchStepResult(
            task_id=self.task_id,
            observation=projected,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            diagnostics=copy.deepcopy(dict(info)),
        )

    def _validate_info_task_id(self, info: Any) -> None:
        if not isinstance(info, Mapping):
            raise UserBenchEnvironmentError("TravelGym info must be a mapping")
        if info.get("task_id") != self.task_id:
            raise UserBenchEnvironmentError(
                f"TravelGym returned task ID {info.get('task_id')!r}, expected {self.task_id!r}"
            )

    def _require_open(self) -> None:
        if self._closed:
            raise UserBenchLifecycleError("UserBench environment is closed")

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
