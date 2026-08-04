"""Project-owned lifecycle wrapper around the pinned ``travelgym.TravelEnv``."""

from __future__ import annotations

import asyncio
import copy
import importlib
import io
import math
from contextlib import redirect_stdout
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, cast

from travel_grpo.envs.observation import UserBenchObservation, UserBenchStepResult
from travel_grpo.envs.reward import TravelRewardTask, UserBenchRewardSnapshot
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


_UPSTREAM_STDOUT_LOCK = asyncio.Lock()


def _classify_upstream_stdout(output: str) -> dict[str, int]:
    """Convert known pinned-UserBench fallback prints into safe counters."""

    return {
        "userbench_judgment_fallbacks": output.count(
            "By default turn to normal conversation"
        ),
        "userbench_search_fallbacks": output.count("By default will return N/A"),
    }


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
    capture_upstream_diagnostics: bool = False

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


def _fallback_diagnostics(observation: Any, info: Mapping[str, Any]) -> dict[str, int]:
    """Detect pinned-UserBench fallback paths without capturing process stdout."""

    feedback = observation.get("feedback", "") if isinstance(observation, Mapping) else ""
    history = info.get("conversation_history", ())
    latest_agent: Mapping[str, Any] | None = None
    if isinstance(history, list):
        for entry in reversed(history):
            if isinstance(entry, Mapping) and entry.get("role") == "agent":
                latest_agent = entry
                break
    note = str(latest_agent.get("note", "")) if latest_agent else ""
    return {
        "userbench_judgment_fallbacks": int(
            note.startswith("Judging of agent's latest utterance failed")
        ),
        "userbench_response_fallbacks": int(
            note.startswith("Responding system met some issues")
            or "not sure how to respond to your latest utterance" in str(feedback)
        ),
        "userbench_search_fallbacks": int(
            "searching backend is experiencing some issues" in str(feedback)
        ),
    }


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

    async def areset(self) -> UserBenchObservation:
        """Reset without blocking an AgentLoop worker's asyncio event loop.

        The pinned TravelGym reset API is synchronous.  Keeping that call on a
        worker thread lets unrelated trajectories continue while a simulator
        client or upstream initialization is slow.
        """

        return await asyncio.to_thread(self.reset)

    def reward_task(self) -> TravelRewardTask:
        """Return only frozen labels required by the internal terminal reward."""

        self._require_open()
        task = getattr(self._environment, "current_task", None)
        if not isinstance(task, Mapping):
            raise UserBenchEnvironmentError(
                "TravelGym does not expose current_task for reward verification"
            )
        reward_task = TravelRewardTask.from_upstream(task)
        if reward_task.task_id != self.task_id:
            raise UserBenchEnvironmentError(
                f"reward task ID {reward_task.task_id!r} does not match {self.task_id!r}"
            )
        return reward_task

    def reward_snapshot(self) -> UserBenchRewardSnapshot:
        """Sample hidden evidence for the reward ledger; never returned to the Actor."""

        self._require_open()
        if not self._reset:
            raise UserBenchLifecycleError("reset() must be called before reward_snapshot()")
        remaining = getattr(self._environment, "remaining_preferences", None)
        state = getattr(self._environment, "state_list", None)
        if not isinstance(remaining, list) or not isinstance(state, Mapping):
            raise UserBenchEnvironmentError(
                "TravelGym does not expose the state required by Travel Reward v2"
            )
        try:
            remaining_ids = frozenset(str(pref["id"]) for pref in remaining)
            return UserBenchRewardSnapshot(
                remaining_preference_ids=remaining_ids,
                active_elicited_count=int(state["active_elicited_preferences"]),
                passive_elicited_count=int(state["passive_elicited_preferences"]),
                remaining_search_aspects=frozenset(
                    str(value) for value in state["search_arguments"]
                ),
                choice_initials=frozenset(
                    str(value) for value in state["choice_initials"]
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise UserBenchEnvironmentError(
                "TravelGym reward state violates the pinned snapshot contract"
            ) from exc

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
        captured = ""
        try:
            async_step = getattr(self._environment, "step_async", None)
            if self.config.capture_upstream_diagnostics:
                buffer = io.StringIO()
                async with _UPSTREAM_STDOUT_LOCK:
                    with redirect_stdout(buffer):
                        if callable(async_step):
                            transition = await async_step(rendered)
                        else:
                            transition = await asyncio.to_thread(
                                self._environment.step, rendered
                            )
                captured = buffer.getvalue()
            elif callable(async_step):
                transition = await async_step(rendered)
            else:
                transition = await asyncio.to_thread(self._environment.step, rendered)
        except Exception as exc:
            raise UserBenchEnvironmentError(
                f"TravelGym async step failed for task {self.task_id!r}"
            ) from exc
        result = self._project_transition(transition)
        if captured:
            counters = _classify_upstream_stdout(captured)
            if any(counters.values()):
                result = replace(
                    result,
                    diagnostics={**dict(result.diagnostics), **counters},
                )
        return result

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
        reconciled_termination = self._async_one_choice_termination(terminated)
        projected = UserBenchObservation.from_upstream(observation)
        self._done = reconciled_termination or truncated
        diagnostics = copy.deepcopy(dict(info))
        diagnostics.update(_fallback_diagnostics(observation, info))
        if reconciled_termination and not terminated:
            # TravelEnv.step_async in the pinned snapshot omits the synchronous
            # one-choice termination check.  Reconcile it from the same public
            # state invariant used by TravelEnv.step; no reward or answer is
            # synthesized here.
            diagnostics["wrapper_async_termination_reconciled"] = 1
        return UserBenchStepResult(
            task_id=self.task_id,
            observation=projected,
            reward=float(reward),
            terminated=reconciled_termination,
            truncated=truncated,
            diagnostics=diagnostics,
        )

    def _async_one_choice_termination(self, terminated: bool) -> bool:
        """Match sync one-choice termination when upstream async omits it."""

        if terminated or not self.config.one_choice_per_aspect:
            return terminated
        task = getattr(self._environment, "current_task", None)
        state = getattr(self._environment, "state_list", None)
        if not isinstance(task, Mapping) or not isinstance(state, Mapping):
            return terminated
        dimensions = task.get("dimensions")
        choices = state.get("choice_initials")
        if not isinstance(dimensions, (list, tuple, set, frozenset)):
            return terminated
        if not isinstance(choices, (list, tuple, set, frozenset)):
            return terminated
        return bool(dimensions) and len(choices) >= len(dimensions)

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

    def __enter__(self) -> UserBenchWrapper:
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
