"""veRL 0.8 rollout metadata and direct UserBench session construction."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from travel_grpo.envs.userbench_context import (
    UserBenchSessionError,
    UserBenchSessionState,
    get_current_session,
    require_current_session,
    set_current_session,
)
from travel_grpo.envs.userbench_interaction import SimulatorRole, UserSimulatorRuntime
from travel_grpo.envs.reward import REWARD_VERSION
from travel_grpo.envs.userbench_wrapper import (
    UserBenchEnvironmentConfig,
    UserBenchWrapper,
)

AGENT_NAME = "userbench_tool_agent"
ENVIRONMENT_NAME = "TravelGym"
PROJECT_ROOT = Path(__file__).resolve().parents[5]


def _non_empty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def build_rollout_extra_info(task_id: str) -> dict[str, Any]:
    """Build the veRL 0.8 extra_info payload with duplicated ID guards."""

    normalized = _non_empty(task_id, "task_id")
    return {
        "task_id": normalized,
        "need_tools_kwargs": True,
        "tools_kwargs": {
            "interact_with_env": {
                "create_kwargs": {
                    "env_name": ENVIRONMENT_NAME,
                    "id": normalized,
                }
            }
        },
    }


def validate_rollout_extra_info(extra_info: Mapping[str, Any]) -> str:
    if not isinstance(extra_info, Mapping):
        raise TypeError("rollout extra_info must be a mapping")
    task_id = _non_empty(extra_info.get("task_id"), "extra_info.task_id")
    tools = extra_info.get("tools_kwargs")
    entry = tools.get("interact_with_env") if isinstance(tools, Mapping) else None
    create = entry.get("create_kwargs") if isinstance(entry, Mapping) else None
    if not isinstance(create, Mapping):
        raise TypeError("rollout extra_info is missing tool create_kwargs")
    tool_id = _non_empty(create.get("id"), "tool task_id")
    if tool_id != task_id:
        raise ValueError(
            f"extra_info task ID {task_id!r} does not match tool task ID {tool_id!r}"
        )
    if create.get("env_name") != ENVIRONMENT_NAME:
        raise ValueError(f"tool env_name must be {ENVIRONMENT_NAME!r}")
    return task_id


def task_id_from_run_kwargs(kwargs: Mapping[str, Any]) -> str:
    extra_info = kwargs.get("extra_info")
    if hasattr(extra_info, "item"):
        extra_info = extra_info.item()
    if not isinstance(extra_info, Mapping):
        raise ValueError("veRL rollout is missing extra_info")
    return validate_rollout_extra_info(extra_info)


def calculate_current_session_score() -> float:
    return float(require_current_session().reward_report()["terminal_reward"])


def _project_path(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    working = candidate.resolve()
    return working if working.exists() else (PROJECT_ROOT / candidate).resolve()


def _load_yaml(path: str | Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required by the UserBench AgentLoop") from exc
    resolved = _project_path(path)
    loaded = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise TypeError(f"configuration must be a mapping: {resolved}")
    return loaded


@dataclass(frozen=True)
class UserBenchRolloutRuntime:
    environment_config: UserBenchEnvironmentConfig
    simulator_runtime: UserSimulatorRuntime
    source_root: Path | None

    @classmethod
    def from_config_files(
        cls, environment_path: str | Path, simulator_path: str | Path
    ) -> "UserBenchRolloutRuntime":
        environment_document = _load_yaml(environment_path)
        simulator_document = _load_yaml(simulator_path)
        environment = environment_document.get("environment", environment_document)
        reward = environment_document.get("reward")
        simulator = simulator_document.get("simulator", simulator_document)
        if not isinstance(environment, Mapping) or not isinstance(simulator, Mapping):
            raise TypeError("invalid UserBench environment/simulator configuration")
        if not isinstance(reward, Mapping) or reward.get("version") != REWARD_VERSION:
            raise ValueError(f"environment reward must use {REWARD_VERSION}")
        if reward.get("terminal_only") is not True or reward.get("range") != [-1.0, 1.0]:
            raise ValueError("Travel Reward v2 must be terminal-only in [-1, 1]")
        role = SimulatorRole(_non_empty(simulator.get("role"), "simulator.role"))
        if role is not SimulatorRole.GRPO:
            raise ValueError("GRPO AgentLoop requires the grpo simulator role")
        expected = {
            "model_env": "GRPO_USER_SIM_MODEL",
            "base_url_env": "GRPO_USER_SIM_BASE_URL",
            "api_key_env": "GRPO_USER_SIM_API_KEY",
            "temperature": 0.0,
            "max_tokens": 2048,
            "timeout": 60.0,
        }
        for name, value in expected.items():
            if simulator.get(name) != value:
                raise ValueError(f"simulator.{name} must be {value!r}")
        allowed = set(UserBenchEnvironmentConfig.__dataclass_fields__)
        config = UserBenchEnvironmentConfig(
            **{key: value for key, value in environment.items() if key in allowed}
        )
        source = environment.get("source_root")
        return cls(
            environment_config=config,
            simulator_runtime=UserSimulatorRuntime.from_environment(role),
            source_root=None if source is None else _project_path(str(source)),
        )

    def start_session(
        self,
        task_id: str,
        *,
        request_id: str | None = None,
        wrapper_factory: Any = UserBenchWrapper,
    ) -> UserBenchSessionState:
        if get_current_session() is not None:
            raise UserBenchSessionError("a UserBench session is already active")
        normalized = _non_empty(task_id, "task_id")
        wrapper = wrapper_factory(
            normalized,
            self.simulator_runtime,
            self.environment_config,
            source_root=self.source_root,
        )
        try:
            wrapper.reset()
            state = UserBenchSessionState(
                request_id=request_id or uuid.uuid4().hex,
                task_id=normalized,
                wrapper=wrapper,
                reward_task=wrapper.reward_task(),
                reward_snapshot=wrapper.reward_snapshot(),
            )
        except Exception:
            wrapper.close()
            raise
        set_current_session(state)
        return state

    async def astart_session(
        self,
        task_id: str,
        *,
        request_id: str | None = None,
        wrapper_factory: Any = UserBenchWrapper,
    ) -> UserBenchSessionState:
        """Create and reset a session on the non-blocking rollout path."""

        if get_current_session() is not None:
            raise UserBenchSessionError("a UserBench session is already active")
        normalized = _non_empty(task_id, "task_id")
        wrapper = wrapper_factory(
            normalized,
            self.simulator_runtime,
            self.environment_config,
            source_root=self.source_root,
        )
        try:
            async_reset = getattr(wrapper, "areset", None)
            if callable(async_reset):
                await async_reset()
            else:
                import asyncio

                await asyncio.to_thread(wrapper.reset)
            state = UserBenchSessionState(
                request_id=request_id or uuid.uuid4().hex,
                task_id=normalized,
                wrapper=wrapper,
                reward_task=wrapper.reward_task(),
                reward_snapshot=wrapper.reward_snapshot(),
            )
        except Exception:
            wrapper.close()
            raise
        set_current_session(state)
        return state


class UserBenchInteraction:
    """Removed veRL 0.6 compatibility marker.

    veRL 0.8 uses :class:`UserBenchAgentLoop` directly.  Construction fails
    loudly so an old interaction config can never silently enter production.
    """

    def __init__(self, *_: Any, **__: Any) -> None:
        raise RuntimeError("UserBenchInteraction was removed; use UserBenchAgentLoop with veRL 0.8")
