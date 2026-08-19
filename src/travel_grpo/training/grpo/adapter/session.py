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


# [项目注释] 功能：`_non_empty`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：strip, ValueError, isinstance。
# [项目注释] 输入：`value`: Any；`name`: str。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
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


# [项目注释] 功能：`validate_rollout_extra_info`：执行输入、状态或产物校验，并在不满足约束时返回诊断或抛出异常。 主要协作调用：_non_empty, isinstance,
# [项目注释]    TypeError, ValueError。
# [项目注释] 输入：`extra_info`: Mapping[str, Any]。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
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


# [项目注释] 功能：`task_id_from_run_kwargs`：编排一个训练、采集、评测或 replay 流程，并汇总其结果。 主要协作调用：hasattr,
# [项目注释]    validate_rollout_extra_info, item, isinstance。
# [项目注释] 输入：`kwargs`: Mapping[str, Any]。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
def task_id_from_run_kwargs(kwargs: Mapping[str, Any]) -> str:
    extra_info = kwargs.get("extra_info")
    if hasattr(extra_info, "item"):
        extra_info = extra_info.item()
    if not isinstance(extra_info, Mapping):
        raise ValueError("veRL rollout is missing extra_info")
    return validate_rollout_extra_info(extra_info)


# [项目注释] 功能：`calculate_current_session_score`：计算奖励、指标或聚合统计，供训练、评测或报告使用。 主要协作调用：float, reward_report,
# [项目注释]    require_current_session。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `float`；具体值由各分支决定。
def calculate_current_session_score() -> float:
    return float(require_current_session().reward_report()["terminal_reward"])


# [项目注释] 功能：`_project_path`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：expanduser, is_absolute, resolve, exists。
# [项目注释] 输入：`value`: str | Path。
# [项目注释] 输出：标注返回 `Path`；具体值由各分支决定。
def _project_path(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    working = candidate.resolve()
    return working if working.exists() else (PROJECT_ROOT / candidate).resolve()


# [项目注释] 功能：`_load_yaml`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：_project_path, safe_load, read_text, isinstance。
# [项目注释] 输入：`path`: str | Path。
# [项目注释] 输出：标注返回 `Mapping[str, Any]`；具体值由各分支决定。
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
# [项目注释] 类型：`UserBenchRolloutRuntime` 封装相关状态、协议或数据结构。类属性和方法共同维护其不变量。
class UserBenchRolloutRuntime:
    environment_config: UserBenchEnvironmentConfig
    simulator_runtime: UserSimulatorRuntime
    source_root: Path | None

    @classmethod
    # [项目注释] 功能：`from_config_files`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：_load_yaml, SimulatorRole, items,
    # [项目注释]    set。
    # [项目注释] 输入：`environment_path`: str | Path；`simulator_path`: str | Path。
    # [项目注释] 输出：标注返回 `'UserBenchRolloutRuntime'`；具体值由各分支决定。
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
            raise ValueError("Travel Reward v3 must be terminal-only in [-1, 1]")
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

    # [项目注释] 功能：`start_session`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：_non_empty, wrapper_factory,
    # [项目注释]    set_current_session, get_current_session。
    # [项目注释] 输入：`task_id`: str；`request_id`: str | None；`wrapper_factory`: Any。
    # [项目注释] 输出：标注返回 `UserBenchSessionState`；具体值由各分支决定。
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
            initial_observation = wrapper.reset()
            state = UserBenchSessionState(
                request_id=request_id or uuid.uuid4().hex,
                task_id=normalized,
                wrapper=wrapper,
                reward_task=wrapper.reward_task(),
                reward_snapshot=wrapper.reward_snapshot(),
                public_initial_message=getattr(initial_observation, "feedback", None),
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
                initial_observation = await async_reset()
            else:
                import asyncio

                initial_observation = await asyncio.to_thread(wrapper.reset)
            state = UserBenchSessionState(
                request_id=request_id or uuid.uuid4().hex,
                task_id=normalized,
                wrapper=wrapper,
                reward_task=wrapper.reward_task(),
                reward_snapshot=wrapper.reward_snapshot(),
                public_initial_message=getattr(initial_observation, "feedback", None),
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

    # [项目注释] 功能：`__init__`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：RuntimeError。
    # [项目注释] 输入：*`_`；**`__`。
    # [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
    def __init__(self, *_: Any, **__: Any) -> None:
        raise RuntimeError("UserBenchInteraction was removed; use UserBenchAgentLoop with veRL 0.8")
