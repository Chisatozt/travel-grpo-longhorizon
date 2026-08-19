"""Process-isolated configuration for UserBench user simulators."""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from enum import Enum


class SimulatorBoundaryError(RuntimeError):
    """Raised when one process attempts to mix simulator runtimes."""


# [项目注释] 类型：`SimulatorRole` 封装相关状态、协议或数据结构。类属性和方法共同维护其不变量。
class SimulatorRole(str, Enum):
    COLLECTION = "collection"
    GRPO = "grpo"
    EVAL = "eval"


DEEPSEEK_V4_FLASH_MODEL = "deepseek-v4-flash"


_ROLE_PREFIX = {
    SimulatorRole.COLLECTION: "COLLECTION_USER_SIM",
    SimulatorRole.GRPO: "GRPO_USER_SIM",
    SimulatorRole.EVAL: "EVAL_USER_SIM",
}


@dataclass(frozen=True)
class UserSimulatorRuntime:
    """Credentials and endpoint for exactly one process-level simulator role."""

    role: SimulatorRole
    model: str
    base_url: str
    api_key: str = field(repr=False)
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: float = 15.0

    # [项目注释] 功能：`__post_init__`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：strip, ValueError。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("user simulator model must be non-empty")
        if not self.base_url.strip():
            raise ValueError("user simulator base_url must be non-empty")
        if self.temperature < 0:
            raise ValueError("user simulator temperature must be non-negative")
        if self.max_tokens <= 0:
            raise ValueError("user simulator max_tokens must be positive")
        if self.timeout <= 0:
            raise ValueError("user simulator timeout must be positive")

    @classmethod
    # [项目注释] 功能：`from_environment`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：SimulatorRole, cls, strip,
    # [项目注释]    casefold。
    # [项目注释] 输入：`role`: SimulatorRole | str；`environ`: Mapping[str, str] | None。
    # [项目注释] 输出：标注返回 `UserSimulatorRuntime`；具体值由各分支决定。
    def from_environment(
        cls,
        role: SimulatorRole | str,
        environ: Mapping[str, str] | None = None,
    ) -> UserSimulatorRuntime:
        resolved_role = SimulatorRole(role)
        values = os.environ if environ is None else environ
        prefix = _ROLE_PREFIX[resolved_role]

        # [项目注释] 功能：`require`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：strip, SimulatorBoundaryError。
        # [项目注释] 输入：`name`: str。
        # [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
        def require(name: str) -> str:
            value = values.get(f"{prefix}_{name}", "").strip()
            if not value:
                raise SimulatorBoundaryError(
                    f"missing environment variable {prefix}_{name}"
                )
            return value

        runtime = cls(
            role=resolved_role,
            model=require("MODEL"),
            base_url=require("BASE_URL"),
            api_key=require("API_KEY"),
            temperature=float(values.get(f"{prefix}_TEMPERATURE", "0")),
            max_tokens=int(values.get(f"{prefix}_MAX_TOKENS", "2048")),
            timeout=float(values.get(f"{prefix}_TIMEOUT", "60")),
        )
        if runtime.model.casefold() != DEEPSEEK_V4_FLASH_MODEL:
            raise SimulatorBoundaryError(
                f"{prefix}_MODEL must be {DEEPSEEK_V4_FLASH_MODEL!r}"
            )
        return runtime


_BINDING_LOCK = threading.Lock()
_BOUND_RUNTIME: UserSimulatorRuntime | None = None


def bind_user_simulator_process(
    runtime: UserSimulatorRuntime,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Bind OpenAI SDK globals once and reject cross-role endpoint mixing."""

    global _BOUND_RUNTIME
    target = os.environ if environ is None else environ
    with _BINDING_LOCK:
        if _BOUND_RUNTIME is not None and _BOUND_RUNTIME != runtime:
            raise SimulatorBoundaryError(
                "this process is already bound to a different user simulator runtime"
            )
        target["OPENAI_API_KEY"] = runtime.api_key
        target["OPENAI_BASE_URL"] = runtime.base_url
        _BOUND_RUNTIME = runtime


# [项目注释] 功能：`_reset_user_simulator_binding_for_tests`：清理运行时资源或恢复边界状态，保证后续调用不会继承脏状态。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def _reset_user_simulator_binding_for_tests() -> None:
    global _BOUND_RUNTIME
    with _BINDING_LOCK:
        _BOUND_RUNTIME = None
