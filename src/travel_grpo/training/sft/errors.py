"""Errors raised by the SFT collection boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class TeacherCollectionError(RuntimeError):
    """Raised when a task pool or collected trajectory violates the contract."""


class TeacherGenerationError(TeacherCollectionError):
    """Raised after request-local action correction attempts are exhausted."""

    # [项目注释] 功能：`__init__`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：__init__, tuple, super, dict。
    # [项目注释] 输入：`message`: str；`diagnostics`: Sequence[Mapping[str, Any]]；`reason_code`: str。
    # [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
    def __init__(
        self,
        message: str,
        diagnostics: Sequence[Mapping[str, Any]],
        *,
        reason_code: str = "teacher_action_exhausted",
    ) -> None:
        super().__init__(message)
        self.diagnostics = tuple(dict(value) for value in diagnostics)
        self.reason_code = reason_code


class TeacherAttemptAbort(TeacherCollectionError):
    """Stops an attempt as soon as strict admission becomes impossible."""

    # [项目注释] 功能：`__init__`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：__init__, super。
    # [项目注释] 输入：`reason_code`: str；`message`: str。
    # [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code
