"""Errors raised by the SFT collection boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class TeacherCollectionError(RuntimeError):
    """Raised when a task pool or collected trajectory violates the contract."""


class TeacherGenerationError(TeacherCollectionError):
    """Raised after request-local action correction attempts are exhausted."""

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

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code
