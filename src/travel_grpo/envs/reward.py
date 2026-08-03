"""Unshaped reward tracing for pinned UserBench transitions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


class UserBenchRewardError(ValueError):
    """Raised when an upstream reward is not a finite number."""


@dataclass
class RawRewardTrace:
    """Preserve upstream step rewards without normalization or shaping."""

    _values: list[float] = field(default_factory=list, repr=False)

    def append(self, value: float) -> None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise UserBenchRewardError("UserBench reward must be numeric")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise UserBenchRewardError("UserBench reward must be finite")
        self._values.append(normalized)

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(self._values)

    @property
    def total(self) -> float:
        return float(sum(self._values))
