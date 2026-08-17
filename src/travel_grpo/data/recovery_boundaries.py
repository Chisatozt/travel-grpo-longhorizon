"""Compatibility facade for :mod:`travel_grpo.data.recovery.boundaries`."""

from travel_grpo.data.recovery import boundaries as _canonical
from travel_grpo.data.recovery.boundaries import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_canonical, name)
