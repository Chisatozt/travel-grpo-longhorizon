"""Compatibility facade for :mod:`travel_grpo.training.sft.collection`."""

from travel_grpo.training.sft import collection as _canonical
from travel_grpo.training.sft.collection import *  # noqa: F401,F403


def __getattr__(name: str):
    """Forward legacy private imports to the canonical collector module."""

    return getattr(_canonical, name)
