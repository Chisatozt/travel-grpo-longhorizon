"""Raw reward preservation contracts."""

import math

import pytest

from travel_grpo.envs.reward import RawRewardTrace, UserBenchRewardError


def test_reward_trace_preserves_values_and_sums_without_shaping():
    trace = RawRewardTrace()
    for reward in (0.2, 0.2, 0.8, 1.0):
        trace.append(reward)
    assert trace.values == (0.2, 0.2, 0.8, 1.0)
    assert trace.total == pytest.approx(2.2)


@pytest.mark.parametrize("value", [True, math.inf, -math.inf, math.nan, "1"])
def test_invalid_rewards_fail(value):
    with pytest.raises(UserBenchRewardError):
        RawRewardTrace().append(value)
