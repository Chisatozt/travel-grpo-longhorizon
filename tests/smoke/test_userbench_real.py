"""Opt-in smoke test for the editable pinned UserBench installation."""

import asyncio
import json
import os
from pathlib import Path

import pytest

from travel_grpo.envs.userbench_interaction import (
    SimulatorRole,
    UserSimulatorRuntime,
    _reset_user_simulator_binding_for_tests,
)
from travel_grpo.envs.userbench_wrapper import UserBenchWrapper

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    os.getenv("TRAVEL_GRPO_REAL_USERBENCH_SMOKE") != "1",
    reason="set TRAVEL_GRPO_REAL_USERBENCH_SMOKE=1 after editable UserBench install",
)
def test_real_userbench_without_network(monkeypatch):
    travel_env = pytest.importorskip("travelgym.env.travel_env")
    task_path = ROOT / "environments/UserBench/travelgym/data/travelgym_data_22.json"
    tasks = json.loads(task_path.read_text(encoding="utf-8"))
    task_id = next(iter(tasks.values()))["id"]

    async def fake_async_evaluator(*args, **kwargs):
        return "offline evaluator feedback", [], 0.2

    monkeypatch.setattr(travel_env, "async_evaluate_action", fake_async_evaluator)
    _reset_user_simulator_binding_for_tests()
    runtime = UserSimulatorRuntime(
        SimulatorRole.TRAIN,
        "offline-model",
        "http://127.0.0.1:9/v1",
        "offline-key",
    )
    wrapper = UserBenchWrapper(task_id, runtime)
    try:
        wrapper.reset()
        result = asyncio.run(
            wrapper.astep(
                {"thought": "offline smoke", "choice": "action", "content": "hello"}
            )
        )
        assert result.observation.feedback == "offline evaluator feedback"
        assert result.reward == 0.2
    finally:
        wrapper.close()
        _reset_user_simulator_binding_for_tests()
