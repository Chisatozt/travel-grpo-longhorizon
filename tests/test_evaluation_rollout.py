from __future__ import annotations

import asyncio
import json

from travel_grpo.envs.observation import UserBenchObservation, UserBenchStepResult
from travel_grpo.envs.userbench_interaction import (
    DEEPSEEK_V4_FLASH_MODEL,
    SimulatorRole,
    UserSimulatorRuntime,
)
from travel_grpo.envs.userbench_tools import UserBenchAction
from travel_grpo.evaluation.rollout import rollout_task
from travel_grpo.models.openai_compatible import TeacherToolCall


class _FakeWrapper:
    instances: list["_FakeWrapper"] = []

    def __init__(self, task_id, simulator, config, *, source_root=None):
        self.task_id = task_id
        self.calls = 0
        self.actions = []
        self.closed = False
        self.__class__.instances.append(self)

    async def areset(self):
        return UserBenchObservation(
            "Initial public travel request.", 0, False, 0.0, {}
        )

    def reward_task(self):
        return None

    def reward_snapshot(self):
        return None

    async def astep(self, action):
        self.calls += 1
        self.actions.append(action)
        if action.choice.value == "search":
            feedback = "Here are all the options: H1, H2"
        else:
            feedback = "Please provide the hotel preference."
        return UserBenchStepResult(
            self.task_id,
            UserBenchObservation(feedback, self.calls, False, 0.0, {}),
            0.0,
            False,
            False,
            {},
        )

    def close(self):
        self.closed = True


class _FakeActor:
    def __init__(self):
        self.messages = []
        self.calls = 0
        self._actions = [
            {"thought": "ask", "choice": "action", "content": "hotel preference"},
            {"thought": "search", "choice": "search", "content": "hotel Paris"},
            {"thought": "wrong phase", "choice": "action", "content": "hotel rating"},
            {"thought": "answer", "choice": "answer", "content": "H1"},
        ]

    async def generate_action(self, messages):
        self.messages.append([dict(message) for message in messages])
        parameters = self._actions[min(self.calls, len(self._actions) - 1)]
        self.calls += 1
        return TeacherToolCall(
            call_id=f"fake-{self.calls}",
            action=UserBenchAction.from_parameters(parameters),
        )


SIMULATOR = UserSimulatorRuntime(
    role=SimulatorRole.EVAL,
    model=DEEPSEEK_V4_FLASH_MODEL,
    base_url="http://eval.invalid/v1",
    api_key="test-key",
)
TASK = {
    "task_id": "task-1",
    "composition": "22",
    "prompt": [
        {"role": "system", "content": "You are a travel agent."},
        {"role": "user", "content": "I need a hotel in Paris."},
    ],
}


def test_guard_rejects_before_simulator_and_renders_public_feedback():
    actor = _FakeActor()
    _FakeWrapper.instances.clear()

    result = asyncio.run(
        rollout_task(
            TASK,
            actor=actor,
            simulator=SIMULATOR,
            wrapper_factory=_FakeWrapper,
            public_control_enabled=True,
        )
    )

    wrapper = _FakeWrapper.instances[-1]
    assert wrapper.calls == 3  # action + search + answer; rejected action never reached UserBench
    assert result["phase_guard_version"] == "public-control-v1"
    assert result["guard_rejections"] == 1
    assert result["guard_rejection_reasons"] == {
        "ANSWER_REQUIRED accepts choice=answer only": 1
    }
    transcript = json.dumps(result["visible_transcript"], ensure_ascii=False)
    assert "Current control state: ANSWER_REQUIRED" in transcript
    assert "Allowed next tool calls: answer (one visible option ID)" in transcript
    for forbidden in (
        "remaining_preference_ids",
        "correct_ids",
        "best_ids",
        "reward_snapshot",
        "reward delta",
    ):
        assert forbidden not in transcript
    assert wrapper.closed is True


def test_raw_rollout_remains_explicitly_available_for_ablation():
    actor = _FakeActor()
    _FakeWrapper.instances.clear()
    result = asyncio.run(
        rollout_task(
            TASK,
            actor=actor,
            simulator=SIMULATOR,
            wrapper_factory=_FakeWrapper,
            public_control_enabled=False,
        )
    )
    assert "phase_guard_version" not in result
    assert _FakeWrapper.instances[-1].calls == 20
