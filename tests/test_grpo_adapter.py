"""Offline tests for provider-neutral portions of the veRL adapter."""

import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from travel_grpo.envs.observation import UserBenchObservation, UserBenchStepResult
from travel_grpo.envs.reward import TravelRewardTask, UserBenchRewardSnapshot
from travel_grpo.envs.userbench_context import (
    UserBenchSessionState,
    clear_current_session,
    set_current_session,
)
from travel_grpo.training.grpo.adapter.agent_loop import (
    finalize_actor_stop,
    reject_parallel_tool_calls,
    select_post_tool_state,
    session_requests_termination,
)
from travel_grpo.training.grpo.adapter.session import (
    build_rollout_extra_info,
    calculate_current_session_score,
    validate_rollout_extra_info,
)
from travel_grpo.training.grpo.adapter.tools import execute_userbench_action

ROOT = Path(__file__).resolve().parents[1]


class FakeWrapper:
    def __init__(self, task_id, result, snapshot=None):
        self.task_id = task_id
        self.result = result
        self.calls = 0
        self.closed = False
        self.snapshot = snapshot

    async def astep(self, action):
        self.calls += 1
        return self.result

    def reward_snapshot(self):
        return self.snapshot

    def close(self):
        self.closed = True


def test_rollout_extra_info_duplicates_and_validates_task_id():
    extra = build_rollout_extra_info("task-7")
    assert validate_rollout_extra_info(extra) == "task-7"
    extra["tools_kwargs"]["interact_with_env"]["create_kwargs"]["id"] = "other"
    with pytest.raises(ValueError, match="does not match"):
        validate_rollout_extra_info(extra)


def test_tool_returns_zero_tool_reward_and_terminal_reward_v2():
    async def scenario():
        observation = UserBenchObservation("accepted", 1, True, 0.8, {})
        before = UserBenchRewardSnapshot(frozenset(), 1, 0, frozenset(), frozenset())
        after = UserBenchRewardSnapshot(frozenset(), 1, 0, frozenset(), frozenset({"F"}))
        wrapper = FakeWrapper(
            "task-1", UserBenchStepResult("task-1", observation, 0.8, True, False, {}), after
        )
        task = TravelRewardTask(
            "task-1", ("flight",), {"flight": "F1"},
            {"flight": frozenset({"F1"})}, {"flight": frozenset({"P1"})}
        )
        state = UserBenchSessionState(
            "request-1", "task-1", wrapper, reward_task=task, reward_snapshot=before,
            active_preference_ids={"P1"}, searched_aspects={"flight"}
        )
        set_current_session(state)
        try:
            result = await execute_userbench_action(
                {"thought": "done", "choice": "answer", "content": "F1"}
            )
            assert result.text == "accepted"
            assert result.reward == 0.0
            assert result.metadata["raw_reward"] == 0.8
            assert result.metadata["raw_cumulative_reward"] == 0.8
            assert state.rewards.total == 0.8
            assert calculate_current_session_score() == 1.0
            assert session_requests_termination()
            assert select_post_tool_state("generate", "terminate") == "terminate"
        finally:
            clear_current_session()

    asyncio.run(scenario())


def test_malformed_tool_call_returns_stable_error_without_stepping_env():
    async def scenario():
        observation = UserBenchObservation("unused", 1, False, 0.0, {})
        wrapper = FakeWrapper(
            "task-1", UserBenchStepResult("task-1", observation, 0.0, False, False, {})
        )
        set_current_session(UserBenchSessionState("request-1", "task-1", wrapper))
        try:
            result = await execute_userbench_action(
                {"thought": "x", "choice": "search", "content": "[answer] x"}
            )
            assert result.text.startswith("Error: invalid interact_with_env call:")
            assert result.reward == 0.0
            assert wrapper.calls == 0
        finally:
            clear_current_session()

    asyncio.run(scenario())


def test_parallel_tool_calls_terminate_before_environment_step():
    observation = UserBenchObservation("unused", 0, False, 0.0, {})
    wrapper = FakeWrapper(
        "task-1", UserBenchStepResult("task-1", observation, 0.0, False, False, {})
    )
    session = UserBenchSessionState("request-1", "task-1", wrapper)
    set_current_session(session)
    try:
        assert reject_parallel_tool_calls(
            SimpleNamespace(tool_calls=[object(), object()])
        )
        assert wrapper.calls == 0
        assert session.protocol_error == "parallel_tool_calls"
        assert session.termination_reason == "parallel_tool_calls"
        assert session.invalid_actions == 1
        assert session.parallel_tool_calls is True
        assert session.metrics()["termination_reason"] == "parallel_tool_calls"
    finally:
        clear_current_session()


def test_no_tool_output_is_a_penalized_protocol_error():
    session = SimpleNamespace(
        num_tool_calls=0,
        protocol_error=None,
        invalid_actions=0,
        termination_reason=None,
        done=False,
    )
    finalize_actor_stop(session)
    assert session.protocol_error == "no_tool_output"
    assert session.termination_reason == "no_tool_output"
    assert session.invalid_actions == 1


def test_verl_yaml_paths_and_simulator_roles_are_consistent():
    config_root = ROOT / "configs"
    grpo = yaml.safe_load(
        (config_root / "train/grpo/vanilla_grpo.yaml").read_text(encoding="utf-8")
    )
    multi_turn = grpo["actor_rollout_ref"]["rollout"]["multi_turn"]
    assert grpo["data"]["apply_chat_template_kwargs"]["enable_thinking"] is False
    assert multi_turn["max_parallel_calls"] == 1
    assert multi_turn["tool_config_path"] == (
        "configs/tool_config/userbench_tools.yaml"
    )

    interaction = yaml.safe_load(
        (config_root / "interaction_config/userbench_interaction.yaml").read_text(
            encoding="utf-8"
        )
    )["interaction"][0]
    module_name, class_name = interaction["class_name"].rsplit(".", 1)
    assert hasattr(importlib.import_module(module_name), class_name)

    environment = yaml.safe_load(
        (config_root / "interaction_config/userbench.yaml").read_text(encoding="utf-8")
    )
    assert environment["reward"]["version"] == "userbench-travel-reward-v2"
    assert environment["reward"]["terminal_only"] is True

    loop = yaml.safe_load(
        (config_root / "interaction_config/agent_loop.yaml").read_text(encoding="utf-8")
    )[0]
    module_name, class_name = loop["_target_"].rsplit(".", 1)
    assert hasattr(importlib.import_module(module_name), class_name)

    train = yaml.safe_load(
        (config_root / "interaction_config/simulator_train.yaml").read_text(
            encoding="utf-8"
        )
    )
    evaluation = yaml.safe_load(
        (config_root / "interaction_config/simulator_eval.yaml").read_text(
            encoding="utf-8"
        )
    )
    collection = yaml.safe_load(
        (config_root / "interaction_config/simulator_collection.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert train["simulator"]["role"] == "grpo"
    assert train["simulator"]["model_env"] == "GRPO_USER_SIM_MODEL"
    assert collection["simulator"]["role"] == "collection"
    assert collection["simulator"]["model_env"] == "COLLECTION_USER_SIM_MODEL"
    assert evaluation["simulator"]["role"] == "eval"
    assert grpo["actor_rollout_ref"]["model"]["path"] == (
        "${oc.env:GRPO_ACTOR_MODEL,Qwen/Qwen3.5-2B}"
    )
