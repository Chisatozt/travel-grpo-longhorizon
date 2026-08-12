"""CPU tests for Agent Loop prompt preparation and policy provenance."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from travel_grpo.prompts.actor_policy import (
    ACTOR_RUNTIME_POLICY,
    ACTOR_RUNTIME_POLICY_MARKER,
    ACTOR_RUNTIME_POLICY_VERSION,
)
from travel_grpo.training.grpo.adapter.agent_loop import (
    UserBenchAgentLoop,
    actor_policy_metadata,
    prepare_actor_prompt,
)


ROOT = Path(__file__).resolve().parents[1]


def prompt() -> list[dict[str, object]]:
    return [
        {"role": "system", "content": "Use interact_with_env."},
        {"role": "user", "content": "Plan the trip."},
    ]


def test_agent_loop_injects_policy_into_the_system_message_only() -> None:
    prepared = prepare_actor_prompt(prompt())
    assert ACTOR_RUNTIME_POLICY in prepared[0]["content"]
    assert prepared[0]["content"].count(ACTOR_RUNTIME_POLICY_MARKER) == 1
    assert ACTOR_RUNTIME_POLICY not in prepared[1]["content"]


def test_agent_loop_policy_injection_is_idempotent() -> None:
    prepared = prepare_actor_prompt(prompt())
    repeated = prepare_actor_prompt(prepared)
    assert repeated == prepared
    assert repeated[0]["content"].count(ACTOR_RUNTIME_POLICY_MARKER) == 1


def test_agent_loop_requires_a_system_message_when_enabled() -> None:
    with pytest.raises(ValueError, match="system message"):
        prepare_actor_prompt([{"role": "user", "content": "Plan."}])


def test_policy_can_be_disabled_without_changing_prompt_content() -> None:
    original = prompt()
    prepared = prepare_actor_prompt(
        original,
        actor_policy_enabled="false",
        actor_policy_version=ACTOR_RUNTIME_POLICY_VERSION,
    )
    assert prepared == original
    assert prepared is not original
    assert prepared[0] is not original[0]


def test_train_and_validation_use_the_same_default_policy() -> None:
    train_prompt = prepare_actor_prompt(
        prompt(), actor_policy_enabled=True, actor_policy_version=ACTOR_RUNTIME_POLICY_VERSION
    )
    validation_prompt = prepare_actor_prompt(
        prompt(), actor_policy_enabled=True, actor_policy_version=ACTOR_RUNTIME_POLICY_VERSION
    )
    assert train_prompt == validation_prompt


def test_raw_prompt_is_never_modified_in_place() -> None:
    original = prompt()
    snapshot = [dict(message) for message in original]
    prepare_actor_prompt(original)
    assert original == snapshot
    assert ACTOR_RUNTIME_POLICY not in original[0]["content"]


def test_policy_version_is_recorded_without_hidden_state() -> None:
    assert actor_policy_metadata(
        actor_policy_enabled=True,
        actor_policy_version=ACTOR_RUNTIME_POLICY_VERSION,
    ) == {
        "actor_policy_enabled": True,
        "actor_policy_version": ACTOR_RUNTIME_POLICY_VERSION,
    }
    assert actor_policy_metadata(
        actor_policy_enabled=False,
        actor_policy_version=ACTOR_RUNTIME_POLICY_VERSION,
    ) == {"actor_policy_enabled": False, "actor_policy_version": "disabled"}


def test_agent_loop_config_pins_one_default_for_train_and_validation() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/interaction_config/agent_loop.yaml").read_text(
            encoding="utf-8"
        )
    )[0]
    assert config["actor_policy_enabled"] == (
        "${oc.env:TRAVEL_GRPO_ACTOR_POLICY_ENABLED,true}"
    )
    assert config["actor_policy_version"] == (
        "${oc.env:TRAVEL_GRPO_ACTOR_POLICY_VERSION,actor-runtime-v2}"
    )
    assert config["turn_credit_mode"] == (
        "${oc.env:TRAVEL_GRPO_TURN_CREDIT_MODE,off}"
    )
    assert config["turn_credit_config_json"] == (
        '${oc.env:TRAVEL_GRPO_TURN_CREDIT_CONFIG_JSON,""}'
    )



def test_agent_loop_run_passes_private_policy_prompt_to_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the project run boundary without constructing veRL or a model."""

    base = UserBenchAgentLoop.__mro__[1]
    if not hasattr(base, "run"):
        pytest.skip("veRL ToolAgentLoop is unavailable in the core-only test environment")

    seen: list[dict] = []

    async def fake_parent_run(self, sampling_params, **kwargs):
        seen.append(kwargs)
        # Simulate a parent that mutates its private prompt while tokenizing.
        kwargs["raw_prompt"][0]["content"] += " parent mutation"
        return SimpleNamespace(extra_fields={}, reward_score=None)

    class FakeRuntime:
        async def astart_session(self, task_id):
            return SimpleNamespace(
                task_id=task_id,
                actor_attempts=0,
                num_tool_calls=1,
                invalid_actions=0,
                exact_repeats=0,
                semantic_repeats=0,
                protocol_error=None,
                termination_reason="environment_terminated",
                done=True,
                infrastructure_errors=(),
                stall_recovery_enabled=False,
                stall_recovery_triggered=False,
                stall_recovery_used=False,
                stall_hard_truncated=False,
                consecutive_no_progress=0,
                max_consecutive_no_progress=0,
                answer_only_generation_started=False,
                visible_answer_options=(),
                metrics=lambda: {},
                reward_report=lambda: {
                    "terminal_reward": 0.0,
                    "reward_valid": True,
                    "quality_by_aspect": {},
                    "simulator_fallback_counts": {},
                    "correct_itinerary": False,
                    "gold_itinerary": False,
                    "user_aligned_success": False,
                    "completion_rate": 0.0,
                    "search_coverage": 0.0,
                    "active_preference_coverage": 0.0,
                    "passive_preference_coverage": 0.0,
                    "efficiency": 0.0,
                    "policy_penalty": 0.0,
                    "effective_steps": 1,
                    "reward_degraded": False,
                },
            )

    monkeypatch.setattr(base, "run", fake_parent_run)
    monkeypatch.setattr(
        "travel_grpo.training.grpo.adapter.agent_loop.finalize_actor_stop",
        lambda session: None,
    )
    monkeypatch.setattr(
        "travel_grpo.training.grpo.adapter.agent_loop.clear_current_session",
        lambda close=True: None,
    )
    loop = object.__new__(UserBenchAgentLoop)
    loop.actor_policy_enabled = True
    loop.actor_policy_version = ACTOR_RUNTIME_POLICY_VERSION
    loop.stall_recovery_enabled = False
    loop.userbench_runtime = FakeRuntime()
    raw = prompt()
    original_content = raw[0]["content"]

    import asyncio

    output = asyncio.run(
        loop.run(
            {"temperature": 0.7},
            raw_prompt=raw,
            extra_info={
                "task_id": "task-1",
                "tools_kwargs": {
                    "interact_with_env": {
                        "create_kwargs": {"env_name": "TravelGym", "id": "task-1"}
                    }
                },
            },
        )
    )

    assert ACTOR_RUNTIME_POLICY in seen[0]["raw_prompt"][0]["content"]
    assert "parent mutation" in seen[0]["raw_prompt"][0]["content"]
    assert raw[0]["content"] == original_content
    assert output.extra_fields["actor_policy_version"] == ACTOR_RUNTIME_POLICY_VERSION
    assert "turn_credit" not in output.extra_fields
