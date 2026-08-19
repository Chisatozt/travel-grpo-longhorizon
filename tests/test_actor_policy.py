"""CPU contracts for the shared production Actor policy."""

from __future__ import annotations

from travel_grpo.prompts.actor_policy import (
    ACTOR_RUNTIME_POLICY,
    ACTOR_RUNTIME_POLICY_MARKER,
    ACTOR_RUNTIME_POLICY_VERSION,
    TEACHER_GENERATION_INSTRUCTION,
    ensure_actor_runtime_policy,
    ensure_teacher_generation_messages,
    strip_teacher_generation_instruction,
)
from travel_grpo.training.sft_collection import (
    TEACHER_ACTOR_POLICY,
    TeacherTrajectory,
    _prepare_teacher_messages,
)


# [项目注释] 功能：`_prompt`：把协议/状态数据转换为模型、用户或日志可见的文本表示。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `list[dict[str, str]]`；具体值由各分支决定。
def _prompt() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Use interact_with_env."},
        {"role": "user", "content": "Plan the trip."},
    ]


# [项目注释] 功能：`test_runtime_policy_contains_the_production_behavior_contract`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_runtime_policy_contains_the_production_behavior_contract() -> None:
    for phrase in (
        "already answered a preference",
        "explicitly has no preference",
        "immediately issue one search",
        "normal candidate list",
        "exactly one option ID",
        "materially rewritten query retry",
        "second fallback",
        "extra services, preferences",
        "public control state",
        "Opening dispatch is mandatory",
        "Current public aspect",
        "target aspect",
        "later aspect early",
        "hard allow-list",
    ):
        assert phrase in ACTOR_RUNTIME_POLICY
    assert ACTOR_RUNTIME_POLICY_VERSION in ACTOR_RUNTIME_POLICY
    # Compatibility import remains the same Actor-facing policy, not the
    # Teacher-only generation instruction.
    assert TEACHER_ACTOR_POLICY == ACTOR_RUNTIME_POLICY
    assert TEACHER_GENERATION_INSTRUCTION not in ACTOR_RUNTIME_POLICY


# [项目注释] 功能：`test_runtime_policy_injection_is_idempotent_and_deduplicated`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：ensure_actor_runtime_policy, _prompt, count。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_runtime_policy_injection_is_idempotent_and_deduplicated() -> None:
    once = ensure_actor_runtime_policy(_prompt())
    twice = ensure_actor_runtime_policy(once)
    assert twice == once
    content = twice[0]["content"]
    assert content.count(ACTOR_RUNTIME_POLICY_MARKER) == 1
    assert content.count(ACTOR_RUNTIME_POLICY) == 1


# [项目注释] 功能：`test_teacher_instruction_is_request_only_and_removed_for_actor_messages`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：ensure_teacher_generation_messages, strip_teacher_generation_instruction,
# [项目注释]    _prepare_teacher_messages, _prompt。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_teacher_instruction_is_request_only_and_removed_for_actor_messages() -> None:
    teacher_messages = ensure_teacher_generation_messages(_prompt())
    teacher_content = teacher_messages[0]["content"]
    assert ACTOR_RUNTIME_POLICY in teacher_content
    assert TEACHER_GENERATION_INSTRUCTION in teacher_content

    actor_messages = strip_teacher_generation_instruction(teacher_messages)
    actor_content = actor_messages[0]["content"]
    assert ACTOR_RUNTIME_POLICY in actor_content
    assert TEACHER_GENERATION_INSTRUCTION not in actor_content

    prepared = _prepare_teacher_messages(_prompt())
    archived = strip_teacher_generation_instruction(prepared)
    assert TEACHER_GENERATION_INSTRUCTION in prepared[0]["content"]
    assert TEACHER_GENERATION_INSTRUCTION not in archived[0]["content"]
    assert ACTOR_RUNTIME_POLICY in archived[0]["content"]


# [项目注释] 功能：`test_sft_trajectory_records_actor_policy_version`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：TeacherTrajectory, tuple, to_record, _prompt。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_sft_trajectory_records_actor_policy_version() -> None:
    trajectory = TeacherTrajectory(
        task_id="hotel:2-1",
        composition="22",
        difficulty="easy",
        source_split="train",
        teacher_model="deepseek-v4-flash",
        simulator_model="deepseek-v4-flash",
        messages=tuple(_prompt()),
        step_rewards=(),
        terminated=False,
        truncated=True,
    )
    assert trajectory.to_record()["actor_policy_version"] == ACTOR_RUNTIME_POLICY_VERSION


# [项目注释] 功能：`test_grpo_rows_use_the_same_runtime_policy_without_hidden_labels`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：str, build_verl_records, count, resolve。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_grpo_rows_use_the_same_runtime_policy_without_hidden_labels() -> None:
    from pathlib import Path

    from travel_grpo.training.grpo.data import build_verl_records

    root = Path(__file__).resolve().parents[1]
    row = build_verl_records(root / "data/grpo/train.parquet", project_split="train")[0]
    assert row["extra_info"]["actor_policy_version"] == ACTOR_RUNTIME_POLICY_VERSION
    assert row["prompt"][0]["content"].count(ACTOR_RUNTIME_POLICY_MARKER) == 1
    assert ACTOR_RUNTIME_POLICY in row["prompt"][0]["content"]
    serialized = str(row)
    for secret in (
        "remaining_preference_ids",
        "correct_ids",
        "best_ids",
        "reward_snapshot",
        "reward delta",
        "reward_delta",
        "SECRET_HIDDEN_PREFERENCE_VALUE",
    ):
        assert secret not in serialized
