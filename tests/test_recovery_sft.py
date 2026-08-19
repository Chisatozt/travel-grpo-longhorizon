# [项目注释] 模块：测试模块，负责验证 test_recovery_sft 的行为契约。
# [项目注释] 该文件的公共边界、输入输出和调用关系由下方实现及架构文档共同定义。

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from travel_grpo.prompts.actor_policy import (
    ACTOR_RUNTIME_POLICY,
    ACTOR_RUNTIME_POLICY_MARKER,
    TEACHER_GENERATION_INSTRUCTION,
)
from travel_grpo.training.recovery_sft import (
    RECOVERY_SFT_SCHEMA_VERSION,
    RecoverySFTError,
    audit_rendered_record,
    build_recovery_sft_dataset,
    render_recovery_record,
)
from travel_grpo.training.sft_dataset import (
    IGNORE_INDEX,
    build_action_only_examples,
    load_tool_schema,
)

ROOT = Path(__file__).resolve().parents[1]
TOOL_CONFIG = ROOT / "configs/tool_config/userbench_tools.yaml"


# [项目注释] 类型：`FakeQwenTokenizer` 封装相关状态、协议或数据结构。类属性和方法共同维护其不变量。
class FakeQwenTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    padding_side = "right"

    # [项目注释] 功能：`apply_chat_template`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：dumps, str, ord。
    # [项目注释] 输入：`conversation`；`tools`；`tokenize`；`add_generation_prompt`；`enable_thinking`。
    # [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
    def apply_chat_template(
        self, conversation, *, tools, tokenize, add_generation_prompt, enable_thinking
    ):
        assert tokenize is True
        assert enable_thinking is False
        text = "<tools>" + json.dumps(tools, sort_keys=True) + "</tools>"
        for message in conversation:
            text += f"<{message['role']}>"
            if message.get("role") == "assistant":
                text += json.dumps(message.get("tool_calls", []), sort_keys=True)
            else:
                text += str(message.get("content", ""))
            text += f"</{message['role']}>"
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(value) + 3 for value in text]


# [项目注释] 功能：`_call`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：dumps。
# [项目注释] 输入：`call_id`: str；`choice`: str；`content`: str。
# [项目注释] 输出：标注返回 `dict`；具体值由各分支决定。
def _call(call_id: str, choice: str, content: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "interact_with_env",
                    "arguments": json.dumps(
                        {"thought": "public step", "choice": choice, "content": content}
                    ),
                },
            }
        ],
    }


# [项目注释] 功能：`_tool`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。
# [项目注释] 输入：`call_id`: str；`content`: str。
# [项目注释] 输出：标注返回 `dict`；具体值由各分支决定。
def _tool(call_id: str, content: str) -> dict:
    return {
        "role": "tool",
        "name": "interact_with_env",
        "tool_call_id": call_id,
        "content": content,
    }


# [项目注释] 功能：`_target_record`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：_call, _tool。
# [项目注释] 输入：`choice`: str；`content`: str；`split`: str。
# [项目注释] 输出：标注返回 `dict`；具体值由各分支决定。
def _target_record(*, choice: str = "answer", content: str = "H1", split: str = "grpo_train") -> dict:
    messages = [
        {"role": "system", "content": "Base production system."},
        {"role": "user", "content": "I need a hotel in Paris."},
        _call("call-search", "search", "Search for a hotel in Paris."),
        _tool("call-search", "Candidate options: H1 and H2."),
    ]
    target = _call("recovery-target", choice, content)
    return {
        "schema_version": "recovery-target-v1",
        "boundary_schema_version": "recovery-boundary-v1",
        "task_id": "hotel:2-1",
        "boundary_type": "valid_search_to_answer",
        "policy_version": "teacher-state-machine-v5",
        "composition": "22",
        "project_split": split,
        "messages": messages,
        "public_state_before": {
            "current_aspect": "hotel",
            "recovery_mode": "ANSWER_REQUIRED",
            "fallback_count": 0,
            "visible_option_ids": ["H1", "H2"],
            "answered_aspects": [],
            "blocked_aspects": [],
            "search_attempts": 1,
            "normal_search_seen": True,
            "consecutive_no_progress": 0,
        },
        "target_assistant": target,
        "target_status": "accepted",
        "source_provenance": [],
        "target_provenance": {"method": "accepted_teacher_or_source_prefix"},
    }


# [项目注释] 功能：`test_recovery_prompt_injects_policy_note_and_does_not_mutate_source`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：_target_record, deepcopy, render_recovery_record, all。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_recovery_prompt_injects_policy_note_and_does_not_mutate_source() -> None:
    source = _target_record()
    original = copy.deepcopy(source)
    rendered = render_recovery_record(source)
    assert source == original
    assert rendered["schema_version"] == RECOVERY_SFT_SCHEMA_VERSION
    system = rendered["messages"][0]["content"]
    assert system.count(ACTOR_RUNTIME_POLICY_MARKER) == 1
    assert system.count(ACTOR_RUNTIME_POLICY) == 1
    assert TEACHER_GENERATION_INSTRUCTION not in system
    assert rendered["control_note"] in rendered["messages"][-2]["content"]
    assert rendered["messages"][-1]["role"] == "assistant"
    assert rendered["messages"][-1].get("loss_mask") is not True
    assert all(
        message.get("loss_mask") is True
        for message in rendered["messages"]
        if message.get("role") == "assistant" and message is not rendered["messages"][-1]
    )


# [项目注释] 功能：`test_recovery_reuses_action_only_loss_mask_for_final_target`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：render_recovery_record, build_action_only_examples, next, all。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_recovery_reuses_action_only_loss_mask_for_final_target() -> None:
    rendered = render_recovery_record(_target_record())
    examples = build_action_only_examples(
        [rendered],
        FakeQwenTokenizer(),
        load_tool_schema(TOOL_CONFIG),
        max_sequence_length=100_000,
        record_format="recovery",
    )
    assert len(examples) == 1
    example = examples[0]
    assert example.label_tokens > 0
    first = next(index for index, value in enumerate(example.labels) if value != IGNORE_INDEX)
    assert all(value == IGNORE_INDEX for value in example.labels[:first])
    assert example.assistant_turn_index == 2


# [项目注释] 功能：`test_recovery_audit_checks_visible_answer_and_teacher_leakage`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：render_recovery_record, audit_rendered_record, dumps, deepcopy。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_recovery_audit_checks_visible_answer_and_teacher_leakage() -> None:
    rendered = render_recovery_record(_target_record())
    result = audit_rendered_record(
        rendered,
        FakeQwenTokenizer(),
        load_tool_schema(TOOL_CONFIG),
        max_sequence_length=100_000,
    )
    assert result.valid
    assert result.metrics["answer_id_visible"] is True
    rendered["target_assistant"]["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"thought": "x", "choice": "answer", "content": "H9"}
    )
    rendered["messages"][-1] = copy.deepcopy(rendered["target_assistant"])
    bad = audit_rendered_record(
        rendered,
        FakeQwenTokenizer(),
        load_tool_schema(TOOL_CONFIG),
        max_sequence_length=100_000,
    )
    assert not bad.valid
    assert "answer_id_not_visible" in bad.reasons


# [项目注释] 功能：`test_recovery_audit_rejects_overlong_without_truncation`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：render_recovery_record, audit_rendered_record, _target_record, FakeQwenTokenizer。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_recovery_audit_rejects_overlong_without_truncation() -> None:
    rendered = render_recovery_record(_target_record())
    rendered["messages"][-2]["content"] = "x" * 20_000
    result = audit_rendered_record(
        rendered,
        FakeQwenTokenizer(),
        load_tool_schema(TOOL_CONFIG),
        max_sequence_length=100,
    )
    assert not result.valid
    assert "overlong_sample" in result.reasons


# [项目注释] 功能：`test_recovery_requires_public_system_and_final_target`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：_target_record, raises, render_recovery_record。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_recovery_requires_public_system_and_final_target() -> None:
    source = _target_record()
    source["messages"] = source["messages"][1:]
    with pytest.raises(RecoverySFTError):
        render_recovery_record(source)


# [项目注释] 功能：`test_public_warning_history_is_kept_as_recovery_context`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：_target_record, render_recovery_record, audit_rendered_record, FakeQwenTokenizer。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_public_warning_history_is_kept_as_recovery_context() -> None:
    source = _target_record()
    source["messages"][3]["content"] = "Your question is too vague and general."
    rendered = render_recovery_record(source)
    result = audit_rendered_record(
        rendered,
        FakeQwenTokenizer(),
        load_tool_schema(TOOL_CONFIG),
        max_sequence_length=100_000,
    )
    assert result.valid


# [项目注释] 功能：`test_build_quarantines_exact_duplicates_but_allows_same_task_boundaries`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：_target_record, mkdir, write_text, build_recovery_sft_dataset。
# [项目注释] 输入：`tmp_path`: Path。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_build_quarantines_exact_duplicates_but_allows_same_task_boundaries(tmp_path: Path) -> None:
    source = _target_record(split="grpo_train")
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    (target_dir / "train.jsonl").write_text(
        json.dumps(source) + "\n" + json.dumps(copy.deepcopy(source)) + "\n",
        encoding="utf-8",
    )
    (target_dir / "validation.jsonl").write_text("", encoding="utf-8")
    paths, manifest = build_recovery_sft_dataset(
        target_dir,
        tmp_path / "out",
        FakeQwenTokenizer(),
        load_tool_schema(TOOL_CONFIG),
        max_sequence_length=100_000,
    )
    assert sum(1 for _ in paths["train"].open(encoding="utf-8")) == 1
    assert manifest["audit"]["counts"]["rejected"] == 1
    assert manifest["audit"]["rejection_reasons"] == {"duplicate_sample_hash": 1}
    assert manifest["audit"]["task_split_checks"]["train_validation_overlap"] == []
