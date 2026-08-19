# [项目注释] 模块：测试模块，负责验证 test_recovery_targets 的行为契约。
# [项目注释] 该文件的公共边界、输入输出和调用关系由下方实现及架构文档共同定义。

from __future__ import annotations

import json
from pathlib import Path

from travel_grpo.data.recovery_targets import (
    TARGET_STATUS_ACCEPTED,
    TARGET_STATUS_EXCLUDED_EVALUATION,
    build_target_dataset,
    construct_target,
)
from travel_grpo.envs.userbench_tools import ActionChoice, UserBenchAction


# [项目注释] 功能：`_call`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：dumps, len。
# [项目注释] 输入：`choice`: str；`content`: str；`thought`: str。
# [项目注释] 输出：标注返回 `dict`；具体值由各分支决定。
def _call(choice: str, content: str, thought: str = "source") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call-{choice}-{len(content)}",
                "type": "function",
                "function": {
                    "name": "interact_with_env",
                    "arguments": json.dumps(
                        {"thought": thought, "choice": choice, "content": content}
                    ),
                },
            }
        ],
    }


# [项目注释] 功能：`_tool`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。
# [项目注释] 输入：`content`: str。
# [项目注释] 输出：标注返回 `dict`；具体值由各分支决定。
def _tool(content: str) -> dict:
    return {"role": "tool", "name": "interact_with_env", "content": content}


# [项目注释] 功能：`_context`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。
# [项目注释] 输入：`boundary_type`: str；`messages`: list[dict]；`state`: dict；`split`: str；`provenance`: list[dict]
# [项目注释]    | None。
# [项目注释] 输出：标注返回 `dict`；具体值由各分支决定。
def _context(
    boundary_type: str,
    messages: list[dict],
    state: dict,
    *,
    split: str = "sft_train",
    provenance: list[dict] | None = None,
) -> dict:
    return {
        "schema_version": "recovery-boundary-v1",
        "task_id": "apartment:2-1|rental_car:2-2",
        "boundary_type": boundary_type,
        "policy_version": "test",
        "messages": messages,
        "public_state_before": state,
        "target_assistant": None,
        "source_provenance": provenance or [],
        "quality_checks": {"public_state_only": True},
        "composition": "22",
        "project_split": split,
    }


# [项目注释] 功能：`test_accepted_teacher_search_and_answer_are_reused`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：write_text, _context, construct_target, loads。
# [项目注释] 输入：`tmp_path`: Path。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_accepted_teacher_search_and_answer_are_reused(tmp_path: Path) -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "I need an apartment."},
        _call("search", "Search for an apartment in Austin."),
        _tool('Here are all the options for <apartment>: {"id": "A1"} {"id": "A2"}'),
        _call("answer", "A1", "choose visible"),
        _tool("recorded"),
    ]
    source = tmp_path / "teacher.jsonl"
    source.write_text(json.dumps({"messages": messages}) + "\n", encoding="utf-8")
    base = messages[:4]
    state = {
        "current_aspect": "apartment",
        "recovery_mode": "ANSWER_REQUIRED",
        "fallback_count": 0,
        "visible_option_ids": ["A1", "A2"],
        "answered_aspects": [],
        "blocked_aspects": [],
        "search_attempts": 1,
        "normal_search_seen": True,
    }
    context = _context(
        "valid_search_to_answer",
        base,
        state,
        provenance=[{"source_kind": "teacher_accepted", "path": "teacher.jsonl", "line": 1}],
    )
    decision = construct_target(context, tmp_path)
    assert decision.status == TARGET_STATUS_ACCEPTED
    assert decision.target_assistant is not None
    arguments = json.loads(
        decision.target_assistant["tool_calls"][0]["function"]["arguments"]
    )
    assert arguments["choice"] == "answer"
    assert arguments["content"] == "A1"
    assert decision.target_assistant["content"] == ""


# [项目注释] 功能：`test_first_fallback_gets_one_substantive_same_aspect_retry`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：_context, construct_target, from_parameters, _call。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_first_fallback_gets_one_substantive_same_aspect_retry() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "I need an apartment."},
        _call("search", "Search for an apartment in Austin."),
        _tool("Currently the searching backend is experiencing some issues."),
    ]
    context = _context(
        "first_fallback",
        messages,
        {
            "current_aspect": "apartment",
            "recovery_mode": "SEARCH_RETRY_REQUIRED",
            "fallback_count": 1,
            "visible_option_ids": [],
            "answered_aspects": [],
            "blocked_aspects": [],
            "search_attempts": 1,
        },
    )
    decision = construct_target(context, ".")
    assert decision.status == TARGET_STATUS_ACCEPTED
    action = UserBenchAction.from_parameters(
        json.loads(decision.target_assistant["tool_calls"][0]["function"]["arguments"])
    )
    assert action.choice is ActionChoice.SEARCH
    assert action.content != "Search for an apartment in Austin."
    assert "apartment" in action.content.casefold()


# [项目注释] 功能：`test_second_fallback_switches_to_next_public_aspect`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：_context, construct_target, loads, _call。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_second_fallback_switches_to_next_public_aspect() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "I need an apartment and a rental car."},
        _call("search", "Search for an apartment in Austin."),
        _tool("fallback"),
        _call("search", "Revised search for an apartment: Austin."),
        _tool("fallback"),
    ]
    context = _context(
        "second_fallback",
        messages,
        {
            "current_aspect": "apartment",
            "recovery_mode": "SWITCH_ASPECT_REQUIRED",
            "fallback_count": 2,
            "visible_option_ids": [],
            "answered_aspects": [],
            "blocked_aspects": ["apartment"],
            "search_attempts": 2,
        },
    )
    decision = construct_target(context, ".")
    assert decision.status == TARGET_STATUS_ACCEPTED
    arguments = json.loads(
        decision.target_assistant["tool_calls"][0]["function"]["arguments"]
    )
    assert arguments["choice"] == "action"
    assert "rental car" in arguments["content"].casefold()
    assert "apartment" not in arguments["content"].casefold()


# [项目注释] 功能：`test_pending_visible_options_is_quarantined_instead_of_guessing`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：_context, construct_target。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_pending_visible_options_is_quarantined_instead_of_guessing() -> None:
    context = _context(
        "visible_options_pending_answer",
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "I need an apartment."},
        ],
        {
            "current_aspect": "apartment",
            "recovery_mode": "ANSWER_REQUIRED",
            "fallback_count": 0,
            "visible_option_ids": ["A1", "A2"],
            "answered_aspects": [],
            "blocked_aspects": [],
        },
    )
    decision = construct_target(context, ".")
    assert decision.target_assistant is None
    assert "answer_id_not_determinable_without_hidden_correctness" in decision.rejection_reasons


# [项目注释] 功能：`test_evaluation_targets_are_excluded_from_train_and_hidden_keys_absent`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：_context, build_target_dataset, len。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_evaluation_targets_are_excluded_from_train_and_hidden_keys_absent() -> None:
    context = _context(
        "repeated_no_progress_action",
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "I need an apartment."},
        ],
        {
            "current_aspect": "apartment",
            "recovery_mode": "SEARCH_REQUIRED",
            "fallback_count": 0,
            "visible_option_ids": [],
            "answered_aspects": [],
            "blocked_aspects": [],
        },
        split="evaluation",
    )
    train, validation, rejected, manifest = build_target_dataset([context], ".")
    assert not train and not validation
    assert len(rejected) == 1
    assert rejected[0]["target_status"] == TARGET_STATUS_EXCLUDED_EVALUATION
    assert manifest["split_checks"]["evaluation_in_train"] == 0
    assert manifest["quality_checks"]["target_hidden_key_hits"] == 0
