# [项目注释] 模块：测试模块，负责验证 test_inference_gate 的行为契约。
# [项目注释] 该文件的公共边界、输入输出和调用关系由下方实现及架构文档共同定义。

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "inference_gate", ROOT / "scripts/eval/inference_gate.py"
)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


# [项目注释] 功能：`_args`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：type。
# [项目注释] 输入：`tmp_path`: Path。
# [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
def _args(tmp_path: Path):
    return type(
        "Args",
        (),
        {
            "boundary_file": ROOT / "outputs/recovery_boundaries/recovery-boundary-v1/contexts.jsonl",
            "dataset": ROOT / "data/evaluation/tasks.parquet",
            "model": "outputs/models/sft-merged",
            "output": tmp_path,
            "max_tokens": 4096,
        },
    )()


@pytest.mark.skipif(
    not (ROOT / "outputs/recovery_boundaries/recovery-boundary-v1/contexts.jsonl").exists(),
    reason="derived boundary fixture is not present",
)
# [项目注释] 功能：`test_fixed_manifest_counts_and_task_ids`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。 主要协作调用：skipif,
# [项目注释]    build_manifest, _args, len。
# [项目注释] 输入：`tmp_path`: Path。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_fixed_manifest_counts_and_task_ids(tmp_path: Path) -> None:
    manifest, samples, tasks = gate.build_manifest(_args(tmp_path))
    assert {key: len(value) for key, value in samples.items()} == gate.FIXED_COUNTS
    assert len(tasks) == 8
    assert manifest["actor_policy_version"] == gate.ACTOR_RUNTIME_POLICY_VERSION
    assert manifest["inference_config"]["parameter_updates"] is False
    assert manifest["inference_config"]["grpo"] is False


@pytest.mark.skipif(
    not (ROOT / "outputs/recovery_boundaries/recovery-boundary-v1/contexts.jsonl").exists(),
    reason="derived boundary fixture is not present",
)
# [项目注释] 功能：`test_prompt_conditions_are_public_and_nonduplicating`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：skipif, build_manifest, items, _args。
# [项目注释] 输入：`tmp_path`: Path。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_prompt_conditions_are_public_and_nonduplicating(tmp_path: Path) -> None:
    _, samples, _ = gate.build_manifest(_args(tmp_path))
    for category, records in samples.items():
        for record in records:
            prompt_a, _ = gate._public_messages(record, "A")
            prompt_b, _ = gate._public_messages(record, "B")
            assert gate.ACTOR_RUNTIME_POLICY not in prompt_a[0]["content"]
            assert prompt_b[0]["content"].count(gate.ACTOR_RUNTIME_POLICY_MARKER) == 1
            text = json.dumps(prompt_b, ensure_ascii=False).casefold()
            for forbidden in (
                "remaining_preference_ids",
                "correct_ids",
                "best_ids",
                "reward_snapshot",
                "reward delta",
                "hidden preference",
            ):
                assert forbidden not in text, (category, forbidden)


@pytest.mark.skipif(
    not (ROOT / "outputs/recovery_boundaries/recovery-boundary-v1/contexts.jsonl").exists(),
    reason="derived boundary fixture is not present",
)
# [项目注释] 功能：`test_normal_result_probes_are_answer_required_with_visible_ids`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：skipif, load_boundary_records, choose_probe_records, len。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_normal_result_probes_are_answer_required_with_visible_ids() -> None:
    records = gate.load_boundary_records(gate.BOUNDARY_FILE)
    selected = gate.choose_probe_records(
        records,
        boundary_type="valid_search_to_answer",
        count=gate.FIXED_COUNTS["normal_search_result"],
    )
    assert len(selected) == gate.FIXED_COUNTS["normal_search_result"]
    for record in selected:
        _, payload = gate._public_messages(record, "B")
        assert payload["recovery_mode"] == "ANSWER_REQUIRED"
        assert payload["visible_option_ids"]


@pytest.mark.skipif(
    not (ROOT / "data/evaluation/tasks.parquet").exists(),
    reason="frozen evaluation fixture is not present",
)
# [项目注释] 功能：`test_manifest_supports_32_task_closed_loop_validation`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：skipif, _args, build_manifest, len。
# [项目注释] 输入：`tmp_path`: Path。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_manifest_supports_32_task_closed_loop_validation(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.closed_loop_task_count = 32
    manifest, _, tasks = gate.build_manifest(args)
    assert len(tasks) == 32
    assert manifest["closed_loop_task_count"] == 32
    assert set(manifest["closed_loop_compositions"]) <= {"22", "33", "44"}


# [项目注释] 功能：`test_probe_metric_definitions`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。 主要协作调用：UserBenchAction,
# [项目注释]    _classify_probe。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def test_probe_metric_definitions() -> None:
    state = {"current_aspect": "restaurant", "visible_option_ids": ["R1"]}
    action = gate.UserBenchAction("answer", gate.ActionChoice.ANSWER, "R1")
    value = gate._classify_probe(
        "normal_search_result",
        {"public_state_before": state, "messages": []},
        action,
        state_payload=state,
        previous_actions=(),
    )
    assert value["answer_at_1"] is True
    assert value["visible_id_only"] is True
