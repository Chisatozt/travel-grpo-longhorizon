# [项目注释] 模块：测试模块，负责验证 test_evaluation_pipeline 的行为契约。
# [项目注释] 该文件的公共边界、输入输出和调用关系由下方实现及架构文档共同定义。

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from travel_grpo.evaluation.checkpoint_selection import select_checkpoint
from travel_grpo.evaluation.artifacts import attach_attempt_history
from travel_grpo.evaluation.contracts import build_contract, build_subset_contract
from travel_grpo.evaluation.comparison import compare_stage_results
from travel_grpo.evaluation.summary import summarize_results
from travel_grpo.evaluation.validation import summarize_validation_rows
from travel_grpo.models.vllm_policy import ActorRuntime

ROOT = Path(__file__).resolve().parents[1]


# [项目注释] 功能：`result`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。
# [项目注释] 输入：`task_id`: str；`reward`: float；`valid`: bool；`composition`: str。
# [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
def result(task_id: str, reward: float = 0.5, *, valid: bool = True, composition: str = "22"):
    return {
        "task_id": task_id,
        "composition": composition,
        "infrastructure_valid": valid,
        "actor_attempts": 2,
        "environment_steps": 2,
        "termination_reason": "environment_terminated" if valid else "api_failure",
        "reward": {
            "terminal_reward": reward,
            "reward_valid": valid,
            "quality_by_aspect": {"hotel": 1.0},
            "gold_itinerary": True,
            "correct_itinerary": True,
            "user_aligned_success": True,
            "completion_rate": 1.0,
            "correct_answer_rate": 1.0,
            "answer_submission_rate": 1.0,
            "active_preference_coverage": 1.0,
            "passive_preference_coverage": 0.0,
            "efficiency": 1.0,
            "policy_penalty": 0.0,
        },
    }


# [项目注释] 功能：`test_summary_uses_fixed_denominator_for_invalid_and_missing`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：summarize_results, result。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_summary_uses_fixed_denominator_for_invalid_and_missing():
    summary = summarize_results(
        [result("a", 1.0), result("b", valid=False)],
        expected_task_ids=("a", "b", "c"),
    )
    assert summary["denominator"] == 3
    assert summary["valid_tasks"] == 1
    assert summary["fixed_denominator"]["terminal_reward"] == 1 / 3
    assert summary["valid_only"]["terminal_reward"] == 1.0


# [项目注释] 功能：`test_summary_keeps_metric_schema_when_every_task_is_invalid`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：summarize_results, result。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_summary_keeps_metric_schema_when_every_task_is_invalid():
    summary = summarize_results(
        [result("a", valid=False)], expected_task_ids=("a", "b")
    )
    assert summary["fixed_denominator"]["terminal_reward"] == 0.0
    assert summary["fixed_denominator"]["micro_avg"] == 0.0
    assert summary["valid_only"]["terminal_reward"] == 0.0


# [项目注释] 功能：`test_actor_runtime_rejects_wrong_frozen_stage_model`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：ActorRuntime, require_model, raises。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_actor_runtime_rejects_wrong_frozen_stage_model():
    runtime = ActorRuntime(
        model="Qwen/Qwen3.5-2B",
        base_url="http://127.0.0.1:8000/v1",
        api_key="local",
    )
    runtime.require_model("Qwen/Qwen3.5-2B")
    with pytest.raises(ValueError, match="does not match"):
        runtime.require_model("outputs/models/sft-merged")


# [项目注释] 功能：`test_explicit_infrastructure_retry_preserves_attempt_diagnostics`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：result, attach_attempt_history, len。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_explicit_infrastructure_retry_preserves_attempt_diagnostics():
    first = result("a", valid=False)
    first["termination_reason"] = "actor_infrastructure_failure"
    first["reward"]["infrastructure_errors"] = ["actor_TimeoutError"]
    attach_attempt_history(first)
    second = result("a", valid=True)
    attach_attempt_history(second, first)
    assert len(second["attempt_history"]) == 2
    assert second["attempt_history"][0]["infrastructure_errors"] == [
        "actor_TimeoutError"
    ]
    assert second["attempt_history"][1]["infrastructure_valid"] is True


# [项目注释] 功能：`summary`：计算奖励、指标或聚合统计，供训练、评测或报告使用。
# [项目注释] 输入：`correct`；`aligned`；`reward`；`valid`；`efficiency`。
# [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
def summary(correct=0.8, aligned=0.7, reward=0.4, valid=1.0, efficiency=0.5):
    return {
        "denominator": 132,
        "infrastructure_valid_rate": valid,
        "fixed_denominator": {
            "correct_itinerary": correct,
            "user_aligned_success": aligned,
            "terminal_reward": reward,
            "efficiency": efficiency,
        },
    }


# [项目注释] 功能：`test_checkpoint_selection_applies_gates_and_tiebreaks`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：select_checkpoint, summary。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_checkpoint_selection_applies_gates_and_tiebreaks():
    selected = select_checkpoint(
        [
            {"step": 50, "summary_path": "step_50", "summary": summary(reward=0.5)},
            {"step": 100, "summary_path": "step_100", "summary": summary(reward=0.5)},
            {"step": 150, "summary_path": "step_150", "summary": summary(valid=0.97, reward=0.9)},
        ],
        summary(),
    )
    assert selected["passed"] is True
    assert selected["selected_step"] == 50
    assert selected["candidates"][2]["rejection_reasons"] == ["valid_rate_below_0.98"]


# [项目注释] 功能：`test_validation_directory_is_summarized_and_selected_atomically`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：importorskip, write_table, mkdir, run。
# [项目注释] 输入：`tmp_path`。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_validation_directory_is_summarized_and_selected_atomically(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    tasks = [
        {"task_id": f"task-{index}", "composition": "22"}
        for index in range(132)
    ]
    tasks_path = tmp_path / "tasks.parquet"
    pq.write_table(pa.Table.from_pylist(tasks), tasks_path)
    validation_dir = tmp_path / "run" / "validation_rollouts"
    validation_dir.mkdir(parents=True)

    # [项目注释] 功能：`rows`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。
    # [项目注释] 输入：`reward`。
    # [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
    def rows(reward):
        return [
            {
                "task_id": task["task_id"],
                "reward_valid": True,
                "terminal_reward": reward,
                "quality_by_aspect": {"hotel": 1.0},
                "correct_itinerary": True,
                "gold_itinerary": True,
                "user_aligned_success": True,
                "completion_rate": 1.0,
                "efficiency": 1.0,
            }
            for task in tasks
        ]

    assert summarize_validation_rows(rows(0.4), tasks)["denominator"] == 132
    for step, reward in ((0, 0.4), (50, 0.5)):
        (validation_dir / f"{step}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows(reward)),
            encoding="utf-8",
        )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/eval/select_checkpoint.py"),
            "--validation-dir",
            str(validation_dir),
            "--tasks",
            str(tasks_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    selected = json.loads((tmp_path / "run/checkpoint_selection.json").read_text())
    assert selected["passed"] is True
    assert selected["selected_step"] == 50
    assert (tmp_path / "run/validation_summaries/step_0.summary.json").is_file()


# [项目注释] 功能：`test_subset_contract_and_comparison_use_subset_denominator`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：build_subset_contract, list, compare_stage_results, raises。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_subset_contract_and_comparison_use_subset_denominator():
    records = [
        {"task_id": "a", "composition": "22", "source_split": "test"},
        {"task_id": "b", "composition": "33", "source_split": "test"},
    ]
    with pytest.raises(ValueError, match="471"):
        build_contract(records)
    subset = build_subset_contract(records)
    task_ids = list(subset.task_ids)
    stages = {
        stage: {
            "contract": {
                "contract_hash": subset.contract_hash,
                "task_ids": task_ids,
                "evaluation_mode": "subset",
            },
            "results": [result(task_id) for task_id in task_ids],
        }
        for stage in ("baseline", "sft", "grpo")
    }
    comparison = compare_stage_results(stages, allow_subset=True)
    assert comparison["evaluation_mode"] == "subset"
    assert comparison["expected_tasks"] == 2


# [项目注释] 功能：`test_formal_comparison_requires_complete_matching_contract`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：compare_stage_results, range, result, enumerate。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_formal_comparison_requires_complete_matching_contract():
    task_ids = [f"t{index}" for index in range(471)]
    stages = {
        stage: {
            "contract": {"contract_hash": "same", "task_ids": task_ids},
            "results": [result(task_id, reward=index / 1000) for index, task_id in enumerate(task_ids)],
        }
        for stage in ("baseline", "sft", "grpo")
    }
    value = compare_stage_results(stages)
    assert value["contract_hash"] == "same"
    assert value["paired_deltas"]["baseline_to_sft"]["terminal_reward"] == 0.0
