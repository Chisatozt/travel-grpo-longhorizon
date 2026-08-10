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
