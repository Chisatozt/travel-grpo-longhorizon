from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "recovery_sft_decision", ROOT / "scripts/eval/recovery_sft_decision.py"
)
assert SPEC is not None and SPEC.loader is not None
decision = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(decision)


def test_followup_gate_is_not_vacuously_passed(tmp_path: Path) -> None:
    (tmp_path / "A/closed_loop").mkdir(parents=True)
    (tmp_path / "B/closed_loop").mkdir(parents=True)
    row = {
        "task_id": "t",
        "visible_transcript": [
            {"role": "assistant", "tool_calls": [{"function": {"arguments": '{"choice":"action"}'}}]}
        ],
    }
    (tmp_path / "B/closed_loop/01.json").write_text(__import__("json").dumps(row), encoding="utf-8")
    value = decision.closed_loop_followup(tmp_path)["B"]
    assert value["answer_calls"] == 0
    assert value["answer_followup_gate_observable"] is False


def test_leakage_scan_detects_forbidden_public_text(tmp_path: Path) -> None:
    (tmp_path / "A/probes").mkdir(parents=True)
    (tmp_path / "B/probes").mkdir(parents=True)
    (tmp_path / "A/closed_loop").mkdir(parents=True)
    (tmp_path / "B/closed_loop").mkdir(parents=True)
    (tmp_path / "B/probes/x.json").write_text(
        '{"action":{"content":"remaining_preference_ids"}}', encoding="utf-8"
    )
    value = decision.leakage_scan(tmp_path)
    assert value["hit_count"] == 1


def test_gate_rows_fail_known_recovery_boundaries() -> None:
    comparison = {
        "single_step": {
            "B": {
                "normal_search_result": {"answer_at_1": 1.0, "visible_id_only": 0.875},
                "preference_complete": {"clean_search_at_1": 2 / 24},
                "first_fallback": {"exact_query_repeat": 8 / 24},
                "second_fallback": {"same_aspect_search": 8 / 24},
            }
        },
        "closed_loop": {"B": {"max_steps": 0.25}},
    }
    followup = {"B": {"answer_followup_gate_observable": False, "answer_followed_by_action_or_search_rate_over_tasks": 0.0}}
    leakage = {"hit_count": 0}
    rows = decision.gate_rows(comparison, followup, leakage)
    assert rows[0]["pass"] is True
    assert rows[1]["pass"] is False
    assert rows[2]["pass"] is False
    assert rows[3]["pass"] is False
    assert rows[4]["pass"] is False
    assert rows[5]["pass"] is False
    assert rows[6]["pass"] is True
    assert rows[7]["pass"] is True
