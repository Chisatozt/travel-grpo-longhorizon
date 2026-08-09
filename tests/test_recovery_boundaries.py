from __future__ import annotations

import json
from pathlib import Path

from travel_grpo.data.recovery_boundaries import (
    BOUNDARY_TYPES,
    SCHEMA_VERSION,
    SourceSpec,
    extract_message_boundaries,
    extract_recovery_boundaries,
    load_task_split_map,
    normalize_actor_messages,
    parse_grpo_transcript,
    write_extraction,
)


def _call(thought: str, choice: str, content: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call-{len(thought)}-{choice}",
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


def _tool(content: str) -> dict:
    return {"role": "tool", "name": "interact_with_env", "content": content}


def _messages() -> list[dict]:
    return [
        {"role": "system", "content": "public system"},
        {
            "role": "user",
            "content": "I need an apartment and a rental car in Austin.",
        },
        _call("ask", "action", "Do you have a preferred apartment room?"),
        _tool("I do not have specific preference for that."),
        _call("search apartment", "search", "Search for an apartment in Austin."),
        _tool('Here are all the options for <apartment>: {"id": "A1"} {"id": "A2"}'),
        _call("answer", "answer", "A1"),
        _tool("Your chosen option is recorded. Continue with another aspect."),
        _call("repeat", "action", "Do you have a preferred rental car brand?"),
        _tool("I do not have specific preference for that."),
        _call("search car", "search", "Search for a rental car in Austin."),
        _tool("Currently the searching backend is experiencing some issues. Please try again later."),
        _call("retry", "search", "Search for a compact rental vehicle in Austin."),
        _tool("Currently the searching backend is experiencing some issues. Please try again later."),
    ]


def test_schema_boundaries_and_public_replay() -> None:
    records = extract_message_boundaries(
        task_id="apartment:2-1|rental_car:2-2",
        messages=_messages(),
        policy_version="teacher-v-test",
        provenance={"source_kind": "test", "path": "synthetic.jsonl"},
        composition="22",
        project_split="sft_train",
    )
    types = {record.boundary_type for record in records}
    assert {
        "explicit_no_preference",
        "preference_complete_to_search",
        "valid_search_to_answer",
        "first_fallback",
        "second_fallback",
    } <= types
    answer = next(record for record in records if record.boundary_type == "valid_search_to_answer")
    payload = answer.public_state_before
    assert payload["recovery_mode"] == "ANSWER_REQUIRED"
    assert payload["visible_option_ids"] == ["A1", "A2"]
    assert payload["fallback_count"] == 0

    first = next(record for record in records if record.boundary_type == "first_fallback")
    second = next(record for record in records if record.boundary_type == "second_fallback")
    assert first.public_state_before["fallback_count"] == 1
    assert first.public_state_before["recovery_mode"] == "SEARCH_RETRY_REQUIRED"
    assert second.public_state_before["fallback_count"] == 2
    assert second.public_state_before["blocked_aspects"] == ["rental_car"]


def test_grpo_transcript_parser_keeps_only_public_call_fields() -> None:
    input_text = "system\npublic system\nuser\nI need an apartment.\nassistant\n"
    output_text = (
        "<tool_call>\n<function=interact_with_env>\n"
        "<parameter=thought>ask</parameter>\n<parameter=choice>action</parameter>\n"
        "<parameter=content>Do you prefer a room?</parameter>\n"
        "</function>\n</tool_call>user\n<tool_response>No preference.</tool_response>"
    )
    messages = parse_grpo_transcript(input_text, output_text)
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "tool"]
    arguments = json.loads(messages[2]["tool_calls"][0]["function"]["arguments"])
    assert set(arguments) == {"thought", "choice", "content"}


def test_split_assignment_precedes_extraction_and_eval_is_not_training(tmp_path: Path) -> None:
    rows = {
        "data/sft/tasks_train.jsonl": {"task_id": "train-task", "composition": "22"},
        "data/sft/tasks_validation.jsonl": {"task_id": "holdout-task", "composition": "22"},
        "data/grpo/train.jsonl": {"task_id": "grpo-task", "composition": "33"},
        "data/grpo/validation.jsonl": {"task_id": "grpo-val", "composition": "33"},
        "data/evaluation/tasks.jsonl": {"task_id": "eval-task", "composition": "44"},
    }
    for relative, row in rows.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assignment = load_task_split_map(tmp_path)
    assert assignment["train-task"]["project_split"] == "sft_train"
    assert assignment["holdout-task"]["project_split"] == "sft_validation"
    assert assignment["eval-task"]["project_split"] == "evaluation"

    source = tmp_path / "outputs/evaluation/ab_prompt_test/A/01.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        json.dumps(
            {
                "task_id": "train-task",
                "composition": "22",
                "ab_condition": "A",
                "visible_transcript": _messages(),
            }
        ),
        encoding="utf-8",
    )
    records, manifest = extract_recovery_boundaries(
        tmp_path,
        sources=[SourceSpec(source, "ab_offline", "evaluation", True)],
    )
    assert records
    assert all(record["project_split"] == "evaluation" for record in records)
    assert all(not record["quality_checks"]["training_eligible"] for record in records)
    assert manifest["split_checks"]["evaluation_contexts_marked_training"] == 0
    assert manifest["split_checks"]["sample_level_random_split"] is False


def test_normalizer_drops_record_level_hidden_fields() -> None:
    messages = normalize_actor_messages(
        [
            {
                "role": "assistant",
                "content": "visible",
                "reward_snapshot": {"best_ids": ["A1"]},
                "tool_calls": [],
            },
            {"role": "tool", "content": "visible feedback", "correct_ids": ["A1"]},
        ]
    )
    assert messages == [
        {"role": "assistant", "content": "visible"},
        {"role": "tool", "content": "visible feedback"},
    ]


def test_write_extraction_manifest_and_targets_deferred(tmp_path: Path) -> None:
    records = extract_message_boundaries(
        task_id="apartment:2-1",
        messages=_messages()[:6],
        policy_version="v1",
        provenance={"source_kind": "test", "path": "synthetic"},
        composition="22",
        project_split="sft_train",
    )
    assert records
    # Internal candidates are intentionally target-free; the public writer
    # is exercised with the schema-shaped records produced by the full API.
    from travel_grpo.data.recovery_boundaries import _dedupe_candidates

    shaped, _ = _dedupe_candidates(records)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "counts": {"unique_contexts": len(shaped)},
    }
    contexts, manifest_path = write_extraction(shaped, manifest, tmp_path / "out")
    assert contexts.exists() and manifest_path.exists()
    row = json.loads(contexts.read_text(encoding="utf-8").splitlines()[0])
    assert row["schema_version"] == SCHEMA_VERSION
    assert row["target_assistant"] is None
    assert set(BOUNDARY_TYPES) >= {row["boundary_type"]}
