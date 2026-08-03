"""Offline action-only SFT dataset and dry-run contracts."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from travel_grpo.envs.reward import REWARD_VERSION
from travel_grpo.training.sft_collection import TRAJECTORY_SCHEMA_VERSION
from travel_grpo.training.sft_dataset import (
    IGNORE_INDEX,
    ActionOnlyDataCollator,
    SFTDatasetError,
    assert_train_validation_disjoint,
    audit_trajectory_file,
    build_action_only_examples,
    load_sft_trajectories,
    load_tool_schema,
    trajectory_rejection_reasons,
)

ROOT = Path(__file__).resolve().parents[1]
TOOL_CONFIG = ROOT / "configs/tool_config/userbench_tools.yaml"


class FakeQwenTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    padding_side = "right"

    @staticmethod
    def encode(text):
        return [ord(value) + 3 for value in text]

    @staticmethod
    def decode(tokens):
        return "".join(chr(value - 3) for value in tokens)

    def apply_chat_template(
        self, conversation, *, tools, tokenize, add_generation_prompt, enable_thinking
    ):
        assert tokenize is True
        assert enable_thinking is False
        assert tools == [load_tool_schema(TOOL_CONFIG)]
        for message in conversation:
            if message.get("role") == "assistant":
                assert isinstance(
                    message["tool_calls"][0]["function"]["arguments"], dict
                )
        text = "<tools>" + json.dumps(tools, sort_keys=True) + "</tools>"
        for message in conversation:
            role = message["role"]
            text += f"<{role}>"
            if role == "assistant":
                text += json.dumps(message["tool_calls"], sort_keys=True)
            else:
                text += str(message.get("content", ""))
            text += f"</{role}>"
        if add_generation_prompt:
            text += "<assistant>"
        return self.encode(text)


def _call(call_id, choice, content):
    arguments = json.dumps(
        {"thought": "next step", "choice": choice, "content": content},
        separators=(",", ":"),
    )
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "interact_with_env", "arguments": arguments},
            }
        ],
    }


def valid_record(task_id="hotel:2-1"):
    reward = {
        "reward_version": REWARD_VERSION,
        "reward_valid": True,
        "terminal_reward": 1.0,
        "completion_rate": 1.0,
        "correct_itinerary": True,
        "gold_itinerary": True,
        "fully_grounded": True,
        "active_preference_coverage": 1.0,
        "passive_preference_coverage": 0.0,
        "policy_penalty": 0.0,
        "invalid_actions": 0,
        "exact_repeats": 0,
        "semantic_repeats": 0,
        "ambiguous_actions": 0,
        "unsearched_answers": 0,
        "wrong_answers": 0,
        "infrastructure_errors": [],
    }
    return {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "task_id": task_id,
        "composition": "22",
        "difficulty": "easy",
        "source_split": "train",
        "teacher_model": "deepseek-v4-flash",
        "simulator_model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "Use one tool."},
            {"role": "user", "content": "Find a hotel."},
            _call("call-1", "search", "hotels in Paris"),
            {
                "role": "tool",
                "name": "interact_with_env",
                "tool_call_id": "call-1",
                "content": "H1 is available.",
            },
            _call("call-2", "answer", "H1"),
            {
                "role": "tool",
                "name": "interact_with_env",
                "tool_call_id": "call-2",
                "content": "Best option recorded.",
            },
        ],
        "step_rewards": [0.2, 1.0],
        "total_reward": 1.2,
        "num_steps": 2,
        "terminated": True,
        "truncated": False,
        "expected_aspects": ["hotel"],
        "answered_aspects": ["hotel"],
        "reward_breakdown": reward,
        **reward,
    }


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in records),
        encoding="utf-8",
    )


def test_valid_multiturn_empty_content_renders_only_tool_calls_as_labels():
    record = valid_record()
    tokenizer = FakeQwenTokenizer()
    examples = build_action_only_examples(
        [record], tokenizer, load_tool_schema(TOOL_CONFIG), max_sequence_length=100000
    )
    assert len(examples) == 2
    assert [value.assistant_turn_index for value in examples] == [1, 2]
    for example in examples:
        first_label = next(
            index for index, value in enumerate(example.labels) if value != IGNORE_INDEX
        )
        assert all(value == IGNORE_INDEX for value in example.labels[:first_label])
        target = tokenizer.decode(example.labels[first_label:])
        assert "interact_with_env" in target
        assert "choice" in target and "content" in target
        assert "H1 is available" not in target


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda value: value["messages"][3].update(tool_call_id="wrong"),
            "tool_call_id_mismatch",
        ),
        (
            lambda value: value["messages"][2]["tool_calls"].append(
                copy.deepcopy(value["messages"][2]["tool_calls"][0])
            ),
            "assistant_must_contain_one_tool_call",
        ),
        (
            lambda value: value["messages"][2]["tool_calls"][0]["function"].update(
                name="wrong_tool"
            ),
            "wrong_function_name",
        ),
        (
            lambda value: value["messages"][2]["tool_calls"][0]["function"].update(
                arguments="{bad"
            ),
            "invalid_tool_arguments_json",
        ),
        (
            lambda value: value["messages"][2].update(content="extra prose"),
            "assistant_content_must_be_empty",
        ),
        (
            lambda value: value["messages"][2]["tool_calls"][0].update(
                type="custom"
            ),
            "wrong_tool_call_type",
        ),
    ],
)
def test_message_contract_failures(mutate, reason):
    record = valid_record()
    mutate(record)
    assert reason in trajectory_rejection_reasons(record)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("reward_valid", False, "reward_invalid"),
        ("policy_penalty", 0.1, "policy_penalty"),
        ("terminal_reward", 0.69, "terminal_reward_below_threshold"),
        ("invalid_actions", 1, "invalid_actions"),
        ("exact_repeats", 1, "exact_repeats"),
        ("semantic_repeats", 1, "semantic_repeats"),
        ("ambiguous_actions", 1, "ambiguous_actions"),
        ("unsearched_answers", 1, "unsearched_answers"),
        ("wrong_answers", 1, "wrong_answers"),
    ],
)
def test_reward_admission_failures(field, value, reason):
    record = valid_record()
    record[field] = value
    assert reason in trajectory_rejection_reasons(record)


def test_missing_reward_and_vague_feedback_are_both_reported():
    record = valid_record()
    record["schema_version"] = "userbench-teacher-trajectory-v3"
    record["reward_breakdown"] = None
    record["messages"][3]["content"] = "Your question is too vague and general."
    reasons = trajectory_rejection_reasons(record)
    assert "legacy_or_unknown_schema" in reasons
    assert "missing_reward_evidence" in reasons
    assert "vague_action_feedback" in reasons


def test_current_smoke_is_audited_as_legacy_and_vague():
    source = ROOT / "outputs/teacher_trajectories/smoke_strict_v2_deepseek_v4_flash.accepted.jsonl"
    audit = audit_trajectory_file(source)
    summary = audit.summary()
    assert summary["accepted_trajectories"] == 0
    assert summary["rejection_reasons"]["legacy_or_unknown_schema"] == 1
    assert summary["rejection_reasons"]["missing_reward_evidence"] == 1
    assert summary["rejection_reasons"]["vague_action_feedback"] == 1


def test_loader_rejects_duplicate_task_ids(tmp_path):
    source = tmp_path / "duplicate.jsonl"
    write_jsonl(source, [valid_record(), valid_record()])
    audit = audit_trajectory_file(source)
    assert audit.summary()["rejection_reasons"]["duplicate_task_id"] == 1
    with pytest.raises(SFTDatasetError, match="not trainable"):
        load_sft_trajectories(source)


def test_overlength_fails_without_truncation():
    tokenizer = FakeQwenTokenizer()
    with pytest.raises(SFTDatasetError, match="silent truncation is forbidden"):
        build_action_only_examples(
            [valid_record()], tokenizer, load_tool_schema(TOOL_CONFIG), max_sequence_length=10
        )


def test_collator_uses_minus_100_for_label_padding():
    tokenizer = FakeQwenTokenizer()
    examples = build_action_only_examples(
        [valid_record()], tokenizer, load_tool_schema(TOOL_CONFIG), max_sequence_length=100000
    )
    batch = ActionOnlyDataCollator(pad_token_id=0)(
        [examples[0], examples[1]]
    )
    assert batch["labels"].shape == batch["input_ids"].shape
    shorter = min(range(2), key=lambda index: examples[index].sequence_length)
    padding = batch["attention_mask"][shorter] == 0
    assert (batch["labels"][shorter][padding] == IGNORE_INDEX).all()


def test_train_validation_overlap_fails():
    with pytest.raises(SFTDatasetError, match="overlap"):
        assert_train_validation_disjoint([valid_record()], [valid_record()])


def test_chat_template_prefix_mismatch_fails():
    class BrokenTokenizer(FakeQwenTokenizer):
        def apply_chat_template(self, conversation, **kwargs):
            result = super().apply_chat_template(conversation, **kwargs)
            if not kwargs["add_generation_prompt"]:
                result[0] += 1
            return result

    with pytest.raises(SFTDatasetError, match="verified assistant completion prefix"):
        build_action_only_examples(
            [valid_record()],
            BrokenTokenizer(),
            load_tool_schema(TOOL_CONFIG),
            max_sequence_length=100000,
        )


def test_no_assistant_completion_tokens_fails():
    class EmptyCompletionTokenizer(FakeQwenTokenizer):
        def apply_chat_template(self, conversation, **kwargs):
            if conversation and conversation[-1].get("role") == "assistant":
                return super().apply_chat_template(
                    conversation[:-1],
                    tools=kwargs["tools"],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            return super().apply_chat_template(conversation, **kwargs)

    with pytest.raises(SFTDatasetError, match="no supervised tokens"):
        build_action_only_examples(
            [valid_record()],
            EmptyCompletionTokenizer(),
            load_tool_schema(TOOL_CONFIG),
            max_sequence_length=100000,
        )


def test_rendering_does_not_rewrite_archived_json_arguments():
    record = valid_record()
    original = copy.deepcopy(record)
    build_action_only_examples(
        [record],
        FakeQwenTokenizer(),
        load_tool_schema(TOOL_CONFIG),
        max_sequence_length=100000,
    )
    assert record == original
    assert isinstance(
        record["messages"][2]["tool_calls"][0]["function"]["arguments"], str
    )


def test_dry_run_is_offline_and_writes_no_checkpoint(tmp_path):
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    write_jsonl(train_path, [valid_record("hotel:2-1")])
    write_jsonl(validation_path, [valid_record("hotel:2-2")])
    output = ROOT / "outputs/test-sft-dry-run-never-created"
    config = tmp_path / "sft.yaml"
    config.write_text(
        f"""
model:
  base: Qwen/Qwen3.5-2B
  cache_dir: outputs/cache/huggingface
  trust_remote_code: false
  qlora: false
data:
  train_trajectories: {train_path.as_posix()}
  validation_trajectories: {validation_path.as_posix()}
  tool_schema_path: {TOOL_CONFIG.as_posix()}
  assistant_loss: action_only
  example_unit: assistant_turn
  max_sequence_length: 16384
lora:
  rank: 8
  alpha: 16
  dropout: 0.05
  target_modules: [q_proj]
training:
  output_dir: {output.as_posix()}
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 0.0001
  weight_decay: 0.01
  warmup_ratio: 0.03
  lr_scheduler_type: cosine
  optim: adamw_torch
  num_train_epochs: 1
  max_steps: -1
  gradient_checkpointing: true
  bf16: true
  fp16: false
  logging_steps: 1
  eval_strategy: steps
  eval_steps: 1
  save_strategy: steps
  save_steps: 1
  save_total_limit: 1
  seed: 42
  report_to: none
""",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["HF_HUB_OFFLINE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/train/sft/sft_train.py"),
            "--config",
            str(config),
            "--dry-run",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["mode"] == "dry-run"
    assert summary["train"]["accepted_trajectories"] == 1
    assert not output.exists()
