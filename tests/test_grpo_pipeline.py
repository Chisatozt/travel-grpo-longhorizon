from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from travel_grpo.training.grpo.dynamic_sampling import (
    BoundedSamplingState,
    _ordered_unique,
    extract_userbench_group_signals,
    install_verl_bounded_sampler,
    select_reward_varying_groups,
)
from travel_grpo.training.grpo.preflight import run_preflight

ROOT = Path(__file__).resolve().parents[1]


def test_dynamic_sampling_discards_invalid_and_equal_groups():
    uids = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
    rewards = [0.1, 0.2, 0.1, 0.3] + [0.4] * 4 + [-0.1, 0.2, 0.3, 0.4]
    invalid = [False] * 8 + [False, True, False, False]
    indices, stats = select_reward_varying_groups(
        uids, rewards, sampling_invalid=invalid, expected_group_size=4
    )
    assert indices == [0, 1, 2, 3]
    assert stats["kept_group_count"] == 1
    assert stats["constant_reward_group_count"] == 1
    assert stats["sampling_invalid_group_count"] == 1


def test_bounded_sampling_three_batches_and_skip_limit():
    state = BoundedSamplingState()
    for _ in range(3):
        assert not state.record_batch({"kept_group_count": 0})
    assert not state.may_generate
    assert state.finish_update() is False
    for _ in range(9):
        state.generation_batches = 3
        assert state.finish_update() is False
    state.generation_batches = 3
    with pytest.raises(RuntimeError, match="consecutive"):
        state.finish_update()


def test_reward_valid_false_is_sampling_invalid():
    rewards, invalid, reasons = extract_userbench_group_signals(
        [{"reward": {"terminal_reward": 0.0, "reward_valid": False}}]
    )
    assert rewards == [0.0]
    assert invalid == [True]
    assert reasons == [("reward_invalid",)]


def test_groups_are_restored_to_original_prompt_order():
    class Scalar:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    assert _ordered_unique([Scalar("a"), "a", Scalar("b"), "b"]) == ["a", "b"]


def test_bounded_sampler_restores_cross_batch_group_order(monkeypatch):
    class Scores:
        def __init__(self, values):
            self.values = values

        def sum(self, dim=-1):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return list(self.values)

    class Output:
        def __init__(self, uids, rewards):
            self.rows = list(uids)
            self.batch = {"rm_scores": Scores(rewards)}
            self.non_tensor_batch = {
                "uid": list(uids),
                "userbench": [
                    {"reward": {"terminal_reward": reward, "reward_valid": True}}
                    for reward in rewards
                ],
            }
            self.meta_info = {"timing": {}}

        def slice(self, start, stop):
            return Output(
                self.rows[start:stop], self.batch["rm_scores"].values[start:stop]
            )

    class DataProto:
        @staticmethod
        def concat(outputs):
            uids = [uid for output in outputs for uid in output.rows]
            rewards = [
                reward
                for output in outputs
                for reward in output.batch["rm_scores"].values
            ]
            return Output(uids, rewards)

    monkeypatch.setitem(sys.modules, "verl", types.SimpleNamespace(DataProto=DataProto))
    uids = ["a"] * 4 + ["b"] * 4
    outputs = iter(
        [
            Output(uids, [0.1] * 4 + [0.1, 0.2, 0.3, 0.4]),
            Output(uids, [0.1, 0.2, 0.3, 0.4] + [0.2] * 4),
        ]
    )
    manager = types.SimpleNamespace(generate_sequences=lambda _: next(outputs))
    install_verl_bounded_sampler(
        manager,
        {
            "enable": True,
            "group_size": 4,
            "required_groups": 2,
            "max_generation_batches": 3,
            "max_consecutive_skips": 10,
            "reward_tolerance": 1e-6,
        },
    )
    batch = types.SimpleNamespace(
        meta_info={}, non_tensor_batch={"uid": uids}
    )
    result = manager.generate_sequences(batch)
    assert result.rows == uids
    assert result.meta_info["travel_dynamic_sampling"]["sampled_batches"] == 2


def test_grpo_profile_dry_preflight_is_static_only(tmp_path):
    yaml = pytest.importorskip("yaml")
    profile = yaml.safe_load((ROOT / "configs/train/grpo/grpo.yaml").read_text(encoding="utf-8"))
    report = run_preflight(
        profile,
        project_root=ROOT,
        output_dir=tmp_path / "new-output",
        resume=False,
        strict_runtime=False,
        environ={},
    )
    assert report == {
        "static_contract": "ok",
        "strict_runtime": False,
        "runtime": "skipped by dry-run",
    }


def test_verl_data_has_no_hidden_labels():
    pa = pytest.importorskip("pyarrow")
    from travel_grpo.training.grpo.data import build_verl_records

    records = build_verl_records(ROOT / "data/grpo/train.parquet", project_split="train")
    assert len(records) == 1723
    assert [row["extra_info"]["index"] for row in records] == list(range(1723))
    for row in records[:10]:
        task_id = row["extra_info"]["task_id"]
        assert row["reward_model"]["id"] == task_id
        assert row["reward_model"]["ground_truth"] == ""
        assert row["extra_info"]["tools_kwargs"]["interact_with_env"]["create_kwargs"]["id"] == task_id
        serialized = json.dumps(row).casefold()
        assert "best_id" not in serialized
        assert "correct_ids" not in serialized
        assert "remaining_preference_ids" not in serialized
        assert "preference_ids_by_aspect" not in serialized


def test_verl_data_manifest_pins_runtime_and_generator_versions(tmp_path):
    from travel_grpo.training.grpo.data import (
        VERL_DATA_GENERATOR_VERSION,
        VERL_RUNTIME_VERSION,
        prepare_verl_datasets,
    )

    summary = prepare_verl_datasets(
        train_source=ROOT / "data/grpo/train.parquet",
        validation_source=ROOT / "data/grpo/validation.parquet",
        output_root=tmp_path,
    )
    manifest = json.loads(Path(summary["manifest"]).read_text(encoding="utf-8"))
    assert manifest["verl_version"] == VERL_RUNTIME_VERSION == "0.8.0"
    assert manifest["generator_version"] == VERL_DATA_GENERATOR_VERSION
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_verl_datasets(
            train_source=ROOT / "data/grpo/train.parquet",
            validation_source=ROOT / "data/grpo/validation.parquet",
            output_root=tmp_path,
        )


def test_actor_export_requires_passed_selected_checkpoint(tmp_path):
    actor = tmp_path / "run" / "global_step_100" / "actor"
    actor.mkdir(parents=True)
    selection = tmp_path / "run" / "checkpoint_selection.json"
    selection.write_text(
        json.dumps({"passed": True, "selected_step": 100}), encoding="utf-8"
    )
    command = [
        sys.executable,
        str(ROOT / "scripts/train/grpo/export_actor.py"),
        str(actor),
        str(tmp_path / "merged"),
        "--dry-run",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["selected_step"] == 100

    selection.write_text(
        json.dumps({"passed": True, "selected_step": 50}), encoding="utf-8"
    )
    rejected = subprocess.run(command, capture_output=True, text=True, check=False)
    assert rejected.returncode != 0
    assert "not the passed selected step" in rejected.stderr
