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
from travel_grpo.training.grpo.preflight import is_validation_sampling
from travel_grpo.training.grpo.preflight import PINNED

ROOT = Path(__file__).resolve().parents[1]


def test_grpo_numpy_pin_is_compatible_with_verl_080():
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"numpy==1.26.4; platform_system == \'Linux\'"' in project
    assert PINNED["numpy"] == "1.26.4"


def test_dynamic_sampling_discards_invalid_and_equal_groups():
    uids = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
    rewards = [0.1, 0.2, 0.1, 0.3] + [0.4] * 4 + [-0.1, 0.2, 0.3, 0.4]
    invalid = [False] * 8 + [False, True, False, False]
    indices, stats = select_reward_varying_groups(
        uids, rewards, sampling_invalid=invalid, expected_group_size=4
    )
    assert indices == [0, 1, 2, 3, 8, 10, 11]
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


def test_stalled_valid_trajectory_remains_a_dynamic_sampling_candidate():
    rewards, invalid, reasons = extract_userbench_group_signals(
        [
            {
                "reward": {
                    "terminal_reward": 0.2,
                    "reward_valid": True,
                    "termination_reason": "stalled_no_progress",
                }
            }
        ]
    )
    assert rewards == [0.2]
    assert invalid == [False]
    assert reasons == [()]


def test_training_and_validation_sampling_profiles_are_disjoint():
    assert is_validation_sampling(
        {"temperature": 0.0, "top_p": 1.0, "do_sample": False}
    ) is True
    assert is_validation_sampling({"temperature": 0.7, "top_p": 0.9}) is False


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


def test_bounded_sampler_keeps_valid_rows_across_batches_and_preserves_fields(monkeypatch):
    class Scores:
        def __init__(self, values):
            self.values = list(values)

        def sum(self, dim=-1):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return list(self.values)

    class Output:
        def __init__(self, row_ids, uids, rewards, valid=None, degraded=None):
            self.rows = list(row_ids)
            self.batch = {
                "rm_scores": Scores(rewards),
                "token_ids": [[f"token-{row}"] for row in row_ids],
                "attention_mask": [[1] for _ in row_ids],
                "response_mask": [[1] for _ in row_ids],
            }
            valid = [True] * len(row_ids) if valid is None else list(valid)
            degraded = [False] * len(row_ids) if degraded is None else list(degraded)
            self.non_tensor_batch = {
                "uid": list(uids),
                "userbench": [
                    {
                        "reward": {
                            "terminal_reward": reward,
                            "reward_valid": item_valid,
                            "reward_degraded": item_degraded,
                        },
                        "tool_metadata": {"row": row},
                    }
                    for row, reward, item_valid, item_degraded in zip(
                        row_ids, rewards, valid, degraded, strict=True
                    )
                ],
                "tool_metadata": [{"row": row} for row in row_ids],
            }
            self.meta_info = {"timing": {}}

        def slice(self, start, stop):
            valid = [
                item["reward"]["reward_valid"]
                for item in self.non_tensor_batch["userbench"][start:stop]
            ]
            degraded = [
                item["reward"]["reward_degraded"]
                for item in self.non_tensor_batch["userbench"][start:stop]
            ]
            return Output(
                self.rows[start:stop],
                self.non_tensor_batch["uid"][start:stop],
                self.batch["rm_scores"].values[start:stop],
                valid,
                degraded,
            )

    class DataProto:
        @staticmethod
        def concat(outputs):
            return Output(
                [row for output in outputs for row in output.rows],
                [
                    uid
                    for output in outputs
                    for uid in output.non_tensor_batch["uid"]
                ],
                [
                    reward
                    for output in outputs
                    for reward in output.batch["rm_scores"].values
                ],
                [
                    item["reward"]["reward_valid"]
                    for output in outputs
                    for item in output.non_tensor_batch["userbench"]
                ],
                [
                    item["reward"]["reward_degraded"]
                    for output in outputs
                    for item in output.non_tensor_batch["userbench"]
                ],
            )

    monkeypatch.setitem(sys.modules, "verl", types.SimpleNamespace(DataProto=DataProto))
    input_uids = ["a"] * 4 + ["b"] * 4
    batches = [
        Output(
            ["a0", "a1", "a2", "a-invalid", "b0", "b1", "b2", "b3"],
            input_uids,
            [0.1, 0.2, 0.3, 0.0, 0.1, 0.1, 0.1, 0.1],
            [True, True, True, False, True, True, True, True],
        ),
        Output(
            ["a4", "a5", "a6", "a7", "b4", "b5", "b6", "b7"],
            input_uids,
            [0.4, 0.5, 0.6, 0.7, 0.1, 0.2, 0.3, 0.4],
        ),
    ]
    seen_batches = []

    def generate(batch):
        seen_batches.append(batch)
        return batches.pop(0)

    manager = types.SimpleNamespace(generate_sequences=generate)
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
        meta_info={}, non_tensor_batch={"uid": input_uids}
    )
    result = manager.generate_sequences(batch)
    assert seen_batches == [batch, batch]
    assert len(result.rows) == 8
    assert result.rows[:4] == ["a0", "a1", "a2", "a7"]
    assert result.rows[4:] == ["b0", "b5", "b6", "b7"]
    assert result.non_tensor_batch["uid"] == ["a"] * 4 + ["b"] * 4
    assert set(result.batch) == {
        "rm_scores",
        "token_ids",
        "attention_mask",
        "response_mask",
    }
    assert set(result.non_tensor_batch) == {"uid", "userbench", "tool_metadata"}
    diagnostics = result.meta_info["travel_dynamic_sampling"]
    assert diagnostics["sampled_batches"] == 2
    assert diagnostics["candidate_count"] == 15
    assert diagnostics["degraded_candidate_count"] == 0


def test_bounded_sampler_uses_degraded_rows_only_when_clean_candidates_are_insufficient(monkeypatch):
    class Scores:
        def __init__(self, values):
            self.values = list(values)

        def sum(self, dim=-1):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return list(self.values)

    class Output:
        def __init__(self, uids, rewards, degraded):
            self.rows = list(range(len(uids)))
            self.batch = {"rm_scores": Scores(rewards)}
            self.non_tensor_batch = {
                "uid": list(uids),
                "userbench": [
                    {
                        "reward": {
                            "terminal_reward": reward,
                            "reward_valid": True,
                            "reward_degraded": item_degraded,
                        }
                    }
                    for reward, item_degraded in zip(rewards, degraded, strict=True)
                ],
            }
            self.meta_info = {"timing": {}}

        def slice(self, start, stop):
            values = self.batch["rm_scores"].values[start:stop]
            items = self.non_tensor_batch["userbench"][start:stop]
            return Output(
                self.non_tensor_batch["uid"][start:stop],
                values,
                [item["reward"]["reward_degraded"] for item in items],
            )

    class DataProto:
        @staticmethod
        def concat(outputs):
            return Output(
                [
                    uid
                    for output in outputs
                    for uid in output.non_tensor_batch["uid"]
                ],
                [
                    reward
                    for output in outputs
                    for reward in output.batch["rm_scores"].values
                ],
                [
                    item["reward"]["reward_degraded"]
                    for output in outputs
                    for item in output.non_tensor_batch["userbench"]
                ],
            )

    monkeypatch.setitem(sys.modules, "verl", types.SimpleNamespace(DataProto=DataProto))
    uids = ["a"] * 4 + ["b"] * 4
    output = Output(
        uids,
        [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
        [False, False, False, True, False, False, False, False],
    )
    manager = types.SimpleNamespace(generate_sequences=lambda _: output)
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
    batch = types.SimpleNamespace(meta_info={}, non_tensor_batch={"uid": uids})
    result = manager.generate_sequences(batch)
    assert len(result.rows) == 8
    assert result.meta_info["travel_dynamic_sampling"]["degraded_candidate_count"] == 1


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


def test_preflight_rejects_sampling_profile_drift(tmp_path):
    yaml = pytest.importorskip("yaml")
    profile = yaml.safe_load((ROOT / "configs/train/grpo/grpo.yaml").read_text(encoding="utf-8"))
    profile["rollout"]["temperature"] = 0.0
    with pytest.raises(RuntimeError, match="sampling profile classification failed"):
        run_preflight(
            profile,
            project_root=ROOT,
            output_dir=tmp_path / "drift-output",
            resume=False,
            strict_runtime=False,
            environ={},
        )


@pytest.mark.parametrize(
    "flags, enabled",
    [
        (["--no-stall-recovery"], False),
        (["--stall-recovery", "--stall-threshold", "4"], True),
    ],
)
def test_grpo_dry_run_exposes_stall_configuration(tmp_path, flags, enabled):
    command = [
        sys.executable,
        str(ROOT / "scripts/train/grpo/train_grpo.py"),
        "--config",
        "configs/train/grpo/grpo.yaml",
        "--dry-run",
        *flags,
        "--output",
        str(tmp_path / ("enabled" if enabled else "disabled")),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    document = json.loads(completed.stdout)
    assert document["preflight"]["static_contract"] == "ok"
    assert document["stall_recovery"] == {"enabled": enabled, "threshold": 4}
    assert document["agent_loop_env"] == {
        "TRAVEL_GRPO_STALL_RECOVERY": "true" if enabled else "false",
        "TRAVEL_GRPO_STALL_THRESHOLD": "4",
    }


def _write_fake_sft_adapter(path: Path) -> None:
    path.mkdir()
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"dry-run placeholder")


def test_grpo_from_sft_dry_run_chains_merge_data_and_training(tmp_path):
    adapter = tmp_path / "stage2-adapter"
    _write_fake_sft_adapter(adapter)
    script = ROOT / "scripts/train/grpo/run_grpo_from_sft.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--sft-adapter",
            str(adapter),
            "--merged-model",
            str(tmp_path / "merged"),
            "--data-output",
            str(tmp_path / "grpo-data"),
            "--output",
            str(tmp_path / "grpo-output"),
            "--dry-run",
            "--stall-recovery",
            "--stall-threshold",
            "5",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "merge_lora.py" in completed.stdout
    assert "prepare_data.py" in completed.stdout
    assert "train_grpo.py" in completed.stdout
    assert '"status": "dry-run"' in completed.stdout
    assert '"enabled": true' in completed.stdout
    assert '"threshold": 5' in completed.stdout
    assert not (tmp_path / "merged").exists()
    assert not (tmp_path / "grpo-data").exists()


def test_grpo_from_sft_rejects_missing_adapter_before_any_write(tmp_path):
    script = ROOT / "scripts/train/grpo/run_grpo_from_sft.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--sft-adapter",
            str(tmp_path / "not-generated-yet"),
            "--merged-model",
            str(tmp_path / "merged"),
            "--data-output",
            str(tmp_path / "grpo-data"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "not generated yet" in completed.stderr
    assert not (tmp_path / "merged").exists()
    assert not (tmp_path / "grpo-data").exists()


def test_grpo_dry_run_accepts_custom_model_and_data_paths(tmp_path):
    data_output = tmp_path / "prepared-data"
    model_path = tmp_path / "merged-model"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/train/grpo/train_grpo.py"),
            "--config",
            "configs/train/grpo/grpo.yaml",
            "--model-path",
            str(model_path),
            "--data-output",
            str(data_output),
            "--dry-run",
            "--output",
            str(tmp_path / "run"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    document = json.loads(completed.stdout)
    assert (
        f"data.train_files=['{data_output.resolve() / 'train.parquet'}']"
        in document["command"]
    )
    assert (
        f"actor_rollout_ref.model.path={model_path.resolve()}"
        in document["command"]
    )


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
