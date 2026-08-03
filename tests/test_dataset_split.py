"""Contracts for the pinned UserBench task split builder."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from travel_grpo.data.userbench import (
    ALL_PROJECT_SPLITS,
    OUTPUT_COLUMNS,
    SOURCE_COLUMNS,
    DatasetSplitError,
    build_dataset_splits,
    compute_jsonl_sha256,
    load_onechoice_tasks,
    load_split_spec,
    verify_dataset_splits,
    write_dataset_splits,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "data" / "dataset_split.toml"
SOURCE_ROOT = ROOT / "environments" / "UserBench" / "data"
EXPECTED_COUNTS = {
    "sft_train": 716,
    "sft_validation": 80,
    "grpo_train": 1723,
    "grpo_validation": 132,
    "evaluation": 471,
}
EXPECTED_BY_COMPOSITION = {
    "22": (154, 17, 371, 28, 101),
    "33": (132, 15, 318, 24, 87),
    "44": (102, 12, 246, 19, 67),
    "2222": (26, 3, 62, 5, 17),
    "233": (92, 10, 222, 17, 61),
    "333": (81, 9, 195, 15, 53),
    "334": (72, 8, 173, 13, 47),
    "444": (57, 6, 136, 11, 38),
}


@pytest.fixture(scope="session")
def split_spec():
    return load_split_spec(CONFIG)


@pytest.fixture(scope="session")
def split_bundle(split_spec):
    return build_dataset_splits(split_spec, SOURCE_ROOT)


def test_exact_counts_and_disjointness(split_bundle):
    assert split_bundle.manifest_base["counts"]["project_splits"] == EXPECTED_COUNTS
    assert split_bundle.manifest_base["counts"]["total"] == 3122
    assert split_bundle.manifest_base["counts"]["upstream_train"] == 2651
    assert split_bundle.manifest_base["counts"]["upstream_test"] == 471
    assert not any(
        split_bundle.manifest_base["checks"]["project_split_intersections"].values()
    )
    assert split_bundle.manifest_base["checks"]["upstream_train_test_overlap"] == 0


def test_exact_composition_quotas(split_bundle):
    by_composition = split_bundle.manifest_base["counts"]["by_composition"]
    for composition, expected in EXPECTED_BY_COMPOSITION.items():
        assert tuple(by_composition[composition][name] for name in ALL_PROJECT_SPLITS) == expected


def test_records_follow_example_contract(split_bundle):
    for project_split, records in split_bundle.records.items():
        expected_source_split = "test" if project_split == "evaluation" else "train"
        for record in records:
            assert tuple(record) == OUTPUT_COLUMNS
            assert record["source_split"] == expected_source_split
            assert record["difficulty"] in {"easy", "medium", "hard"}
            assert [message["role"] for message in record["prompt"]] == ["system", "user"]


def test_reference_example_records_are_reproduced(split_bundle):
    examples = [
        json.loads(line)
        for line in (ROOT / "data" / "example.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    evaluation_by_id = {
        record["task_id"]: record for record in split_bundle.records["evaluation"]
    }
    assert examples
    for example in examples:
        assert tuple(example) == OUTPUT_COLUMNS
        assert evaluation_by_id[example["task_id"]] == example


def test_hash_split_is_deterministic(split_spec, split_bundle):
    rebuilt = build_dataset_splits(split_spec, SOURCE_ROOT)
    for project_split in ALL_PROJECT_SPLITS:
        assert [row["task_id"] for row in rebuilt.records[project_split]] == [
            row["task_id"] for row in split_bundle.records[project_split]
        ]
        assert compute_jsonl_sha256(rebuilt.records[project_split]) == compute_jsonl_sha256(
            split_bundle.records[project_split]
        )


def test_write_verify_and_refuse_overwrite(tmp_path, split_spec, split_bundle):
    manifest = write_dataset_splits(split_bundle, tmp_path)
    assert manifest["counts"]["project_splits"] == EXPECTED_COUNTS
    report = verify_dataset_splits(split_spec, SOURCE_ROOT, tmp_path)
    assert report["valid"] is True
    assert report["counts"] == EXPECTED_COUNTS
    before = {
        path.relative_to(tmp_path).as_posix(): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    verify_dataset_splits(split_spec, SOURCE_ROOT, tmp_path)
    after = {
        path.relative_to(tmp_path).as_posix(): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before
    with pytest.raises(FileExistsError, match="without force"):
        write_dataset_splits(split_bundle, tmp_path)


def test_source_count_drift_fails():
    with pytest.raises(DatasetSplitError, match="expected 999"):
        load_onechoice_tasks(SOURCE_ROOT, "22", "train", expected_count=999)


def _write_small_source(tmp_path: Path, rows: list[dict], task: dict, task_id: str) -> Path:
    source_root = tmp_path / "UserBench" / "data"
    parquet_dir = source_root / "travel22_multiturn_onechoice"
    parquet_dir.mkdir(parents=True)
    original_schema = pq.read_table(
        SOURCE_ROOT / "travel22_multiturn_onechoice" / "train.parquet"
    ).schema
    pq.write_table(pa.Table.from_pylist(rows, schema=original_schema), parquet_dir / "train.parquet")
    task_dir = source_root.parent / "travelgym" / "data"
    task_dir.mkdir(parents=True)
    (task_dir / "travelgym_data_22.json").write_text(
        json.dumps({task_id: task}), encoding="utf-8"
    )
    return source_root


def _first_source_row_and_task():
    row = pq.read_table(
        SOURCE_ROOT / "travel22_multiturn_onechoice" / "train.parquet"
    ).slice(0, 1).to_pylist()[0]
    task_id = row["reward_model"]["id"]
    task_data = json.loads(
        (SOURCE_ROOT.parent / "travelgym" / "data" / "travelgym_data_22.json").read_text(
            encoding="utf-8"
        )
    )
    return row, task_data[task_id], task_id


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("roles", "prompt roles"),
        ("ground_truth", "ground_truth does not match"),
        ("tool_id", "task ID does not match"),
    ],
)
def test_corrupt_source_rows_fail(tmp_path, mutation, message):
    row, task, task_id = _first_source_row_and_task()
    row = copy.deepcopy(row)
    if mutation == "roles":
        row["prompt"][1]["role"] = "assistant"
    elif mutation == "ground_truth":
        row["reward_model"]["ground_truth"] = "['NOT_A_REAL_OPTION']"
    else:
        row["extra_info"]["tools_kwargs"]["interact_with_env"]["create_kwargs"][
            "id"
        ] = "different-task"
    source_root = _write_small_source(tmp_path, [row], task, task_id)
    with pytest.raises(DatasetSplitError, match=message):
        load_onechoice_tasks(source_root, "22", "train", expected_count=1)


def test_duplicate_source_task_id_fails(tmp_path):
    row, task, task_id = _first_source_row_and_task()
    source_root = _write_small_source(tmp_path, [row, copy.deepcopy(row)], task, task_id)
    with pytest.raises(DatasetSplitError, match="duplicate task ID"):
        load_onechoice_tasks(source_root, "22", "train", expected_count=2)


def test_missing_source_column_fails(tmp_path):
    source_root = tmp_path / "UserBench" / "data"
    parquet_dir = source_root / "travel22_multiturn_onechoice"
    parquet_dir.mkdir(parents=True)
    pq.write_table(pa.table({"data_source": ["interact_travelgym"]}), parquet_dir / "train.parquet")
    with pytest.raises(DatasetSplitError, match="columns are"):
        load_onechoice_tasks(source_root, "22", "train", expected_count=1)


def test_cli_dry_run_writes_nothing(tmp_path):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "data" / "build_dataset_splits.py"),
            "--config",
            str(CONFIG),
            "--source-root",
            str(SOURCE_ROOT),
            "--output-root",
            str(tmp_path),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
    )
    summary = json.loads(result.stdout)
    assert summary["dry_run"] is True
    assert summary["counts"]["project_splits"] == EXPECTED_COUNTS
    repeated = subprocess.run(
        result.args,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
    )
    assert json.loads(repeated.stdout)["planned_jsonl_sha256"] == summary[
        "planned_jsonl_sha256"
    ]
    assert list(tmp_path.iterdir()) == []
