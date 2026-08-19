"""Runtime orchestration for the frozen UserBench evaluation stage.

The command-line compatibility shim remains at
``scripts/eval/evaluate_userbench.py``. Keeping the runner here makes task
loading, contract construction, resumable execution, and summary writing
part of the evaluation feature instead of the launcher directory.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
import pyarrow.parquet as pq

from travel_grpo.envs.userbench_interaction import SimulatorRole, UserSimulatorRuntime
from travel_grpo.evaluation.artifacts import (
    atomic_json,
    attach_attempt_history,
    load_completed,
    task_path,
    write_results_jsonl,
)
from travel_grpo.evaluation.contracts import (
    FORMAL_EVALUATION_TASK_COUNT,
    STAGES,
    build_contract,
    build_subset_contract,
)
from travel_grpo.evaluation.rollout import (
    PUBLIC_CONTROL_PHASE_GUARD_VERSION,
    rollout_task,
)
from travel_grpo.evaluation.summary import summarize_results
from travel_grpo.models.vllm_policy import ActorRuntime, OpenAICompatibleActorClient
from travel_grpo.prompts.actor_policy import ACTOR_RUNTIME_POLICY_VERSION

MODELS = {
    "baseline": "Qwen/Qwen3.5-2B",
    "sft": "outputs/models/sft-merged",
    "grpo": "outputs/models/grpo-merged",
}


# [项目注释] 功能：`load_tasks`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：read_table, to_pylist, tuple, ValueError。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `list[dict]`；具体值由各分支决定。
def load_tasks(path: Path) -> list[dict]:
    table = pq.read_table(path)
    expected = ("task_id", "composition", "difficulty", "source_split", "prompt")
    if tuple(table.column_names) != expected:
        raise ValueError(f"evaluation Parquet schema drift: {table.column_names}")
    return table.to_pylist()


# [项目注释] 功能：`_sha256`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：sha256, hexdigest, open, iter。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# [项目注释] 功能：`_load_subset_manifest`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：tuple, Counter, loads, isinstance。
# [项目注释] 输入：`path`: Path；`records`: Sequence[Mapping[str, object]]。
# [项目注释] 输出：标注返回 `dict`；具体值由各分支决定。
def _load_subset_manifest(path: Path, records: Sequence[Mapping[str, object]]) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"subset manifest does not exist: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("subset manifest must contain a JSON object")
    if manifest.get("schema_version") != "travel-evaluation-subset-v1":
        raise ValueError("unsupported subset manifest schema")
    if manifest.get("source_task_count") != 471:
        raise ValueError("subset manifest must originate from the 471-task test pool")

    manifest_ids = manifest.get("task_ids")
    manifest_compositions = manifest.get("compositions")
    if not isinstance(manifest_ids, list) or not isinstance(manifest_compositions, list):
        raise ValueError("subset manifest must contain task_ids and compositions lists")
    dataset_ids = tuple(str(row["task_id"]) for row in records)
    dataset_compositions = tuple(str(row["composition"]) for row in records)
    if tuple(str(value) for value in manifest_ids) != dataset_ids:
        raise ValueError("subset manifest task IDs do not match the dataset")
    if tuple(str(value) for value in manifest_compositions) != dataset_compositions:
        raise ValueError("subset manifest compositions do not match the dataset")
    if manifest.get("selected_task_count") != len(records):
        raise ValueError("subset manifest count does not match the dataset")
    if len(set(dataset_ids)) != len(dataset_ids):
        raise ValueError("subset dataset task IDs must be unique")

    actual_counts = Counter(dataset_compositions)
    quotas = manifest.get("quotas")
    if not isinstance(quotas, dict):
        raise ValueError("subset manifest must contain composition quotas")
    normalized_quotas = {str(key): int(value) for key, value in quotas.items()}
    if dict(actual_counts) != normalized_quotas:
        raise ValueError(
            "subset dataset composition counts do not match manifest quotas"
        )
    return manifest


async def _run_pending_tasks(
    pending: Sequence[dict],
    *,
    selected_count: int,
    completed: dict[str, dict],
    actor: OpenAICompatibleActorClient,
    simulator: UserSimulatorRuntime,
    output: Path,
    concurrency: int,
    retry_infrastructure_invalid: bool,
    public_control_enabled: bool,
) -> None:
    """Run independent tasks with a bounded number of active rollouts.

    Each task keeps its existing sequential 20-turn rollout. The semaphore
    only overlaps different tasks, so per-task environment state and atomic
    checkpoints remain isolated while the Actor and eval simulator can serve
    more than one task at a time.
    """

    semaphore = asyncio.Semaphore(concurrency)
    finished = selected_count - len(pending)

    # [项目注释] 功能：`run_one`：异步地编排一个训练、采集、评测或 replay 流程，并汇总其结果。 主要协作调用：str, attach_attempt_history,
    # [项目注释]    atomic_json, print。
    # [项目注释] 输入：`task`: dict。
    # [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
    async def run_one(task: dict) -> None:
        nonlocal finished
        async with semaphore:
            task_id = str(task["task_id"])
            old = completed.get(task_id)
            if old is not None and (
                old.get("infrastructure_valid") is True
                or not retry_infrastructure_invalid
            ):
                return
            result = await rollout_task(
                task,
                actor=actor,
                simulator=simulator,
                source_root=ROOT / "environments/UserBench",
                public_control_enabled=public_control_enabled,
            )
            result = attach_attempt_history(result, old)
            atomic_json(task_path(output, task_id), result)
            completed[task_id] = result
            finished += 1
            print(
                f"[evaluation] completed {finished}/{selected_count} task={task_id} "
                f"valid={result.get('infrastructure_valid') is True}",
                flush=True,
            )

    # Wait for every task before propagating an exception. This lets already
    # successful tasks finish and persist their checkpoints, so --resume can
    # continue cleanly after a single task failure.
    outcomes = await asyncio.gather(
        *(run_one(task) for task in pending), return_exceptions=True
    )
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome


# [项目注释] 功能：`run`：异步地编排一个训练、采集、评测或 replay 流程，并汇总其结果。 主要协作调用：load_tasks, to_dict, loads, from_environment。
# [项目注释] 输入：`args`: argparse.Namespace。
# [项目注释] 输出：标注返回 `int`；具体值由各分支决定。
async def run(args: argparse.Namespace) -> int:
    records = load_tasks(args.dataset)
    subset_manifest = None
    if args.subset_manifest is not None:
        subset_manifest = _load_subset_manifest(args.subset_manifest, records)
        contract = build_subset_contract(
            records, simulator_endpoint=os.environ.get("EVAL_USER_SIM_BASE_URL")
        )
        evaluation_mode = "subset"
    else:
        contract = build_contract(
            records, simulator_endpoint=os.environ.get("EVAL_USER_SIM_BASE_URL")
        )
        evaluation_mode = "formal"
    selected = records[: args.limit] if args.limit is not None else records
    if args.output is not None:
        output = args.output
    elif subset_manifest is not None:
        output = ROOT / "outputs/evaluation" / f"{args.stage}-subset-{args.subset_manifest.stem}"
    else:
        output = ROOT / "outputs/evaluation" / args.stage
    contract_document = contract.to_dict(stage=args.stage, model=args.model)
    contract_document["evaluation_mode"] = evaluation_mode
    public_control_enabled = not args.raw_open_loop
    phase_guard_version = (
        PUBLIC_CONTROL_PHASE_GUARD_VERSION
        if public_control_enabled
        else "none"
    )
    contract_document["actor_policy_version"] = ACTOR_RUNTIME_POLICY_VERSION
    contract_document["public_control_enabled"] = public_control_enabled
    contract_document["phase_guard_version"] = phase_guard_version
    if subset_manifest is not None:
        contract_document["subset_manifest_sha256"] = _sha256(args.subset_manifest)
        contract_document["subset_manifest_schema"] = subset_manifest["schema_version"]
    # Normalize tuples and other JSON-compatible containers before comparing
    # with a contract loaded from disk. Without this, resume can reject an
    # identical contract because JSON reloads tuples as lists.
    contract_document = json.loads(json.dumps(contract_document))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "valid": True,
                    "stage": args.stage,
                    "model": args.model,
                    "contract_hash": contract.contract_hash,
                    "evaluation_mode": evaluation_mode,
                    "contract_tasks": len(contract.task_ids),
                    "frozen_tasks": len(records),
                    "would_run": len(selected),
                    "composition_counts": dict(
                        Counter(str(task["composition"]) for task in selected)
                    ),
                    "concurrency": args.concurrency,
                    "output": str(output),
                    "actor_policy_version": ACTOR_RUNTIME_POLICY_VERSION,
                    "public_control_enabled": public_control_enabled,
                    "phase_guard_version": phase_guard_version,
                    "subset_manifest": str(args.subset_manifest)
                    if args.subset_manifest is not None
                    else None,
                },
                indent=2,
            )
        )
        return 0
    simulator = UserSimulatorRuntime.from_environment(SimulatorRole.EVAL)
    actor_runtime = ActorRuntime.from_environment()
    actor_runtime.require_model(args.model)
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract_document:
            raise ValueError("existing evaluation contract differs from this run")
    else:
        if any(output.iterdir()) and not args.resume:
            raise FileExistsError("evaluation output is non-empty; use --resume")
        atomic_json(contract_path, contract_document)
    completed = load_completed(output)
    actor = OpenAICompatibleActorClient(actor_runtime)
    try:
        pending = [
            task
            for task in selected
            if not (
                (old := completed.get(task["task_id"])) is not None
                and (
                    old.get("infrastructure_valid") is True
                    or not args.retry_infrastructure_invalid
                )
            )
        ]
        print(
            f"[evaluation] stage={args.stage} selected={len(selected)} "
            f"pending={len(pending)} concurrency={args.concurrency}",
            flush=True,
        )
        await _run_pending_tasks(
            pending,
            selected_count=len(selected),
            completed=completed,
            actor=actor,
            simulator=simulator,
            output=output,
            concurrency=args.concurrency,
            retry_infrastructure_invalid=args.retry_infrastructure_invalid,
            public_control_enabled=public_control_enabled,
        )
    finally:
        await actor.close()
    ordered = [completed[task_id] for task_id in contract.task_ids if task_id in completed]
    write_results_jsonl(output / "results.jsonl", ordered)
    summary = summarize_results(
        ordered,
        expected_task_ids=contract.task_ids,
        expected_compositions=contract.compositions,
    )
    atomic_json(output / "summary.json", summary)
    atomic_json(
        output / "run_manifest.json",
        {
            "schema_version": "travel-evaluation-run-v1",
            "stage": args.stage,
            "model": args.model,
            "evaluation_mode": evaluation_mode,
            "contract_hash": contract.contract_hash,
            "actor_policy_version": ACTOR_RUNTIME_POLICY_VERSION,
            "public_control_enabled": public_control_enabled,
            "phase_guard_version": phase_guard_version,
            "completed_tasks": len(ordered),
            "expected_tasks": len(contract.task_ids),
            "formal_complete": (
                evaluation_mode == "formal"
                and len(ordered) == FORMAL_EVALUATION_TASK_COUNT
            ),
            "subset_complete": (
                evaluation_mode == "subset" and len(ordered) == len(contract.task_ids)
            ),
        },
    )
    print(json.dumps(summary, indent=2))
    return 0
