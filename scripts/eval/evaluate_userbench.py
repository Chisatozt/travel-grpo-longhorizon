#!/usr/bin/env python3
"""Run or dry-run the frozen 471-task UserBench evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pyarrow.parquet as pq

from travel_grpo.envs.userbench_interaction import SimulatorRole, UserSimulatorRuntime
from travel_grpo.evaluation.artifacts import (
    atomic_json,
    attach_attempt_history,
    load_completed,
    task_path,
    write_results_jsonl,
)
from travel_grpo.evaluation.contracts import STAGES, build_contract
from travel_grpo.evaluation.rollout import rollout_task
from travel_grpo.evaluation.summary import summarize_results
from travel_grpo.models.vllm_policy import ActorRuntime, OpenAICompatibleActorClient

MODELS = {
    "baseline": "Qwen/Qwen3.5-2B",
    "sft": "outputs/models/sft-merged",
    "grpo": "outputs/models/grpo-merged",
}


def load_tasks(path: Path) -> list[dict]:
    table = pq.read_table(path)
    expected = ("task_id", "composition", "difficulty", "source_split", "prompt")
    if tuple(table.column_names) != expected:
        raise ValueError(f"evaluation Parquet schema drift: {table.column_names}")
    return table.to_pylist()


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
) -> None:
    """Run independent tasks with a bounded number of active rollouts.

    Each task keeps its existing sequential 20-turn rollout. The semaphore
    only overlaps different tasks, so per-task environment state and atomic
    checkpoints remain isolated while the Actor and eval simulator can serve
    more than one task at a time.
    """

    semaphore = asyncio.Semaphore(concurrency)
    finished = selected_count - len(pending)

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


async def run(args: argparse.Namespace) -> int:
    records = load_tasks(args.dataset)
    contract = build_contract(
        records, simulator_endpoint=os.environ.get("EVAL_USER_SIM_BASE_URL")
    )
    selected = records[: args.limit] if args.limit is not None else records
    output = args.output or ROOT / "outputs/evaluation" / args.stage
    contract_document = contract.to_dict(stage=args.stage, model=args.model)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "valid": True,
                    "stage": args.stage,
                    "model": args.model,
                    "contract_hash": contract.contract_hash,
                    "frozen_tasks": len(records),
                    "would_run": len(selected),
                    "concurrency": args.concurrency,
                    "output": str(output),
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
    atomic_json(output / "run_manifest.json", {"schema_version": "travel-evaluation-run-v1", "stage": args.stage, "model": args.model, "contract_hash": contract.contract_hash, "completed_tasks": len(ordered), "formal_complete": len(ordered) == 471})
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--model")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/evaluation/tasks.parquet")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="maximum number of tasks evaluated concurrently (default: 1)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-infrastructure-invalid", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.model = args.model or MODELS[args.stage]
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
