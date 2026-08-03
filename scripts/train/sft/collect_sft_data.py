"""Collect DeepSeek-V4-Flash teacher trajectories from UserBench SFT tasks."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from travel_grpo.envs.userbench_interaction import (  # noqa: E402
    SimulatorRole,
    UserSimulatorRuntime,
)
from travel_grpo.models.openai_compatible import (  # noqa: E402
    OpenAICompatibleTeacherClient,
    TeacherRuntime,
)
from travel_grpo.training.sft_collection import (  # noqa: E402
    assert_disjoint_from_evaluation,
    collect_teacher_outcomes,
    initialize_teacher_run,
    load_teacher_outcome_checkpoints,
    load_teacher_task_pool,
    summarize_teacher_outcomes,
    validate_teacher_collection_config,
    write_teacher_outcome_checkpoint,
    write_teacher_collection_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/train/sft/teacher_collection.yaml",
    )
    parser.add_argument(
        "--input", type=Path, default=ROOT / "data/sft/tasks_train.jsonl"
    )
    parser.add_argument(
        "--evaluation-tasks",
        type=Path,
        default=ROOT / "data/evaluation/tasks.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/teacher_trajectories/sft_train.accepted.jsonl",
    )
    parser.add_argument("--rejected-output", type=Path)
    parser.add_argument("--diagnostics-output", type=Path)
    parser.add_argument(
        "--source-root", type=Path, default=ROOT / "environments/UserBench"
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="per-task atomic checkpoint directory (defaults beside --output)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an existing run with the same ordered task IDs and policy",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate task/model/runtime contracts without making API calls",
    )
    return parser


async def run(args: argparse.Namespace) -> dict[str, object]:
    validate_teacher_collection_config(args.config)
    artifact_stem = args.output.stem
    if artifact_stem.endswith(".accepted"):
        artifact_stem = artifact_stem[: -len(".accepted")]
    rejected_output = args.rejected_output or args.output.with_name(
        f"{artifact_stem}.rejected.jsonl"
    )
    diagnostics_output = args.diagnostics_output or args.output.with_name(
        f"{artifact_stem}.diagnostics.jsonl"
    )
    outputs = (args.output, rejected_output, diagnostics_output)
    run_dir = args.run_dir or args.output.parent / f".{artifact_stem}.run"
    if len(set(outputs)) != 3:
        raise ValueError("accepted, rejected, and diagnostics outputs must be distinct")
    if not args.force and not args.dry_run and not args.resume:
        existing = next((path for path in outputs if path.exists()), None)
        if existing is not None:
            raise FileExistsError(f"trajectory output already exists: {existing}")
    if args.attempts <= 0:
        raise ValueError("--attempts must be positive")
    tasks = load_teacher_task_pool(args.input)
    assert_disjoint_from_evaluation(tasks, args.evaluation_tasks)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        tasks = tasks[: args.limit]

    teacher_runtime = TeacherRuntime.from_environment()
    simulator_runtime = UserSimulatorRuntime.from_environment(SimulatorRole.COLLECTION)
    summary: dict[str, object] = {
        "task_count": len(tasks),
        "teacher_model": teacher_runtime.model,
        "simulator_model": simulator_runtime.model,
        "simulator_role": simulator_runtime.role.value,
        "output": str(args.output),
        "rejected_output": str(rejected_output),
        "diagnostics_output": str(diagnostics_output),
        "trajectory_attempts": args.attempts,
        "run_dir": str(run_dir),
        "resume": bool(args.resume),
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        return summary

    task_ids = [str(task["task_id"]) for task in tasks]
    initialize_teacher_run(run_dir, task_ids, resume=args.resume)
    checkpointed = load_teacher_outcome_checkpoints(run_dir, task_ids)
    pending = [task for task in tasks if str(task["task_id"]) not in checkpointed]
    new_outcomes = ()
    if pending:
        teacher = OpenAICompatibleTeacherClient(teacher_runtime)
        completed_now = 0

        def checkpoint(outcome):
            nonlocal completed_now
            path = write_teacher_outcome_checkpoint(outcome, run_dir)
            completed_now += 1
            print(
                json.dumps(
                    {
                        "event": "teacher_task_completed",
                        "completed": len(checkpointed) + completed_now,
                        "total": len(tasks),
                        "accepted": outcome.accepted,
                        "attempts": len(outcome.attempts),
                        "checkpoint": str(path),
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
                flush=True,
            )

        try:
            new_outcomes = await collect_teacher_outcomes(
                pending,
                teacher=teacher,
                simulator=simulator_runtime,
                concurrency=args.concurrency,
                max_attempts=args.attempts,
                source_root=args.source_root,
                on_outcome=checkpoint,
            )
        finally:
            await teacher.close()
    combined = {**checkpointed, **{value.task_id: value for value in new_outcomes}}
    outcomes = tuple(combined[task_id] for task_id in task_ids)
    write_teacher_collection_artifacts(
        outcomes,
        accepted_path=args.output,
        rejected_path=rejected_output,
        diagnostics_path=diagnostics_output,
        force=args.force or args.resume,
    )
    accepted = [value.trajectory for value in outcomes if value.trajectory is not None]
    summary.update(
        {
            "accepted": len(accepted),
            "rejected": len(outcomes) - len(accepted),
            "attempts_used": sum(len(value.attempts) for value in outcomes),
            "resumed_tasks": len(checkpointed),
            "new_tasks": len(new_outcomes),
            "quality": summarize_teacher_outcomes(outcomes),
        }
    )
    return summary


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
