"""Collect DeepSeek-V4-Flash teacher trajectories from UserBench SFT tasks."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections import Counter
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
from travel_grpo.training.sft.collection import (  # noqa: E402
    TeacherCollectionError,
    assert_disjoint_from_evaluation,
    build_stratified_task_plan,
    collect_teacher_outcomes,
    initialize_teacher_run,
    load_teacher_outcome_checkpoints,
    load_teacher_task_pool,
    select_stratified_task_wave,
    summarize_teacher_outcomes,
    validate_teacher_collection_config,
    write_stratified_selection_manifest,
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
    parser.add_argument("--silver-output", type=Path)
    parser.add_argument("--diagnostics-output", type=Path)
    parser.add_argument(
        "--source-root", type=Path, default=ROOT / "environments/UserBench"
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--stratify-by",
        choices=("composition",),
        help="enable adaptive accepted-trajectory collection by composition",
    )
    parser.add_argument(
        "--target-accepted",
        "--stratified-target",
        dest="target_accepted",
        type=int,
        help="stop after this many unique Gold+Silver trajectories are admitted",
    )
    parser.add_argument(
        "--stratified-wave-size",
        type=int,
        default=32,
        help="maximum number of new tasks attempted in each adaptive wave",
    )
    parser.add_argument(
        "--sampling-seed",
        default="sft-stratified-v1",
        help="stable seed for ordering candidates within each stratum",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="adaptive selection manifest (defaults under --run-dir)",
    )
    parser.add_argument(
        "--batch-config",
        type=Path,
        default=ROOT / "configs/train/sft/teacher_smoke_batches.json",
        help="fixed task-batch manifest used with --batch",
    )
    parser.add_argument(
        "--batch",
        help="named three-task batch from --batch-config; mutually exclusive with --limit",
    )
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stratified_manifest_core(
    *,
    input_path: Path,
    tasks: tuple[dict[str, object], ...],
    quotas: dict[str, int],
    target: int,
    field: str,
    seed: str,
    wave_size: int,
) -> dict[str, object]:
    return {
        "schema_version": "userbench-teacher-stratified-selection-v1",
        "source_path": str(input_path.resolve()),
        "source_sha256": _sha256_file(input_path),
        "stratification_field": field,
        "target_accepted": target,
        "sampling_seed": seed,
        "wave_size": wave_size,
        "candidate_count": len(tasks),
        "candidate_task_ids": [str(task["task_id"]) for task in tasks],
        "quotas": dict(sorted(quotas.items())),
    }


def _write_stratified_state(
    path: Path,
    *,
    core: dict[str, object],
    status: str,
    outcomes: dict[str, object],
    waves: list[dict[str, object]],
) -> None:
    accepted_ids = [
        task_id
        for task_id in core["candidate_task_ids"]
        if getattr(outcomes.get(task_id), "accepted", False)
    ]
    accepted_by_stratum: Counter[str] = Counter()
    for task_id in accepted_ids:
        outcome = outcomes[task_id]
        trajectory = getattr(outcome, "trajectory", None)
        if trajectory is not None:
            accepted_by_stratum[str(trajectory.composition)] += 1
    quotas = {str(key): int(value) for key, value in core["quotas"].items()}
    state = {
        **core,
        "status": status,
        "completed_task_ids": [
            task_id for task_id in core["candidate_task_ids"] if task_id in outcomes
        ],
        "accepted_task_ids": accepted_ids,
        "accepted_by_stratum": dict(sorted(accepted_by_stratum.items())),
        "remaining_by_stratum": {
            key: max(value - accepted_by_stratum[key], 0)
            for key, value in sorted(quotas.items())
        },
        "waves": waves,
    }
    write_stratified_selection_manifest(path, state)


async def _run_stratified_collection(
    args: argparse.Namespace,
    *,
    tasks: tuple[dict[str, object], ...],
    quotas: dict[str, int],
    teacher_runtime: TeacherRuntime,
    simulator_runtime: UserSimulatorRuntime,
    output: Path,
    silver_output: Path,
    rejected_output: Path,
    diagnostics_output: Path,
    run_dir: Path,
    summary: dict[str, object],
) -> dict[str, object]:
    """Collect waves until every composition's accepted quota is satisfied."""

    task_ids = [str(task["task_id"]) for task in tasks]
    initialize_teacher_run(run_dir, task_ids, resume=args.resume)
    manifest_path = args.selection_manifest or run_dir / "selection_manifest.json"
    core = _stratified_manifest_core(
        input_path=args.input,
        tasks=tasks,
        quotas=quotas,
        target=int(args.target_accepted),
        field=str(args.stratify_by),
        seed=args.sampling_seed,
        wave_size=args.stratified_wave_size,
    )
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"stratified selection manifest already exists; pass --resume: {manifest_path}"
            )
        try:
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TeacherCollectionError(
                f"stratified selection manifest is unreadable: {manifest_path}"
            ) from exc
        if not isinstance(current, dict) or any(current.get(key) != value for key, value in core.items()):
            raise TeacherCollectionError(
                "stratified selection manifest does not match input, seed, quota, or candidate order"
            )
        waves = list(current.get("waves", []))
    else:
        if args.resume:
            raise FileNotFoundError(f"stratified selection manifest does not exist: {manifest_path}")
        waves = []
        _write_stratified_state(
            manifest_path,
            core=core,
            status="running",
            outcomes={},
            waves=waves,
        )

    initial_checkpointed = load_teacher_outcome_checkpoints(run_dir, task_ids)
    outcomes: dict[str, object] = dict(initial_checkpointed)
    completed_this_run = 0
    teacher = OpenAICompatibleTeacherClient(teacher_runtime)
    try:
        while True:
            checkpointed = load_teacher_outcome_checkpoints(run_dir, task_ids)
            outcomes = dict(checkpointed)
            accepted_ids = {
                task_id for task_id, value in outcomes.items() if value.accepted
            }
            accepted_by_stratum = Counter(
                str(value.trajectory.composition)
                for value in outcomes.values()
                if value.accepted and value.trajectory is not None
            )
            if all(accepted_by_stratum[key] >= value for key, value in quotas.items()):
                break
            wave_tasks = select_stratified_task_wave(
                tasks,
                quotas=quotas,
                attempted_task_ids=set(outcomes),
                accepted_task_ids=accepted_ids,
                field=str(args.stratify_by),
                wave_size=args.stratified_wave_size,
            )
            if not wave_tasks:
                _write_stratified_state(
                    manifest_path,
                    core=core,
                    status="exhausted",
                    outcomes=outcomes,
                    waves=waves,
                )
                missing = {
                    key: max(value - accepted_by_stratum[key], 0)
                    for key, value in quotas.items()
                    if accepted_by_stratum[key] < value
                }
                raise TeacherCollectionError(
                    f"stratified candidate pool exhausted before quotas were met: {missing}"
                )
            wave_index = len(waves) + 1
            completed_before_wave = len(outcomes)
            waves.append(
                {
                    "wave": wave_index,
                    "task_ids": [str(task["task_id"]) for task in wave_tasks],
                    "status": "running",
                    "completed_before_wave": completed_before_wave,
                }
            )
            _write_stratified_state(
                manifest_path,
                core=core,
                status="running",
                outcomes=outcomes,
                waves=waves,
            )

            def checkpoint(outcome: object) -> None:
                nonlocal completed_this_run
                write_teacher_outcome_checkpoint(outcome, run_dir)
                completed_this_run += 1
                print(
                    json.dumps(
                        {
                            "event": "teacher_stratified_task_completed",
                            "wave": wave_index,
                            "task_id": outcome.task_id,
                            "accepted": outcome.accepted,
                            "quality_tier": outcome.quality_tier,
                            "attempts": len(outcome.attempts),
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )

            new_outcomes = await collect_teacher_outcomes(
                wave_tasks,
                teacher=teacher,
                simulator=simulator_runtime,
                concurrency=args.concurrency,
                max_attempts=args.attempts,
                source_root=args.source_root,
                on_outcome=checkpoint,
            )
            waves[-1] = {
                **waves[-1],
                "status": "complete",
                "accepted": sum(value.accepted for value in new_outcomes),
                "attempted": len(new_outcomes),
            }
            outcomes = dict(load_teacher_outcome_checkpoints(run_dir, task_ids))
            ordered_outcomes = tuple(outcomes[task_id] for task_id in task_ids if task_id in outcomes)
            write_teacher_collection_artifacts(
                ordered_outcomes,
                accepted_path=output,
                silver_path=silver_output,
                rejected_path=rejected_output,
                diagnostics_path=diagnostics_output,
                force=True,
            )
            _write_stratified_state(
                manifest_path,
                core=core,
                status="running",
                outcomes=outcomes,
                waves=waves,
            )
    finally:
        await teacher.close()

    ordered_outcomes = tuple(outcomes[task_id] for task_id in task_ids if task_id in outcomes)
    write_teacher_collection_artifacts(
        ordered_outcomes,
        accepted_path=output,
        silver_path=silver_output,
        rejected_path=rejected_output,
        diagnostics_path=diagnostics_output,
        force=True,
    )
    _write_stratified_state(
        manifest_path,
        core=core,
        status="complete",
        outcomes=outcomes,
        waves=waves,
    )
    summary.update(
        {
            "accepted": sum(value.accepted for value in outcomes.values()),
            "gold": sum(value.quality_tier == "gold" for value in outcomes.values()),
            "silver": sum(value.quality_tier == "silver" for value in outcomes.values()),
            "rejected": sum(not value.accepted for value in outcomes.values()),
            "attempts_used": sum(len(value.attempts) for value in outcomes.values()),
            "resumed_tasks": len(initial_checkpointed),
            "new_tasks": completed_this_run,
            "quality": summarize_teacher_outcomes(ordered_outcomes),
            "selection_manifest": str(manifest_path),
            "stratified_complete": True,
            "waves": len(waves),
        }
    )
    return summary


async def run(args: argparse.Namespace) -> dict[str, object]:
    validate_teacher_collection_config(args.config)
    artifact_stem = args.output.stem
    if artifact_stem.endswith(".accepted"):
        artifact_stem = artifact_stem[: -len(".accepted")]
    rejected_output = args.rejected_output or args.output.with_name(
        f"{artifact_stem}.rejected.jsonl"
    )
    silver_output = args.silver_output or args.output.with_name(
        f"{artifact_stem}.silver.jsonl"
    )
    diagnostics_output = args.diagnostics_output or args.output.with_name(
        f"{artifact_stem}.diagnostics.jsonl"
    )
    outputs = (args.output, silver_output, rejected_output, diagnostics_output)
    run_dir = args.run_dir or args.output.parent / f".{artifact_stem}.run"
    if len(set(outputs)) != 4:
        raise ValueError("gold, silver, rejected, and diagnostics outputs must be distinct")
    if not args.force and not args.dry_run and not args.resume:
        existing = next((path for path in outputs if path.exists()), None)
        if existing is not None:
            raise FileExistsError(f"trajectory output already exists: {existing}")
    if args.attempts <= 0:
        raise ValueError("--attempts must be positive")
    if args.target_accepted is not None and args.stratify_by is None:
        args.stratify_by = "composition"
    stratified = args.stratify_by is not None or args.target_accepted is not None
    if stratified and args.target_accepted is None:
        raise ValueError("--stratify-by requires --target-accepted")
    if stratified and (args.limit is not None or args.batch is not None):
        raise ValueError("adaptive stratification cannot be combined with --limit or --batch")
    if stratified and args.stratified_wave_size <= 0:
        raise ValueError("--stratified-wave-size must be positive")
    tasks = load_teacher_task_pool(args.input)
    assert_disjoint_from_evaluation(tasks, args.evaluation_tasks)
    if args.batch is not None and args.limit is not None:
        raise ValueError("--batch and --limit are mutually exclusive")
    if args.batch is not None:
        with args.batch_config.open("r", encoding="utf-8") as handle:
            batch_config = json.load(handle)
        source = batch_config.get("source")
        expected_source = args.input.resolve()
        configured_source = (ROOT / str(source)).resolve() if source else None
        if configured_source != expected_source:
            raise ValueError(
                f"batch source {configured_source} does not match input {expected_source}"
            )
        batches = batch_config.get("batches")
        selected = batches.get(args.batch) if isinstance(batches, dict) else None
        selected_ids = selected.get("task_ids") if isinstance(selected, dict) else None
        if (
            not isinstance(selected_ids, list)
            or len(selected_ids) != 3
            or len(set(selected_ids)) != 3
            or not all(isinstance(value, str) and value for value in selected_ids)
        ):
            raise ValueError(f"batch {args.batch!r} must contain exactly three unique task IDs")
        task_by_id = {str(task["task_id"]): task for task in tasks}
        missing = [task_id for task_id in selected_ids if task_id not in task_by_id]
        if missing:
            raise ValueError(f"batch {args.batch!r} contains unknown task IDs: {missing}")
        tasks = [task_by_id[task_id] for task_id in selected_ids]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        tasks = tasks[: args.limit]

    quotas: dict[str, int] = {}
    if stratified:
        tasks, quotas = build_stratified_task_plan(
            tasks,
            target=args.target_accepted,
            field=args.stratify_by,
            seed=args.sampling_seed,
        )

    teacher_runtime = TeacherRuntime.from_environment()
    simulator_runtime = UserSimulatorRuntime.from_environment(SimulatorRole.COLLECTION)
    summary: dict[str, object] = {
        "task_count": len(tasks),
        "task_ids": [str(task["task_id"]) for task in tasks],
        "batch": args.batch,
        "teacher_model": teacher_runtime.model,
        "simulator_model": simulator_runtime.model,
        "simulator_role": simulator_runtime.role.value,
        "output": str(args.output),
        "silver_output": str(silver_output),
        "rejected_output": str(rejected_output),
        "diagnostics_output": str(diagnostics_output),
        "trajectory_attempts": args.attempts,
        "run_dir": str(run_dir),
        "resume": bool(args.resume),
        "dry_run": bool(args.dry_run),
        "stratified": stratified,
    }
    if stratified:
        summary.update(
            {
                "stratification_field": args.stratify_by,
                "target_accepted": args.target_accepted,
                "sampling_seed": args.sampling_seed,
                "stratified_wave_size": args.stratified_wave_size,
                "quotas": quotas,
            }
        )
    if args.dry_run:
        return summary

    if stratified:
        return await _run_stratified_collection(
            args,
            tasks=tasks,
            quotas=quotas,
            teacher_runtime=teacher_runtime,
            simulator_runtime=simulator_runtime,
            output=args.output,
            silver_output=silver_output,
            rejected_output=rejected_output,
            diagnostics_output=diagnostics_output,
            run_dir=run_dir,
            summary=summary,
        )

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
                        "quality_tier": outcome.quality_tier,
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
        silver_path=silver_output,
        rejected_path=rejected_output,
        diagnostics_path=diagnostics_output,
        force=args.force or args.resume,
    )
    accepted = [value.trajectory for value in outcomes if value.trajectory is not None]
    summary.update(
        {
            "accepted": len(accepted),
            "gold": sum(value.quality_tier == "gold" for value in outcomes),
            "silver": sum(value.quality_tier == "silver" for value in outcomes),
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
