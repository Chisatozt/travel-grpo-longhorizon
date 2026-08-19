#!/usr/bin/env python3
"""Export and evaluate GRPO checkpoints sequentially, then shut down.

The driver is intended to run inside tmux after the GPU is enabled.  It:

1. exports each veRL FSDP actor checkpoint to a standalone HF model;
2. starts a fresh local vLLM Actor for that model;
3. resumes the frozen 200-task public-control evaluation;
4. retries infrastructure-invalid tasks until all 200 are valid;
5. shuts the host down after all checkpoints complete or progress stalls.

Progress means an increase in the number of task artifacts whose
``infrastructure_valid`` field is true.  Merely writing an invalid task does
not reset the stall timer.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = Path(
    "outputs/models/grpo-sft-merged-turn-credit-20-recover50-to200-save25"
)
DEFAULT_SUBSET_DATASET = Path(
    "outputs/evaluation/subsets/tasks_200_proportional_v1.parquet"
)
DEFAULT_SUBSET_MANIFEST = Path(
    "outputs/evaluation/subsets/tasks_200_proportional_v1.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/evaluation/turn-credit-checkpoints-200-task-public-guarded-c4"
)
DEFAULT_STEPS = (100, 150, 200)


@dataclass(frozen=True)
# [项目注释] 类型：`Progress` 封装相关状态、协议或数据结构。类属性和方法共同维护其不变量。
class Progress:
    total: int
    valid: int
    invalid: int


# [项目注释] 功能：`parse_args`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：ArgumentParser, add_argument, parse_args, list。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `argparse.Namespace`；具体值由各分支决定。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--subset-dataset", type=Path, default=DEFAULT_SUBSET_DATASET)
    parser.add_argument("--subset-manifest", type=Path, default=DEFAULT_SUBSET_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_STEPS))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--expected-tasks", type=int, default=200)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--stall-minutes", type=float, default=20.0)
    parser.add_argument("--shutdown-delay-minutes", type=float, default=5.0)
    parser.add_argument("--actor-startup-minutes", type=float, default=10.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print commands without exporting, evaluating, or shutting down.",
    )
    parser.add_argument(
        "--no-shutdown",
        action="store_true",
        help="Run the sequence but never shut down the host (for debugging only).",
    )
    return parser.parse_args()


# [项目注释] 功能：`utc_now`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：isoformat, now。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# [项目注释] 功能：`resolve_from_root`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：is_absolute。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `Path`；具体值由各分支决定。
def resolve_from_root(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def relative_model_name(path: Path) -> str:
    """Return the stable repo-relative model name required by the eval contract."""
    return path.resolve().relative_to(ROOT).as_posix()


# [项目注释] 功能：`read_subset_count`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：loads, int, read_text, ValueError。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `int`；具体值由各分支决定。
def read_subset_count(path: Path) -> int:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != "travel-evaluation-subset-v1":
        raise ValueError(f"unsupported subset manifest: {path}")
    return int(document["selected_task_count"])


# [项目注释] 功能：`task_progress`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：glob, Progress, is_dir, loads。
# [项目注释] 输入：`run_dir`: Path。
# [项目注释] 输出：标注返回 `Progress`；具体值由各分支决定。
def task_progress(run_dir: Path) -> Progress:
    task_dir = run_dir / "tasks"
    total = valid = invalid = 0
    if not task_dir.is_dir():
        return Progress(total=0, valid=0, invalid=0)
    for path in task_dir.glob("*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        total += 1
        if row.get("infrastructure_valid") is True:
            valid += 1
        else:
            invalid += 1
    return Progress(total=total, valid=valid, invalid=invalid)


# [项目注释] 功能：`evaluation_complete`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：task_progress, bool, is_file,
# [项目注释]    loads。
# [项目注释] 输入：`run_dir`: Path；`expected_tasks`: int；`model_name`: str。
# [项目注释] 输出：标注返回 `bool`；具体值由各分支决定。
def evaluation_complete(run_dir: Path, expected_tasks: int, model_name: str) -> bool:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    progress = task_progress(run_dir)
    return bool(
        manifest.get("schema_version") == "travel-evaluation-run-v1"
        and manifest.get("model") == model_name
        and manifest.get("public_control_enabled") is True
        and manifest.get("subset_complete") is True
        and int(manifest.get("completed_tasks", -1)) == expected_tasks
        and progress.valid == expected_tasks
        and progress.invalid == 0
    )


# [项目注释] 功能：`merged_model_complete`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：bool, is_file, any, glob。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `bool`；具体值由各分支决定。
def merged_model_complete(path: Path) -> bool:
    return bool(
        (path / "config.json").is_file()
        and (path / "tokenizer_config.json").is_file()
        and (
            any(path.glob("*.safetensors"))
            or (path / "pytorch_model.bin").is_file()
        )
    )


# [项目注释] 功能：`atomic_status`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：mkdir, with_suffix, write_text, replace。
# [项目注释] 输入：`path`: Path；**`values`。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def atomic_status(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"schema_version": "checkpoint-eval-sequence-v1", "updated_at": utc_now(), **values}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


# [项目注释] 功能：`stop_process`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：print, killpg, wait, poll。
# [项目注释] 输入：`process`: subprocess.Popen[Any] | None；`name`: str。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def stop_process(process: subprocess.Popen[Any] | None, name: str) -> None:
    if process is None or process.poll() is not None:
        return
    print(f"[sequence] stopping {name} pid={process.pid}", flush=True)
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


# [项目注释] 功能：`actor_ready`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：Request, build_opener, rstrip, ProxyHandler。
# [项目注释] 输入：`base_url`: str；`api_key`: str；`model_name`: str。
# [项目注释] 输出：标注返回 `bool`；具体值由各分支决定。
def actor_ready(base_url: str, api_key: str, model_name: str) -> bool:
    url = base_url.rstrip("/") + "/models"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    # The Actor endpoint is local.  Explicitly bypass inherited HTTP proxies.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=5) as response:
            document = json.loads(response.read())
    except Exception:
        return False
    return model_name in {
        str(item.get("id"))
        for item in document.get("data", [])
        if isinstance(item, dict)
    }


# [项目注释] 功能：`wait_for_actor`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：TimeoutError, monotonic, poll,
# [项目注释]    actor_ready。
# [项目注释] 输入：`process`: subprocess.Popen[Any]；`base_url`: str；`api_key`: str；`model_name`:
# [项目注释]    str；`timeout_seconds`: float；`poll_seconds`: float。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def wait_for_actor(
    process: subprocess.Popen[Any],
    *,
    base_url: str,
    api_key: str,
    model_name: str,
    timeout_seconds: float,
    poll_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"vLLM Actor exited during startup with code {code}")
        if actor_ready(base_url, api_key, model_name):
            print(f"[sequence] Actor ready: {model_name}", flush=True)
            return
        time.sleep(min(poll_seconds, 10.0))
    raise TimeoutError(f"vLLM Actor was not ready after {timeout_seconds / 60:.1f} minutes")


# [项目注释] 功能：`export_command`：把内部结果序列化或导出到指定介质，并保持项目约定的格式。 主要协作调用：str。
# [项目注释] 输入：`actor_dir`: Path；`merged_dir`: Path。
# [项目注释] 输出：标注返回 `list[str]`；具体值由各分支决定。
def export_command(actor_dir: Path, merged_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor_dir),
        "--target_dir",
        str(merged_dir),
    ]


# [项目注释] 功能：`actor_command`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。
# [项目注释] 输入：`model_name`: str。
# [项目注释] 输出：标注返回 `list[str]`；具体值由各分支决定。
def actor_command(model_name: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_name,
        "--served-model-name",
        model_name,
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "32768",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
    ]


# [项目注释] 功能：`evaluation_command`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：str。
# [项目注释] 输入：`model_name`: str；`dataset`: Path；`subset_manifest`: Path；`run_dir`: Path；`concurrency`: int。
# [项目注释] 输出：标注返回 `list[str]`；具体值由各分支决定。
def evaluation_command(
    *,
    model_name: str,
    dataset: Path,
    subset_manifest: Path,
    run_dir: Path,
    concurrency: int,
) -> list[str]:
    return [
        sys.executable,
        "scripts/eval/evaluate_userbench.py",
        "--stage",
        "grpo",
        "--model",
        model_name,
        "--dataset",
        str(dataset),
        "--subset-manifest",
        str(subset_manifest),
        "--output",
        str(run_dir),
        "--concurrency",
        str(concurrency),
        "--resume",
        "--retry-infrastructure-invalid",
    ]


def monitor_evaluation(
    process: subprocess.Popen[Any],
    *,
    run_dir: Path,
    expected_tasks: int,
    stall_seconds: float,
    poll_seconds: float,
    initial_progress_at: float,
    status_path: Path,
    step: int,
) -> tuple[bool, float]:
    """Return (stalled, last_progress_at) for one evaluator invocation."""
    progress = task_progress(run_dir)
    best_valid = progress.valid
    last_progress_at = initial_progress_at
    print(
        f"[sequence] step={step} progress valid={progress.valid}/{expected_tasks} "
        f"total={progress.total} invalid={progress.invalid}",
        flush=True,
    )
    while process.poll() is None:
        time.sleep(poll_seconds)
        progress = task_progress(run_dir)
        if progress.valid > best_valid:
            best_valid = progress.valid
            last_progress_at = time.monotonic()
            print(
                f"[sequence] step={step} progress valid={progress.valid}/{expected_tasks} "
                f"total={progress.total} invalid={progress.invalid}",
                flush=True,
            )
        atomic_status(
            status_path,
            state="evaluating",
            step=step,
            valid_tasks=progress.valid,
            total_task_files=progress.total,
            invalid_tasks=progress.invalid,
            expected_tasks=expected_tasks,
        )
        if time.monotonic() - last_progress_at >= stall_seconds:
            return True, last_progress_at
    return False, last_progress_at


# [项目注释] 功能：`shutdown_after_grace`：清理运行时资源或恢复边界状态，保证后续调用不会继承脏状态。 主要协作调用：atomic_status, print, sleep, call。
# [项目注释] 输入：`reason`: str；`delay_seconds`: float；`no_shutdown`: bool；`status_path`: Path。
# [项目注释] 输出：标注返回 `int`；具体值由各分支决定。
def shutdown_after_grace(
    *,
    reason: str,
    delay_seconds: float,
    no_shutdown: bool,
    status_path: Path,
) -> int:
    atomic_status(status_path, state="shutdown_pending", reason=reason)
    print(
        f"[sequence] trigger: {reason}; shutdown in {delay_seconds / 60:.1f} minutes",
        flush=True,
    )
    time.sleep(delay_seconds)
    if no_shutdown:
        print(f"[sequence] --no-shutdown: would shut down because {reason}", flush=True)
        atomic_status(status_path, state="finished_no_shutdown", reason=reason)
        return 0
    print(f"[sequence] shutting down host: {reason}", flush=True)
    atomic_status(status_path, state="shutting_down", reason=reason)
    return subprocess.call(["/usr/bin/shutdown", "-h", "now"])


# [项目注释] 功能：`validate_inputs`：执行输入、状态或产物校验，并在不满足约束时返回诊断或抛出异常。 主要协作调用：resolve_from_root, read_subset_count,
# [项目注释]    ValueError, is_file。
# [项目注释] 输入：`args`: argparse.Namespace。
# [项目注释] 输出：标注返回 `tuple[Path, Path, Path, Path, int]`；具体值由各分支决定。
def validate_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, int]:
    run_root = resolve_from_root(args.run_root)
    dataset = resolve_from_root(args.subset_dataset)
    subset_manifest = resolve_from_root(args.subset_manifest)
    output_root = resolve_from_root(args.output_root)
    expected = read_subset_count(subset_manifest)
    if expected != args.expected_tasks:
        raise ValueError(
            f"subset contains {expected} tasks, expected {args.expected_tasks}"
        )
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    if args.concurrency <= 0 or args.poll_seconds <= 0:
        raise ValueError("concurrency and poll-seconds must be positive")
    if args.stall_minutes <= 0 or args.shutdown_delay_minutes < 0:
        raise ValueError("stall-minutes must be positive and shutdown delay non-negative")
    if len(set(args.steps)) != len(args.steps):
        raise ValueError("steps must be unique")
    for step in args.steps:
        actor_dir = run_root / f"global_step_{step}" / "actor"
        if not actor_dir.is_dir():
            raise FileNotFoundError(actor_dir)
    return run_root, dataset, subset_manifest, output_root, expected


# [项目注释] 功能：`print_dry_run`：编排一个训练、采集、评测或 replay 流程，并汇总其结果。 主要协作调用：print, relative_model_name, dumps, str。
# [项目注释] 输入：`args`: argparse.Namespace；`run_root`: Path；`dataset`: Path；`subset_manifest`:
# [项目注释]    Path；`output_root`: Path；`expected`: int。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
def print_dry_run(
    args: argparse.Namespace,
    *,
    run_root: Path,
    dataset: Path,
    subset_manifest: Path,
    output_root: Path,
    expected: int,
) -> None:
    plan = []
    for step in args.steps:
        actor_dir = run_root / f"global_step_{step}" / "actor"
        merged_dir = run_root / f"global_step_{step}-merged-eval"
        model_name = relative_model_name(merged_dir)
        run_dir = output_root / f"step-{step}" / "run"
        plan.append(
            {
                "step": step,
                "actor_checkpoint": str(actor_dir),
                "merged_model": str(merged_dir),
                "already_exported": merged_model_complete(merged_dir),
                "already_complete": evaluation_complete(run_dir, expected, model_name),
                "export_command": export_command(actor_dir, merged_dir),
                "actor_command": actor_command(model_name),
                "evaluation_command": evaluation_command(
                    model_name=model_name,
                    dataset=dataset,
                    subset_manifest=subset_manifest,
                    run_dir=run_dir,
                    concurrency=args.concurrency,
                ),
            }
        )
    print(
        json.dumps(
            {
                "valid": True,
                "dry_run": True,
                "expected_tasks": expected,
                "steps": list(args.steps),
                "stall_minutes": args.stall_minutes,
                "shutdown_delay_minutes": args.shutdown_delay_minutes,
                "public_control_enabled": True,
                "plan": plan,
            },
            indent=2,
        )
    )


# [项目注释] 功能：`main`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：parse_args, chdir, validate_inputs, strip。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `int`；具体值由各分支决定。
def main() -> int:
    args = parse_args()
    os.chdir(ROOT)
    run_root, dataset, subset_manifest, output_root, expected = validate_inputs(args)
    if args.dry_run:
        print_dry_run(
            args,
            run_root=run_root,
            dataset=dataset,
            subset_manifest=subset_manifest,
            output_root=output_root,
            expected=expected,
        )
        return 0

    actor_base_url = os.environ.get("ACTOR_BASE_URL", "").strip()
    actor_api_key = os.environ.get("ACTOR_API_KEY", "").strip()
    if not actor_base_url or not actor_api_key:
        raise ValueError("ACTOR_BASE_URL and ACTOR_API_KEY must be loaded from .env")

    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "sequence_status.json"
    atomic_status(status_path, state="starting", steps=list(args.steps))

    actor_process: subprocess.Popen[Any] | None = None
    evaluator_process: subprocess.Popen[Any] | None = None
    actor_log: IO[str] | None = None
    evaluator_log: IO[str] | None = None
    try:
        for step in args.steps:
            actor_dir = run_root / f"global_step_{step}" / "actor"
            merged_dir = run_root / f"global_step_{step}-merged-eval"
            model_name = relative_model_name(merged_dir)
            step_root = output_root / f"step-{step}"
            run_dir = step_root / "run"
            step_root.mkdir(parents=True, exist_ok=True)

            if evaluation_complete(run_dir, expected, model_name):
                print(f"[sequence] step={step} already complete; skipping", flush=True)
                continue

            if not merged_model_complete(merged_dir):
                if merged_dir.exists() and any(merged_dir.iterdir()):
                    raise RuntimeError(f"partial merged model requires inspection: {merged_dir}")
                atomic_status(status_path, state="exporting", step=step)
                command = export_command(actor_dir, merged_dir)
                print(f"[sequence] exporting step={step}: {' '.join(command)}", flush=True)
                subprocess.run(command, cwd=ROOT, check=True)
                if not merged_model_complete(merged_dir):
                    raise RuntimeError(f"exported model is incomplete: {merged_dir}")

            actor_env = os.environ.copy()
            actor_env["ACTOR_MODEL"] = model_name
            actor_env["NO_PROXY"] = "127.0.0.1,localhost"
            actor_env["no_proxy"] = "127.0.0.1,localhost"
            actor_log = (step_root / "actor.log").open("a", encoding="utf-8")
            actor_process = subprocess.Popen(
                actor_command(model_name),
                cwd=ROOT,
                env=actor_env,
                stdout=actor_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            atomic_status(status_path, state="actor_starting", step=step)
            wait_for_actor(
                actor_process,
                base_url=actor_base_url,
                api_key=actor_api_key,
                model_name=model_name,
                timeout_seconds=args.actor_startup_minutes * 60.0,
                poll_seconds=args.poll_seconds,
            )

            last_progress_at = time.monotonic()
            while not evaluation_complete(run_dir, expected, model_name):
                progress = task_progress(run_dir)
                eval_env = actor_env.copy()
                evaluator_log = (step_root / "evaluation.log").open("a", encoding="utf-8")
                command = evaluation_command(
                    model_name=model_name,
                    dataset=dataset,
                    subset_manifest=subset_manifest,
                    run_dir=run_dir,
                    concurrency=args.concurrency,
                )
                print(
                    f"[sequence] evaluating step={step}; valid={progress.valid}/{expected}",
                    flush=True,
                )
                evaluator_process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=eval_env,
                    stdout=evaluator_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                stalled, last_progress_at = monitor_evaluation(
                    evaluator_process,
                    run_dir=run_dir,
                    expected_tasks=expected,
                    stall_seconds=args.stall_minutes * 60.0,
                    poll_seconds=args.poll_seconds,
                    initial_progress_at=last_progress_at,
                    status_path=status_path,
                    step=step,
                )
                if stalled:
                    # Allow one recovery window while the evaluator remains alive.
                    grace_progress = task_progress(run_dir).valid
                    grace_deadline = time.monotonic() + args.shutdown_delay_minutes * 60.0
                    print(
                        f"[sequence] step={step} stalled; entering recovery grace period",
                        flush=True,
                    )
                    while time.monotonic() < grace_deadline and evaluator_process.poll() is None:
                        time.sleep(args.poll_seconds)
                        current = task_progress(run_dir).valid
                        if current > grace_progress:
                            print(
                                f"[sequence] step={step} recovered: {grace_progress}->{current}",
                                flush=True,
                            )
                            last_progress_at = time.monotonic()
                            stalled = False
                            break
                    if stalled:
                        stop_process(evaluator_process, "evaluator")
                        stop_process(actor_process, "Actor")
                        return shutdown_after_grace(
                            reason=f"step {step} evaluation made no valid-task progress",
                            delay_seconds=0,
                            no_shutdown=args.no_shutdown,
                            status_path=status_path,
                        )

                code = evaluator_process.wait()
                evaluator_process = None
                evaluator_log.close()
                evaluator_log = None
                if code != 0:
                    raise RuntimeError(f"step {step} evaluator exited with code {code}")
                progress = task_progress(run_dir)
                if progress.valid < expected:
                    print(
                        f"[sequence] step={step} retrying infrastructure-invalid tasks: "
                        f"valid={progress.valid}/{expected} invalid={progress.invalid}",
                        flush=True,
                    )

            print(f"[sequence] step={step} complete: {expected}/{expected} valid", flush=True)
            stop_process(actor_process, "Actor")
            actor_process = None
            actor_log.close()
            actor_log = None

        atomic_status(status_path, state="all_complete", steps=list(args.steps))
        return shutdown_after_grace(
            reason="all checkpoint evaluations completed",
            delay_seconds=args.shutdown_delay_minutes * 60.0,
            no_shutdown=args.no_shutdown,
            status_path=status_path,
        )
    except BaseException as exc:
        atomic_status(status_path, state="failed", reason=f"{type(exc).__name__}: {exc}")
        print(f"[sequence] fatal error: {type(exc).__name__}: {exc}", flush=True)
        stop_process(evaluator_process, "evaluator")
        stop_process(actor_process, "Actor")
        return shutdown_after_grace(
            reason=f"checkpoint evaluation failed: {type(exc).__name__}",
            delay_seconds=args.shutdown_delay_minutes * 60.0,
            no_shutdown=args.no_shutdown,
            status_path=status_path,
        )
    finally:
        if evaluator_log is not None:
            evaluator_log.close()
        if actor_log is not None:
            actor_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
