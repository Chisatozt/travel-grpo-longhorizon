#!/usr/bin/env python3
"""Validate a project GRPO profile and launch veRL 0.8."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml

from travel_grpo.training.grpo.compat import TORCH_PADDING_WORKER_SETUP_HOOK
from travel_grpo.training.grpo.preflight import run_preflight


def load_profile(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("GRPO profile must contain a mapping")
    return value


def hydra_overrides(profile: dict[str, Any], output: Path, resume: bool, logger: str) -> list[str]:
    d, r, v, a, alg, t = (profile[name] for name in ("data", "rollout", "validation", "actor", "algorithm", "trainer"))
    absolute = lambda value: str((ROOT / str(value)).resolve())
    values = {
        "data.train_files": f"['{absolute(d['train_files'])}']",
        "data.val_files": f"['{absolute(d['val_files'])}']",
        "data.train_batch_size": d["train_batch_size"],
        "data.val_batch_size": d["val_batch_size"],
        "data.max_prompt_length": d["max_prompt_length"],
        "data.max_response_length": d["max_response_length"],
        "data.truncation": d["truncation"],
        "actor_rollout_ref.model.path": absolute(profile["model_path"]),
        "actor_rollout_ref.model.lora_rank": a["lora_rank"],
        "actor_rollout_ref.model.lora_alpha": a["lora_alpha"],
        "actor_rollout_ref.model.target_modules": "all-linear",
        "actor_rollout_ref.actor.optim.lr": a["learning_rate"],
        "actor_rollout_ref.actor.ppo_mini_batch_size": a["mini_batch_size"],
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": a["micro_batch_size_per_gpu"],
        "actor_rollout_ref.model.enable_gradient_checkpointing": a["gradient_checkpointing"],
        "actor_rollout_ref.actor.use_kl_loss": alg["kl_loss"],
        "actor_rollout_ref.rollout.name": r["backend"],
        "actor_rollout_ref.rollout.mode": r["mode"],
        "actor_rollout_ref.rollout.n": r["n"],
        "actor_rollout_ref.rollout.temperature": r["temperature"],
        "actor_rollout_ref.rollout.top_p": r["top_p"],
        "actor_rollout_ref.rollout.max_model_len": r["max_model_len"],
        "actor_rollout_ref.rollout.gpu_memory_utilization": r["gpu_memory_utilization"],
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.agent.num_workers": r["workers"],
        "actor_rollout_ref.rollout.agent.default_agent_loop": "userbench_tool_agent",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path": str((ROOT / "configs/interaction_config/agent_loop.yaml").resolve()),
        "actor_rollout_ref.rollout.multi_turn.enable": True,
        "actor_rollout_ref.rollout.multi_turn.max_parallel_calls": r["max_parallel_calls"],
        "actor_rollout_ref.rollout.multi_turn.max_assistant_turns": r["max_assistant_turns"],
        "actor_rollout_ref.rollout.multi_turn.format": r["format"],
        "actor_rollout_ref.rollout.multi_turn.tool_config_path": str((ROOT / "configs/tool_config/userbench_tools.yaml").resolve()),
        "actor_rollout_ref.rollout.val_kwargs.n": v["n"],
        "actor_rollout_ref.rollout.val_kwargs.temperature": v["temperature"],
        "actor_rollout_ref.rollout.val_kwargs.top_p": v["top_p"],
        "actor_rollout_ref.rollout.val_kwargs.do_sample": v["do_sample"],
        "algorithm.adv_estimator": alg["adv_estimator"],
        "algorithm.norm_adv_by_std_in_grpo": alg["norm_adv_by_std_in_grpo"],
        "algorithm.use_kl_in_reward": alg["kl_in_reward"],
        "trainer.total_training_steps": t["total_training_steps"],
        "trainer.save_freq": t["save_freq"],
        "trainer.test_freq": t["test_freq"],
        "trainer.val_before_train": t["val_before_train"],
        "trainer.logger": f"[{logger}]",
        "trainer.n_gpus_per_node": 1,
        "trainer.nnodes": 1,
        "trainer.default_local_dir": str(output),
        "trainer.validation_data_dir": str(output / "validation_rollouts"),
        "trainer.rollout_data_dir": str(output / "training_rollouts"),
        "trainer.resume_mode": "auto" if resume else "disable",
    }
    overrides = [f"{key}={str(value).lower() if isinstance(value, bool) else value}" for key, value in values.items()]
    overrides.append("+data.apply_chat_template_kwargs.enable_thinking=false")
    dynamic = profile.get("dynamic", {})
    overrides.append(f"+travel_dynamic_sampling.enable={str(bool(profile.get('dynamic_sampling'))).lower()}")
    for key, value in dynamic.items():
        overrides.append(f"+travel_dynamic_sampling.{key}={value}")
    # veRL 0.8 calls attention_utils from Ray workers.  Install the project's
    # pure-Torch padding implementation in those workers so flash-attn remains
    # optional.  This is injected only by the GRPO launcher; SFT and evaluation
    # entry points do not receive this runtime environment.
    overrides.append(
        "+ray_kwargs.ray_init.runtime_env.worker_process_setup_hook="
        f"{TORCH_PADDING_WORKER_SETUP_HOOK}"
    )
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        help="override the local merged model path from the GRPO profile",
    )
    parser.add_argument(
        "--data-output",
        type=Path,
        help="override the prepared GRPO data directory from the profile",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--logger", choices=("console", "swanlab"), default="console")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stall-recovery",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--stall-threshold", type=int, default=4)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    if args.stall_threshold < 1:
        parser.error("--stall-threshold must be >= 1")
    profile = load_profile((ROOT / args.config).resolve() if not args.config.is_absolute() else args.config)
    if args.model_path is not None:
        profile["model_path"] = str(args.model_path.expanduser().resolve())
    data_output = None
    if args.data_output is not None:
        data_output = args.data_output.expanduser().resolve()
        profile["data"] = dict(profile["data"])
        profile["data"]["train_files"] = str(data_output / "train.parquet")
        profile["data"]["val_files"] = str(data_output / "validation.parquet")
    output = (ROOT / (args.output or Path(profile["output_dir"]))).resolve()
    report = run_preflight(
        profile,
        project_root=ROOT,
        output_dir=output,
        resume=args.resume,
        strict_runtime=not args.dry_run,
        stall_threshold=args.stall_threshold,
        data_output_dir=data_output,
    )
    if not args.dry_run and args.logger == "swanlab":
        try:
            __import__("swanlab")
        except ImportError as exc:
            raise RuntimeError(
                "SwanLab logging is optional; install `pip install -e .[logging]`"
            ) from exc
    command = [sys.executable, "-m", "verl.trainer.main_ppo", *hydra_overrides(profile, output, args.resume, args.logger), *args.overrides]
    launch_env = {
        **os.environ,
        "PYTHONPATH": str(SRC),
        "TRAVEL_GRPO_STALL_RECOVERY": (
            "true" if args.stall_recovery else "false"
        ),
        "TRAVEL_GRPO_STALL_THRESHOLD": str(args.stall_threshold),
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "valid": True,
                    "preflight": report,
                    "command": command,
                    "stall_recovery": {
                        "enabled": args.stall_recovery,
                        "threshold": args.stall_threshold,
                    },
                    "agent_loop_env": {
                        "TRAVEL_GRPO_STALL_RECOVERY": launch_env[
                            "TRAVEL_GRPO_STALL_RECOVERY"
                        ],
                        "TRAVEL_GRPO_STALL_THRESHOLD": launch_env[
                            "TRAVEL_GRPO_STALL_THRESHOLD"
                        ],
                    },
                },
                indent=2,
            )
        )
        return 0
    output.mkdir(parents=True, exist_ok=True)
    return subprocess.call(command, cwd=ROOT, env=launch_env)


if __name__ == "__main__":
    raise SystemExit(main())
