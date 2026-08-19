"""GRPO profile validation and veRL launch orchestration.

The compatibility command remains at ``scripts/train/grpo/train_grpo.py``.
This module owns the training-stage configuration translation and subprocess
boundary so the script directory contains no GRPO business logic.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
import yaml

from travel_grpo.training.grpo.compat import TORCH_PADDING_WORKER_SETUP_HOOK
from travel_grpo.training.grpo.preflight import run_preflight


# [项目注释] 功能：`load_profile`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：safe_load, read_text, isinstance, TypeError。
# [项目注释] 输入：`path`: Path。
# [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
def load_profile(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("GRPO profile must contain a mapping")
    return value


# [项目注释] 功能：`hydra_overrides`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：bool, int, str, items。
# [项目注释] 输入：`profile`: dict[str, Any]；`output`: Path；`resume`: bool；`logger`: str。
# [项目注释] 输出：标注返回 `list[str]`；具体值由各分支决定。
def hydra_overrides(profile: dict[str, Any], output: Path, resume: bool, logger: str) -> list[str]:
    d, r, v, a, alg, t = (profile[name] for name in ("data", "rollout", "validation", "actor", "algorithm", "trainer"))
    multi_turn = profile["actor_rollout_ref"]["rollout"]["multi_turn"]
    reuse_rollout_updates = bool(a.get("reuse_rollout_updates", False))
    configured_ppo_epochs = int(a.get("ppo_epochs", 1))
    ppo_epochs = configured_ppo_epochs if reuse_rollout_updates else 1
    max_tool_response_length = int(multi_turn.get("max_tool_response_length", 256))
    tool_response_truncate_side = str(multi_turn.get("tool_response_truncate_side", "middle"))
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
        "actor_rollout_ref.actor.ppo_epochs": ppo_epochs,
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
        "actor_rollout_ref.rollout.multi_turn.max_tool_response_length": max_tool_response_length,
        "actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side": tool_response_truncate_side,
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
    turn_credit = profile.get("turn_credit", {})
    overrides.append(
        f'+algorithm.travel_turn_credit.mode="{turn_credit.get("mode", "off")}"'
    )
    overrides.append(
        "+algorithm.travel_turn_credit.version="
        f"{turn_credit.get('version', 'conservative-turn-credit-v2')}"
    )
    for key in ("evidence_clip", "mix_lambda", "multiplier_band", "epsilon"):
        overrides.append(f"+algorithm.travel_turn_credit.{key}={turn_credit[key]}")
    # veRL 0.8 calls attention_utils from Ray workers.  Install the project's
    # pure-Torch padding implementation in those workers so flash-attn remains
    # optional.  This is injected only by the GRPO launcher; SFT and evaluation
    # entry points do not receive this runtime environment.
    overrides.append(
        "+ray_kwargs.ray_init.runtime_env.worker_process_setup_hook="
        f"{TORCH_PADDING_WORKER_SETUP_HOOK}"
    )
    return overrides


# [项目注释] 功能：`main`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：ArgumentParser, add_argument, parse_args,
# [项目注释]    load_profile。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `int`；具体值由各分支决定。
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
    parser.add_argument("--reuse-rollout-updates", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ppo-epochs", type=int)
    parser.add_argument(
        "--turn-credit-mode", choices=("off", "shadow", "train")
    )
    parser.add_argument("--turn-credit-lambda", type=float)
    parser.add_argument("--turn-credit-band", type=float)
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
    turn_credit = dict(profile.get("turn_credit", {}))
    if args.reuse_rollout_updates is not None or args.ppo_epochs is not None:
        actor = dict(profile["actor"])
        if args.reuse_rollout_updates is not None:
            actor["reuse_rollout_updates"] = args.reuse_rollout_updates
        if args.ppo_epochs is not None:
            actor["ppo_epochs"] = args.ppo_epochs
        profile["actor"] = actor
    if args.turn_credit_mode is not None:
        turn_credit["mode"] = args.turn_credit_mode
    if args.turn_credit_lambda is not None:
        turn_credit["mix_lambda"] = args.turn_credit_lambda
    if args.turn_credit_band is not None:
        turn_credit["multiplier_band"] = args.turn_credit_band
    turn_credit["enabled"] = turn_credit.get("mode", "off") != "off"
    profile["turn_credit"] = turn_credit
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
        "TRAVEL_GRPO_TURN_CREDIT_MODE": str(turn_credit["mode"]),
        "TRAVEL_GRPO_TURN_CREDIT_CONFIG_JSON": json.dumps(
            turn_credit, sort_keys=True, separators=(",", ":")
        ),
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
                        "TRAVEL_GRPO_TURN_CREDIT_MODE": launch_env[
                            "TRAVEL_GRPO_TURN_CREDIT_MODE"
                        ],
                        "TRAVEL_GRPO_TURN_CREDIT_CONFIG_JSON": launch_env[
                            "TRAVEL_GRPO_TURN_CREDIT_CONFIG_JSON"
                        ],
                    },
                    "rollout_reuse": {
                        "enabled": bool(profile["actor"].get("reuse_rollout_updates", False)),
                        "ppo_epochs": int(profile["actor"].get("ppo_epochs", 1)),
                        "rollout_batches_per_update": 1,
                        "old_policy_anchor": "fixed per sampled batch",
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
