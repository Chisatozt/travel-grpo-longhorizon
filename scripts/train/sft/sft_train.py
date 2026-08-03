"""Validate, render, or LoRA-finetune Qwen3.5 on strict UserBench trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from travel_grpo.training.sft_dataset import (  # noqa: E402
    ActionOnlyDataCollator,
    SFTDatasetError,
    assert_train_validation_disjoint,
    audit_trajectory_file,
    build_action_only_examples,
    load_sft_trajectories,
    load_tool_schema,
    rendered_dataset_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/train/sft/sft_lora.yaml"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--render-smoke", action="store_true")
    parser.add_argument("--audit-only", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    return parser


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _project_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("SFT configuration requires PyYAML") from exc
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read SFT config: {path}") from exc
    root = _mapping(document, "config")
    model = _mapping(root.get("model"), "model")
    data = _mapping(root.get("data"), "data")
    lora = _mapping(root.get("lora"), "lora")
    training = _mapping(root.get("training"), "training")
    if model.get("base") != "Qwen/Qwen3.5-2B":
        raise ValueError("model.base must be 'Qwen/Qwen3.5-2B'")
    cache_dir = _project_path(
        model.get("cache_dir", "outputs/cache/huggingface"), "model.cache_dir"
    )
    try:
        cache_dir.relative_to((ROOT / "outputs").resolve())
    except ValueError as exc:
        raise ValueError("model.cache_dir must be under outputs/") from exc
    if data.get("assistant_loss") != "action_only":
        raise ValueError("data.assistant_loss must be action_only")
    if data.get("example_unit") != "assistant_turn":
        raise ValueError("data.example_unit must be assistant_turn")
    if not isinstance(data.get("max_sequence_length"), int) or data["max_sequence_length"] <= 0:
        raise ValueError("data.max_sequence_length must be positive")
    if training.get("bf16") and training.get("fp16"):
        raise ValueError("bf16 and fp16 cannot both be enabled")
    if not isinstance(lora.get("target_modules"), list) or not lora["target_modules"]:
        raise ValueError("lora.target_modules must be a non-empty list")
    for key in ("rank", "alpha"):
        if not isinstance(lora.get(key), int) or lora[key] <= 0:
            raise ValueError(f"lora.{key} must be a positive integer")
    dropout = lora.get("dropout")
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not 0 <= dropout < 1:
        raise ValueError("lora.dropout must be in [0, 1)")
    required_training = {
        "per_device_train_batch_size": int,
        "per_device_eval_batch_size": int,
        "gradient_accumulation_steps": int,
        "learning_rate": (int, float),
        "weight_decay": (int, float),
        "warmup_ratio": (int, float),
        "lr_scheduler_type": str,
        "optim": str,
        "num_train_epochs": (int, float),
        "max_steps": int,
        "gradient_checkpointing": bool,
        "bf16": bool,
        "fp16": bool,
        "logging_steps": int,
        "eval_strategy": str,
        "eval_steps": int,
        "save_strategy": str,
        "save_steps": int,
        "save_total_limit": int,
        "seed": int,
        "report_to": str,
    }
    for key, expected_type in required_training.items():
        value = training.get(key)
        if not isinstance(value, expected_type) or isinstance(value, bool) and expected_type is not bool:
            raise ValueError(f"training.{key} has an invalid or missing value")
    for key in (
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "logging_steps",
        "eval_steps",
        "save_steps",
        "save_total_limit",
    ):
        if training[key] <= 0:
            raise ValueError(f"training.{key} must be positive")
    if training["learning_rate"] <= 0 or training["num_train_epochs"] <= 0:
        raise ValueError("learning_rate and num_train_epochs must be positive")
    if training["weight_decay"] < 0:
        raise ValueError("training.weight_decay must be non-negative")
    if not 0 <= training["warmup_ratio"] <= 1:
        raise ValueError("training.warmup_ratio must be in [0, 1]")
    output = _project_path(training.get("output_dir"), "training.output_dir")
    try:
        output.relative_to((ROOT / "outputs").resolve())
    except ValueError as exc:
        raise ValueError("training.output_dir must be under outputs/") from exc
    return dict(root)


def _audit(config: Mapping[str, Any], *, limit: int | None) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    data = _mapping(config["data"], "data")
    schema = load_tool_schema(_project_path(data["tool_schema_path"], "tool schema"))
    train_path = _project_path(data["train_trajectories"], "train trajectories")
    validation_path = _project_path(
        data["validation_trajectories"], "validation trajectories"
    )
    train_audit = audit_trajectory_file(train_path, limit=limit)
    validation_audit = audit_trajectory_file(validation_path, limit=limit)
    train = load_sft_trajectories(train_path, limit=limit)
    validation = load_sft_trajectories(validation_path, limit=limit)
    assert_train_validation_disjoint(train, validation)
    return (
        {
            "train": train_audit.summary(),
            "validation": validation_audit.summary(),
            "train_validation_intersection": 0,
        },
        train,
        validation,
        schema,
    )


def _load_tokenizer(model: Mapping[str, Any]):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "tokenizer rendering requires the SFT extra; run `pip install -e .[sft]`"
        ) from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model["base"],
            trust_remote_code=bool(model.get("trust_remote_code", False)),
            local_files_only=True,
            cache_dir=str(
                _project_path(
                    model.get("cache_dir", "outputs/cache/huggingface"),
                    "model.cache_dir",
                )
            ),
        )
    except Exception as exc:
        raise RuntimeError(
            "Qwen3.5 tokenizer is not available in the local cache; download it explicitly "
            "before using --render-smoke or formal SFT"
        ) from exc
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _render(config, train, validation, schema, *, limit):
    model = _mapping(config["model"], "model")
    data = _mapping(config["data"], "data")
    tokenizer = _load_tokenizer(model)
    train_examples = build_action_only_examples(
        train[:limit] if limit else train,
        tokenizer,
        schema,
        max_sequence_length=int(data["max_sequence_length"]),
    )
    validation_examples = build_action_only_examples(
        validation[:limit] if limit else validation,
        tokenizer,
        schema,
        max_sequence_length=int(data["max_sequence_length"]),
    )
    return tokenizer, train_examples, validation_examples


def _train(config, tokenizer, train_examples, validation_examples, resume):
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("formal SFT requires `pip install -e .[sft]`") from exc
    model_config = _mapping(config["model"], "model")
    lora = _mapping(config["lora"], "lora")
    training = _mapping(config["training"], "training")
    qlora = bool(model_config.get("qlora", False))
    quantization = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if qlora
        else None
    )
    dtype = torch.bfloat16 if training.get("bf16") else torch.float16 if training.get("fp16") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_config["base"],
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        torch_dtype=dtype,
        quantization_config=quantization,
        cache_dir=str(
            _project_path(
                model_config.get("cache_dir", "outputs/cache/huggingface"),
                "model.cache_dir",
            )
        ),
    )
    if qlora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=bool(training["gradient_checkpointing"])
        )
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora["rank"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=list(lora["target_modules"]),
            task_type="CAUSAL_LM",
        ),
    )
    if training["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    arguments = TrainingArguments(
        output_dir=str(_project_path(training["output_dir"], "output dir")),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        warmup_ratio=float(training["warmup_ratio"]),
        lr_scheduler_type=str(training["lr_scheduler_type"]),
        optim=str(training["optim"]),
        num_train_epochs=float(training["num_train_epochs"]),
        max_steps=int(training["max_steps"]),
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        bf16=bool(training["bf16"]),
        fp16=bool(training["fp16"]),
        logging_steps=int(training["logging_steps"]),
        eval_strategy=str(training["eval_strategy"]),
        eval_steps=int(training["eval_steps"]),
        save_strategy=str(training["save_strategy"]),
        save_steps=int(training["save_steps"]),
        save_total_limit=int(training["save_total_limit"]),
        seed=int(training["seed"]),
        report_to=[] if training.get("report_to") == "none" else [training["report_to"]],
        remove_unused_columns=False,
    )
    collator = ActionOnlyDataCollator(
        pad_token_id=int(tokenizer.pad_token_id), padding_side="right"
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=[value.to_trainer_dict() for value in train_examples],
        eval_dataset=[value.to_trainer_dict() for value in validation_examples],
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    trainer.save_model()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config.resolve())
    data = _mapping(config["data"], "data")
    load_tool_schema(_project_path(data["tool_schema_path"], "tool schema"))
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.resume_from_checkpoint is not None:
        checkpoint = args.resume_from_checkpoint.resolve()
        try:
            checkpoint.relative_to((ROOT / "outputs").resolve())
        except ValueError as exc:
            raise ValueError("--resume-from-checkpoint must be under outputs/") from exc
    if args.audit_only is not None:
        return {"mode": "audit", **audit_trajectory_file(args.audit_only, limit=args.limit).summary()}
    audit, train, validation, schema = _audit(config, limit=args.limit)
    summary: dict[str, Any] = {"mode": "dry-run" if args.dry_run else "train", **audit}
    if args.dry_run and not args.render_smoke:
        return summary
    tokenizer, train_examples, validation_examples = _render(
        config, train, validation, schema, limit=args.limit
    )
    summary["rendered"] = {
        "train": rendered_dataset_summary(train_examples),
        "validation": rendered_dataset_summary(validation_examples),
    }
    if args.dry_run or args.render_smoke:
        summary["mode"] = "render-smoke"
        return summary
    _train(
        config,
        tokenizer,
        train_examples,
        validation_examples,
        args.resume_from_checkpoint,
    )
    return summary


def main() -> None:
    try:
        print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))
    except (SFTDatasetError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"SFT error: {exc}") from exc


if __name__ == "__main__":
    main()
