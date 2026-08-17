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

from travel_grpo.training.sft.dataset import (  # noqa: E402
    ActionOnlyDataCollator,
    SFTDatasetError,
    assert_sft_readiness,
    assert_task_ids_within_split,
    assert_train_validation_disjoint,
    audit_trajectory_file,
    build_action_only_dataset,
    load_sft_trajectory_files,
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
    parser.add_argument(
        "--audit-format",
        choices=("trajectory", "prefix"),
        default="trajectory",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--allow-small-smoke",
        action="store_true",
        help="bypass formal trajectory-count gates for an explicit non-formal smoke",
    )
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


def _project_paths(value: Any, name: str) -> tuple[Path, ...]:
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty path list")
    return tuple(_project_path(item, name) for item in values)


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
    tiers = data.get("accepted_quality_tiers", ["gold"])
    if (
        not isinstance(tiers, list)
        or not tiers
        or set(tiers) - {"gold", "silver"}
    ):
        raise ValueError("data.accepted_quality_tiers must contain gold and/or silver")
    for key in ("train_format", "validation_format"):
        value = data.get(key, "trajectory")
        if value not in {"trajectory", "prefix"}:
            raise ValueError(f"data.{key} must be trajectory or prefix")
    _project_paths(data.get("train_trajectories"), "data.train_trajectories")
    _project_paths(
        data.get("validation_trajectories"), "data.validation_trajectories"
    )
    for key in ("train_tasks", "validation_tasks"):
        if key in data:
            _project_path(data[key], f"data.{key}")
    for key in ("minimum_train_trajectories", "minimum_validation_trajectories"):
        value = data.get(key, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"data.{key} must be a non-negative integer")
    if not isinstance(data.get("max_sequence_length"), int) or data["max_sequence_length"] <= 0:
        raise ValueError("data.max_sequence_length must be positive")
    if training.get("bf16") and training.get("fp16"):
        raise ValueError("bf16 and fp16 cannot both be enabled")
    targets = lora.get("target_modules")
    if not (
        targets == "all-linear"
        or isinstance(targets, list)
        and targets
        and all(isinstance(value, str) and value for value in targets)
    ):
        raise ValueError("lora.target_modules must be all-linear or a non-empty string list")
    init_from = lora.get("init_from")
    if init_from is not None:
        init_path = _project_path(init_from, "lora.init_from")
        try:
            init_path.relative_to((ROOT / "outputs").resolve())
        except ValueError as exc:
            raise ValueError("lora.init_from must be under outputs/") from exc
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
    if init_from is not None and output == _project_path(init_from, "lora.init_from"):
        raise ValueError("training.output_dir must differ from lora.init_from")
    return dict(root)


def _audit(
    config: Mapping[str, Any],
    *,
    limit: int | None,
    allow_small_smoke: bool,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    data = _mapping(config["data"], "data")
    schema = load_tool_schema(_project_path(data["tool_schema_path"], "tool schema"))
    train_paths = _project_paths(data["train_trajectories"], "train trajectories")
    validation_paths = _project_paths(
        data["validation_trajectories"], "validation trajectories"
    )
    tiers = tuple(data.get("accepted_quality_tiers", ["gold"]))
    train_format = str(data.get("train_format", "trajectory"))
    validation_format = str(data.get("validation_format", "trajectory"))
    train_audits = [
        audit_trajectory_file(
            path,
            limit=limit,
            accepted_quality_tiers=tiers,
            record_format=train_format,
        )
        for path in train_paths
    ]
    validation_audits = [
        audit_trajectory_file(
            path,
            limit=limit,
            accepted_quality_tiers=tiers,
            record_format=validation_format,
        )
        for path in validation_paths
    ]
    train = load_sft_trajectory_files(
        train_paths,
        limit=limit,
        accepted_quality_tiers=tiers,
        record_format=train_format,
    )
    validation = load_sft_trajectory_files(
        validation_paths,
        limit=limit,
        accepted_quality_tiers=tiers,
        record_format=validation_format,
    )
    assert_train_validation_disjoint(train, validation)
    if "train_tasks" in data or "validation_tasks" in data:
        if not {"train_tasks", "validation_tasks"} <= set(data):
            raise ValueError(
                "data.train_tasks and data.validation_tasks must be configured together"
            )
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("SFT split isolation requires the data extra") from exc
        train_task_ids = pq.read_table(
            _project_path(data["train_tasks"], "data.train_tasks"),
            columns=["task_id"],
        ).column("task_id").to_pylist()
        validation_task_ids = pq.read_table(
            _project_path(data["validation_tasks"], "data.validation_tasks"),
            columns=["task_id"],
        ).column("task_id").to_pylist()
        assert_task_ids_within_split(train, train_task_ids, split_name="train")
        assert_task_ids_within_split(
            validation, validation_task_ids, split_name="validation"
        )
    minimum_train = int(data.get("minimum_train_trajectories", 0))
    minimum_validation = int(data.get("minimum_validation_trajectories", 0))
    required_compositions = tuple(data.get("required_compositions", ()))
    if not allow_small_smoke and (minimum_train or minimum_validation or required_compositions):
        assert_sft_readiness(
            train,
            validation,
            minimum_train=minimum_train,
            minimum_validation=minimum_validation,
            required_compositions=required_compositions,
        )
    return (
        {
            "train": {
                "record_format": train_format,
                "files": [value.summary() for value in train_audits],
                "accepted_trajectories": len(train),
            },
            "validation": {
                "record_format": validation_format,
                "files": [value.summary() for value in validation_audits],
                "accepted_trajectories": len(validation),
            },
            "train_validation_intersection": 0,
            "formal_readiness_enforced": not allow_small_smoke,
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


def _render(config, train, validation, schema, *, limit, allow_small_smoke):
    model = _mapping(config["model"], "model")
    data = _mapping(config["data"], "data")
    tokenizer = _load_tokenizer(model)
    train_format = str(data.get("train_format", "trajectory"))
    validation_format = str(data.get("validation_format", "trajectory"))
    train_examples, train_overlong = build_action_only_dataset(
        train[:limit] if limit else train,
        tokenizer,
        schema,
        max_sequence_length=int(data["max_sequence_length"]),
        accepted_quality_tiers=tuple(data.get("accepted_quality_tiers", ["gold"])),
        record_format=train_format,
    )
    validation_examples, validation_overlong = build_action_only_dataset(
        validation[:limit] if limit else validation,
        tokenizer,
        schema,
        max_sequence_length=int(data["max_sequence_length"]),
        accepted_quality_tiers=tuple(data.get("accepted_quality_tiers", ["gold"])),
        record_format=validation_format,
    )
    retained_train_ids = {value.task_id for value in train_examples}
    retained_validation_ids = {value.task_id for value in validation_examples}
    retained_train = tuple(value for value in train if str(value["task_id"]) in retained_train_ids)
    retained_validation = tuple(
        value for value in validation if str(value["task_id"]) in retained_validation_ids
    )
    if not allow_small_smoke:
        assert_sft_readiness(
            retained_train,
            retained_validation,
            minimum_train=int(data.get("minimum_train_trajectories", 0)),
            minimum_validation=int(data.get("minimum_validation_trajectories", 0)),
            required_compositions=tuple(data.get("required_compositions", ())),
        )
    return (
        tokenizer,
        train_examples,
        validation_examples,
        {"train": list(train_overlong), "validation": list(validation_overlong)},
    )


def _train(config, tokenizer, train_examples, validation_examples, resume):
    try:
        import torch
        from peft import (
            LoraConfig,
            PeftModel,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForMultimodalLM,
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
    architecture = AutoConfig.from_pretrained(
        model_config["base"],
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        cache_dir=str(
            _project_path(
                model_config.get("cache_dir", "outputs/cache/huggingface"),
                "model.cache_dir",
            )
        ),
    )
    model_class = (
        AutoModelForMultimodalLM
        if str(getattr(architecture, "model_type", "")).startswith("qwen3_5")
        else AutoModelForCausalLM
    )
    model = model_class.from_pretrained(
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
    init_from = lora.get("init_from")
    if init_from is not None:
        adapter = _project_path(init_from, "lora.init_from")
        model_files = (
            adapter / "adapter_model.safetensors",
            adapter / "adapter_model.bin",
        )
        if not (adapter / "adapter_config.json").is_file() or not any(
            path.is_file() for path in model_files
        ):
            raise RuntimeError(
                f"Stage-2 initialization adapter is incomplete: {adapter}"
            )
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
        adapter_config = model.peft_config.get("default")
        if adapter_config is None:
            raise RuntimeError("Stage-2 initialization adapter has no default config")
        expected = (
            int(lora["rank"]),
            int(lora["alpha"]),
            float(lora["dropout"]),
        )
        observed = (
            int(adapter_config.r),
            int(adapter_config.lora_alpha),
            float(adapter_config.lora_dropout),
        )
        if observed != expected:
            raise RuntimeError(
                "Stage-2 LoRA hyperparameters do not match the Stage-1 adapter: "
                f"expected {expected}, found {observed}"
            )
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(lora["rank"]),
                lora_alpha=int(lora["alpha"]),
                lora_dropout=float(lora["dropout"]),
                target_modules=(
                    lora["target_modules"]
                    if isinstance(lora["target_modules"], str)
                    else list(lora["target_modules"])
                ),
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
        return {
            "mode": "audit",
            "record_format": args.audit_format,
            **audit_trajectory_file(
                args.audit_only,
                limit=args.limit,
                record_format=args.audit_format,
            ).summary(),
        }
    audit, train, validation, schema = _audit(
        config,
        limit=args.limit,
        allow_small_smoke=bool(args.allow_small_smoke),
    )
    summary: dict[str, Any] = {
        "mode": "dry-run" if args.dry_run else "train",
        "stage": config.get("stage", {}).get("name")
        if isinstance(config.get("stage"), Mapping)
        else None,
        "initial_adapter": _mapping(config["lora"], "lora").get("init_from"),
        **audit,
    }
    if args.dry_run and not args.render_smoke:
        return summary
    tokenizer, train_examples, validation_examples, overlong = _render(
        config,
        train,
        validation,
        schema,
        limit=args.limit,
        allow_small_smoke=bool(args.allow_small_smoke),
    )
    summary["rendered"] = {
        "train": rendered_dataset_summary(train_examples),
        "validation": rendered_dataset_summary(validation_examples),
        "overlong_rejections": overlong,
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
