#!/usr/bin/env python3
"""Merge the SFT LoRA adapter into a standalone Qwen3.5 GRPO starting point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]


def choose_model_class(config: Any, causal_class: Any, multimodal_class: Any) -> Any:
    """Use Qwen3.5's multimodal Auto class while keeping ordinary CausalLM support."""

    return (
        multimodal_class
        if str(getattr(config, "model_type", "")).startswith("qwen3_5")
        else causal_class
    )


def build_merge_manifest(
    *, base_model: str, adapter: Path, output: Path, model_type: str, dtype: str
) -> dict[str, Any]:
    return {
        "operation": "peft_merge_and_unload",
        "base_model": base_model,
        "adapter": str(adapter.resolve()),
        "output": str(output.resolve()),
        "model_type": model_type,
        "dtype": dtype,
        "next_stage": "veRL 0.8 GRPO with a fresh rank-16 LoRA adapter",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-2B")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=ROOT / "outputs/sft/qwen3.5-2b-lora",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/models/sft-merged"
    )
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    adapter = args.adapter.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not adapter.is_dir() or not (adapter / "adapter_config.json").is_file():
        raise SystemExit(f"SFT LoRA adapter is incomplete: {adapter}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"merge output directory must be new or empty: {output}")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry-run",
                    "base_model": args.base_model,
                    "adapter": str(adapter),
                    "output": str(output),
                    "dtype": "bfloat16" if args.bf16 else "float32",
                },
                indent=2,
            )
        )
        return
    try:
        import torch
        from peft import PeftModel
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForMultimodalLM,
            AutoProcessor,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise SystemExit("LoRA merge requires the pinned SFT dependencies") from exc

    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    model_class = choose_model_class(
        config, AutoModelForCausalLM, AutoModelForMultimodalLM
    )
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    base = model_class.from_pretrained(
        args.base_model, torch_dtype=dtype, trust_remote_code=True
    )
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    output.mkdir(parents=True, exist_ok=False)
    merged.save_pretrained(
        str(output), safe_serialization=True, max_shard_size=args.max_shard_size
    )
    try:
        processor = AutoProcessor.from_pretrained(
            args.base_model, trust_remote_code=True
        )
    except Exception:
        processor = AutoTokenizer.from_pretrained(
            args.base_model, trust_remote_code=True
        )
    processor.save_pretrained(str(output))
    manifest = build_merge_manifest(
        base_model=args.base_model,
        adapter=adapter,
        output=output,
        model_type=str(config.model_type),
        dtype="bfloat16" if args.bf16 else "float32",
    )
    (output / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
