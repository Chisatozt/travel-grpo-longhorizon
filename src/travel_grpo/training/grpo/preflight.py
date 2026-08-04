"""Static and production runtime checks performed before Ray or CUDA starts."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from travel_grpo.envs.userbench_context import validate_embedded_userbench
from travel_grpo.envs.userbench_interaction import DEEPSEEK_V4_FLASH_MODEL
from travel_grpo.envs.userbench_tools import TOOL_NAME, get_interact_with_env_schema
from travel_grpo.training.grpo.data import verify_verl_datasets

PINNED = {
    "verl": "0.8.0",
    "vllm": "0.25.1",
    "torch": "2.11.0",
    "torchvision": "0.26.0",
    "torchaudio": "2.11.0",
    "ray": "2.56.1",
    "tensordict": "0.10.0",
    "numpy": "2.2.6",
}
TRANSFORMERS_COMMIT = "7ea2320c76117e6742364808a666ef6f2fb40a67"


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _complete_model(path: Path) -> None:
    _check(path.is_dir(), f"merged SFT model directory is missing: {path}")
    _check((path / "config.json").is_file(), f"model config.json is missing: {path}")
    _check((path / "tokenizer_config.json").is_file(), f"model tokenizer is missing: {path}")
    _check(
        any(path.glob("*.safetensors")) or (path / "pytorch_model.bin").is_file(),
        f"model weights are missing: {path}",
    )


def _simulator_environment(environ: Mapping[str, str]) -> None:
    for name in ("GRPO_USER_SIM_MODEL", "GRPO_USER_SIM_BASE_URL", "GRPO_USER_SIM_API_KEY"):
        _check(bool(environ.get(name, "").strip()), f"missing environment variable {name}")
    _check(
        environ["GRPO_USER_SIM_MODEL"].casefold() == DEEPSEEK_V4_FLASH_MODEL,
        f"GRPO_USER_SIM_MODEL must be {DEEPSEEK_V4_FLASH_MODEL}",
    )
    _check(float(environ.get("GRPO_USER_SIM_TEMPERATURE", "0")) == 0.0, "GRPO simulator temperature must be 0")


def run_preflight(
    profile: Mapping[str, Any],
    *,
    project_root: Path,
    output_dir: Path,
    resume: bool,
    strict_runtime: bool,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate the training contract; strict mode is used only for real runs."""

    env = os.environ if environ is None else environ
    data = profile["data"]
    rollout = profile["rollout"]
    _check(profile.get("profile_version") == "travel-grpo-verl08-v1", "unknown GRPO profile version")
    _check(int(data["train_batch_size"]) == 2 and int(data["val_batch_size"]) == 2, "GRPO batch sizes must be 2/2")
    _check(int(rollout["n"]) == 4 and int(rollout["max_parallel_calls"]) == 1, "rollout must use n=4 and one tool call")
    _check(rollout["format"] == "qwen3_coder" and rollout["enable_thinking"] is False, "Qwen tool format/thinking contract drift")
    _check(int(data["max_prompt_length"]) + int(data["max_response_length"]) == 32768, "context budget must be 32768")
    tool_document = __import__("yaml").safe_load((project_root / "configs/tool_config/userbench_tools.yaml").read_text(encoding="utf-8"))
    tools = tool_document if isinstance(tool_document, list) else tool_document.get("tools")
    _check(isinstance(tools, list) and len(tools) == 1, "interact_with_env must be the only tool")
    _check(tools[0]["tool_schema"] == get_interact_with_env_schema(), "tool YAML and Python schema differ")
    _check(tools[0]["tool_schema"]["function"]["name"] == TOOL_NAME, "tool name drift")
    if output_dir.exists() and any(output_dir.iterdir()):
        _check(resume and (output_dir / "latest_checkpointed_iteration.txt").is_file(), "output directory is non-empty and is not a legal resume directory")

    report: dict[str, Any] = {"static_contract": "ok", "strict_runtime": strict_runtime}
    if not strict_runtime:
        report["runtime"] = "skipped by dry-run"
        return report
    _check(platform.system() == "Linux", "formal GRPO requires Linux")
    _check(sys.version_info[:2] == (3, 12), "formal GRPO requires Python 3.12")
    for distribution, expected in PINNED.items():
        found = importlib.metadata.version(distribution)
        _check(found == expected, f"{distribution}=={expected} required, found {found}")
    direct_url = importlib.metadata.distribution("transformers").read_text("direct_url.json")
    _check(
        isinstance(direct_url, str) and TRANSFORMERS_COMMIT in direct_url,
        f"Transformers must be installed from commit {TRANSFORMERS_COMMIT}",
    )
    import torch

    _check(torch.cuda.device_count() == 1, "formal profile requires exactly one visible GPU")
    properties = torch.cuda.get_device_properties(0)
    gib = properties.total_memory / 1024**3
    _check(gib >= 80.0, f"visible GPU has {gib:.1f} GiB; at least 80 GiB is required")
    _check(torch.cuda.is_bf16_supported(), "visible GPU does not support BF16")
    model_path = (project_root / str(profile["model_path"])).resolve()
    _complete_model(model_path)
    source = validate_embedded_userbench(project_root / "environments/UserBench")
    _check(not any(source.root.rglob(".git")), "embedded UserBench contains a nested .git")
    verify_verl_datasets(project_root / "outputs/grpo/data")
    _simulator_environment(env)
    from travel_grpo.training.grpo.compat import (
        require_verl_080,
        require_verl_dynamic_sampling_patch,
    )

    require_verl_080()
    require_verl_dynamic_sampling_patch()
    from vllm.entrypoints.openai.tool_parsers import ToolParserManager

    parsers = getattr(ToolParserManager, "tool_parsers", getattr(ToolParserManager, "_tool_parsers", {}))
    _check("qwen3_coder" in parsers, "vLLM qwen3_coder tool parser is unavailable")
    report.update({"runtime": "ok", "gpu_memory_gib": round(gib, 2), "userbench_root": str(source.root)})
    return report
