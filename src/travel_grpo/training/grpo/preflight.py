"""Static and production runtime checks performed before Ray or CUDA starts."""

from __future__ import annotations

import importlib.metadata
import math
import os
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from travel_grpo.envs.userbench_context import validate_embedded_userbench
from travel_grpo.envs.userbench_interaction import DEEPSEEK_V4_FLASH_MODEL
from travel_grpo.envs.userbench_tools import TOOL_NAME, get_interact_with_env_schema
from travel_grpo.training.grpo.turn_credit import (
    TURN_CREDIT_VERSION,
    TurnCreditConfig,
    validate_turn_credit_mode,
)
from travel_grpo.training.grpo.data import verify_verl_datasets

PINNED = {
    "verl": "0.8.0",
    "vllm": "0.25.1",
    "torch": "2.11.0",
    "torchvision": "0.26.0",
    "torchaudio": "2.11.0",
    "ray": "2.56.1",
    "tensordict": "0.10.0",
    "opencv-python-headless": "4.13.0.90",
    "numpy": "2.2.6",
}
TRANSFORMERS_COMMIT = "7ea2320c76117e6742364808a666ef6f2fb40a67"


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sampling_value(sampling_params: Mapping[str, Any], name: str) -> Any:
    if isinstance(sampling_params, Mapping):
        return sampling_params.get(name)
    return getattr(sampling_params, name, None)


def _sampling_float(sampling_params: Mapping[str, Any], name: str) -> float:
    raw = _sampling_value(sampling_params, name)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sampling profile is missing numeric {name}") from exc
    if not math.isfinite(value):
        raise ValueError(f"sampling profile {name} must be finite")
    return value


def _sampling_profile(sampling_params: Mapping[str, Any]) -> str:
    """Classify only the pinned training and validation sampling profiles."""

    temperature = _sampling_float(sampling_params, "temperature")
    top_p = _sampling_float(sampling_params, "top_p")
    do_sample = _sampling_value(sampling_params, "do_sample")
    if do_sample is not None and not isinstance(do_sample, bool):
        raise ValueError("sampling profile do_sample must be a boolean when present")
    if temperature == 0.0 and top_p == 1.0:
        if do_sample is not None and do_sample is not False:
            raise ValueError("validation sampling must set do_sample=false")
        return "validation"
    if temperature == 0.7 and top_p == 0.9:
        return "training"
    raise ValueError(
        "sampling profile is neither the pinned GRPO training profile "
        "(temperature=0.7, top_p=0.9) nor validation profile "
        "(temperature=0.0, top_p=1.0, do_sample=false)"
    )


def is_validation_sampling(sampling_params: Mapping[str, Any]) -> bool:
    """Return whether a rollout uses the pinned deterministic validation profile."""

    return _sampling_profile(sampling_params) == "validation"


def validate_sampling_profiles(
    training_sampling: Mapping[str, Any],
    validation_sampling: Mapping[str, Any],
) -> None:
    """Fail loudly if the two fixed profiles stop being distinguishable."""

    try:
        training_profile = _sampling_profile(training_sampling)
        validation_profile = _sampling_profile(validation_sampling)
    except ValueError as exc:
        raise RuntimeError(f"GRPO sampling profile classification failed: {exc}") from exc
    _check(training_profile == "training", "GRPO training sampling profile drifted")
    _check(
        validation_profile == "validation",
        "GRPO validation sampling profile drifted or is not deterministic",
    )


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
    stall_threshold: int = 4,
    data_output_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate the training contract; strict mode is used only for real runs."""

    env = os.environ if environ is None else environ
    data = profile["data"]
    rollout = profile["rollout"]
    if isinstance(stall_threshold, bool) or int(stall_threshold) < 1:
        raise RuntimeError("stall threshold must be an integer >= 1")
    validate_sampling_profiles(
        {
            "temperature": rollout["temperature"],
            "top_p": rollout["top_p"],
        },
        profile["validation"],
    )
    _check(profile.get("profile_version") == "travel-grpo-verl08-v1", "unknown GRPO profile version")
    _check(int(data["train_batch_size"]) == 2 and int(data["val_batch_size"]) == 2, "GRPO batch sizes must be 2/2")
    _check(int(rollout["n"]) == 4 and int(rollout["max_parallel_calls"]) == 1, "rollout must use n=4 and one tool call")
    _check(rollout["format"] == "qwen3_coder" and rollout["enable_thinking"] is False, "Qwen tool format/thinking contract drift")
    _check(int(data["max_prompt_length"]) + int(data["max_response_length"]) == 32768, "context budget must be 32768")
    turn_credit = profile.get("turn_credit", {})
    _check(isinstance(turn_credit, Mapping), "turn_credit must be a mapping")
    turn_credit_mode = validate_turn_credit_mode(turn_credit.get("mode", "off"))
    _check(
        turn_credit.get("version") == TURN_CREDIT_VERSION,
        "turn-credit version drifted",
    )
    _check(
        bool(turn_credit.get("enabled", False)) == (turn_credit_mode != "off"),
        "turn_credit.enabled must agree with turn_credit.mode",
    )
    parsed_turn_credit = TurnCreditConfig.from_mapping(turn_credit)
    _check(
        math.isclose(
            parsed_turn_credit.preference_chain
            + parsed_turn_credit.successful_search
            + parsed_turn_credit.correct_answer,
            1.0,
        ),
        "completed-aspect turn-credit allocation must sum to 1.0",
    )
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
    verify_verl_datasets(
        (data_output_dir or (project_root / "outputs/grpo/data")).resolve()
    )
    _simulator_environment(env)
    from travel_grpo.training.grpo.compat import (
        require_verl_080,
        require_verl_dynamic_sampling_patch,
    )

    require_verl_080()
    require_verl_dynamic_sampling_patch()
    from vllm.tool_parsers import ToolParserManager

    try:
        ToolParserManager.get_tool_parser("qwen3_coder")
    except (KeyError, ImportError, AttributeError) as exc:
        raise RuntimeError(
            "vLLM qwen3_coder tool parser is unavailable"
        ) from exc
    report.update({"runtime": "ok", "gpu_memory_gib": round(gib, 2), "userbench_root": str(source.root)})
    return report
