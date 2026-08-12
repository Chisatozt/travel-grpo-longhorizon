"""Narrow compatibility boundary for the pinned external veRL runtime."""

from __future__ import annotations

from importlib import metadata
import hashlib
from importlib.util import find_spec
from pathlib import Path

VERL_REQUIRED_VERSION = "0.8.0"
VERL_INSTALL_HINT = (
    "run `bash scripts/setup.sh` on the supported Linux Python 3.12 runtime"
)
VERL_TRAINER_PATCHED_SHA256 = "51E774CC9E112EEE00EBBEDDAB99FBF9D89C34C900F4473E1254E9CA8637CF64"
# Ray resolves this dotted path in every worker process created for the GRPO
# job.  Keep it in the project-owned compatibility boundary so the launch
# script cannot accidentally point at an external veRL implementation.
TORCH_PADDING_WORKER_SETUP_HOOK = (
    "travel_grpo.training.grpo.compat.install_torch_padding_fallback"
)

_TORCH_PADDING_FALLBACK_INSTALLED = False


class VerlCompatibilityError(RuntimeError):
    """Raised when the external veRL installation is missing or incompatible."""


def require_verl_080() -> str:
    """Require exactly the supported veRL release without a lightweight import dependency."""

    try:
        version = metadata.version("verl")
    except metadata.PackageNotFoundError as exc:
        raise VerlCompatibilityError(
            f"veRL is not installed; {VERL_INSTALL_HINT}"
        ) from exc
    if version != VERL_REQUIRED_VERSION:
        raise VerlCompatibilityError(
            f"unsupported veRL version {version!r}; expected {VERL_REQUIRED_VERSION}; "
            f"{VERL_INSTALL_HINT}"
        )
    return version


def require_verl_dynamic_sampling_patch() -> str:
    """Prove the pinned trainer contains only the reviewed connection patch."""

    require_verl_080()
    spec = find_spec("verl.trainer.ppo.ray_trainer")
    if spec is None or spec.origin is None:
        raise VerlCompatibilityError("cannot locate veRL ray_trainer.py")
    path = Path(spec.origin).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    if digest != VERL_TRAINER_PATCHED_SHA256:
        raise VerlCompatibilityError(
            "veRL project connection patches are missing or drifted; rerun `bash scripts/setup.sh`"
        )
    return digest


def install_torch_padding_fallback() -> None:
    """Install veRL's pure-Torch padding helpers for the pinned Qwen runtime.

    veRL 0.8 imports FlashAttention padding helpers eagerly on CUDA systems.
    The project uses the upstream-provided NPU/PyTorch implementation instead
    of mutating veRL source files.  This hook is invoked through Ray's worker
    process setup configuration.
    """

    global _TORCH_PADDING_FALLBACK_INSTALLED
    if _TORCH_PADDING_FALLBACK_INSTALLED:
        return

    require_verl_080()
    from verl.utils import attention_utils

    # Keep the pinned baseline unchanged when a working FlashAttention
    # installation is available.  The fallback is only for environments such
    # as the supported cu130 image where flash-attn is intentionally absent (or
    # cannot be imported because its compiled extension is incompatible).
    try:
        from flash_attn.bert_padding import (  # noqa: F401
            index_first_axis,
            pad_input,
            rearrange,
            unpad_input,
        )
    except (ImportError, ModuleNotFoundError, OSError):
        pass
    else:
        _TORCH_PADDING_FALLBACK_INSTALLED = True
        return

    from verl.utils import npu_flash_attn_utils as fallback

    functions = (
        fallback.index_first_axis,
        fallback.pad_input,
        fallback.rearrange,
        fallback.unpad_input,
    )
    attention_utils._get_attention_functions = lambda: functions
    _TORCH_PADDING_FALLBACK_INSTALLED = True
