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
VERL_TRAINER_PATCHED_SHA256 = "84C334738B82ABA8B57A2D735DD0C17CC48C6D5852247E12546CBC7987C7DC36"


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
            "veRL dynamic-sampling patch is missing or drifted; rerun `bash scripts/setup.sh`"
        )
    return digest


def install_torch_padding_fallback() -> None:
    """Install veRL's pure-Torch padding helpers for the pinned Qwen runtime.

    veRL 0.8 imports FlashAttention padding helpers eagerly on CUDA systems.
    The project uses the upstream-provided NPU/PyTorch implementation instead
    of mutating veRL source files.  This hook is invoked through Ray's worker
    process setup configuration.
    """

    require_verl_080()
    from verl.utils import attention_utils
    from verl.utils import npu_flash_attn_utils as fallback

    functions = (
        fallback.index_first_axis,
        fallback.pad_input,
        fallback.rearrange,
        fallback.unpad_input,
    )
    attention_utils._get_attention_functions = lambda: functions
