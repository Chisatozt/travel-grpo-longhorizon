"""Narrow compatibility boundary for the externally installed veRL runtime."""

from __future__ import annotations

from importlib import metadata

VERL_REQUIRED_VERSION = "0.6.1"
VERL_INSTALL_HINT = (
    "install veRL 0.6.1 from an external checkout with `pip install -e /path/to/verl`"
)


class VerlCompatibilityError(RuntimeError):
    """Raised when the external veRL installation is missing or incompatible."""


def require_verl_061() -> str:
    """Require exactly the supported veRL release without making it a core dependency."""

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
