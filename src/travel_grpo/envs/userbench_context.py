"""Pinned-source validation and per-trajectory UserBench context."""

from __future__ import annotations

import contextvars
import json
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from travel_grpo.envs.observation import UserBenchStepResult
from travel_grpo.envs.reward import RawRewardTrace

if TYPE_CHECKING:
    from travel_grpo.envs.userbench_wrapper import UserBenchWrapper


PINNED_USERBENCH_COMMIT = "80506d2ab484cab843e60a2401ff3e0290d05b87"
DEFAULT_USERBENCH_ROOT = (
    Path(__file__).resolve().parents[3] / "environments" / "UserBench"
)


class UserBenchSourceError(RuntimeError):
    """Raised when the embedded UserBench source is missing or untrusted."""


class UserBenchSessionError(RuntimeError):
    """Raised when a rollout violates its UserBench session lifecycle."""


@dataclass(frozen=True)
class EmbeddedUserBench:
    root: Path
    upstream_commit: str
    license: str
    license_file: Path


def validate_embedded_userbench(root: str | Path | None = None) -> EmbeddedUserBench:
    """Validate provenance and licensing for the project-pinned snapshot."""

    source_root = Path(root) if root is not None else DEFAULT_USERBENCH_ROOT
    source_root = source_root.resolve()
    return _validate_embedded_userbench_cached(source_root)


@cache
def _validate_embedded_userbench_cached(source_root: Path) -> EmbeddedUserBench:
    manifest_path = source_root / "EMBEDDED_SOURCE.json"
    if not manifest_path.is_file():
        raise UserBenchSourceError(
            f"missing UserBench provenance manifest: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UserBenchSourceError(
            f"invalid UserBench provenance manifest: {manifest_path}"
        ) from exc
    commit = manifest.get("upstream_commit")
    if commit != PINNED_USERBENCH_COMMIT:
        raise UserBenchSourceError(
            f"UserBench commit is {commit!r}, expected {PINNED_USERBENCH_COMMIT!r}"
        )
    if manifest.get("license") != "Apache-2.0":
        raise UserBenchSourceError("embedded UserBench license must be Apache-2.0")
    license_name = manifest.get("license_file")
    if not isinstance(license_name, str) or not license_name:
        raise UserBenchSourceError("UserBench provenance is missing license_file")
    license_file = source_root / license_name
    if not license_file.is_file():
        raise UserBenchSourceError(
            f"missing embedded UserBench license: {license_file}"
        )
    try:
        license_text = license_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise UserBenchSourceError(
            f"cannot read embedded UserBench license: {license_file}"
        ) from exc
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise UserBenchSourceError("embedded UserBench license text is not Apache-2.0")
    if not (source_root / "travelgym" / "env" / "travel_env.py").is_file():
        raise UserBenchSourceError(f"missing TravelGym source under {source_root}")
    if any(source_root.rglob(".git")):
        raise UserBenchSourceError(
            "embedded UserBench snapshot must not contain a nested .git"
        )
    return EmbeddedUserBench(
        root=source_root,
        upstream_commit=commit,
        license="Apache-2.0",
        license_file=license_file,
    )


@dataclass
class UserBenchSessionState:
    """Mutable state shared by one AgentLoop task and its tool child tasks."""

    request_id: str
    task_id: str
    wrapper: UserBenchWrapper
    rewards: RawRewardTrace = field(default_factory=RawRewardTrace)
    num_tool_calls: int = 0
    terminated: bool = False
    truncated: bool = False
    protocol_error: str | None = None
    invalid_actions: int = 0
    termination_reason: str | None = None

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def record_step(self, result: UserBenchStepResult) -> None:
        if result.task_id != self.task_id:
            raise UserBenchSessionError(
                f"step task ID {result.task_id!r} does not match session {self.task_id!r}"
            )
        self.rewards.append(result.reward)
        self.num_tool_calls += 1
        self.terminated = result.terminated
        self.truncated = result.truncated

    def metrics(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "step_rewards": list(self.rewards.values),
            "cumulative_reward": self.rewards.total,
            "num_tool_calls": self.num_tool_calls,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "protocol_error": self.protocol_error,
            "invalid_actions": self.invalid_actions,
            "termination_reason": self.termination_reason,
        }


CURRENT_USERBENCH_SESSION: contextvars.ContextVar[UserBenchSessionState | None] = (
    contextvars.ContextVar("current_userbench_session", default=None)
)


def set_current_session(session: UserBenchSessionState) -> None:
    CURRENT_USERBENCH_SESSION.set(session)


def get_current_session() -> UserBenchSessionState | None:
    return CURRENT_USERBENCH_SESSION.get()


def require_current_session() -> UserBenchSessionState:
    session = get_current_session()
    if session is None:
        raise UserBenchSessionError(
            "UserBench session is missing from the current asyncio context"
        )
    return session


def clear_current_session(*, close: bool = True) -> None:
    session = get_current_session()
    CURRENT_USERBENCH_SESSION.set(None)
    if close and session is not None:
        session.wrapper.close()
