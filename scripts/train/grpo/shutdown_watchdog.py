#!/usr/bin/env python3
"""Shut down the host after a GRPO run exits or stops making progress."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from pathlib import Path


STEP_PATTERN = re.compile(r"training/global_step:(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tmux-session", required=True)
    parser.add_argument("--log-path", required=True, type=Path)
    parser.add_argument("--stall-minutes", type=float, default=15.0)
    parser.add_argument("--shutdown-delay-minutes", type=float, default=5.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Monitor normally but log instead of shutting down the host.",
    )
    return parser.parse_args()


def tmux_session_alive(session: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def latest_completed_step(log_path: Path) -> int | None:
    if not log_path.is_file():
        return None
    matches = STEP_PATTERN.findall(log_path.read_text(errors="replace"))
    return int(matches[-1]) if matches else None


def main() -> int:
    args = parse_args()
    stall_seconds = args.stall_minutes * 60.0
    delay_seconds = args.shutdown_delay_minutes * 60.0

    last_step = latest_completed_step(args.log_path)
    last_progress_at = time.monotonic()
    print(
        f"[watchdog] armed: session={args.tmux_session!r}, "
        f"initial_step={last_step}, stall_minutes={args.stall_minutes:g}, "
        f"shutdown_delay_minutes={args.shutdown_delay_minutes:g}, "
        f"dry_run={args.dry_run}",
        flush=True,
    )

    while True:
        alive = tmux_session_alive(args.tmux_session)
        step = latest_completed_step(args.log_path)

        if step is not None and step != last_step:
            print(f"[watchdog] progress: {last_step} -> {step}", flush=True)
            last_step = step
            last_progress_at = time.monotonic()

        if not alive:
            reason = "training tmux session exited"
            allow_recovery = False
            break

        idle_seconds = time.monotonic() - last_progress_at
        if idle_seconds >= stall_seconds:
            reason = f"no completed global step for {idle_seconds / 60.0:.1f} minutes"
            allow_recovery = True
            break

        time.sleep(args.poll_seconds)

    grace_step = latest_completed_step(args.log_path)
    print(f"[watchdog] trigger: {reason}", flush=True)
    print(
        f"[watchdog] shutdown grace period: {args.shutdown_delay_minutes:g} minutes",
        flush=True,
    )

    grace_deadline = time.monotonic() + delay_seconds
    while time.monotonic() < grace_deadline:
        time.sleep(min(args.poll_seconds, grace_deadline - time.monotonic()))
        current_step = latest_completed_step(args.log_path)
        if (
            allow_recovery
            and tmux_session_alive(args.tmux_session)
            and current_step is not None
            and current_step != grace_step
        ):
            print(
                f"[watchdog] training recovered at step {current_step}; "
                "shutdown cancelled",
                flush=True,
            )
            return 0

    if args.dry_run:
        print(f"[watchdog] dry-run: would shut down because {reason}", flush=True)
        return 0

    print(f"[watchdog] shutting down host: {reason}", flush=True)
    return os.system("/usr/bin/shutdown -h now")


if __name__ == "__main__":
    raise SystemExit(main())
