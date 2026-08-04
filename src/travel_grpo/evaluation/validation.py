"""Normalize veRL validation dumps into the fixed UserBench summary contract."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from travel_grpo.evaluation.summary import summarize_results


def summarize_validation_rows(
    rows: Sequence[Mapping[str, Any]], tasks: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(tasks) != 132:
        raise ValueError("checkpoint selection validation must contain 132 tasks")
    composition = {str(row["task_id"]): str(row["composition"]) for row in tasks}
    if len(composition) != 132:
        raise ValueError("checkpoint validation task IDs must be unique")
    results = []
    seen: set[str] = set()
    for row in rows:
        task_id = str(row.get("task_id", ""))
        if task_id not in composition:
            raise ValueError(f"validation dump contains an unknown task ID: {task_id!r}")
        if task_id in seen:
            raise ValueError(f"validation dump contains duplicate task ID: {task_id!r}")
        seen.add(task_id)
        reward_valid = row.get("reward_valid") is True
        results.append(
            {
                "task_id": task_id,
                "composition": composition[task_id],
                "infrastructure_valid": reward_valid,
                "actor_attempts": row.get("actor_attempts", 0),
                "environment_steps": row.get("environment_steps", 0),
                "termination_reason": row.get("termination_reason"),
                "reward": {
                    "reward_valid": reward_valid,
                    "terminal_reward": row.get("terminal_reward", row.get("score", 0.0)),
                    "quality_by_aspect": row.get("quality_by_aspect", {}),
                    "correct_itinerary": row.get("correct_itinerary") is True,
                    "gold_itinerary": row.get("gold_itinerary") is True,
                    "user_aligned_success": row.get("user_aligned_success") is True,
                    "completion_rate": row.get("completion_rate", 0.0),
                    "active_preference_coverage": row.get("active_preference_coverage", 0.0),
                    "passive_preference_coverage": row.get("passive_preference_coverage", 0.0),
                    "efficiency": row.get("efficiency", 0.0),
                    "policy_penalty": row.get("policy_penalty", 0.0),
                    "invalid_actions": row.get("invalid_actions", 0),
                    "exact_repeats": row.get("exact_repeats", 0),
                    "semantic_repeats": row.get("semantic_repeats", 0),
                },
            }
        )
    return summarize_results(
        results,
        expected_task_ids=[str(row["task_id"]) for row in tasks],
        expected_compositions=[str(row["composition"]) for row in tasks],
    )


def summarize_validation_file(
    path: Path, tasks: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"validation dump rows must be JSON objects: {path}")
    return summarize_validation_rows(rows, tasks)
