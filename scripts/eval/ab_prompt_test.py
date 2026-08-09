#!/usr/bin/env python3
"""Small paired A/B test for the SFT Actor prompt suffix.

Both conditions use the same SFT-merged Actor and the same frozen UserBench
task rows.  Condition A sends the task prompt unchanged.  Condition B appends
the exact ``TEACHER_ACTOR_POLICY`` suffix used during SFT data collection.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from travel_grpo.envs.userbench_interaction import SimulatorRole, UserSimulatorRuntime
from travel_grpo.evaluation.artifacts import atomic_json
from travel_grpo.evaluation.rollout import rollout_task
from travel_grpo.models.vllm_policy import ActorRuntime, OpenAICompatibleActorClient
from travel_grpo.training.sft_collection import TEACHER_ACTOR_POLICY


CHOICES = frozenset(("action", "search", "answer"))


def load_tasks(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    expected = ("task_id", "composition", "difficulty", "source_split", "prompt")
    if tuple(table.column_names) != expected:
        raise ValueError(f"unexpected evaluation schema: {table.column_names}")
    rows = table.to_pylist()
    if len(rows) != 471:
        raise ValueError(f"expected the frozen 471-row test set, found {len(rows)}")
    return rows


def choose_batch(rows: Sequence[Mapping[str, Any]], size: int) -> list[dict[str, Any]]:
    """Choose a deterministic, composition-balanced batch from the test set."""

    if size <= 0 or size > 32:
        raise ValueError("size must be between 1 and 32")
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        composition = str(row["composition"])
        if composition in {"333", "334", "444", "2222"}:
            continue
        groups.setdefault(composition, []).append(row)
    preferred = [key for key in ("22", "33", "44") if key in groups]
    selected: list[Mapping[str, Any]] = []
    # Round-robin gives a less composition-skewed smoke sample while keeping
    # task selection deterministic and identical across A and B.
    while len(selected) < size:
        progressed = False
        for composition in preferred:
            index = sum(1 for row in selected if str(row["composition"]) == composition)
            if index < len(groups[composition]):
                selected.append(groups[composition][index])
                progressed = True
                if len(selected) == size:
                    break
        if not progressed:
            raise ValueError("not enough eligible tasks for requested batch")
    return [dict(row) for row in selected]


def with_condition_prompt(task: Mapping[str, Any], condition: str) -> dict[str, Any]:
    value = copy.deepcopy(dict(task))
    value["prompt"] = copy.deepcopy(list(task["prompt"]))
    if condition == "B":
        system = value["prompt"][0]
        system["content"] = f"{system['content']}\n\n{TEACHER_ACTOR_POLICY}"
    return value


def action_choices(result: Mapping[str, Any]) -> list[str]:
    choices: list[str] = []
    for message in result.get("visible_transcript", []):
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1:
            continue
        function = calls[0].get("function") if isinstance(calls[0], Mapping) else None
        raw = function.get("arguments") if isinstance(function, Mapping) else None
        try:
            parameters = json.loads(raw) if isinstance(raw, str) else None
        except json.JSONDecodeError:
            parameters = None
        choice = parameters.get("choice") if isinstance(parameters, Mapping) else None
        if isinstance(choice, str) and choice.lower() in CHOICES:
            choices.append(choice.lower())
    return choices


def condition_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record.get("infrastructure_valid") is True]
    task_count = len(records)
    valid_count = len(valid)
    transitions = Counter()
    transition_denominators = Counter()
    task_transitions = Counter()
    answers_per_task: list[int] = []
    for record in valid:
        choices = action_choices(record)
        for source, target, name in (
            ("action", "search", "action_to_search"),
            ("search", "answer", "search_to_answer"),
        ):
            transition_denominators[name] += sum(value == source for value in choices)
            transitions[name] += sum(
                left == source and right == target
                for left, right in zip(choices, choices[1:])
            )
            if any(
                left == source and right == target
                for left, right in zip(choices, choices[1:])
            ):
                task_transitions[name] += 1
        answers_per_task.append(sum(value == "answer" for value in choices))

    def rate(name: str) -> float | None:
        denominator = transition_denominators[name]
        return transitions[name] / denominator if denominator else None

    reward_values = [record.get("reward", {}) for record in valid]

    def mean_metric(name: str) -> float:
        return sum(float(value.get(name, 0.0)) for value in reward_values) / valid_count if valid_count else 0.0

    return {
        "tasks": task_count,
        "valid_tasks": valid_count,
        "invalid_tasks": task_count - valid_count,
        "action_to_search": {
            "transitions": transitions["action_to_search"],
            "source_actions": transition_denominators["action_to_search"],
            "transition_rate": rate("action_to_search"),
            "tasks_with_transition": task_transitions["action_to_search"],
            "task_rate": task_transitions["action_to_search"] / valid_count if valid_count else None,
        },
        "search_to_answer": {
            "transitions": transitions["search_to_answer"],
            "source_searches": transition_denominators["search_to_answer"],
            "transition_rate": rate("search_to_answer"),
            "tasks_with_transition": task_transitions["search_to_answer"],
            "task_rate": task_transitions["search_to_answer"] / valid_count if valid_count else None,
        },
        "answer_call_rate": {
            "tasks_with_answer": sum(value > 0 for value in answers_per_task),
            "task_rate": sum(value > 0 for value in answers_per_task) / valid_count if valid_count else None,
            "total_answer_calls": sum(answers_per_task),
            "mean_answer_calls_per_task": sum(answers_per_task) / valid_count if valid_count else 0.0,
        },
        "completion": mean_metric("completion_rate"),
        "exact_repeats": {"total": sum(float(value.get("exact_repeats", 0)) for value in reward_values), "mean_per_task": mean_metric("exact_repeats")},
        "semantic_repeats": {"total": sum(float(value.get("semantic_repeats", 0)) for value in reward_values), "mean_per_task": mean_metric("semantic_repeats")},
        "termination_reasons": dict(sorted(Counter(str(record.get("termination_reason")) for record in records).items())),
    }


async def run(args: argparse.Namespace) -> None:
    rows = load_tasks(args.dataset)
    tasks = choose_batch(rows, args.tasks)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "manifest.json", {
        "schema_version": "ab-prompt-test-v1",
        "batch_size": len(tasks),
        "task_ids": [task["task_id"] for task in tasks],
        "compositions": [str(task["composition"]) for task in tasks],
        "excluded_compositions": ["333", "334", "444", "2222"],
        "model": args.model,
        "suffix": TEACHER_ACTOR_POLICY,
    })
    actor_runtime = ActorRuntime.from_environment()
    actor_runtime.require_model(args.model)
    simulator = UserSimulatorRuntime.from_environment(SimulatorRole.EVAL)
    actor = OpenAICompatibleActorClient(actor_runtime)
    try:
        for condition in ("A", "B"):
            condition_dir = output / condition
            condition_dir.mkdir(parents=True, exist_ok=True)
            records: list[dict[str, Any]] = []
            for index, task in enumerate(tasks, start=1):
                result = await rollout_task(
                    with_condition_prompt(task, condition),
                    actor=actor,
                    simulator=simulator,
                    source_root=ROOT / "environments/UserBench",
                )
                result["ab_condition"] = condition
                result["action_choices"] = action_choices(result)
                atomic_json(condition_dir / f"{index:02d}.json", result)
                records.append(result)
                print(f"condition={condition} task={index}/{len(tasks)} id={task['task_id']} valid={result.get('infrastructure_valid')} choices={result['action_choices']}", flush=True)
            atomic_json(condition_dir / "summary.json", condition_summary(records))
    finally:
        await actor.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/evaluation/tasks.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation/ab_prompt_test")
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--model", default="outputs/models/sft-merged")
    args = parser.parse_args()
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
