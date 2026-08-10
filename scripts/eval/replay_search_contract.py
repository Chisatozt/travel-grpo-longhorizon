#!/usr/bin/env python3
"""Replay Actor search calls against the patched UserBench search contract.

This is an inference-free diagnostic.  It reuses actor-visible search calls
from an existing closed-loop trace, checks the deterministic base-argument
compatibility path first, and optionally lets the configured evaluation
simulator judge only the ambiguous calls.  It never loads an Actor model and
never writes hidden correctness or reward state to the report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
USERBENCH = ROOT / "environments" / "UserBench"
for path in (SRC, USERBENCH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from travel_grpo.envs.userbench_interaction import (  # noqa: E402
    SimulatorRole,
    UserSimulatorRuntime,
    bind_user_simulator_process,
)
from travel_grpo.evaluation.artifacts import atomic_json  # noqa: E402
from travel_grpo.prompts.actor_policy import ACTOR_RUNTIME_POLICY_VERSION  # noqa: E402
from travelgym.env.search_contract import (  # noqa: E402
    deterministic_search_judgement,
)
from travelgym.env.prompt_async import async_evaluate_action  # noqa: E402
from travelgym.env.task_data import get_task_by_id  # noqa: E402


_SEARCH_FALLBACK_MARKERS = (
    "something wrong within your searching request",
    "arguments are not related to any of the ground truth",
    "searching backend is experiencing some issues",
)


def _extract_search_calls(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    transcript = value.get("visible_transcript", [])
    calls: list[str] = []
    if not isinstance(transcript, list):
        return calls
    for message in transcript:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls", ()):
            if not isinstance(tool_call, Mapping):
                continue
            function = tool_call.get("function")
            if not isinstance(function, Mapping):
                continue
            raw = function.get("arguments")
            try:
                arguments = json.loads(raw) if isinstance(raw, str) else raw
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(arguments, Mapping) and arguments.get("choice") == "search":
                content = arguments.get("content")
                if isinstance(content, str) and content.strip():
                    calls.append(content.strip())
    return calls


def _candidate_ids(task: Mapping[str, Any], aspect: str | None) -> set[str]:
    if not aspect:
        return set()
    all_options = task.get("all_options")
    options = all_options.get(aspect, []) if isinstance(all_options, Mapping) else []
    return {
        str(option.get("id"))
        for option in options
        if isinstance(option, Mapping) and isinstance(option.get("id"), str)
    }


def _has_candidate_list(feedback: str, ids: set[str]) -> bool:
    if not ids:
        return False
    return "Here are all the options" in feedback and any(
        re.search(rf"(?<![A-Za-z0-9]){re.escape(option_id)}(?![A-Za-z0-9])", feedback)
        for option_id in ids
    )


def _is_fallback(feedback: str) -> bool:
    lowered = feedback.casefold()
    return any(marker in lowered for marker in _SEARCH_FALLBACK_MARKERS)


def _initial_state(task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "search_times": 0,
        "search_arguments": list(task.get("dimensions", ())),
        "search_results": {},
        "search_queries": {},
    }


def _model_config(runtime: UserSimulatorRuntime) -> dict[str, Any]:
    return {
        "api_key": runtime.api_key,
        "model_name": runtime.model,
        "temperature": runtime.temperature,
        "max_tokens": runtime.max_tokens,
        "timeout": runtime.timeout,
    }


async def replay(
    trace_dir: Path,
    output: Path,
    *,
    api_fallback: bool,
) -> dict[str, Any]:
    paths = sorted(trace_dir.glob("[0-9][0-9].json"))
    if not paths:
        raise FileNotFoundError(f"no closed-loop records under {trace_dir}")

    runtime: UserSimulatorRuntime | None = None
    model_config: dict[str, Any] = {}
    if api_fallback:
        runtime = UserSimulatorRuntime.from_environment(SimulatorRole.EVAL)
        bind_user_simulator_process(runtime)
        model_config = _model_config(runtime)

    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for path in paths:
        source = json.loads(path.read_text(encoding="utf-8"))
        task_id = str(source.get("task_id"))
        task = get_task_by_id(task_id)
        state = _initial_state(task)
        for index, query in enumerate(_extract_search_calls(path), start=1):
            deterministic = deterministic_search_judgement(query, task)
            used_api = deterministic is None and api_fallback
            feedback = ""
            error: str | None = None
            if deterministic is None and not api_fallback:
                feedback = "Replay skipped: no deterministic base-argument match."
            else:
                try:
                    feedback, _, _internal_reward = await async_evaluate_action(
                        "[search] " + query,
                        task,
                        {
                            "search_failure_interval": 10**9,
                            "search_correct_reward": 0.2,
                        },
                        model_config,
                        [],
                        [],
                        state,
                    )
                except Exception as exc:  # pragma: no cover - defensive API boundary
                    error = f"{exc.__class__.__name__}: {exc}"

            aspects = state.get("search_results", {})
            aspect = deterministic.get("alignment_aspect") if deterministic else None
            if aspect is None and isinstance(aspects, Mapping) and aspects:
                aspect = str(next(reversed(aspects)))
            ids = _candidate_ids(task, aspect)
            rows.append(
                {
                    "source_record": path.name,
                    "task_id": task_id,
                    "composition": str(source.get("composition", "")),
                    "query_index": index,
                    "query": query,
                    "deterministic_base_match": deterministic is not None,
                    "api_judgement_used": used_api,
                    "matched_aspect": aspect,
                    "candidate_list_returned": _has_candidate_list(feedback, ids),
                    "fallback_returned": _is_fallback(feedback),
                    "error": error,
                }
            )

    by_path = Counter(
        "candidate" if row["candidate_list_returned"] else "fallback_or_no_result"
        for row in rows
    )
    summary = {
        "schema_version": "userbench-search-replay-v1",
        "source_trace_dir": str(trace_dir),
        "task_count": len(paths),
        "query_count": len(rows),
        "actor_policy_version": ACTOR_RUNTIME_POLICY_VERSION,
        "userbench_search_contract": "base_args_plus_preferences_v1",
        "api_fallback_enabled": api_fallback,
        "deterministic_base_matches": sum(row["deterministic_base_match"] for row in rows),
        "api_judgement_queries": sum(row["api_judgement_used"] for row in rows),
        "candidate_list_returned": sum(row["candidate_list_returned"] for row in rows),
        "fallback_or_no_result": sum(not row["candidate_list_returned"] for row in rows),
        "fallback_returned": sum(row["fallback_returned"] for row in rows),
        "errors": sum(bool(row["error"]) for row in rows),
        "outcome_counts": dict(sorted(by_path.items())),
        "compositions": dict(sorted(Counter(row["composition"] for row in rows).items())),
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(
        output / "manifest.json",
        {
            "schema_version": "userbench-search-replay-v1",
            "source_trace_dir": str(trace_dir),
            "task_records": len(paths),
            "query_records": len(rows),
            "api_fallback_enabled": api_fallback,
            "model_parameters_not_recorded": True,
        },
    )
    with (output / "queries.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=ROOT / "outputs/evaluation/validation32_sft_merged_b/B/closed_loop",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/evaluation/userbench-search-replay-v1",
    )
    parser.add_argument(
        "--no-api-fallback",
        action="store_true",
        help="only exercise the deterministic base-argument compatibility path",
    )
    args = parser.parse_args()
    summary = asyncio.run(
        replay(args.trace_dir, args.output, api_fallback=not args.no_api_fallback)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
