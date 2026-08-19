#!/usr/bin/env python3
"""Probe one-step search-to-answer behavior without running live UserBench.

Contexts are cut from accepted SFT trajectories immediately after a normal
search result. The Actor receives only that transcript and must emit exactly
one next tool call. A removes the shared runtime policy; B restores the
versioned production runtime policy. Teacher-only generation instructions are
never added to either Actor context.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from travel_grpo.evaluation.artifacts import atomic_json
from travel_grpo.models.vllm_policy import ActorRuntime, OpenAICompatibleActorClient
from travel_grpo.prompts.actor_policy import (
    ACTOR_RUNTIME_POLICY,
    ACTOR_RUNTIME_POLICY_VERSION,
    ensure_actor_runtime_policy,
    strip_actor_runtime_policy,
)


ID_RE = re.compile(r"^[ACFHR]\d+$")
JSON_ID_RE = re.compile(r'"id"\s*:\s*"([ACFHR]\d+)"')
NORMAL_MARKER = "Here are all the options for <"
FALLBACK_MARKERS = (
    "Normally simulate a system error",
    "searching backend is experiencing some issues",
    "By default will return N/A",
)
DEFAULT_SOURCES = (
    ROOT / "outputs/teacher_trajectories/sft_validation.from_train.accepted.jsonl",
    ROOT / "outputs/teacher_trajectories/sft_validation.from_train.silver.jsonl",
)


# [项目注释] 功能：`load_task_map`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：glob, loads, isinstance, read_text。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `dict[str, dict[str, Any]]`；具体值由各分支决定。
def load_task_map() -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for path in (ROOT / "environments/UserBench/travelgym/data").glob(
        "travelgym_data_*.json"
    ):
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            tasks.update(value)
    return tasks


# [项目注释] 功能：`clean_messages`：把协议/状态数据转换为模型、用户或日志可见的文本表示。 主要协作调用：deepcopy, pop, dict。
# [项目注释] 输入：`messages`: list[Mapping[str, Any]]。
# [项目注释] 输出：标注返回 `list[dict[str, Any]]`；具体值由各分支决定。
def clean_messages(messages: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for message in messages:
        value = copy.deepcopy(dict(message))
        value.pop("loss_mask", None)
        cleaned.append(value)
    return cleaned


def without_teacher_suffix(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compatibility name: remove current or historical policy blocks."""

    return strip_actor_runtime_policy(messages)


def with_teacher_suffix(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compatibility name: append exactly one production runtime policy."""

    return ensure_actor_runtime_policy(messages)


# [项目注释] 功能：`collect_contexts`：编排一个训练、采集、评测或 replay 流程，并汇总其结果。 主要协作调用：enumerate, len, ValueError,
# [项目注释]    splitlines。
# [项目注释] 输入：`sources`: tuple[Path, ...]；`task_map`: Mapping[str, Mapping[str, Any]]；`limit`: int。
# [项目注释] 输出：标注返回 `list[dict[str, Any]]`；具体值由各分支决定。
def collect_contexts(
    sources: tuple[Path, ...], task_map: Mapping[str, Mapping[str, Any]], limit: int
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for source in sources:
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            task_id = str(record["task_id"])
            task = task_map.get(task_id)
            if not isinstance(task, Mapping):
                continue
            messages = record.get("messages")
            if not isinstance(messages, list):
                continue
            for index in range(len(messages) - 1):
                assistant, tool = messages[index : index + 2]
                if assistant.get("role") != "assistant" or tool.get("role") != "tool":
                    continue
                calls = assistant.get("tool_calls")
                if not isinstance(calls, list) or len(calls) != 1:
                    continue
                function = calls[0].get("function") if isinstance(calls[0], Mapping) else None
                raw_arguments = function.get("arguments") if isinstance(function, Mapping) else None
                try:
                    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else None
                except json.JSONDecodeError:
                    arguments = None
                if not isinstance(arguments, Mapping) or arguments.get("choice") != "search":
                    continue
                tool_content = tool.get("content")
                if not isinstance(tool_content, str) or NORMAL_MARKER not in tool_content:
                    continue
                if any(marker in tool_content for marker in FALLBACK_MARKERS):
                    continue
                aspect = tool_content.split(NORMAL_MARKER, 1)[1].split(">", 1)[0]
                aspect_data = task.get(aspect)
                if not isinstance(aspect_data, Mapping):
                    continue
                candidate_ids = list(dict.fromkeys(JSON_ID_RE.findall(tool_content)))
                all_ids = {str(value) for value in aspect_data.get("all_ids", ())}
                correct_ids = [str(value) for value in aspect_data.get("correct_ids", ())]
                if not candidate_ids or not set(candidate_ids) <= all_ids or not correct_ids:
                    continue
                context_messages = clean_messages(messages[: index + 2])
                contexts.append(
                    {
                        "context_id": len(contexts) + 1,
                        "source_file": str(source.relative_to(ROOT)),
                        "source_line": line_number,
                        "task_id": task_id,
                        "quality_tier": str(record.get("quality_tier", "unknown")),
                        "composition": str(record.get("composition", "unknown")),
                        "message_index": index,
                        "aspect": aspect,
                        "candidate_ids": candidate_ids,
                        "correct_ids": correct_ids,
                        "messages": context_messages,
                    }
                )
    if len(contexts) < limit:
        raise ValueError(f"only {len(contexts)} eligible contexts, need {limit}")
    return contexts[:limit]


# [项目注释] 功能：`classify`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：bool, lower, str, fullmatch。
# [项目注释] 输入：`parameters`: Mapping[str, Any] | None；`context`: Mapping[str, Any]。
# [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
def classify(parameters: Mapping[str, Any] | None, context: Mapping[str, Any]) -> dict[str, Any]:
    choice = str(parameters.get("choice", "")).strip().lower() if parameters else ""
    content = str(parameters.get("content", "")) if parameters else ""
    exact_one_id = bool(ID_RE.fullmatch(content.strip()))
    option_id = content.strip() if exact_one_id else None
    candidates = {str(value) for value in context["candidate_ids"]}
    correct = {str(value) for value in context["correct_ids"]}
    return {
        "choice": choice,
        "content": content,
        "answer_at_1": choice == "answer",
        "content_is_one_option_id": exact_one_id,
        "option_id": option_id,
        "id_in_candidates": bool(option_id and option_id in candidates),
        "id_is_correct": bool(option_id and option_id in correct),
        "wrong_repeat_search": choice == "search",
        "wrong_repeat_action": choice == "action",
        "protocol_invalid": not bool(choice),
    }


# [项目注释] 功能：`summarize`：计算奖励、指标或聚合统计，供训练、评测或报告使用。 主要协作调用：len, sum, dict, count。
# [项目注释] 输入：`results`: list[Mapping[str, Any]]。
# [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
def summarize(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(results)
    answer_calls = [value for value in results if value.get("answer_at_1") is True]

    # [项目注释] 功能：`count`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：sum。
    # [项目注释] 输入：`key`: str。
    # [项目注释] 输出：标注返回 `int`；具体值由各分支决定。
    def count(key: str) -> int:
        return sum(value.get(key) is True for value in results)

    # [项目注释] 功能：`pct`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。
    # [项目注释] 输入：`value`: int；`denominator`: int。
    # [项目注释] 输出：标注返回 `float`；具体值由各分支决定。
    def pct(value: int, denominator: int = n) -> float:
        return value / denominator if denominator else 0.0

    return {
        "contexts": n,
        "answer_at_1": {"count": count("answer_at_1"), "rate": pct(count("answer_at_1"))},
        "answer_content_one_option_id": {
            "count": count("content_is_one_option_id"),
            "rate_all_contexts": pct(count("content_is_one_option_id")),
            "count_given_answer": sum(value.get("content_is_one_option_id") is True for value in answer_calls),
            "rate_given_answer": pct(
                sum(value.get("content_is_one_option_id") is True for value in answer_calls),
                len(answer_calls),
            ),
        },
        "id_in_candidates": {
            "count": count("id_in_candidates"),
            "rate_all_contexts": pct(count("id_in_candidates")),
            "count_given_answer": sum(value.get("id_in_candidates") is True for value in answer_calls),
            "rate_given_answer": pct(
                sum(value.get("id_in_candidates") is True for value in answer_calls),
                len(answer_calls),
            ),
        },
        "id_is_correct": {
            "count": count("id_is_correct"),
            "rate_all_contexts": pct(count("id_is_correct")),
            "count_given_answer": sum(value.get("id_is_correct") is True for value in answer_calls),
            "rate_given_answer": pct(
                sum(value.get("id_is_correct") is True for value in answer_calls),
                len(answer_calls),
            ),
        },
        "wrong_repeat_search": {"count": count("wrong_repeat_search"), "rate": pct(count("wrong_repeat_search"))},
        "wrong_repeat_action": {"count": count("wrong_repeat_action"), "rate": pct(count("wrong_repeat_action"))},
        "protocol_invalid": {"count": count("protocol_invalid"), "rate": pct(count("protocol_invalid"))},
        "choice_distribution": dict(sorted(Counter(str(value.get("choice", "")) for value in results).items())),
    }


# [项目注释] 功能：`run`：异步地编排一个训练、采集、评测或 replay 流程，并汇总其结果。 主要协作调用：load_task_map, collect_contexts, mkdir,
# [项目注释]    atomic_json。
# [项目注释] 输入：`args`: argparse.Namespace。
# [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
async def run(args: argparse.Namespace) -> None:
    task_map = load_task_map()
    contexts = collect_contexts(tuple(args.source), task_map, args.contexts)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "search-answer-probe-v1",
        "model": args.model,
        "contexts": len(contexts),
        "sources": [str(path.relative_to(ROOT)) for path in args.source],
        "actor_policy_version": ACTOR_RUNTIME_POLICY_VERSION,
        "actor_runtime_policy": ACTOR_RUNTIME_POLICY,
        "contexts_metadata": [
            {key: value for key, value in context.items() if key != "messages"}
            for context in contexts
        ],
    }
    atomic_json(output / "manifest.json", manifest)
    actor_runtime = ActorRuntime.from_environment()
    actor_runtime.require_model(args.model)
    actor = OpenAICompatibleActorClient(actor_runtime)
    try:
        for condition, transform in (("A", without_teacher_suffix), ("B", with_teacher_suffix)):
            results: list[dict[str, Any]] = []
            for context in contexts:
                error: str | None = None
                parameters: Mapping[str, Any] | None = None
                try:
                    call = await actor.generate_action(transform(context["messages"]))
                    parameters = call.parameters
                except Exception as exc:  # retain one result per context for diagnostics
                    error = f"{exc.__class__.__name__}: {exc}"
                result = {
                    "context_id": context["context_id"],
                    "task_id": context["task_id"],
                    "aspect": context["aspect"],
                    "candidate_ids": context["candidate_ids"],
                    "correct_ids": context["correct_ids"],
                    "error": error,
                    "parameters": dict(parameters) if parameters is not None else None,
                    **classify(parameters, context),
                }
                atomic_json(output / condition / f"{context['context_id']:03d}.json", result)
                results.append(result)
                print(
                    f"condition={condition} context={context['context_id']}/{len(contexts)} "
                    f"task={context['task_id']} choice={result['choice']} "
                    f"answer={result['answer_at_1']} error={error is not None}",
                    flush=True,
                )
            atomic_json(output / condition / "summary.json", summarize(results))
    finally:
        await actor.close()


# [项目注释] 功能：`main`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：ArgumentParser, add_argument, parse_args, run。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `int`；具体值由各分支决定。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=int, default=24)
    parser.add_argument("--source", type=Path, action="append", default=list(DEFAULT_SOURCES))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation/search_answer_probe")
    parser.add_argument("--model", default="outputs/models/sft-merged")
    args = parser.parse_args()
    if not 20 <= args.contexts <= 50:
        parser.error("--contexts must be between 20 and 50")
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
