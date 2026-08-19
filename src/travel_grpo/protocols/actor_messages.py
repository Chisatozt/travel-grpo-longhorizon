"""Normalize messages to the actor-visible UserBench protocol."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


# [项目注释] 功能：`_clean_tool_call`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：isinstance, str, set, loads。
# [项目注释] 输入：`call`: Mapping[str, Any]。
# [项目注释] 输出：标注返回 `dict[str, Any] | None`；具体值由各分支决定。
def _clean_tool_call(call: Mapping[str, Any]) -> dict[str, Any] | None:
    function = call.get("function")
    if not isinstance(function, Mapping):
        return None
    name = function.get("name")
    raw_arguments = function.get("arguments")
    if name != "interact_with_env":
        return None
    try:
        parameters = (
            json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        )
    except json.JSONDecodeError:
        parameters = None
    if not isinstance(parameters, Mapping):
        return None
    public_parameters = {
        key: str(parameters[key])
        for key in ("thought", "choice", "content")
        if isinstance(parameters.get(key), str)
    }
    if set(public_parameters) != {"thought", "choice", "content"}:
        return None
    return {
        "id": str(call.get("id", "offline-call")),
        "type": "function",
        "function": {
            "name": "interact_with_env",
            "arguments": json.dumps(public_parameters, ensure_ascii=False, separators=(",", ":")),
        },
    }


def normalize_actor_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep only fields that are part of the actor-visible message contract."""

    cleaned: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        item: dict[str, Any] = {"role": role}
        content = message.get("content", "")
        if isinstance(content, str):
            item["content"] = content
        elif content is not None:
            item["content"] = str(content)
        if role == "assistant":
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                normalized_calls = [
                    clean_call
                    for call in calls
                    if isinstance(call, Mapping)
                    for clean_call in [_clean_tool_call(call)]
                    if clean_call is not None
                ]
                if normalized_calls:
                    item["tool_calls"] = normalized_calls
        if role == "tool":
            for key in ("name", "tool_call_id"):
                value = message.get(key)
                if isinstance(value, str) and value:
                    item[key] = value
        cleaned.append(item)
    return cleaned

__all__ = ["normalize_actor_messages"]

