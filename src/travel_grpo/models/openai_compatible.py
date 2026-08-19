"""OpenAI-compatible teacher API boundary for UserBench trajectory collection."""

from __future__ import annotations

import copy
import inspect
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from travel_grpo.envs.userbench_interaction import DEEPSEEK_V4_FLASH_MODEL
from travel_grpo.envs.userbench_tools import (
    ActionChoice,
    TOOL_NAME,
    UserBenchAction,
    UserBenchActionError,
    get_interact_with_env_schema,
)


class TeacherApiError(RuntimeError):
    """Raised when the teacher endpoint cannot produce a trustworthy action."""


class TeacherProtocolError(TeacherApiError):
    """Raised when a teacher response violates the single-tool protocol."""

    rejections: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class TeacherRuntime:
    """Credentials and decoding parameters for the external teacher model."""

    model: str
    base_url: str
    api_key: str = field(repr=False)
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 60.0
    max_retries: int = 2
    protocol_retries: int = 2
    action_retries: int = 3
    thought_max_chars: int = 200
    thinking: str | None = "disabled"

    # [项目注释] 功能：`__post_init__`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：casefold, ValueError, strip。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
    def __post_init__(self) -> None:
        if self.model.casefold() != DEEPSEEK_V4_FLASH_MODEL:
            raise ValueError(
                f"teacher model must be {DEEPSEEK_V4_FLASH_MODEL!r}, got {self.model!r}"
            )
        if not self.base_url.strip():
            raise ValueError("teacher base_url must be non-empty")
        if not self.api_key.strip():
            raise ValueError("teacher api_key must be non-empty")
        if self.temperature < 0:
            raise ValueError("teacher temperature must be non-negative")
        if self.max_tokens <= 0:
            raise ValueError("teacher max_tokens must be positive")
        if self.timeout <= 0:
            raise ValueError("teacher timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("teacher max_retries must be non-negative")
        if self.protocol_retries < 0:
            raise ValueError("teacher protocol_retries must be non-negative")
        if self.action_retries < 0:
            raise ValueError("teacher action_retries must be non-negative")
        if self.thought_max_chars <= 0:
            raise ValueError("teacher thought_max_chars must be positive")
        if self.thinking not in {None, "enabled", "disabled"}:
            raise ValueError("teacher thinking must be enabled, disabled, or None")

    @classmethod
    # [项目注释] 功能：`from_environment`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：lower, cls, strip,
    # [项目注释]    TeacherApiError。
    # [项目注释] 输入：`environ`: Mapping[str, str] | None。
    # [项目注释] 输出：标注返回 `TeacherRuntime`；具体值由各分支决定。
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> TeacherRuntime:
        values = os.environ if environ is None else environ

        # [项目注释] 功能：`require`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：strip, TeacherApiError。
        # [项目注释] 输入：`name`: str。
        # [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
        def require(name: str) -> str:
            value = values.get(name, "").strip()
            if not value:
                raise TeacherApiError(f"missing environment variable {name}")
            return value

        thinking = values.get("TEACHER_THINKING", "disabled").strip().lower()
        return cls(
            model=require("TEACHER_MODEL"),
            base_url=require("TEACHER_BASE_URL"),
            api_key=require("TEACHER_API_KEY"),
            temperature=float(values.get("TEACHER_TEMPERATURE", "0")),
            max_tokens=int(values.get("TEACHER_MAX_TOKENS", "4096")),
            timeout=float(values.get("TEACHER_TIMEOUT", "60")),
            max_retries=int(values.get("TEACHER_MAX_RETRIES", "2")),
            protocol_retries=int(values.get("TEACHER_PROTOCOL_RETRIES", "2")),
            action_retries=int(values.get("TEACHER_ACTION_RETRIES", "3")),
            thought_max_chars=int(values.get("TEACHER_THOUGHT_MAX_CHARS", "200")),
            thinking=thinking or None,
        )


@dataclass(frozen=True)
class TeacherRequestConstraint:
    """Request-only phase constraints; never added to the archived tool schema."""

    choice: ActionChoice
    allowed_contents: tuple[str, ...] = ()

    # [项目注释] 功能：`__post_init__`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：any, ValueError, isinstance, strip。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip() for value in self.allowed_contents):
            raise ValueError("allowed_contents must contain non-empty strings")


@dataclass(frozen=True)
class TeacherToolCall:
    """Validated assistant tool call, independent of the OpenAI SDK types."""

    call_id: str
    action: UserBenchAction
    content: str | None = None
    protocol_attempts: int = 1
    protocol_rejections: tuple[dict[str, Any], ...] = ()
    latency_seconds: float = 0.0
    usage: Mapping[str, int] = field(default_factory=dict)

    @property
    # [项目注释] 功能：`parameters`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：标注返回 `dict[str, str]`；具体值由各分支决定。
    def parameters(self) -> dict[str, str]:
        return {
            "thought": self.action.thought,
            "choice": self.action.choice.value,
            "content": self.action.content,
        }

    # [项目注释] 功能：`to_assistant_message`：把协议/状态数据转换为模型、用户或日志可见的文本表示。 主要协作调用：dumps。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：标注返回 `dict[str, Any]`；具体值由各分支决定。
    def to_assistant_message(self) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {
                    "id": self.call_id,
                    "type": "function",
                    "function": {
                        "name": TOOL_NAME,
                        "arguments": json.dumps(
                            self.parameters, ensure_ascii=False, separators=(",", ":")
                        ),
                    },
                }
            ],
        }


# [项目注释] 类型：`TeacherClientProtocol` 封装相关状态、协议或数据结构。类属性和方法共同维护其不变量。
class TeacherClientProtocol(Protocol):
    runtime: TeacherRuntime

    # [项目注释] 功能：`generate_action`：异步地根据输入配置和中间状态构建或生成新的项目产物。
    # [项目注释] 输入：`messages`: Sequence[Mapping[str, Any]]；`force_answer`: bool；`constraint`:
    # [项目注释]    TeacherRequestConstraint | None。
    # [项目注释] 输出：标注返回 `TeacherToolCall`；具体值由各分支决定。
    async def generate_action(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        force_answer: bool = False,
        constraint: TeacherRequestConstraint | None = None,
    ) -> TeacherToolCall: ...

    # [项目注释] 功能：`close`：异步地清理运行时资源或恢复边界状态，保证后续调用不会继承脏状态。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
    async def close(self) -> None: ...


# [项目注释] 功能：`_attribute`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：isinstance, getattr。
# [项目注释] 输入：`value`: Any；`name`: str。
# [项目注释] 输出：标注返回 `Any`；具体值由各分支决定。
def _attribute(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


# [项目注释] 功能：`_usage_record`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：_attribute, int, isinstance。
# [项目注释] 输入：`response`: Any。
# [项目注释] 输出：标注返回 `dict[str, int]`；具体值由各分支决定。
def _usage_record(response: Any) -> dict[str, int]:
    usage = _attribute(response, "usage")
    return {
        name: int(value)
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        if isinstance((value := _attribute(usage, name)), int)
        and not isinstance(value, bool)
    }


class OpenAICompatibleTeacherClient:
    """Strict single-tool client for a DeepSeek-V4-Flash compatible endpoint."""

    # [项目注释] 功能：`__init__`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：AsyncOpenAI, TeacherApiError。
    # [项目注释] 输入：`runtime`: TeacherRuntime；`client`: Any | None。
    # [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
    def __init__(self, runtime: TeacherRuntime, *, client: Any | None = None) -> None:
        self.runtime = runtime
        self._owns_client = client is None
        if client is None:
            try:
                from openai import AsyncOpenAI
            except (
                ImportError
            ) as exc:  # pragma: no cover - optional runtime dependency.
                raise TeacherApiError(
                    "teacher collection requires the API extra; run "
                    "`pip install -e .[api]`"
                ) from exc
            client = AsyncOpenAI(
                api_key=runtime.api_key,
                base_url=runtime.base_url,
                timeout=runtime.timeout,
                max_retries=runtime.max_retries,
            )
        self._client = client

    # [项目注释] 功能：`generate_action`：异步地根据输入配置和中间状态构建或生成新的项目产物。 主要协作调用：range, AssertionError,
    # [项目注释]    TeacherProtocolError, TeacherRequestConstraint。
    # [项目注释] 输入：`messages`: Sequence[Mapping[str, Any]]；`force_answer`: bool；`constraint`:
    # [项目注释]    TeacherRequestConstraint | None。
    # [项目注释] 输出：标注返回 `TeacherToolCall`；具体值由各分支决定。
    async def generate_action(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        force_answer: bool = False,
        constraint: TeacherRequestConstraint | None = None,
    ) -> TeacherToolCall:
        if not messages:
            raise TeacherProtocolError("teacher messages must not be empty")
        rejections: list[dict[str, Any]] = []
        correction: str | None = None
        locked_action: tuple[ActionChoice, str] | None = None
        if force_answer:
            forced = TeacherRequestConstraint(ActionChoice.ANSWER)
            if constraint is not None and constraint.choice is not ActionChoice.ANSWER:
                raise TeacherProtocolError("force_answer conflicts with request constraint")
            constraint = constraint or forced
        elapsed = 0.0
        for attempt in range(self.runtime.protocol_retries + 1):
            schema = copy.deepcopy(get_interact_with_env_schema())
            parameters = schema["function"]["parameters"]
            parameters["additionalProperties"] = False
            thought = parameters["properties"]["thought"]
            thought["maxLength"] = self.runtime.thought_max_chars
            thought["description"] = (
                "One short operational sentence for the next action, at most "
                f"{self.runtime.thought_max_chars} characters. Do not include hidden chain of thought."
            )
            if constraint is not None:
                parameters["properties"]["choice"]["enum"] = [constraint.choice.value]
                if constraint.allowed_contents:
                    parameters["properties"]["content"]["enum"] = list(
                        constraint.allowed_contents
                    )
            request: dict[str, Any] = {
                "model": self.runtime.model,
                "messages": self._protocol_messages(messages, attempt, correction),
                "tools": [schema],
                "tool_choice": "required",
                "parallel_tool_calls": False,
                "temperature": self.runtime.temperature,
                "max_tokens": self.runtime.max_tokens,
            }
            if self.runtime.thinking is not None:
                request["extra_body"] = {"thinking": {"type": self.runtime.thinking}}
            try:
                started = time.perf_counter()
                response = await self._client.chat.completions.create(**request)
                request_latency = time.perf_counter() - started
                elapsed += request_latency
            except Exception as exc:
                raise TeacherApiError(
                    f"teacher API request failed with {exc.__class__.__name__}"
                ) from exc
            try:
                parsed = self._parse_response(response)
                current_action = (parsed.action.choice, parsed.action.content)
                if locked_action is not None and current_action != locked_action:
                    error = TeacherProtocolError(
                        "teacher changed choice/content during a thought-only retry"
                    )
                    error.reason_code = "thought_retry_action_drift"
                    raise error
                if len(parsed.action.thought) > self.runtime.thought_max_chars:
                    locked_action = current_action
                    error = TeacherProtocolError(
                        "teacher thought exceeds configured character limit"
                    )
                    error.reason_code = "thought_too_long"
                    error.thought_length = len(parsed.action.thought)
                    raise error
                if constraint is not None and parsed.action.choice is not constraint.choice:
                    error = TeacherProtocolError(
                        f"teacher choice must be {constraint.choice.value!r} in the current phase"
                    )
                    error.reason_code = "phase_requires_choice"
                    raise error
                if (
                    constraint is not None
                    and constraint.allowed_contents
                    and parsed.action.content not in constraint.allowed_contents
                ):
                    error = TeacherProtocolError(
                        "teacher content is outside the current phase allowlist"
                    )
                    error.reason_code = "phase_content_not_allowed"
                    raise error
                return TeacherToolCall(
                    call_id=parsed.call_id,
                    action=parsed.action,
                    content=parsed.content,
                    protocol_attempts=attempt + 1,
                    protocol_rejections=tuple(rejections),
                    latency_seconds=elapsed,
                    usage=_usage_record(response),
                )
            except TeacherProtocolError as exc:
                diagnostic: dict[str, Any] = {
                    "attempt": attempt + 1,
                    "reason": str(exc),
                    "reason_code": getattr(exc, "reason_code", "invalid_tool_call"),
                    "latency_seconds": request_latency,
                    "usage": _usage_record(response),
                }
                thought_length = getattr(exc, "thought_length", None)
                if thought_length is not None:
                    diagnostic["thought_length"] = thought_length
                    diagnostic["thought_limit"] = self.runtime.thought_max_chars
                if locked_action is not None:
                    diagnostic["locked_choice"] = locked_action[0].value
                    diagnostic["locked_content"] = locked_action[1]
                rejections.append(diagnostic)
                if attempt >= self.runtime.protocol_retries:
                    exc.rejections = tuple(rejections)
                    raise
                correction = self._retry_correction(exc, locked_action)
        raise AssertionError("unreachable teacher protocol retry state")

    @staticmethod
    # [项目注释] 功能：`_protocol_messages`：把协议/状态数据转换为模型、用户或日志可见的文本表示。 主要协作调用：dict, insert, isinstance,
    # [项目注释]    TeacherProtocolError。
    # [项目注释] 输入：`messages`: Sequence[Mapping[str, Any]]；`attempt`: int；`correction`: str | None。
    # [项目注释] 输出：标注返回 `list[dict[str, Any]]`；具体值由各分支决定。
    def _protocol_messages(
        messages: Sequence[Mapping[str, Any]],
        attempt: int,
        correction: str | None = None,
    ) -> list[dict[str, Any]]:
        prepared = [dict(message) for message in messages]
        reminder = (
            "Sequential environment protocol: for this completion, emit exactly one "
            "interact_with_env tool call. Never batch or parallelize multiple calls. "
            "The result of the current action must be observed before choosing the next "
            "action."
        )
        if attempt:
            reminder += (
                f" This is protocol retry {attempt}; the previous completion was "
                "discarded. Follow the correction below exactly."
            )
        if prepared and prepared[0].get("role") == "system":
            content = prepared[0].get("content")
            if not isinstance(content, str):
                raise TeacherProtocolError("teacher system content must be text")
            prepared[0]["content"] = f"{content}\n\n{reminder}"
        else:
            prepared.insert(0, {"role": "system", "content": reminder})
        if correction is not None:
            prepared.append({"role": "user", "content": correction})
        return prepared

    # [项目注释] 功能：`_retry_correction`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：getattr, dumps。
    # [项目注释] 输入：`error`: TeacherProtocolError；`locked_action`: tuple[ActionChoice, str] | None。
    # [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
    def _retry_correction(
        self,
        error: TeacherProtocolError,
        locked_action: tuple[ActionChoice, str] | None,
    ) -> str:
        reason_code = getattr(error, "reason_code", "invalid_tool_call")
        if reason_code in {"thought_too_long", "thought_retry_action_drift"}:
            assert locked_action is not None
            action_json = json.dumps(
                {
                    "choice": locked_action[0].value,
                    "content": locked_action[1],
                },
                ensure_ascii=False,
            )
            return (
                "Only the previous `thought` was invalid. Emit exactly one tool call, "
                f"preserve this choice/content exactly: {action_json}, and replace only "
                "`thought` with one operational sentence no longer than "
                f"{self.runtime.thought_max_chars} characters."
            )
        if reason_code in {"reserved_phase_requires_answer", "phase_requires_choice"}:
            return (
                "Follow the current phase choice enum exactly and emit one "
                "interact_with_env call."
            )
        if reason_code == "phase_content_not_allowed":
            return "Copy one content value from the tool schema enum exactly."
        return (
            f"The previous completion was invalid: {error}. Emit exactly one valid "
            "interact_with_env tool call with only thought, choice, and content."
        )

    @staticmethod
    # [项目注释] 功能：`_parse_response`：读取并解析外部数据，将其转换为项目内部可消费的结构。 主要协作调用：_attribute, TeacherToolCall, isinstance,
    # [项目注释]    TeacherProtocolError。
    # [项目注释] 输入：`response`: Any。
    # [项目注释] 输出：标注返回 `TeacherToolCall`；具体值由各分支决定。
    def _parse_response(response: Any) -> TeacherToolCall:
        choices = _attribute(response, "choices")
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
            raise TeacherProtocolError("teacher response has no choices")
        if len(choices) != 1:
            raise TeacherProtocolError(
                "teacher response must contain exactly one choice"
            )
        message = _attribute(choices[0], "message")
        tool_calls = _attribute(message, "tool_calls")
        if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes)):
            raise TeacherProtocolError("teacher response contains no tool call")
        if len(tool_calls) != 1:
            raise TeacherProtocolError(
                "teacher response must contain exactly one interact_with_env call; "
                f"received {len(tool_calls)}"
            )
        tool_call = tool_calls[0]
        function = _attribute(tool_call, "function")
        name = _attribute(function, "name")
        if name != TOOL_NAME:
            raise TeacherProtocolError(f"teacher called unsupported tool {name!r}")
        raw_arguments = _attribute(function, "arguments")
        if not isinstance(raw_arguments, str):
            raise TeacherProtocolError("teacher tool arguments must be a JSON string")
        try:
            parameters = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise TeacherProtocolError(
                "teacher tool arguments are invalid JSON"
            ) from exc
        try:
            action = UserBenchAction.from_parameters(parameters)
        except UserBenchActionError as exc:
            raise TeacherProtocolError(f"invalid teacher action: {exc}") from exc
        call_id = _attribute(tool_call, "id")
        if not isinstance(call_id, str) or not call_id.strip():
            raise TeacherProtocolError("teacher tool call is missing an ID")
        content = _attribute(message, "content")
        if content is not None and not isinstance(content, str):
            raise TeacherProtocolError("teacher assistant content must be text or null")
        if isinstance(content, str) and content.strip():
            error = TeacherProtocolError(
                "teacher assistant content must be empty when a tool call is present"
            )
            error.reason_code = "assistant_content_not_empty"
            raise error
        return TeacherToolCall(call_id=call_id, action=action, content=content)

    # [项目注释] 功能：`close`：异步地清理运行时资源或恢复边界状态，保证后续调用不会继承脏状态。 主要协作调用：getattr, callable, close, isawaitable。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
    async def close(self) -> None:
        if not self._owns_client:
            return
        close = getattr(self._client, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
