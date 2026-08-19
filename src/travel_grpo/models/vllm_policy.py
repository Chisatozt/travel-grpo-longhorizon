"""Strict OpenAI-compatible Actor boundary used by frozen evaluation."""

from __future__ import annotations

import inspect
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from travel_grpo.envs.userbench_tools import get_interact_with_env_schema
from travel_grpo.models.openai_compatible import (
    OpenAICompatibleTeacherClient,
    TeacherApiError,
    TeacherProtocolError,
    TeacherToolCall,
)


@dataclass(frozen=True)
# [项目注释] 类型：`ActorRuntime` 封装相关状态、协议或数据结构。类属性和方法共同维护其不变量。
class ActorRuntime:
    model: str
    base_url: str
    api_key: str = field(repr=False)
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 120.0

    # [项目注释] 功能：`__post_init__`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：ValueError, strip。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
    def __post_init__(self) -> None:
        if not self.model.strip() or not self.base_url.strip() or not self.api_key.strip():
            raise ValueError("Actor model, base URL, and API key must be non-empty")
        if self.temperature != 0.0:
            raise ValueError("frozen evaluation requires Actor temperature=0")

    @classmethod
    # [项目注释] 功能：`from_environment`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：cls, strip, ValueError, required。
    # [项目注释] 输入：`environ`: Mapping[str, str] | None。
    # [项目注释] 输出：标注返回 `'ActorRuntime'`；具体值由各分支决定。
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "ActorRuntime":
        env = os.environ if environ is None else environ
        # [项目注释] 功能：`required`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：strip, ValueError。
        # [项目注释] 输入：`name`: str。
        # [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
        def required(name: str) -> str:
            value = env.get(name, "").strip()
            if not value:
                raise ValueError(f"missing environment variable {name}")
            return value
        return cls(
            model=required("ACTOR_MODEL"),
            base_url=required("ACTOR_BASE_URL"),
            api_key=required("ACTOR_API_KEY"),
            temperature=float(env.get("ACTOR_TEMPERATURE", "0")),
            max_tokens=int(env.get("ACTOR_MAX_TOKENS", "4096")),
            timeout=float(env.get("ACTOR_TIMEOUT", "120")),
        )

    def require_model(self, expected_model: str) -> None:
        """Reject a stale Actor service before any frozen task is charged."""

        if self.model != expected_model:
            raise ValueError(
                f"ACTOR_MODEL={self.model!r} does not match the frozen stage model "
                f"{expected_model!r}; restart the Actor server and evaluation process"
            )


# [项目注释] 类型：`OpenAICompatibleActorClient` 封装相关状态、协议或数据结构。类属性和方法共同维护其不变量。
class OpenAICompatibleActorClient:
    # [项目注释] 功能：`__init__`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：AsyncOpenAI, TeacherApiError。
    # [项目注释] 输入：`runtime`: ActorRuntime；`client`: Any | None。
    # [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
    def __init__(self, runtime: ActorRuntime, *, client: Any | None = None) -> None:
        self.runtime = runtime
        self._owns_client = client is None
        if client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise TeacherApiError("evaluation requires `pip install -e .[eval]`") from exc
            client = AsyncOpenAI(api_key=runtime.api_key, base_url=runtime.base_url, timeout=runtime.timeout)
        self._client = client

    # [项目注释] 功能：`generate_action`：异步地根据输入配置和中间状态构建或生成新的项目产物。 主要协作调用：_parse_response, create,
    # [项目注释]    TeacherApiError, dict。
    # [项目注释] 输入：`messages`: Sequence[Mapping[str, Any]]。
    # [项目注释] 输出：标注返回 `TeacherToolCall`；具体值由各分支决定。
    async def generate_action(self, messages: Sequence[Mapping[str, Any]]) -> TeacherToolCall:
        try:
            response = await self._client.chat.completions.create(
                model=self.runtime.model,
                messages=[dict(value) for value in messages],
                tools=[get_interact_with_env_schema()],
                tool_choice="required",
                parallel_tool_calls=False,
                temperature=0,
                max_tokens=self.runtime.max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception as exc:
            raise TeacherApiError(f"Actor API request failed with {exc.__class__.__name__}") from exc
        return OpenAICompatibleTeacherClient._parse_response(response)

    # [项目注释] 功能：`close`：异步地清理运行时资源或恢复边界状态，保证后续调用不会继承脏状态。 主要协作调用：getattr, callable, close, isawaitable。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：标注返回 `None`；具体值由各分支决定。
    async def close(self) -> None:
        if not self._owns_client:
            return
        close = getattr(self._client, "close", None)
        if callable(close):
            value = close()
            if inspect.isawaitable(value):
                await value
