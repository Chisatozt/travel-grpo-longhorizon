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
class ActorRuntime:
    model: str
    base_url: str
    api_key: str = field(repr=False)
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 120.0

    def __post_init__(self) -> None:
        if not self.model.strip() or not self.base_url.strip() or not self.api_key.strip():
            raise ValueError("Actor model, base URL, and API key must be non-empty")
        if self.temperature != 0.0:
            raise ValueError("frozen evaluation requires Actor temperature=0")

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "ActorRuntime":
        env = os.environ if environ is None else environ
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


class OpenAICompatibleActorClient:
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

    async def close(self) -> None:
        if not self._owns_client:
            return
        close = getattr(self._client, "close", None)
        if callable(close):
            value = close()
            if inspect.isawaitable(value):
                await value
