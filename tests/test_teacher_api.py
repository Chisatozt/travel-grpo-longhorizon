"""Offline tests for the DeepSeek teacher API boundary."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from travel_grpo.models.openai_compatible import (
    OpenAICompatibleTeacherClient,
    TeacherApiError,
    TeacherProtocolError,
    TeacherRequestConstraint,
    TeacherRuntime,
)
from travel_grpo.envs.userbench_tools import ActionChoice


def runtime():
    return TeacherRuntime(
        model="deepseek-v4-flash",
        base_url="https://provider.example/v1",
        api_key="top-secret",
    )


class FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        if isinstance(self.response, list):
            return self.response.pop(0)
        return self.response


class FakeClient:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)


def response(arguments, *, name="interact_with_env", count=1):
    calls = [
        SimpleNamespace(
            id=f"call-{index}",
            function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
        )
        for index in range(count)
    ]
    message = SimpleNamespace(content=None, tool_calls=calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_teacher_runtime_is_secret_safe_and_role_is_pinned():
    value = runtime()
    assert "top-secret" not in repr(value)
    assert value.action_retries == 3
    with pytest.raises(ValueError, match="teacher model"):
        TeacherRuntime("other-model", "https://provider.example/v1", "secret")


def test_teacher_runtime_environment_defaults_to_three_action_retries():
    value = TeacherRuntime.from_environment(
        {
            "TEACHER_MODEL": "deepseek-v4-flash",
            "TEACHER_BASE_URL": "https://provider.example/v1",
            "TEACHER_API_KEY": "top-secret",
        }
    )
    assert value.action_retries == 3


def test_teacher_client_sends_required_official_tool_and_parses_action():
    async def scenario():
        completions = FakeCompletions(
            response(
                {"thought": "search first", "choice": "search", "content": "hotels"}
            )
        )
        client = OpenAICompatibleTeacherClient(
            runtime(), client=FakeClient(completions)
        )
        call = await client.generate_action([{"role": "user", "content": "trip"}])
        request = completions.requests[0]
        assert request["model"] == "deepseek-v4-flash"
        assert request["tool_choice"] == "required"
        assert request["parallel_tool_calls"] is False
        assert request["tools"][0]["function"]["name"] == "interact_with_env"
        parameters = request["tools"][0]["function"]["parameters"]
        assert parameters["additionalProperties"] is False
        assert parameters["properties"]["thought"]["maxLength"] == 200
        assert request["extra_body"] == {"thinking": {"type": "disabled"}}
        assert "exactly one" in request["messages"][0]["content"]
        assert call.action.to_environment_action() == "[search] hotels"
        assert call.to_assistant_message()["tool_calls"][0]["id"] == "call-0"

    asyncio.run(scenario())


def test_teacher_protocol_requires_exactly_one_tool_call():
    async def scenario():
        completions = FakeCompletions(
            response({"thought": "x", "choice": "search", "content": "x"}, count=2)
        )
        client = OpenAICompatibleTeacherClient(
            runtime(), client=FakeClient(completions)
        )
        with pytest.raises(TeacherProtocolError, match="exactly one"):
            await client.generate_action([{"role": "user", "content": "trip"}])
        assert len(completions.requests) == 3

    asyncio.run(scenario())


def test_teacher_protocol_retries_before_accepting_one_call():
    async def scenario():
        arguments = {"thought": "x", "choice": "search", "content": "hotels"}
        completions = FakeCompletions(
            [response(arguments, count=2), response(arguments, count=1)]
        )
        client = OpenAICompatibleTeacherClient(
            runtime(), client=FakeClient(completions)
        )

        call = await client.generate_action([{"role": "user", "content": "trip"}])

        assert call.action.content == "hotels"
        assert call.protocol_attempts == 2
        assert call.protocol_rejections[0]["attempt"] == 1
        assert len(completions.requests) == 2
        retry_content = completions.requests[1]["messages"][0]["content"]
        assert "protocol retry 1" in retry_content

    asyncio.run(scenario())


def test_thought_retry_locks_action_and_gives_specific_length_correction():
    async def scenario():
        long_arguments = {
            "thought": "x" * 201,
            "choice": "search",
            "content": "Los Angeles apartments",
        }
        short_arguments = {
            "thought": "Search apartments.",
            "choice": "search",
            "content": "Los Angeles apartments",
        }
        completions = FakeCompletions(
            [response(long_arguments), response(short_arguments)]
        )
        client = OpenAICompatibleTeacherClient(
            runtime(), client=FakeClient(completions)
        )

        call = await client.generate_action([{"role": "user", "content": "trip"}])

        assert call.action.content == "Los Angeles apartments"
        rejection = call.protocol_rejections[0]
        assert rejection["reason_code"] == "thought_too_long"
        assert rejection["thought_length"] == 201
        correction = completions.requests[1]["messages"][-1]["content"]
        assert "replace only `thought`" in correction
        assert '"content": "Los Angeles apartments"' in correction
        assert "200 characters" in correction

    asyncio.run(scenario())


def test_teacher_force_answer_narrows_tool_schema():
    async def scenario():
        completions = FakeCompletions(
            response({"thought": "finish", "choice": "answer", "content": "H1"})
        )
        client = OpenAICompatibleTeacherClient(
            runtime(), client=FakeClient(completions)
        )
        await client.generate_action(
            [{"role": "user", "content": "trip"}], force_answer=True
        )
        choice = completions.requests[0]["tools"][0]["function"]["parameters"][
            "properties"
        ]["choice"]
        assert choice["enum"] == ["answer"]

    asyncio.run(scenario())


def test_teacher_request_constraint_narrows_choice_and_content():
    async def scenario():
        content = "Which hotel amenities are important to you?"
        completions = FakeCompletions(
            response({"thought": "ask", "choice": "action", "content": content})
        )
        client = OpenAICompatibleTeacherClient(
            runtime(), client=FakeClient(completions)
        )
        await client.generate_action(
            [{"role": "user", "content": "trip"}],
            constraint=TeacherRequestConstraint(ActionChoice.ACTION, (content,)),
        )
        properties = completions.requests[0]["tools"][0]["function"]["parameters"][
            "properties"
        ]
        assert properties["choice"]["enum"] == ["action"]
        assert properties["content"]["enum"] == [content]

    asyncio.run(scenario())


def test_teacher_rejects_nonempty_assistant_prose_with_tool_call():
    async def scenario():
        value = response(
            {"thought": "x", "choice": "search", "content": "hotels"}
        )
        value.choices[0].message.content = "I will search."
        client = OpenAICompatibleTeacherClient(
            runtime(), client=FakeClient(FakeCompletions(value))
        )
        with pytest.raises(TeacherProtocolError, match="must be empty"):
            await client.generate_action([{"role": "user", "content": "trip"}])

    asyncio.run(scenario())


def test_teacher_api_error_does_not_echo_secret_or_endpoint():
    async def scenario():
        completions = FakeCompletions(error=RuntimeError("provider unavailable"))
        client = OpenAICompatibleTeacherClient(
            runtime(), client=FakeClient(completions)
        )
        with pytest.raises(TeacherApiError) as caught:
            await client.generate_action([{"role": "user", "content": "trip"}])
        assert "top-secret" not in str(caught.value)
        assert "provider.example" not in str(caught.value)

    asyncio.run(scenario())
