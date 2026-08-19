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


# [项目注释] 功能：`runtime`：编排一个训练、采集、评测或 replay 流程，并汇总其结果。 主要协作调用：TeacherRuntime。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
def runtime():
    return TeacherRuntime(
        model="deepseek-v4-flash",
        base_url="https://provider.example/v1",
        api_key="top-secret",
    )


# [项目注释] 类型：`FakeCompletions` 封装相关状态、协议或数据结构。类属性和方法共同维护其不变量。
class FakeCompletions:
    # [项目注释] 功能：`__init__`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。
    # [项目注释] 输入：`response`；`error`。
    # [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.requests = []

    # [项目注释] 功能：`create`：异步地根据输入配置和中间状态构建或生成新的项目产物。 主要协作调用：isinstance, pop。
    # [项目注释] 输入：**`kwargs`。
    # [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
    async def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        if isinstance(self.response, list):
            return self.response.pop(0)
        return self.response


# [项目注释] 类型：`FakeClient` 封装相关状态、协议或数据结构。类属性和方法共同维护其不变量。
class FakeClient:
    # [项目注释] 功能：`__init__`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：SimpleNamespace。
    # [项目注释] 输入：`completions`。
    # [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)


# [项目注释] 功能：`response`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：SimpleNamespace, range, dumps。
# [项目注释] 输入：`arguments`；`name`；`count`。
# [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
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


# [项目注释] 功能：`test_teacher_runtime_is_secret_safe_and_role_is_pinned`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：runtime, repr, raises, TeacherRuntime。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_teacher_runtime_is_secret_safe_and_role_is_pinned():
    value = runtime()
    assert "top-secret" not in repr(value)
    assert value.action_retries == 3
    with pytest.raises(ValueError, match="teacher model"):
        TeacherRuntime("other-model", "https://provider.example/v1", "secret")


# [项目注释] 功能：`test_teacher_runtime_environment_defaults_to_three_action_retries`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：from_environment。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_teacher_runtime_environment_defaults_to_three_action_retries():
    value = TeacherRuntime.from_environment(
        {
            "TEACHER_MODEL": "deepseek-v4-flash",
            "TEACHER_BASE_URL": "https://provider.example/v1",
            "TEACHER_API_KEY": "top-secret",
        }
    )
    assert value.action_retries == 3


# [项目注释] 功能：`test_teacher_client_sends_required_official_tool_and_parses_action`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：run, FakeCompletions, OpenAICompatibleTeacherClient, scenario。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_teacher_client_sends_required_official_tool_and_parses_action():
    # [项目注释] 功能：`scenario`：异步地实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：FakeCompletions,
    # [项目注释]    OpenAICompatibleTeacherClient, response, runtime。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
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


# [项目注释] 功能：`test_teacher_protocol_requires_exactly_one_tool_call`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。 主要协作调用：run,
# [项目注释]    FakeCompletions, OpenAICompatibleTeacherClient, scenario。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_teacher_protocol_requires_exactly_one_tool_call():
    # [项目注释] 功能：`scenario`：异步地实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：FakeCompletions,
    # [项目注释]    OpenAICompatibleTeacherClient, response, runtime。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
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


# [项目注释] 功能：`test_teacher_protocol_retries_before_accepting_one_call`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：run, FakeCompletions, OpenAICompatibleTeacherClient, scenario。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_teacher_protocol_retries_before_accepting_one_call():
    # [项目注释] 功能：`scenario`：异步地实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：FakeCompletions,
    # [项目注释]    OpenAICompatibleTeacherClient, runtime, generate_action。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
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


# [项目注释] 功能：`test_thought_retry_locks_action_and_gives_specific_length_correction`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：run, FakeCompletions, OpenAICompatibleTeacherClient, scenario。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_thought_retry_locks_action_and_gives_specific_length_correction():
    # [项目注释] 功能：`scenario`：异步地实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：FakeCompletions,
    # [项目注释]    OpenAICompatibleTeacherClient, runtime, generate_action。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
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


# [项目注释] 功能：`test_teacher_force_answer_narrows_tool_schema`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。 主要协作调用：run,
# [项目注释]    FakeCompletions, OpenAICompatibleTeacherClient, scenario。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_teacher_force_answer_narrows_tool_schema():
    # [项目注释] 功能：`scenario`：异步地实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：FakeCompletions,
    # [项目注释]    OpenAICompatibleTeacherClient, response, runtime。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
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


# [项目注释] 功能：`test_teacher_request_constraint_narrows_choice_and_content`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：run, FakeCompletions, OpenAICompatibleTeacherClient, scenario。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_teacher_request_constraint_narrows_choice_and_content():
    # [项目注释] 功能：`scenario`：异步地实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：FakeCompletions,
    # [项目注释]    OpenAICompatibleTeacherClient, response, runtime。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
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


# [项目注释] 功能：`test_teacher_rejects_nonempty_assistant_prose_with_tool_call`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：run, response, OpenAICompatibleTeacherClient, scenario。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_teacher_rejects_nonempty_assistant_prose_with_tool_call():
    # [项目注释] 功能：`scenario`：异步地实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：response,
    # [项目注释]    OpenAICompatibleTeacherClient, runtime, raises。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
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


# [项目注释] 功能：`test_teacher_api_error_does_not_echo_secret_or_endpoint`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：run, FakeCompletions, OpenAICompatibleTeacherClient, scenario。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_teacher_api_error_does_not_echo_secret_or_endpoint():
    # [项目注释] 功能：`scenario`：异步地实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：FakeCompletions,
    # [项目注释]    OpenAICompatibleTeacherClient, runtime, raises。
    # [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
    # [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
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
