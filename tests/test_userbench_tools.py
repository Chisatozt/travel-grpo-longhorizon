"""UserBench action and official tool-schema contracts."""

from pathlib import Path

import pytest
import yaml

from travel_grpo.envs.userbench_tools import (
    UserBenchAction,
    UserBenchActionError,
    action_field_matches,
    action_query_issue,
    extract_visible_option_ids,
    get_interact_with_env_schema,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("choice", ["search", "action", "answer"])
# [项目注释] 功能：`test_action_renders_the_upstream_protocol`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。 主要协作调用：parametrize,
# [项目注释]    from_parameters, to_environment_action。
# [项目注释] 输入：`choice`。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_action_renders_the_upstream_protocol(choice):
    action = UserBenchAction.from_parameters(
        {"thought": " reason ", "choice": choice, "content": f"[{choice}] payload "}
    )
    assert action.thought == "reason"
    assert action.content == "payload"
    assert action.to_environment_action() == f"[{choice}] payload"


@pytest.mark.parametrize(
    "parameters, message",
    [
        ({"choice": "search", "content": "x"}, "missing required"),
        (
            {"thought": "x", "choice": "search", "content": "x", "extra": 1},
            "unexpected",
        ),
        ({"thought": "", "choice": "search", "content": "x"}, "thought"),
        ({"thought": "x", "choice": "finish", "content": "x"}, "choice"),
        (
            {"thought": "x", "choice": "answer", "content": "[search] x"},
            "conflicts",
        ),
        (
            {"thought": "x", "choice": "answer", "content": "[finish] x"},
            "not exposed",
        ),
    ],
)
# [项目注释] 功能：`test_invalid_actions_fail_stably`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。 主要协作调用：parametrize, raises,
# [项目注释]    from_parameters。
# [项目注释] 输入：`parameters`；`message`。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_invalid_actions_fail_stably(parameters, message):
    with pytest.raises(UserBenchActionError, match=message):
        UserBenchAction.from_parameters(parameters)


# [项目注释] 功能：`test_python_and_yaml_schemas_match_the_pinned_official_schema`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：safe_load, get_interact_with_env_schema, read_text。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_python_and_yaml_schemas_match_the_pinned_official_schema():
    official = yaml.safe_load(
        (ROOT / "environments/UserBench/schema/interact_tool.yaml").read_text(
            encoding="utf-8"
        )
    )["tool_schema"]
    project = yaml.safe_load(
        (ROOT / "configs/tool_config/userbench_tools.yaml").read_text(encoding="utf-8")
    )["tools"][0]
    assert get_interact_with_env_schema() == official
    assert project["tool_schema"] == official
    assert project["class_name"] == (
        "travel_grpo.training.grpo.adapter.tools.UserBenchTool"
    )


# [项目注释] 功能：`test_action_query_issue_normalizes_plural_field_words`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：from_parameters, action_query_issue。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_action_query_issue_normalizes_plural_field_words():
    bundled = UserBenchAction.from_parameters(
        {
            "thought": "ask",
            "choice": "action",
            "content": (
                "How many bedrooms and bathrooms, and which amenities do you need "
                "for the apartment?"
            ),
        }
    )
    assert action_query_issue(bundled, ("apartment",)) == "bundled"


# [项目注释] 功能：`test_longer_service_hint_shadows_embedded_seat_hint_only_at_same_span`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：from_parameters, action_query_issue。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_longer_service_hint_shadows_embedded_seat_hint_only_at_same_span():
    service_only = UserBenchAction.from_parameters(
        {
            "thought": "ask",
            "choice": "action",
            "content": "Do you need a child seat service for the rental car?",
        }
    )
    truly_bundled = UserBenchAction.from_parameters(
        {
            "thought": "ask",
            "choice": "action",
            "content": "How many seats, and do you need a child seat for the rental car?",
        }
    )
    assert action_query_issue(service_only, ("rental_car",)) is None
    assert action_query_issue(truly_bundled, ("rental_car",)) == "bundled"


# [项目注释] 功能：`test_layover_duration_is_a_flight_time_question_not_a_path_bundle`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：action_field_matches。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_layover_duration_is_a_flight_time_question_not_a_path_bundle():
    assert action_field_matches(
        "Do you prefer a longer layover duration for the flight?", ("flight",)
    ) == {("flight", "time")}


# [项目注释] 功能：`test_carry_on_allowance_is_flight_amenities_not_service`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：action_field_matches。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_carry_on_allowance_is_flight_amenities_not_service():
    assert action_field_matches(
        "Do you need a carry-on baggage allowance and Wi-Fi for the flight?",
        ("flight",),
    ) == {("flight", "amenities")}


# [项目注释] 功能：`test_delivery_is_a_restaurant_tags_question`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：action_field_matches。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_delivery_is_a_restaurant_tags_question():
    assert action_field_matches(
        "Would restaurant delivery be important for you?", ("restaurant",)
    ) == {("restaurant", "tags")}


# [项目注释] 功能：`test_visible_option_extraction_uses_official_boundaries_only`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：extract_visible_option_ids。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_visible_option_extraction_uses_official_boundaries_only():
    assert extract_visible_option_ids(
        "Results: H1, H2, C3 and R4. Ignore xH9 and H4A."
    ) == {"H1", "H2", "C3", "R4"}
