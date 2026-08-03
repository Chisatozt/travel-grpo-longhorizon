"""UserBench action and official tool-schema contracts."""

from pathlib import Path

import pytest
import yaml

from travel_grpo.envs.userbench_tools import (
    UserBenchAction,
    UserBenchActionError,
    action_query_issue,
    get_interact_with_env_schema,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("choice", ["search", "action", "answer"])
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
def test_invalid_actions_fail_stably(parameters, message):
    with pytest.raises(UserBenchActionError, match=message):
        UserBenchAction.from_parameters(parameters)


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
