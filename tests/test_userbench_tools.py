"""UserBench action and official tool-schema contracts."""

from pathlib import Path

import pytest
import yaml

from travel_grpo.envs.userbench_tools import (
    UserBenchAction,
    UserBenchActionError,
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
