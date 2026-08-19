"""CPU tests for the project-authorized UserBench search compatibility patch."""

import asyncio

from travelgym.env.prompt_async import async_generate_judge_search
from travelgym.env.prompts import evaluate_action, generate_judge_search
from travelgym.env.search_contract import match_base_search_aspect


# [项目注释] 功能：`_task`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
def _task():
    return {
        "dimensions": ["apartment"],
        "arguments": {
            "apartment": {
                "city": "Austin",
                "date_start": "November 10th",
                "date_end": "November 15th",
            }
        },
        "all_options": {
            "apartment": [
                {"id": "A1", "rating": 7, "service": {"cost_late_checkout_fee": 50}},
                {"id": "A2", "rating": 6, "service": {"cost_late_checkout_fee": None}},
            ]
        },
    }


# [项目注释] 功能：`_query`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：标注返回 `str`；具体值由各分支决定。
def _query() -> str:
    return (
        "Search for apartments in Austin from November 10th to November 15th "
        "with rating between 7 and 8 and late checkout."
    )


# [项目注释] 功能：`test_base_arguments_plus_preferences_match_one_aspect`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：_task, match_base_search_aspect, generate_judge_search, _query。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_base_arguments_plus_preferences_match_one_aspect():
    task = _task()
    assert match_base_search_aspect(_query(), task) == "apartment"
    assert generate_judge_search(_query(), task, {}) == {
        "alignment_judgement": "True",
        "alignment_aspect": "apartment",
        "alignment_mode": "base_plus_preferences",
    }


# [项目注释] 功能：`test_wrong_base_argument_does_not_get_positive_deterministic_judgement`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：_task, replace, match_base_search_aspect, _query。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_wrong_base_argument_does_not_get_positive_deterministic_judgement():
    task = _task()
    wrong = _query().replace("Austin", "Denver")
    assert match_base_search_aspect(wrong, task) is None


# [项目注释] 功能：`test_sync_search_and_preference_refinement_reuse_visible_candidates`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：_task, evaluate_action, setattr, _query。
# [项目注释] 输入：`monkeypatch`。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_sync_search_and_preference_refinement_reuse_visible_candidates(monkeypatch):
    task = _task()
    state = {
        "search_times": 0,
        "search_arguments": ["apartment"],
        "search_results": {},
        "search_queries": {},
    }
    config = {"search_failure_interval": 99, "search_correct_reward": 0.2}

    first, _, first_reward = evaluate_action(
        "[search] " + _query(), task, config, {}, [], [], state
    )
    assert first_reward == 0.2
    assert '"id": "A1"' in first
    assert state["search_arguments"] == []

    # A valid base+preference refinement must not call the legacy LLM judge.
    monkeypatch.setattr(
        "travelgym.env.prompts.model_call",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("judge called")),
    )
    second, _, second_reward = evaluate_action(
        "[search] " + _query() + " cheapest option first", task, config, {}, [], [], state
    )
    assert second_reward == 0.0
    assert "existing visible candidates" in second
    assert '"id": "A1"' in second
    assert state["search_results"]["apartment"][0]["id"] == "A1"


# [项目注释] 功能：`test_async_judge_uses_same_base_plus_preferences_contract`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释]    主要协作调用：run, async_generate_judge_search, _query, _task。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_async_judge_uses_same_base_plus_preferences_contract():
    result = asyncio.run(async_generate_judge_search(_query(), _task(), {}))
    assert result["alignment_judgement"] == "True"
    assert result["alignment_aspect"] == "apartment"
    assert result["alignment_mode"] == "base_plus_preferences"

