"""Project import checks."""


# [项目注释] 功能：`test_stage_packages_import`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_stage_packages_import():
    import travel_grpo.envs
    import travel_grpo.evaluation
    import travel_grpo.models
    import travel_grpo.training
    import travel_grpo.utils

    assert travel_grpo.envs is not None
    assert travel_grpo.evaluation is not None
    assert travel_grpo.models is not None
    assert travel_grpo.training is not None
    assert travel_grpo.utils is not None


# [项目注释] 功能：`test_dataset_split_public_api_imports`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。 主要协作调用：callable。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_dataset_split_public_api_imports():
    from travel_grpo.data import (
        build_dataset_splits,
        load_onechoice_tasks,
        load_split_spec,
        verify_dataset_splits,
        write_dataset_splits,
    )

    assert callable(load_onechoice_tasks)
    assert callable(build_dataset_splits)
    assert callable(write_dataset_splits)
    assert callable(verify_dataset_splits)
    assert callable(load_split_spec)


# [项目注释] 功能：`test_refactored_stage_facades_preserve_public_symbols`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_refactored_stage_facades_preserve_public_symbols():
    from travel_grpo.data.recovery.boundaries import SourceSpec as CanonicalSourceSpec
    from travel_grpo.data.recovery_boundaries import SourceSpec as LegacySourceSpec
    from travel_grpo.data.recovery.targets import TargetDecision as CanonicalTargetDecision
    from travel_grpo.data.recovery_targets import TargetDecision as LegacyTargetDecision
    from travel_grpo.training.grpo.turn_credit import TurnCreditConfig as LegacyTurnCreditConfig
    from travel_grpo.training.sft.contracts import TeacherTaskOutcome as CanonicalTeacherTaskOutcome
    from travel_grpo.training.sft.planning import build_stratified_task_plan as canonical_build_plan
    from travel_grpo.training.sft.collection import TeacherTrajectory as CanonicalTeacherTrajectory
    from travel_grpo.training.sft_collection import TeacherTrajectory as LegacyTeacherTrajectory
    from travel_grpo.training.sft.dataset import ActionOnlyExample as CanonicalActionOnlyExample
    from travel_grpo.training.sft_dataset import ActionOnlyExample as LegacyActionOnlyExample
    from travel_grpo.training.sft.recovery import render_recovery_record as canonical_render
    from travel_grpo.training.recovery_sft import render_recovery_record as legacy_render
    from travel_grpo.trajectory.turn_credit import TurnCreditConfig as CanonicalTurnCreditConfig

    assert LegacySourceSpec is CanonicalSourceSpec
    assert LegacyTargetDecision is CanonicalTargetDecision
    assert LegacyTurnCreditConfig is CanonicalTurnCreditConfig
    assert LegacyTeacherTrajectory is CanonicalTeacherTrajectory
    assert LegacyActionOnlyExample is CanonicalActionOnlyExample
    assert legacy_render is canonical_render
    assert CanonicalTeacherTrajectory.__module__ == "travel_grpo.training.sft.contracts"
    assert CanonicalTeacherTaskOutcome.__module__ == "travel_grpo.training.sft.contracts"
    assert canonical_build_plan.__module__ == "travel_grpo.training.sft.planning"
