"""Repository-layout contracts adapted from the stage-oriented reference."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


# [项目注释] 功能：`test_stage_oriented_directories_exist`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。 主要协作调用：all, is_dir。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_stage_oriented_directories_exist():
    required = (
        "configs/data",
        "configs/interaction_config",
        "configs/tool_config",
        "configs/train/sft",
        "configs/train/grpo",
        "configs/eval",
        "scripts/data",
        "scripts/train/sft",
        "scripts/train/grpo",
        "scripts/eval",
        "scripts/vllm_server",
        "src/travel_grpo/data",
        "src/travel_grpo/envs",
        "src/travel_grpo/models",
        "src/travel_grpo/training",
        "src/travel_grpo/evaluation",
        "src/travel_grpo/utils",
    )
    assert all((ROOT / path).is_dir() for path in required)


# [项目注释] 功能：`test_runtime_boundaries_have_separate_configs`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。 主要协作调用：is_file。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_runtime_boundaries_have_separate_configs():
    interaction = ROOT / "configs" / "interaction_config"
    assert (interaction / "simulator_train.yaml").is_file()
    assert (interaction / "simulator_eval.yaml").is_file()
    assert (ROOT / "scripts" / "vllm_server" / "actor.sh").is_file()
    assert (ROOT / "scripts" / "vllm_server" / "train_user_simulator.sh").is_file()


# [项目注释] 功能：`test_legacy_flat_entrypoints_are_absent`：构造测试输入并断言目标行为，失败时暴露回归或契约不一致。 主要协作调用：any, list, exists,
# [项目注释]    glob。
# [项目注释] 输入：无显式业务参数（仅使用实例/类状态）。
# [项目注释] 输出：主要通过副作用更新状态或写出产物，默认返回 `None`。
def test_legacy_flat_entrypoints_are_absent():
    legacy_paths = (
        "configs/dataset_split.toml",
        "configs/grpo.yaml",
        "scripts/build_dataset_splits.py",
        "scripts/grpo.sh",
        "scripts/sft.sh",
        "scripts/evaluate.sh",
    )
    assert not any((ROOT / path).exists() for path in legacy_paths)
    for legacy_package in ("environment", "collection"):
        package_path = ROOT / "src" / "travel_grpo" / legacy_package
        assert not list(package_path.glob("*.py"))
