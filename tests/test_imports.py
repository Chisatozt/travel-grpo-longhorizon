"""Project import checks."""


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
