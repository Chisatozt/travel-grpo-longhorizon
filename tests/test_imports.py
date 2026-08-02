"""Project import checks."""


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
