"""Offline teacher-trajectory collection tests."""

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import pytest

from travel_grpo.envs.observation import UserBenchObservation, UserBenchStepResult
from travel_grpo.envs.userbench_interaction import SimulatorRole, UserSimulatorRuntime
from travel_grpo.envs.userbench_tools import UserBenchAction
from travel_grpo.models.openai_compatible import TeacherRuntime, TeacherToolCall
from travel_grpo.training.sft_collection import (
    TeacherCollectionError,
    assert_disjoint_from_evaluation,
    collect_teacher_trajectory,
    collect_teacher_task_with_retries,
    load_teacher_task_pool,
    validate_teacher_collection_config,
    trajectory_rejection_reasons,
    write_teacher_collection_artifacts,
    write_teacher_trajectories,
)


def test_collection_cli_rejects_existing_output_before_runtime_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_path = ROOT / "scripts/train/sft/collect_sft_data.py"
    spec = importlib.util.spec_from_file_location(
        "collect_sft_data_script", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output = tmp_path / "existing.jsonl"
    output.write_text("do not replace\n", encoding="utf-8")
    args = module.build_parser().parse_args(["--output", str(output)])

    def unexpected_runtime(*args: object, **kwargs: object) -> None:
        raise AssertionError("runtime credentials must not be loaded")

    monkeypatch.setattr(module, "load_teacher_task_pool", unexpected_runtime)
    with pytest.raises(FileExistsError, match="already exists"):
        asyncio.run(module.run(args))


ROOT = Path(__file__).resolve().parents[1]


def test_collection_script_runs_from_source_checkout() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "TEACHER_MODEL": "deepseek-v4-flash",
            "TEACHER_BASE_URL": "https://teacher.example/v1",
            "TEACHER_API_KEY": "test-only",
            "COLLECTION_USER_SIM_MODEL": "deepseek-v4-flash",
            "COLLECTION_USER_SIM_BASE_URL": "https://simulator.example/v1",
            "COLLECTION_USER_SIM_API_KEY": "test-only",
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(ROOT / "scripts/train/sft/collect_sft_data.py"),
            "--dry-run",
            "--limit",
            "1",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["dry_run"] is True


class FakeTeacher:
    def __init__(self):
        self.runtime = TeacherRuntime(
            "deepseek-v4-flash", "https://teacher.example/v1", "teacher-secret"
        )
        self.index = 0

    async def generate_action(self, messages, *, force_answer=False):
        choice = "search" if self.index == 0 else "answer"
        content = "hotel in Paris" if self.index == 0 else "H1"
        call = TeacherToolCall(
            f"call-{self.index}",
            UserBenchAction.from_parameters(
                {"thought": "next", "choice": choice, "content": content}
            ),
        )
        self.index += 1
        return call

    async def close(self):
        return None


class FakeWrapper:
    instances: ClassVar[list] = []

    def __init__(self, task_id, runtime, config, **kwargs):
        self.task_id = task_id
        self.runtime = runtime
        self.config = config
        self.steps = 0
        self.closed = False
        self.__class__.instances.append(self)

    def reset(self):
        return UserBenchObservation("initial", 0, False, 0.0, {})

    async def astep(self, action):
        self.steps += 1
        done = self.steps == 2
        reward = 0.2 if self.steps == 1 else 1.0
        return UserBenchStepResult(
            self.task_id,
            UserBenchObservation(
                f"feedback-{self.steps}", self.steps, done, reward, {}
            ),
            reward,
            done,
            False,
            {},
        )

    def close(self):
        self.closed = True


def task(task_id="hotel:2-1"):
    return {
        "task_id": task_id,
        "composition": "22",
        "difficulty": "easy",
        "source_split": "train",
        "prompt": [
            {"role": "system", "content": "Use the tool."},
            {"role": "user", "content": "Plan a trip."},
        ],
    }


def simulator():
    return UserSimulatorRuntime(
        SimulatorRole.COLLECTION,
        "deepseek-v4-flash",
        "https://simulator.example/v1",
        "simulator-secret",
    )


def test_collects_tool_only_messages_and_raw_rewards():
    FakeWrapper.instances.clear()
    trajectory = asyncio.run(
        collect_teacher_trajectory(
            task(),
            teacher=FakeTeacher(),
            simulator=simulator(),
            wrapper_factory=FakeWrapper,
        )
    )
    record = trajectory.to_record()
    assert record["teacher_model"] == "deepseek-v4-flash"
    assert record["simulator_model"] == "deepseek-v4-flash"
    assert record["step_rewards"] == [0.2, 1.0]
    assert record["total_reward"] == pytest.approx(1.2)
    assert [message["role"] for message in record["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert record["messages"][-1]["content"] == "feedback-2"
    assert FakeWrapper.instances[0].closed


def write_pool(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_task_pool_disjointness_and_atomic_output(tmp_path):
    train_path = tmp_path / "train.jsonl"
    evaluation_path = tmp_path / "evaluation.jsonl"
    evaluation_task = {**task("hotel:2-2"), "source_split": "test"}
    write_pool(train_path, [task()])
    write_pool(evaluation_path, [evaluation_task])
    tasks = load_teacher_task_pool(train_path)
    assert_disjoint_from_evaluation(tasks, evaluation_path)

    trajectory = asyncio.run(
        collect_teacher_trajectory(
            task(),
            teacher=FakeTeacher(),
            simulator=simulator(),
            wrapper_factory=FakeWrapper,
        )
    )
    output = write_teacher_trajectories([trajectory], tmp_path / "output.jsonl")
    assert json.loads(output.read_text(encoding="utf-8"))["task_id"] == "hotel:2-1"
    with pytest.raises(FileExistsError):
        write_teacher_trajectories([trajectory], output)


def test_collection_config_pins_both_deepseek_roles():
    config = validate_teacher_collection_config(
        ROOT / "configs/train/sft/teacher_collection.yaml"
    )
    assert config["teacher"]["model"] == "deepseek-v4-flash"
    assert config["simulator"]["model"] == "deepseek-v4-flash"


def test_teacher_pool_rejects_frozen_test_rows(tmp_path):
    source = tmp_path / "wrong.jsonl"
    write_pool(source, [{**task(), "source_split": "test"}])
    with pytest.raises(TeacherCollectionError, match="official train"):
        load_teacher_task_pool(source)


class SequenceTeacher(FakeTeacher):
    def __init__(self, actions):
        super().__init__()
        self.actions = list(actions)
        self.requests = []

    async def generate_action(self, messages, *, force_answer=False):
        self.requests.append(list(messages))
        choice, content = self.actions.pop(0)
        index = self.index
        self.index += 1
        return TeacherToolCall(
            f"call-{index}",
            UserBenchAction.from_parameters(
                {"thought": "next", "choice": choice, "content": content}
            ),
        )


def test_duplicate_action_is_retried_without_stepping_environment():
    teacher = SequenceTeacher(
        [("search", "Paris hotels"), ("search", "Paris hotels"), ("answer", "H1")]
    )
    trajectory = asyncio.run(
        collect_teacher_trajectory(
            task(), teacher=teacher, simulator=simulator(), wrapper_factory=FakeWrapper
        )
    )
    assert len(trajectory.step_rewards) == 2
    assert trajectory.generation_diagnostics[0]["reason"] == "duplicate_action"


def test_semantic_duplicate_correction_names_completed_and_available_fields():
    teacher = SequenceTeacher(
        [
            ("action", "Do you need hotel amenities such as Wi-Fi?"),
            ("action", "Which hotel amenities do you prefer?"),
            ("answer", "H1"),
        ]
    )
    trajectory = asyncio.run(
        collect_teacher_trajectory(
            task(), teacher=teacher, simulator=simulator(), wrapper_factory=FakeWrapper
        )
    )
    assert len(trajectory.step_rewards) == 2
    correction = teacher.requests[2][-1]["content"]
    assert "hotel/amenities" in correction
    assert "Completed preference fields" in correction
    assert "Available unasked fields" in correction


def test_strict_gate_rejects_truncation_fallback_and_missing_answer():
    trajectory = asyncio.run(
        collect_teacher_trajectory(
            task(),
            teacher=FakeTeacher(),
            simulator=simulator(),
            wrapper_factory=FakeWrapper,
        )
    )
    invalid = trajectory.__class__(
        **{
            **trajectory.__dict__,
            "terminated": False,
            "truncated": True,
            "answered_aspects": (),
            "simulator_fallbacks": 1,
            "simulator_judgment_fallbacks": 1,
        }
    )
    assert set(trajectory_rejection_reasons(invalid)) == {
        "not_terminated",
        "truncated",
        "incomplete_aspect_answers",
        "simulator_fallback",
        "simulator_judgment_fallback",
    }


class RetryWrapper(FakeWrapper):
    attempt = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.this_attempt = self.__class__.attempt
        self.__class__.attempt += 1

    async def astep(self, action):
        result = await super().astep(action)
        if self.this_attempt == 0 and result.done:
            return UserBenchStepResult(
                result.task_id,
                result.observation,
                result.reward,
                False,
                True,
                result.diagnostics,
            )
        return result


def test_whole_trajectory_retry_and_artifact_routing(tmp_path):
    RetryWrapper.attempt = 0
    outcome = asyncio.run(
        collect_teacher_task_with_retries(
            task(),
            teacher=SequenceTeacher(
                [
                    ("search", "hotels first"),
                    ("answer", "H1"),
                    ("search", "hotels second"),
                    ("answer", "H1"),
                ]
            ),
            simulator=simulator(),
            wrapper_factory=RetryWrapper,
            max_attempts=2,
        )
    )
    assert outcome.accepted
    assert [attempt.accepted for attempt in outcome.attempts] == [False, True]
    paths = write_teacher_collection_artifacts(
        [outcome],
        accepted_path=tmp_path / "accepted.jsonl",
        rejected_path=tmp_path / "rejected.jsonl",
        diagnostics_path=tmp_path / "diagnostics.jsonl",
    )
    assert json.loads(paths[0].read_text(encoding="utf-8"))["trajectory_attempt"] == 2
    assert paths[1].read_text(encoding="utf-8") == ""
    assert len(paths[2].read_text(encoding="utf-8").splitlines()) == 2


def test_failed_attempt_diagnostic_contains_safe_partial_trajectory():
    outcome = asyncio.run(
        collect_teacher_task_with_retries(
            task(),
            teacher=SequenceTeacher(
                [
                    ("search", "Los Angeles hotels"),
                    ("search", "Los Angeles hotels"),
                    ("search", "Los Angeles hotels"),
                    ("search", "Los Angeles hotels"),
                ]
            ),
            simulator=simulator(),
            wrapper_factory=FakeWrapper,
            max_attempts=1,
        )
    )
    diagnostic = outcome.attempts[0]
    assert not outcome.accepted
    partial = diagnostic.partial_trajectory
    assert partial is not None
    assert partial["failure_environment_turn"] == 2
    assert partial["environment_steps_completed"] == 1
    assert partial["step_rewards"] == [0.2]
    assert partial["committed_actions"][0]["choice"] == "search"
    serialized = json.dumps(diagnostic.to_record())
    assert "teacher-secret" not in serialized
    assert "simulator-secret" not in serialized
