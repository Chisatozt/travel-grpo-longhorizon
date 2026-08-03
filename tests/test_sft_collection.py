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
from travel_grpo.envs.reward import TravelRewardTask, UserBenchRewardSnapshot
from travel_grpo.envs.userbench_interaction import SimulatorRole, UserSimulatorRuntime
from travel_grpo.envs.userbench_tools import UserBenchAction
from travel_grpo.models.openai_compatible import TeacherRuntime, TeacherToolCall
from travel_grpo.training.teacher_policy import TeacherPhase, TeacherTurnPlan
from travel_grpo.envs.userbench_tools import FIELD_QUERY_HINTS
from travel_grpo.training.sft_collection import (
    TeacherCollectionError,
    assert_disjoint_from_evaluation,
    collect_teacher_trajectory,
    collect_teacher_task_with_retries,
    initialize_teacher_run,
    load_teacher_outcome_checkpoints,
    load_teacher_task_pool,
    validate_teacher_collection_config,
    trajectory_rejection_reasons,
    write_teacher_collection_artifacts,
    write_teacher_outcome_checkpoint,
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


def test_collection_cli_checkpoints_and_resumes_without_recollecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_path = ROOT / "scripts/train/sft/collect_sft_data.py"
    spec = importlib.util.spec_from_file_location("collect_resume_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "evaluation.jsonl"
    write_pool(train, [task()])
    write_pool(evaluation, [{**task("hotel:2-2"), "source_split": "test"}])
    for name, value in {
        "TEACHER_MODEL": "deepseek-v4-flash",
        "TEACHER_BASE_URL": "https://teacher.example/v1",
        "TEACHER_API_KEY": "test-only",
        "COLLECTION_USER_SIM_MODEL": "deepseek-v4-flash",
        "COLLECTION_USER_SIM_BASE_URL": "https://simulator.example/v1",
        "COLLECTION_USER_SIM_API_KEY": "test-only",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(module, "OpenAICompatibleTeacherClient", lambda runtime: FakeTeacher())

    async def collect(tasks, *, teacher, simulator, on_outcome, **kwargs):
        values = []
        for value in tasks:
            outcome = await collect_teacher_task_with_retries(
                value,
                teacher=teacher,
                simulator=simulator,
                wrapper_factory=FakeWrapper,
                max_attempts=1,
            )
            on_outcome(outcome)
            values.append(outcome)
        return tuple(values)

    monkeypatch.setattr(module, "collect_teacher_outcomes", collect)
    output = tmp_path / "accepted.jsonl"
    run_dir = tmp_path / "run"
    common = [
        "--input",
        str(train),
        "--evaluation-tasks",
        str(evaluation),
        "--output",
        str(output),
        "--run-dir",
        str(run_dir),
        "--attempts",
        "1",
    ]
    first = asyncio.run(module.run(module.build_parser().parse_args(common)))
    assert first["new_tasks"] == 1
    assert first["quality"]["accepted"] == 1
    assert len(list((run_dir / "tasks").glob("*.json"))) == 1

    async def unexpected_collect(*args, **kwargs):
        raise AssertionError("resume must not recollect checkpointed tasks")

    monkeypatch.setattr(module, "collect_teacher_outcomes", unexpected_collect)
    resumed = asyncio.run(
        module.run(module.build_parser().parse_args([*common, "--resume"]))
    )
    assert resumed["resumed_tasks"] == 1
    assert resumed["new_tasks"] == 0


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

    async def generate_action(self, messages, *, force_answer=False, constraint=None):
        choice = constraint.choice.value
        if constraint.allowed_contents:
            content = constraint.allowed_contents[0]
        elif choice == "action":
            instruction = messages[-1]["content"]
            field = next(
                value
                for value in ("name", "room", "amenities", "service", "rating")
                if f"`{value}`" in instruction
            )
            content = {
                "name": "Do you prefer a specific hotel name?",
                "room": "What hotel room configuration do you need?",
                "amenities": "Which hotel amenities are important?",
                "service": "Which hotel service is important?",
                "rating": "What hotel rating do you prefer?",
            }[field]
        elif choice == "search":
            content = "Search for hotel options in Paris"
        else:
            content = "H1"
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
        self.actions = []
        self.closed = False
        self.__class__.instances.append(self)

    def reset(self):
        self.steps = 0
        return UserBenchObservation("initial", 0, False, 0.0, {})

    def reward_task(self):
        return TravelRewardTask(
            self.task_id,
            ("hotel",),
            {"hotel": "H1"},
            {"hotel": frozenset({"H1"})},
            {"hotel": frozenset({"P1"})},
        )

    def reward_snapshot(self):
        choices = [value.choice.value for value in self.actions]
        return UserBenchRewardSnapshot(
            remaining_preference_ids=(frozenset() if "action" in choices else frozenset({"P1"})),
            active_elicited_count=1 if "action" in choices else 0,
            passive_elicited_count=0,
            remaining_search_aspects=(frozenset() if "search" in choices else frozenset({"hotel"})),
            choice_initials=frozenset({"H"}) if "answer" in choices else frozenset(),
        )

    async def astep(self, action):
        self.actions.append(action)
        self.steps += 1
        done = action.choice.value == "answer"
        reward = 1.0 if done else 0.2
        feedback = "H1 is a visible hotel option." if action.choice.value == "search" else f"feedback-{self.steps}"
        return UserBenchStepResult(
            self.task_id,
            UserBenchObservation(
                feedback, self.steps, done, reward, {}
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
    assert record["step_rewards"] == [0.2, 0.2, 1.0]
    assert record["total_reward"] == pytest.approx(1.4)
    assert record["schema_version"] == "userbench-teacher-trajectory-v4"
    assert record["reward_version"] == "userbench-travel-reward-v2"
    assert record["reward_valid"] is True
    assert record["terminal_reward"] == pytest.approx(1.0)
    assert record["policy_penalty"] == 0.0
    assert [message["role"] for message in record["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert record["messages"][-1]["content"] == "feedback-3"
    assert [value["phase"] for value in record.get("generation_diagnostics", [])] == []
    assert record["policy_version"] == "teacher-state-machine-v2"
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
    assert config["collection"]["reward_version"] == "userbench-travel-reward-v2"
    assert config["collection"]["minimum_terminal_reward"] == 0.7
    assert config["collection"]["require_zero_policy_penalty"] is True
    assert config["collection"]["policy_version"] == "teacher-state-machine-v2"
    assert config["collection"]["fail_fast_on_strict_violation"] is True
    assert config["collection"]["checkpoint_each_task"] is True
    assert config["collection"]["resume_safe"] is True


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
        self.constraints = []

    async def generate_action(self, messages, *, force_answer=False, constraint=None):
        self.requests.append(list(messages))
        self.constraints.append(constraint)
        choice, content = self.actions.pop(0)
        index = self.index
        self.index += 1
        return TeacherToolCall(
            f"call-{index}",
            UserBenchAction.from_parameters(
                {"thought": "next", "choice": choice, "content": content}
            ),
        )


def test_wrong_phase_choice_is_retried_without_stepping_environment():
    teacher = SequenceTeacher(
        [
            ("search", "Paris hotels"),
            ("action", "Do you prefer a specific hotel name?"),
            ("search", "Search for hotel options in Paris"),
            ("answer", "H1"),
        ]
    )
    trajectory = asyncio.run(
        collect_teacher_trajectory(
            task(), teacher=teacher, simulator=simulator(), wrapper_factory=FakeWrapper
        )
    )
    assert len(trajectory.step_rewards) == 3
    assert trajectory.generation_diagnostics[0]["reason"] == "wrong_phase_choice"
    assert teacher.constraints[0].choice.value == "action"


def test_state_machine_stops_eliciting_after_active_coverage():
    FakeWrapper.instances.clear()
    trajectory = asyncio.run(
        collect_teacher_trajectory(
            task(), teacher=FakeTeacher(), simulator=simulator(), wrapper_factory=FakeWrapper
        )
    )
    assert [value.choice.value for value in FakeWrapper.instances[-1].actions] == [
        "action",
        "search",
        "answer",
    ]
    assert trajectory.reward_breakdown["active_preference_coverage"] == 1.0


def test_bundled_preference_question_is_retried_before_environment_step():
    FakeWrapper.instances.clear()
    teacher = SequenceTeacher(
        [
            ("action", "Which hotel room and amenities do you prefer?"),
            ("action", "Which hotel amenities do you prefer?"),
            ("action", "Do you prefer a specific hotel name?"),
            ("search", "Search for hotel options in Paris"),
            ("answer", "H1"),
        ]
    )
    trajectory = asyncio.run(
        collect_teacher_trajectory(
            task(), teacher=teacher, simulator=simulator(), wrapper_factory=FakeWrapper
        )
    )
    assert FakeWrapper.instances[-1].steps == 3
    assert trajectory.generation_diagnostics[0]["reason"] == "bundled_action"
    assert trajectory.generation_diagnostics[1]["reason"] == "wrong_preference_field"
    assert teacher.requests[0][-1]["content"] != teacher.requests[1][-1]["content"]
    assert teacher.constraints[2].allowed_contents == (
        "Do you prefer a specific hotel name or property?",
    )


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


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("reward_valid", False, "reward_invalid"),
        ("terminal_reward", 0.69, "terminal_reward_below_threshold"),
        ("policy_penalty", 0.1, "policy_penalty"),
        ("exact_repeats", 1, "exact_repeats"),
        ("semantic_repeats", 1, "semantic_repeats"),
        ("ambiguous_actions", 1, "ambiguous_actions"),
        ("unsearched_answers", 1, "unsearched_answers"),
        ("wrong_answers", 1, "wrong_answers"),
    ],
)
def test_teacher_strict_gate_rejects_reward_policy_failures(field, value, reason):
    trajectory = asyncio.run(
        collect_teacher_trajectory(
            task(),
            teacher=FakeTeacher(),
            simulator=simulator(),
            wrapper_factory=FakeWrapper,
        )
    )
    reward = dict(trajectory.reward_breakdown)
    reward[field] = value
    invalid = trajectory.__class__(
        **{**trajectory.__dict__, "reward_breakdown": reward}
    )
    assert reason in trajectory_rejection_reasons(invalid)


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
            teacher=FakeTeacher(),
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
    assert diagnostic.rejection_reasons == (
        "teacher_action_exhausted.wrong_phase_choice",
    )
    assert partial["failure_environment_turn"] == 1
    assert partial["environment_steps_completed"] == 0
    assert partial["step_rewards"] == []
    assert diagnostic.generation_diagnostics[-1]["content"] == "Los Angeles hotels"
    serialized = json.dumps(diagnostic.to_record())
    assert "teacher-secret" not in serialized
    assert "simulator-secret" not in serialized


class ConstraintEchoTeacher(FakeTeacher):
    async def generate_action(self, messages, *, force_answer=False, constraint=None):
        assert constraint is not None and constraint.allowed_contents
        content = constraint.allowed_contents[0]
        call = TeacherToolCall(
            f"canonical-{self.index}",
            UserBenchAction.from_parameters(
                {
                    "thought": "Execute the current phase.",
                    "choice": constraint.choice.value,
                    "content": content,
                }
            ),
        )
        self.index += 1
        return call


class TwoAspectWrapper(FakeWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.elicited = set()
        self.searched = set()
        self.chosen = set()

    def reward_task(self):
        return TravelRewardTask(
            self.task_id,
            ("hotel", "rental_car"),
            {"hotel": "H1", "rental_car": "C1"},
            {
                "hotel": frozenset({"H1"}),
                "rental_car": frozenset({"C1"}),
            },
            {
                "hotel": frozenset({"P1", "P2"}),
                "rental_car": frozenset({"P3", "P4"}),
            },
        )

    def reward_snapshot(self):
        all_preferences = {"P1", "P2", "P3", "P4"}
        return UserBenchRewardSnapshot(
            remaining_preference_ids=frozenset(all_preferences - self.elicited),
            active_elicited_count=len(self.elicited),
            passive_elicited_count=0,
            remaining_search_aspects=frozenset(
                {"hotel", "rental_car"} - self.searched
            ),
            choice_initials=frozenset(self.chosen),
        )

    async def astep(self, action):
        self.actions.append(action)
        self.steps += 1
        content = action.content.casefold()
        if action.choice.value == "action":
            if "hotel" in content:
                candidate = next(value for value in ("P1", "P2") if value not in self.elicited)
            else:
                candidate = next(value for value in ("P3", "P4") if value not in self.elicited)
            self.elicited.add(candidate)
            feedback = "Preference recorded."
            reward = 0.2
        elif action.choice.value == "search":
            aspect = "rental_car" if "rental car" in content else "hotel"
            self.searched.add(aspect)
            feedback = "C1 is a visible rental car option." if aspect == "rental_car" else "H1 is a visible hotel option."
            reward = 0.2
        else:
            self.chosen.add(action.content[0])
            feedback = "Option recorded."
            reward = 1.0
        done = self.chosen == {"H", "C"}
        return UserBenchStepResult(
            self.task_id,
            UserBenchObservation(feedback, self.steps, done, reward, {}),
            reward,
            done,
            False,
            {},
        )


def test_composition_22_state_machine_completes_in_eight_steps():
    trajectory = asyncio.run(
        collect_teacher_trajectory(
            task("hotel:2-1|rental_car:2-2"),
            teacher=ConstraintEchoTeacher(),
            simulator=simulator(),
            wrapper_factory=TwoAspectWrapper,
            trajectory_attempt=3,
        )
    )
    assert len(trajectory.step_rewards) == 8
    choices = [
        json.loads(message["tool_calls"][0]["function"]["arguments"])["choice"]
        for message in trajectory.messages
        if message["role"] == "assistant"
    ]
    assert choices == [
        "action",
        "action",
        "search",
        "answer",
        "action",
        "action",
        "search",
        "answer",
    ]
    assert trajectory.reward_breakdown["terminal_reward"] == pytest.approx(1.0)
    assert trajectory.reward_breakdown["active_preference_coverage"] == 1.0


def test_every_canonical_preference_template_satisfies_its_own_contract():
    for aspect, fields in FIELD_QUERY_HINTS.items():
        for field in fields:
            plan = TeacherTurnPlan(TeacherPhase.ELICIT, aspect, field)
            action = UserBenchAction.from_parameters(
                {
                    "thought": "ask",
                    "choice": "action",
                    "content": plan.canonical_content,
                }
            )
            assert plan.validate(action) is None, (aspect, field, plan.canonical_content)


class JudgmentFallbackWrapper(FakeWrapper):
    async def astep(self, action):
        result = await super().astep(action)
        return UserBenchStepResult(
            result.task_id,
            result.observation,
            result.reward,
            result.terminated,
            result.truncated,
            {"userbench_judgment_fallbacks": 1},
        )


def test_simulator_fallback_aborts_after_the_consumed_step():
    teacher = FakeTeacher()
    outcome = asyncio.run(
        collect_teacher_task_with_retries(
            task(),
            teacher=teacher,
            simulator=simulator(),
            wrapper_factory=JudgmentFallbackWrapper,
            max_attempts=1,
        )
    )
    diagnostic = outcome.attempts[0]
    assert diagnostic.rejection_reasons == ("simulator.judgment_fallback",)
    assert diagnostic.partial_trajectory["environment_steps_completed"] == 1
    assert teacher.index == 1


def test_per_task_checkpoint_round_trip_and_manifest_resume(tmp_path):
    outcome = asyncio.run(
        collect_teacher_task_with_retries(
            task(),
            teacher=FakeTeacher(),
            simulator=simulator(),
            wrapper_factory=FakeWrapper,
            max_attempts=1,
        )
    )
    run_dir = tmp_path / "run"
    initialize_teacher_run(run_dir, [outcome.task_id])
    checkpoint = write_teacher_outcome_checkpoint(outcome, run_dir)
    assert checkpoint.exists()
    restored = load_teacher_outcome_checkpoints(run_dir, [outcome.task_id])
    assert restored[outcome.task_id].trajectory.to_record() == outcome.trajectory.to_record()
    initialize_teacher_run(run_dir, [outcome.task_id], resume=True)
    with pytest.raises(TeacherCollectionError, match="ordered task IDs"):
        initialize_teacher_run(run_dir, ["different-task"], resume=True)
    with pytest.raises(FileExistsError, match="--resume"):
        initialize_teacher_run(run_dir, [outcome.task_id])
