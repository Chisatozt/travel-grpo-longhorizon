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
from travel_grpo.envs.userbench_tools import ActionChoice, UserBenchAction
from travel_grpo.models.openai_compatible import TeacherRuntime, TeacherToolCall
from travel_grpo.training.teacher_policy import TeacherPhase, TeacherTurnPlan
from travel_grpo.envs.userbench_tools import FIELD_QUERY_HINTS
from travel_grpo.training.sft_collection import (
    TeacherCollectionError,
    assert_disjoint_from_evaluation,
    build_stratified_task_plan,
    collect_teacher_trajectory,
    collect_teacher_task_with_retries,
    initialize_teacher_run,
    load_teacher_outcome_checkpoints,
    load_teacher_task_pool,
    select_stratified_task_wave,
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


def test_adaptive_collection_reaches_per_composition_quota_in_waves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_path = ROOT / "scripts/train/sft/collect_sft_data.py"
    spec = importlib.util.spec_from_file_location("collect_stratified_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "evaluation.jsonl"
    train_tasks = [
        {**task(f"hotel:2-{index}"), "composition": composition}
        for composition, indexes in (("22", (1, 2, 3)), ("33", (4, 5, 6)))
        for index in indexes
    ]
    write_pool(train, train_tasks)
    write_pool(evaluation, [{**task("hotel:2-99"), "source_split": "test"}])
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
    args = module.build_parser().parse_args(
        [
            "--input",
            str(train),
            "--evaluation-tasks",
            str(evaluation),
            "--output",
            str(output),
            "--run-dir",
            str(run_dir),
            "--target-accepted",
            "4",
            "--stratified-wave-size",
            "2",
            "--sampling-seed",
            "test-seed",
            "--attempts",
            "1",
        ]
    )
    summary = asyncio.run(module.run(args))
    assert summary["accepted"] == 4
    assert summary["gold"] == 4
    assert summary["stratified_complete"] is True
    assert summary["waves"] == 2
    manifest = json.loads(
        (run_dir / "selection_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "complete"
    assert manifest["quotas"] == {"22": 2, "33": 2}
    assert manifest["accepted_by_stratum"] == {"22": 2, "33": 2}
    assert len(output.read_text(encoding="utf-8").splitlines()) == 4


ROOT = Path(__file__).resolve().parents[1]


def test_stratified_plan_uses_largest_remainder_and_stable_order() -> None:
    tasks = [
        {"task_id": f"{composition}:{index}", "composition": composition}
        for composition, count in (("22", 4), ("33", 3), ("44", 3))
        for index in range(count)
    ]
    first, quotas = build_stratified_task_plan(
        tasks, target=6, field="composition", seed="test-seed"
    )
    second, second_quotas = build_stratified_task_plan(
        list(reversed(tasks)), target=6, field="composition", seed="test-seed"
    )
    assert quotas == {"22": 2, "33": 2, "44": 2}
    assert second_quotas == quotas
    assert [task["task_id"] for task in first] == [
        task["task_id"] for task in second
    ]
    assert [task["composition"] for task in first] == [
        "22",
        "22",
        "22",
        "22",
        "33",
        "33",
        "33",
        "44",
        "44",
        "44",
    ]


def test_stratified_wave_refills_rejected_stratum_without_repeating_tasks() -> None:
    tasks = [
        {"task_id": f"{composition}:{index}", "composition": composition}
        for composition in ("22", "33")
        for index in range(5)
    ]
    first = select_stratified_task_wave(
        tasks,
        quotas={"22": 3, "33": 3},
        attempted_task_ids=set(),
        accepted_task_ids=set(),
        wave_size=4,
    )
    assert len(first) == 4
    assert {task["composition"] for task in first} == {"22", "33"}
    attempted = {str(task["task_id"]) for task in first}
    accepted = {str(first[0]["task_id"])}
    second = select_stratified_task_wave(
        tasks,
        quotas={"22": 3, "33": 3},
        attempted_task_ids=attempted,
        accepted_task_ids=accepted,
        wave_size=3,
    )
    assert len(second) == 3
    assert not attempted & {str(task["task_id"]) for task in second}
    assert sum(task["composition"] == "22" for task in second) == 1
    assert sum(task["composition"] == "33" for task in second) == 2


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


def test_collection_cli_uses_fixed_development_batch() -> None:
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
            "--batch",
            "development",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["batch"] == "development"
    assert summary["task_ids"] == [
        "apartment:2-38|rental_car:2-7",
        "apartment:2-10|rental_car:2-23",
        "hotel:2-69|rental_car:2-21",
    ]


def test_teacher_smoke_batches_are_unique_train_tasks_and_pairwise_disjoint() -> None:
    config = json.loads(
        (ROOT / "configs/train/sft/teacher_smoke_batches.json").read_text(
            encoding="utf-8"
        )
    )
    train_ids = {
        json.loads(line)["task_id"]
        for line in (ROOT / "data/sft/tasks_train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }
    batches = {
        name: tuple(value["task_ids"])
        for name, value in config["batches"].items()
    }
    seen: set[str] = set()
    for task_ids in batches.values():
        assert len(task_ids) == 3
        assert len(set(task_ids)) == 3
        assert set(task_ids) <= train_ids
        assert not (seen & set(task_ids))
        seen.update(task_ids)


def test_upstream_preference_fields_narrow_local_phase_without_exposing_values() -> None:
    task = {
        "id": "hotel:2-1|rental_car:2-2",
        "dimensions": ["hotel", "rental_car"],
        "preferences": {
            "hotel": {
                "best_id": "H1",
                "correct_ids": ["H1"],
                "preferences": [
                    ["hotel", "amenities", "hidden value", "hidden response"],
                    ["hotel", "service", "another hidden value", "hidden response"],
                ],
            },
            "rental_car": {
                "best_id": "C1",
                "correct_ids": ["C1"],
                "preferences": [["rental_car", "brand", "secret", "secret"]],
            },
        },
    }
    reward_task = TravelRewardTask.from_upstream(task)
    assert reward_task.preference_fields_by_aspect == {
        "hotel": ("amenities", "service"),
        "rental_car": ("brand",),
    }
    serialized = json.dumps(reward_task.preference_fields_by_aspect)
    assert "hidden" not in serialized
    assert "secret" not in serialized


def test_generic_amenity_question_is_rejected_before_environment_consumes_a_step() -> None:
    plan = TeacherTurnPlan(TeacherPhase.ELICIT, "apartment", "amenities")
    generic = UserBenchAction.from_parameters(
        {
            "thought": "ask",
            "choice": "action",
            "content": "What amenities are important to you in the apartment, such as a gym, pool, parking, or laundry facilities?",
        }
    )
    assert plan.validate(generic) == "vague_action"

    concrete = UserBenchAction.from_parameters(
        {
            "thought": "ask",
            "choice": "action",
            "content": plan.canonical_content,
        }
    )
    assert plan.validate(concrete) is None


def test_answer_instruction_can_include_only_public_visible_candidate_facts():
    plan = TeacherTurnPlan(
        TeacherPhase.ANSWER,
        "apartment",
        available_option_ids=("A1",),
        visible_option_details=(
            '{"amenities":["Elevator"],"cost":300,"id":"A1"}',
        ),
    )
    instruction = plan.instruction(1)
    assert '"id":"A1"' in instruction
    assert "ground_truth" not in instruction
    assert "best_id" not in instruction


def test_answer_validation_rejects_visible_option_missing_public_search_requirement():
    plan = TeacherTurnPlan(
        TeacherPhase.ANSWER,
        "apartment",
        available_option_ids=("A12", "A18"),
        visible_option_details=(
            '{"amenities":["Air Conditioning"],"id":"A12"}',
            '{"amenities":["Kitchen"],"id":"A18"}',
        ),
        public_requirements=(("air conditioning", ("air conditioning",)),),
    )
    action = UserBenchAction.from_parameters(
        {"thought": "choose", "choice": "answer", "content": "A18"}
    )
    assert plan.validate(action) == "answer_not_matching_public_requirement.air_conditioning"


@pytest.mark.parametrize(
    ("aspect", "field", "required_phrase"),
    [
        ("flight", "amenities", "lounge"),
        ("apartment", "amenities", "garden"),
        ("restaurant", "tags", "late-night"),
        ("hotel", "amenities", "mountain view"),
        ("rental_car", "insurance", "belongings"),
    ],
)
def test_primary_and_repair_questions_cover_global_preference_taxonomy_regressions(
    aspect, field, required_phrase
):
    plan = TeacherTurnPlan(TeacherPhase.ELICIT, aspect, field)
    assert required_phrase in plan.canonical_content.casefold()
    assert required_phrase in plan.elicitation_repair_content.casefold()


def test_search_validation_rejects_preferences_and_years_absent_from_public_context():
    plan = TeacherTurnPlan(
        TeacherPhase.SEARCH,
        "apartment",
        public_search_context=(
            "I will stay in Barcelona from April 15th to April 22nd.",
            "Reliable internet is important, so I need Wi-Fi.",
        ),
    )
    valid = UserBenchAction.from_parameters(
        {
            "thought": "search",
            "choice": "search",
            "content": "Search for an apartment in Barcelona from April 15 to April 22 with Wi-Fi.",
        }
    )
    invented_kitchen = UserBenchAction.from_parameters(
        {
            "thought": "search",
            "choice": "search",
            "content": "Search for an apartment in Barcelona from April 15 to April 22 with Wi-Fi and a kitchen.",
        }
    )
    invented_year = UserBenchAction.from_parameters(
        {
            "thought": "search",
            "choice": "search",
            "content": "Search for an apartment in Barcelona from April 15 to April 22, 2027, with Wi-Fi.",
        }
    )
    assert plan.validate(valid) is None
    assert plan.validate(invented_kitchen) == "search_invents_preference.kitchen"
    assert plan.validate(invented_year) == "search_invents_year"
    assert "Reliable internet" in plan.instruction(1)


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
    assert record["policy_version"] == "teacher-state-machine-v4"
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
    assert config["collection"]["policy_version"] == "teacher-state-machine-v4"
    assert config["collection"]["fail_fast_on_strict_violation"] is True
    assert config["collection"]["checkpoint_each_task"] is True
    assert config["collection"]["resume_safe"] is True
    assert config["collection"]["silver"]["max_elicitation_repairs_per_field"] == 1


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


def test_search_with_invented_preference_is_retried_before_environment_step():
    FakeWrapper.instances.clear()
    teacher = SequenceTeacher(
        [
            ("action", "Do you prefer a specific hotel name?"),
            ("search", "Search for hotel options in Paris with parking."),
            ("search", "Search for hotel options in Paris."),
            ("answer", "H1"),
        ]
    )
    trajectory = asyncio.run(
        collect_teacher_trajectory(
            task(), teacher=teacher, simulator=simulator(), wrapper_factory=FakeWrapper
        )
    )
    assert trajectory.terminated
    assert FakeWrapper.instances[-1].steps == 3
    assert trajectory.generation_diagnostics[0]["reason"] == (
        "search_invents_preference.parking_service"
    )


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
        "Do you prefer a specific hotel name, brand, platform, or property?",
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


class SearchRepairWrapper(FakeWrapper):
    failed_search = False

    def reward_snapshot(self):
        snapshot = super().reward_snapshot()
        search_count = sum(action.choice.value == "search" for action in self.actions)
        if self.failed_search and search_count == 1:
            return UserBenchRewardSnapshot(
                snapshot.remaining_preference_ids,
                snapshot.active_elicited_count,
                snapshot.passive_elicited_count,
                frozenset({"hotel"}),
                snapshot.choice_initials,
            )
        return snapshot

    async def astep(self, action):
        if action.choice.value == "search" and not self.failed_search:
            self.failed_search = True
            self.actions.append(action)
            self.steps += 1
            return UserBenchStepResult(
                self.task_id,
                UserBenchObservation("Search was not recorded.", self.steps, False, 0.0, {}),
                0.0,
                False,
                False,
                {},
            )
        return await super().astep(action)


def test_search_not_recorded_is_retried_once_and_failed_turn_is_loss_masked():
    teacher = SequenceTeacher(
        [
            ("action", "Do you prefer a specific hotel name?"),
            ("search", "Search for hotel options in Paris, first attempt"),
            ("search", "Search for hotel options in Paris, corrected attempt"),
            ("answer", "H1"),
        ]
    )
    trajectory = asyncio.run(
        collect_teacher_trajectory(
            task(),
            teacher=teacher,
            simulator=simulator(),
            wrapper_factory=SearchRepairWrapper,
        )
    )
    assert trajectory.terminated
    assert trajectory.step_rewards == (0.2, 0.0, 0.2, 1.0)
    assistant_messages = [
        message
        for message in trajectory.messages
        if message.get("role") == "assistant"
    ]
    assert [message.get("loss_mask", False) for message in assistant_messages] == [
        False,
        True,
        False,
        False,
    ]
    assert trajectory.generation_diagnostics[-1]["reason"] == "search_repair_allowed"


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
    assert paths[2].read_text(encoding="utf-8") == ""
    assert len(paths[3].read_text(encoding="utf-8").splitlines()) == 2


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
        assert constraint is not None
        if constraint.allowed_contents:
            content = constraint.allowed_contents[0]
        elif constraint.choice is ActionChoice.SEARCH:
            instruction = str(messages[-1]["content"])
            aspect = "hotel" if "search for hotel" in instruction else "rental car"
            content = f"Search for {aspect} options in Los Angeles for the stated dates."
        else:
            raise AssertionError("only SEARCH may omit a deterministic content enum")
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


def test_every_elicitation_repair_template_is_distinct_and_satisfies_contract():
    for aspect, fields in FIELD_QUERY_HINTS.items():
        for field in fields:
            plan = TeacherTurnPlan(TeacherPhase.ELICIT, aspect, field)
            action = UserBenchAction.from_parameters(
                {
                    "thought": "repair",
                    "choice": "action",
                    "content": plan.elicitation_repair_content,
                }
            )
            assert plan.elicitation_repair_content != plan.canonical_content
            assert plan.validate(action) is None, (
                aspect,
                field,
                plan.elicitation_repair_content,
            )


class ElicitationRepairWrapper(FakeWrapper):
    """Ignore the first valid preference question and credit the repair."""

    def reward_snapshot(self):
        choices = [value.choice.value for value in self.actions]
        action_count = choices.count("action")
        return UserBenchRewardSnapshot(
            remaining_preference_ids=(
                frozenset() if action_count >= 2 else frozenset({"P1"})
            ),
            active_elicited_count=1 if action_count >= 2 else 0,
            passive_elicited_count=0,
            remaining_search_aspects=(
                frozenset() if "search" in choices else frozenset({"hotel"})
            ),
            choice_initials=frozenset({"H"}) if "answer" in choices else frozenset(),
        )


class UnrecordedJudgmentThenRecoveryWrapper(ElicitationRepairWrapper):
    """Return one judgment fallback without crediting its elicitation turn."""

    async def astep(self, action):
        result = await super().astep(action)
        action_count = sum(value.choice.value == "action" for value in self.actions)
        if action.choice.value == "action" and action_count == 1:
            return UserBenchStepResult(
                result.task_id,
                result.observation,
                result.reward,
                result.terminated,
                result.truncated,
                {"userbench_judgment_fallbacks": 1},
            )
        return result


class UnrecordedVagueThenRecoveryWrapper(ElicitationRepairWrapper):
    """Reject one elicitation as vague without credit, then accept its rephrase."""

    async def astep(self, action):
        result = await super().astep(action)
        action_count = sum(value.choice.value == "action" for value in self.actions)
        if action.choice.value == "action" and action_count == 1:
            return UserBenchStepResult(
                result.task_id,
                UserBenchObservation(
                    "Your question is too vague and general.",
                    result.observation.step_count,
                    result.observation.episode_complete,
                    result.observation.last_reward,
                    result.observation.diagnostics,
                ),
                result.reward,
                result.terminated,
                result.truncated,
                result.diagnostics,
            )
        return result


@pytest.mark.parametrize(
    ("wrapper_factory", "repair_marker"),
    [
        (UnrecordedJudgmentThenRecoveryWrapper, "judgment_fallback_allowed"),
        (UnrecordedVagueThenRecoveryWrapper, "vague_action_repair_allowed"),
    ],
)
def test_uncommitted_fallback_rephrases_without_duplicate_exhaustion(
    wrapper_factory, repair_marker
):
    teacher = SequenceTeacher(
        [
            ("action", "Do you prefer a specific hotel name?"),
            (
                "action",
                "For the hotel, is there a particular hotel name or property name you want?",
            ),
            ("search", "Search for hotel options in Paris"),
            ("answer", "H1"),
        ]
    )
    trajectory = asyncio.run(
        collect_teacher_trajectory(
            task(),
            teacher=teacher,
            simulator=simulator(),
            wrapper_factory=wrapper_factory,
        )
    )
    reasons = [value["reason"] for value in trajectory.generation_diagnostics]
    assert trajectory.terminated
    assert repair_marker in reasons
    assert "duplicate_action" not in reasons
    assert "semantic_duplicate_action" not in reasons
    assert teacher.constraints[1].allowed_contents == (
        "For the hotel, is there a particular hotel name or property name you want?",
    )


class NeverRecordsElicitationWrapper(ElicitationRepairWrapper):
    """Never credit preference questions, including the one allowed repair."""

    def reward_snapshot(self):
        snapshot = super().reward_snapshot()
        return UserBenchRewardSnapshot(
            remaining_preference_ids=frozenset({"P1"}),
            active_elicited_count=0,
            passive_elicited_count=snapshot.passive_elicited_count,
            remaining_search_aspects=snapshot.remaining_search_aspects,
            choice_initials=snapshot.choice_initials,
        )


def test_elicitation_not_recorded_retries_same_field_once_and_masks_failed_turn():
    teacher = SequenceTeacher(
        [
            ("action", "Do you prefer a specific hotel name?"),
            (
                "action",
                "For the hotel, is there a particular hotel name or property name you want?",
            ),
            ("search", "Search for hotel options in Paris"),
            ("answer", "H1"),
        ]
    )
    trajectory = asyncio.run(
        collect_teacher_trajectory(
            task(),
            teacher=teacher,
            simulator=simulator(),
            wrapper_factory=ElicitationRepairWrapper,
        )
    )
    assert trajectory.terminated
    assert trajectory.step_rewards == (0.2, 0.2, 0.2, 1.0)
    assistant_messages = [
        message for message in trajectory.messages if message.get("role") == "assistant"
    ]
    assert [message.get("loss_mask", False) for message in assistant_messages] == [
        True,
        False,
        False,
        False,
    ]
    assert trajectory.generation_diagnostics[-1]["reason"] == (
        "elicitation_repair_allowed"
    )
    assert trajectory.reward_breakdown["exact_repeats"] == 0
    assert trajectory.reward_breakdown["semantic_repeats"] == 0
    assert teacher.constraints[1].allowed_contents == (
        "For the hotel, is there a particular hotel name or property name you want?",
    )


def test_second_unrecorded_elicitation_aborts_without_consuming_another_field():
    outcome = asyncio.run(
        collect_teacher_task_with_retries(
            task(),
            teacher=SequenceTeacher(
                [
                    ("action", "Do you prefer a specific hotel name?"),
                    (
                        "action",
                        "For the hotel, is there a particular hotel name or property name you want?",
                    ),
                ]
            ),
            simulator=simulator(),
            wrapper_factory=NeverRecordsElicitationWrapper,
            max_attempts=1,
        )
    )
    diagnostic = outcome.attempts[0]
    assert diagnostic.rejection_reasons == ("environment.elicitation_not_recorded",)
    assert diagnostic.partial_trajectory["environment_steps_completed"] == 2
    assert [
        action["field"] for action in diagnostic.partial_trajectory["committed_actions"]
    ] == ["name", "name"]


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


class OneJudgmentFallbackWrapper(FakeWrapper):
    """Inject exactly one simulator judgment fallback, then recover normally."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.injected = False

    async def astep(self, action):
        result = await super().astep(action)
        if self.injected:
            return result
        self.injected = True
        return UserBenchStepResult(
            result.task_id,
            result.observation,
            result.reward,
            result.terminated,
            result.truncated,
            {"userbench_judgment_fallbacks": 1},
        )


def test_one_judgment_fallback_is_admitted_as_silver_and_loss_masked(tmp_path):
    outcome = asyncio.run(
        collect_teacher_task_with_retries(
            task(),
            teacher=FakeTeacher(),
            simulator=simulator(),
            wrapper_factory=OneJudgmentFallbackWrapper,
            max_attempts=1,
        )
    )
    assert outcome.accepted
    assert outcome.quality_tier == "silver"
    assert outcome.trajectory is not None
    assert outcome.trajectory.quality_tier == "silver"
    assert outcome.trajectory.reward_breakdown["reward_valid"] is False
    assert outcome.trajectory.reward_breakdown["raw_terminal_reward"] >= 0.7
    assert any(
        message.get("loss_mask") is True
        for message in outcome.trajectory.messages
        if message.get("role") == "assistant"
    )
    paths = write_teacher_collection_artifacts(
        [outcome],
        accepted_path=tmp_path / "batch.accepted.jsonl",
        silver_path=tmp_path / "batch.silver.jsonl",
        rejected_path=tmp_path / "batch.rejected.jsonl",
        diagnostics_path=tmp_path / "batch.diagnostics.jsonl",
    )
    assert paths[0].read_text(encoding="utf-8") == ""
    assert json.loads(paths[1].read_text(encoding="utf-8"))["quality_tier"] == "silver"
    assert paths[2].read_text(encoding="utf-8") == ""


def test_second_simulator_fallback_aborts_after_the_consumed_step():
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
    assert diagnostic.partial_trajectory["environment_steps_completed"] == 2
    # One fallback is admitted to the silver path; the second fallback is
    # still fail-loud and aborts after its environment step is consumed.
    assert teacher.index == 2


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
