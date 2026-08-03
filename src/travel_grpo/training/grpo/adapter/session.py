"""Per-rollout UserBench session lifecycle and veRL interaction adapter."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from travel_grpo.envs.userbench_context import (
    UserBenchSessionError,
    UserBenchSessionState,
    clear_current_session,
    get_current_session,
    require_current_session,
    set_current_session,
)
from travel_grpo.envs.userbench_interaction import SimulatorRole, UserSimulatorRuntime
from travel_grpo.envs.userbench_wrapper import (
    UserBenchEnvironmentConfig,
    UserBenchWrapper,
)
from travel_grpo.training.grpo.compat import require_verl_061

try:  # The integration package remains importable without the optional runtime.
    from verl.interactions.base import BaseInteraction
except ImportError:  # pragma: no cover - exercised by normal lightweight installs.
    BaseInteraction = object  # type: ignore[assignment,misc]


INTERACTION_NAME = "userbench"
ENVIRONMENT_NAME = "TravelGym"
PROJECT_ROOT = Path(__file__).resolve().parents[5]


def _non_empty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def build_rollout_extra_info(task_id: str) -> dict[str, Any]:
    """Build the duplicated IDs expected by veRL interaction and tool creation."""

    normalized = _non_empty(task_id, "task_id")
    return {
        "interaction_kwargs": {
            "name": INTERACTION_NAME,
            "task_id": normalized,
        },
        "tools_kwargs": {
            "interact_with_env": {
                "create_kwargs": {
                    "env_name": ENVIRONMENT_NAME,
                    "id": normalized,
                }
            }
        },
    }


def validate_rollout_extra_info(extra_info: Mapping[str, Any]) -> str:
    """Return the task ID only when interaction and tool payloads agree."""

    if not isinstance(extra_info, Mapping):
        raise TypeError("rollout extra_info must be a mapping")
    interaction = extra_info.get("interaction_kwargs")
    tools = extra_info.get("tools_kwargs")
    if not isinstance(interaction, Mapping) or not isinstance(tools, Mapping):
        raise TypeError(
            "rollout extra_info is missing interaction_kwargs or tools_kwargs"
        )
    interaction_id = _non_empty(interaction.get("task_id"), "interaction task_id")
    tool_entry = tools.get("interact_with_env")
    create_kwargs = (
        tool_entry.get("create_kwargs") if isinstance(tool_entry, Mapping) else None
    )
    if not isinstance(create_kwargs, Mapping):
        raise TypeError("rollout extra_info is missing interact_with_env create_kwargs")
    tool_id = _non_empty(create_kwargs.get("id"), "tool task_id")
    if interaction_id != tool_id:
        raise ValueError(
            f"interaction task ID {interaction_id!r} does not match tool task ID {tool_id!r}"
        )
    if create_kwargs.get("env_name") != ENVIRONMENT_NAME:
        raise ValueError(f"tool env_name must be {ENVIRONMENT_NAME!r}")
    return interaction_id


def calculate_current_session_score() -> float:
    """Return the exact sum of upstream rewards for the active trajectory."""

    return require_current_session().rewards.total


def _load_yaml_mapping(path: str | Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - veRL environments include PyYAML.
        raise RuntimeError("PyYAML is required by the veRL UserBench adapter") from exc
    config_path = _resolve_project_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, Mapping):
        raise TypeError(f"configuration must be a mapping: {config_path}")
    return loaded


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    working_directory_candidate = candidate.resolve()
    if working_directory_candidate.exists():
        return working_directory_candidate
    return (PROJECT_ROOT / candidate).resolve()


WrapperFactory = Callable[..., UserBenchWrapper]


class UserBenchInteraction(BaseInteraction):  # type: ignore[misc]
    """veRL interaction whose score is the unmodified sum of TravelGym rewards."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        require_verl_061()
        super().__init__(config)
        environment_path = config.get("environment_config_path")
        simulator_path = config.get("simulator_config_path")
        if not environment_path or not simulator_path:
            raise ValueError(
                "UserBench interaction requires environment_config_path and simulator_config_path"
            )
        environment_document = _load_yaml_mapping(environment_path)
        simulator_document = _load_yaml_mapping(simulator_path)
        environment = environment_document.get("environment", environment_document)
        simulator = simulator_document.get("simulator", simulator_document)
        if not isinstance(environment, Mapping) or not isinstance(simulator, Mapping):
            raise TypeError("invalid UserBench environment or simulator configuration")

        allowed_config = set(UserBenchEnvironmentConfig.__dataclass_fields__)
        environment_values = {
            key: value for key, value in environment.items() if key in allowed_config
        }
        self.environment_config = UserBenchEnvironmentConfig(**environment_values)
        source_root = environment.get("source_root")
        self.source_root = (
            _resolve_project_path(source_root) if source_root is not None else None
        )
        role = SimulatorRole(_non_empty(simulator.get("role"), "simulator.role"))
        prefix = "TRAIN_USER_SIM" if role is SimulatorRole.TRAIN else "EVAL_USER_SIM"
        expected_simulator_fields = {
            "model_env": f"{prefix}_MODEL",
            "base_url_env": f"{prefix}_BASE_URL",
            "api_key_env": f"{prefix}_API_KEY",
        }
        for key, expected in expected_simulator_fields.items():
            if simulator.get(key) != expected:
                raise ValueError(f"simulator.{key} must be {expected!r}")
        self.runtime = UserSimulatorRuntime.from_environment(role)
        self._wrapper_factory: WrapperFactory = UserBenchWrapper

    async def start_interaction(
        self, instance_id: str | None = None, **kwargs: Any
    ) -> str:
        if get_current_session() is not None:
            raise UserBenchSessionError(
                "a UserBench session is already active in this context"
            )
        task_id = _non_empty(kwargs.get("task_id"), "task_id")
        request_id = instance_id or uuid.uuid4().hex
        wrapper = self._wrapper_factory(
            task_id,
            self.runtime,
            self.environment_config,
            source_root=self.source_root,
        )
        try:
            wrapper.reset()
        except Exception:
            wrapper.close()
            raise
        set_current_session(
            UserBenchSessionState(
                request_id=request_id, task_id=task_id, wrapper=wrapper
            )
        )
        return request_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> tuple[bool, str, float, dict[str, Any]]:
        session = self._require_request(instance_id)
        if session.done:
            return True, "", 0.0, session.metrics()
        session.protocol_error = "actor produced no interact_with_env tool call"
        return True, "", 0.0, session.metrics()

    async def calculate_score(
        self, instance_id: str | None = None, **kwargs: Any
    ) -> float:
        self._require_request(instance_id)
        return calculate_current_session_score()

    async def finalize_interaction(
        self, instance_id: str | None = None, **kwargs: Any
    ) -> None:
        self._require_request(instance_id)
        clear_current_session(close=True)

    @staticmethod
    def _require_request(instance_id: str | None) -> UserBenchSessionState:
        session = require_current_session()
        if instance_id is not None and session.request_id != instance_id:
            raise UserBenchSessionError(
                f"interaction ID {instance_id!r} does not match active request {session.request_id!r}"
            )
        return session
