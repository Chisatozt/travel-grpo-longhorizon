"""Bounded group selection and the project-owned veRL rollout adapter."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any


def extract_userbench_group_signals(
    infos: Sequence[object],
) -> tuple[list[float], list[bool], list[tuple[str, ...]]]:
    """Extract terminal rewards and explicit invalid flags from AgentLoop metadata."""

    rewards: list[float] = []
    invalid: list[bool] = []
    reasons: list[tuple[str, ...]] = []
    for index, info in enumerate(infos):
        if not isinstance(info, Mapping):
            raise ValueError(f"UserBench extra field {index} must be a mapping")
        reward = info.get("reward")
        if not isinstance(reward, Mapping):
            reward = info
        try:
            value = float(reward["terminal_reward"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"UserBench extra field {index} is missing terminal_reward"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(f"UserBench terminal reward {index} is not finite")
        is_invalid = reward.get("reward_valid") is not True or bool(
            reward.get("infrastructure_invalid", info.get("infrastructure_invalid"))
        )
        raw_reasons = reward.get("infrastructure_errors", ())
        if isinstance(raw_reasons, str):
            raw_reasons = (raw_reasons,)
        if not isinstance(raw_reasons, Sequence):
            raw_reasons = ()
        normalized = tuple(sorted({str(value) for value in raw_reasons if value}))
        if is_invalid and not normalized:
            normalized = ("reward_invalid",)
        rewards.append(value)
        invalid.append(is_invalid)
        reasons.append(normalized)
    return rewards, invalid, reasons


def select_reward_varying_groups(
    uids: Sequence[Hashable],
    rewards: Sequence[float],
    *,
    sampling_invalid: Sequence[bool] | None = None,
    sampling_invalid_reasons: Sequence[Sequence[str]] | None = None,
    expected_group_size: int = 4,
    tolerance: float = 1.0e-6,
) -> tuple[list[int], dict[str, Any]]:
    """Return retained valid candidates and mark complete varying groups.

    A row from an incomplete or sampling-invalid group is still a usable
    candidate when its own reward evidence is valid.  Complete constant groups
    remain dropped because they cannot provide a GRPO advantage signal.
    """

    if len(uids) != len(rewards):
        raise ValueError("uids and rewards must have equal length")
    if expected_group_size <= 1:
        raise ValueError("expected_group_size must be greater than one")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    invalid = list(sampling_invalid or [False] * len(uids))
    reasons = list(sampling_invalid_reasons or [()] * len(uids))
    if len(invalid) != len(uids) or len(reasons) != len(uids):
        raise ValueError("sampling diagnostics must align with uids")

    grouped: dict[Hashable, dict[str, Any]] = {}
    for index, (uid, raw_reward, is_invalid, raw_reasons) in enumerate(
        zip(uids, rewards, invalid, reasons, strict=True)
    ):
        try:
            hash(uid)
        except TypeError as exc:
            raise ValueError(f"uid at index {index} is not hashable") from exc
        reward = float(raw_reward)
        if not math.isfinite(reward):
            raise ValueError(f"reward at index {index} is not finite")
        group = grouped.setdefault(
            uid,
            {"indices": [], "rewards": [], "invalid": [], "reasons": []},
        )
        group["indices"].append(index)
        group["rewards"].append(reward)
        group["invalid"].append(bool(is_invalid))
        group["reasons"].extend(str(value) for value in raw_reasons if value)

    kept: set[Hashable] = set()
    retained_indices: set[int] = set()
    groups: list[dict[str, Any]] = []
    reason_counter: Counter[str] = Counter()
    for uid, group in grouped.items():
        size = len(group["indices"])
        minimum = min(group["rewards"])
        maximum = max(group["rewards"])
        if size != expected_group_size:
            drop_reason = "incomplete_group"
        elif any(group["invalid"]):
            drop_reason = "sampling_invalid"
        elif maximum - minimum <= tolerance:
            drop_reason = "constant_reward"
        else:
            drop_reason = None
            kept.add(uid)
        if drop_reason is None:
            retained_indices.update(group["indices"])
        elif drop_reason in {"incomplete_group", "sampling_invalid"}:
            retained_indices.update(
                index
                for index, is_invalid in zip(
                    group["indices"], group["invalid"], strict=True
                )
                if not is_invalid
            )
        reason_counter.update(set(group["reasons"]))
        groups.append(
            {
                "uid": uid,
                "indices": tuple(group["indices"]),
                "rewards": tuple(group["rewards"]),
                "reward_min": minimum,
                "reward_max": maximum,
                "sampling_invalid": any(group["invalid"]),
                "sampling_invalid_reasons": tuple(sorted(set(group["reasons"]))),
                "drop_reason": drop_reason,
                "kept": drop_reason is None,
            }
        )
    indices = [index for index in range(len(uids)) if index in retained_indices]
    stats = {
        "num_trajectories": len(uids),
        "num_groups": len(grouped),
        "kept_group_count": len(kept),
        "dropped_group_count": len(grouped) - len(kept),
        "constant_reward_group_count": sum(
            group["drop_reason"] == "constant_reward" for group in groups
        ),
        "sampling_invalid_group_count": sum(
            group["drop_reason"] == "sampling_invalid" for group in groups
        ),
        "incomplete_group_count": sum(
            group["drop_reason"] == "incomplete_group" for group in groups
        ),
        "sampling_invalid_reason_counts": dict(sorted(reason_counter.items())),
        "groups": tuple(groups),
    }
    return indices, stats


@dataclass
class BoundedSamplingState:
    """Track the three-batch and ten-consecutive-skip production bounds."""

    required_groups: int = 2
    max_generation_batches: int = 3
    max_consecutive_skips: int = 10
    generation_batches: int = 0
    accepted_groups: int = 0
    consecutive_skips: int = 0
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def record_batch(self, stats: Mapping[str, Any]) -> bool:
        if self.generation_batches >= self.max_generation_batches:
            raise RuntimeError("bounded sampler exceeded its generation-batch limit")
        self.generation_batches += 1
        kept = int(stats.get("kept_group_count", 0))
        self.accepted_groups += kept
        self.diagnostics.append(dict(stats))
        return self.accepted_groups >= self.required_groups

    @property
    def may_generate(self) -> bool:
        return (
            self.accepted_groups < self.required_groups
            and self.generation_batches < self.max_generation_batches
        )

    def finish_update(self) -> bool:
        """Return whether to train; fail after too many consecutive skipped updates."""

        train = self.accepted_groups >= self.required_groups
        if train:
            self.consecutive_skips = 0
        else:
            self.consecutive_skips += 1
            if self.consecutive_skips > self.max_consecutive_skips:
                raise RuntimeError(
                    "bounded sampler exceeded the consecutive skipped-update limit"
                )
        self.generation_batches = 0
        self.accepted_groups = 0
        self.diagnostics.clear()
        return train


class DynamicSamplingExhausted(RuntimeError):
    """Raised once the consecutive skipped-update budget is exhausted."""


@dataclass(frozen=True)
class _RolloutCandidate:
    """One complete DataProto row retained for a single task UID."""

    uid: Hashable
    row: Any = field(repr=False, compare=False)
    reward: float
    degraded: bool
    generation_batch: int
    row_index: int


@dataclass(frozen=True)
class _CandidateSelection:
    candidates: tuple[_RolloutCandidate, ...]
    reward_min: float
    reward_max: float
    unique_reward_count: int

    @property
    def reward_range(self) -> float:
        return self.reward_max - self.reward_min

    @property
    def uses_degraded(self) -> bool:
        return any(candidate.degraded for candidate in self.candidates)


def _python_value(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _ordered_unique(values: Sequence[Any]) -> list[Any]:
    """Preserve prompt order when groups were accepted in different batches."""

    result: list[Any] = []
    seen: set[Any] = set()
    for raw in values:
        value = _python_value(raw)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _select_candidate_group(
    candidates: Sequence[_RolloutCandidate],
    *,
    expected_group_size: int,
) -> _CandidateSelection | None:
    """Choose a deterministic, non-copying group from one UID's candidates."""

    ordered = sorted(candidates, key=lambda item: (item.generation_batch, item.row_index))
    clean = [candidate for candidate in ordered if not candidate.degraded]
    eligible = clean if len(clean) >= expected_group_size else ordered
    if len(eligible) < expected_group_size:
        return None

    best: tuple[float, int, tuple[tuple[int, int], ...], tuple[_RolloutCandidate, ...]] | None = None
    for combination in combinations(eligible, expected_group_size):
        rewards = [candidate.reward for candidate in combination]
        reward_min = min(rewards)
        reward_max = max(rewards)
        reward_range = reward_max - reward_min
        unique_reward_count = len(set(rewards))
        order = tuple(
            (candidate.generation_batch, candidate.row_index)
            for candidate in combination
        )
        if best is None:
            best = (reward_range, unique_reward_count, order, combination)
            continue
        best_range, best_unique, best_order, _ = best
        if reward_range > best_range + 1.0e-12 or (
            abs(reward_range - best_range) <= 1.0e-12
            and (
                unique_reward_count > best_unique
                or (
                    unique_reward_count == best_unique
                    and order < best_order
                )
            )
        ):
            best = (reward_range, unique_reward_count, order, combination)

    assert best is not None
    _, unique_reward_count, _, selected = best
    rewards = [candidate.reward for candidate in selected]
    return _CandidateSelection(
        candidates=tuple(selected),
        reward_min=min(rewards),
        reward_max=max(rewards),
        unique_reward_count=unique_reward_count,
    )


def _verl_sampling_signals(output: Any) -> tuple[list[Any], list[float], list[bool], list[tuple[str, ...]]]:
    """Extract stable prompt IDs and UserBench validity from a veRL ``DataProto``."""

    if "rm_scores" not in output.batch:
        raise ValueError("veRL rollout output is missing terminal rm_scores")
    rewards = output.batch["rm_scores"].sum(dim=-1).detach().cpu().tolist()
    uids = [_python_value(value) for value in output.non_tensor_batch.get("uid", ())]
    metadata = [_python_value(value) for value in output.non_tensor_batch.get("userbench", ())]
    if len(uids) != len(rewards) or len(metadata) != len(rewards):
        raise ValueError("veRL rollout output is missing aligned uid/userbench metadata")
    _, invalid, reasons = extract_userbench_group_signals(metadata)
    return uids, [float(value) for value in rewards], invalid, reasons


def _verl_candidate_signals(
    output: Any,
) -> tuple[list[Any], list[float], list[bool], list[tuple[str, ...]], list[bool]]:
    """Extract row-level validity and optional soft-degradation diagnostics."""

    uids, rewards, invalid, reasons = _verl_sampling_signals(output)
    metadata = [_python_value(value) for value in output.non_tensor_batch["userbench"]]
    degraded: list[bool] = []
    for info in metadata:
        reward = info.get("reward") if isinstance(info, Mapping) else None
        if not isinstance(reward, Mapping):
            reward = info if isinstance(info, Mapping) else {}
        outer_degraded = (
            info.get("reward_degraded", False)
            if isinstance(info, Mapping)
            else False
        )
        degraded.append(
            bool(reward.get("reward_degraded", outer_degraded))
        )
    return uids, rewards, invalid, reasons, degraded


def install_verl_bounded_sampler(manager: Any, config: Mapping[str, Any]) -> None:
    """Wrap veRL generation with bounded, UID-preserving dynamic sampling.

    Every generation call receives the same input ``batch``.  Valid rows are
    retained as complete ``DataProto`` slices under their task UID, so a UID
    can be completed by later generations without ever mixing another task's
    trajectory into its GRPO group.
    """

    if getattr(manager, "_travel_bounded_sampler_installed", False):
        return
    if not bool(config.get("enable", False)):
        return
    expected_group_size = int(config.get("group_size", 4))
    required_groups = int(config.get("required_groups", 2))
    max_batches = int(config.get("max_generation_batches", 3))
    max_skips = int(config.get("max_consecutive_skips", 10))
    tolerance = float(config.get("reward_tolerance", 1e-6))
    if expected_group_size != 4 or required_groups != 2 or max_batches != 3 or max_skips != 10:
        raise ValueError("production bounded-sampling contract is fixed at n=4, groups=2, batches=3, skips=10")
    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("reward_tolerance must be finite and non-negative")
    original = manager.generate_sequences
    state = BoundedSamplingState(
        required_groups=required_groups,
        max_generation_batches=max_batches,
        max_consecutive_skips=max_skips,
    )

    def generate_sequences(batch: Any) -> Any:
        if batch.meta_info.get("validate", False):
            return original(batch)
        raw_input_uids = batch.non_tensor_batch.get("uid", ())
        input_uids = [_python_value(value) for value in raw_input_uids]
        prompt_order = _ordered_unique(input_uids)
        if len(prompt_order) != required_groups:
            raise ValueError(
                "bounded sampler requires exactly two prompt UID groups per update"
            )
        if len(input_uids) != expected_group_size * required_groups:
            raise ValueError(
                "bounded sampler requires group_size rows for every input task UID"
            )
        aggregate = Counter()
        aggregate["degraded_candidate_count"] = 0
        last_output: Any | None = None
        candidates_by_uid: dict[Hashable, list[_RolloutCandidate]] = {
            uid: [] for uid in prompt_order
        }
        for _ in range(max_batches):
            generation_batch = state.generation_batches
            output = original(batch)
            last_output = output
            uids, rewards, invalid, reasons, degraded = _verl_candidate_signals(output)
            if tuple(uids) != tuple(input_uids):
                raise ValueError(
                    "veRL rollout generation must reuse the original task UID batch"
                )
            _, stats = select_reward_varying_groups(
                uids,
                rewards,
                sampling_invalid=invalid,
                sampling_invalid_reasons=reasons,
                expected_group_size=expected_group_size,
                tolerance=tolerance,
            )
            for name in (
                "constant_reward_group_count",
                "sampling_invalid_group_count",
                "incomplete_group_count",
            ):
                aggregate[name] += int(stats[name])
            for reason, count in stats["sampling_invalid_reason_counts"].items():
                aggregate[f"invalid_reason/{reason}"] += int(count)
            for index, (uid, reward, is_invalid, is_degraded) in enumerate(
                zip(uids, rewards, invalid, degraded, strict=True)
            ):
                if is_invalid:
                    continue
                candidates_by_uid[uid].append(
                    _RolloutCandidate(
                        uid=uid,
                        row=output.slice(index, index + 1),
                        reward=float(reward),
                        degraded=bool(is_degraded),
                        generation_batch=generation_batch,
                        row_index=index,
                    )
                )
                aggregate["valid_candidate_count"] += 1
                if is_degraded:
                    aggregate["degraded_candidate_count"] += 1

            selections: dict[Hashable, _CandidateSelection] = {}
            for uid in prompt_order:
                selection = _select_candidate_group(
                    candidates_by_uid[uid],
                    expected_group_size=expected_group_size,
                )
                if selection is not None:
                    aggregate[f"candidate_unique_rewards/{uid}"] = (
                        selection.unique_reward_count
                    )
                    if selection.reward_range <= tolerance:
                        aggregate["constant_candidate_group_count"] += 1
                    else:
                        selections[uid] = selection

            # Do not let complete in-batch groups make the state train early:
            # final acceptance is based on the UID candidate pool below.
            state.record_batch({**stats, "kept_group_count": 0})
            if len(selections) >= required_groups:
                from verl import DataProto

                ordered_rows: list[Any] = []
                for uid in prompt_order:
                    selection = selections.get(uid)
                    if selection is None:
                        raise ValueError(
                            "accepted groups do not match the input prompt UIDs"
                        )
                    ordered_rows.extend(
                        candidate.row for candidate in selection.candidates
                    )
                if len(ordered_rows) != required_groups * expected_group_size:
                    raise ValueError("accepted groups do not have the required row count")
                merged = DataProto.concat(ordered_rows)
                if len(getattr(merged, "non_tensor_batch", {}).get("uid", ())) != len(
                    ordered_rows
                ):
                    raise ValueError("DataProto tensor/non-tensor rows are misaligned")
                if tuple(
                    _python_value(value)
                    for value in merged.non_tensor_batch.get("uid", ())
                ) != tuple(
                    uid for uid in prompt_order for _ in range(expected_group_size)
                ):
                    raise ValueError("final DataProto rows are not grouped by input UID")
                # The trainer expects the same optional metadata contract as
                # before; all trajectory tensors and non-tensor fields came
                # from the complete row slices above.
                if not hasattr(merged, "meta_info"):
                    merged.meta_info = {}
                merged.meta_info.update(output.meta_info)
                sampled_batches = state.generation_batches
                state.accepted_groups = len(selections)
                state.finish_update()
                merged.meta_info["travel_dynamic_sampling"] = {
                    "sampled_batches": sampled_batches,
                    "accepted_groups": len(selections),
                    "candidate_count": sum(
                        len(values) for values in candidates_by_uid.values()
                    ),
                    "cross_batch_candidate_count": sum(
                        len(
                            {
                                candidate.generation_batch
                                for candidate in values
                            }
                        ) > 1
                        for values in candidates_by_uid.values()
                    ),
                    "api_failure_count": sum(
                        count
                        for key, count in aggregate.items()
                        if key.startswith("invalid_reason/")
                        and key != "invalid_reason/reward_invalid"
                    ),
                    "cost_tracking_available": False,
                    **dict(aggregate),
                }
                return merged
        assert last_output is not None
        state.accepted_groups = 0
        try:
            state.finish_update()
        except RuntimeError as exc:
            raise DynamicSamplingExhausted(str(exc)) from exc
        if not hasattr(last_output, "meta_info"):
            last_output.meta_info = {}
        last_output.meta_info["travel_skip_update"] = True
        last_output.meta_info["travel_dynamic_sampling"] = {
            "sampled_batches": max_batches,
            "accepted_groups": 0,
            "candidate_count": sum(
                len(values) for values in candidates_by_uid.values()
            ),
            "cross_batch_candidate_count": sum(
                len({candidate.generation_batch for candidate in values}) > 1
                for values in candidates_by_uid.values()
            ),
            "consecutive_skips": state.consecutive_skips,
            "api_failure_count": sum(
                count
                for key, count in aggregate.items()
                if key.startswith("invalid_reason/")
                and key != "invalid_reason/reward_invalid"
            ),
            "cost_tracking_available": False,
            **dict(aggregate),
        }
        return last_output

    manager.generate_sequences = generate_sequences
    manager._travel_bounded_sampler_installed = True
    manager._travel_bounded_sampling_state = state
