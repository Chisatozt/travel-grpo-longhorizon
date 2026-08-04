"""Bounded group selection and the project-owned veRL rollout adapter."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
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
    """Return indices in complete, valid groups whose terminal reward varies."""

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
    indices = [index for index, uid in enumerate(uids) if uid in kept]
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


def install_verl_bounded_sampler(manager: Any, config: Mapping[str, Any]) -> None:
    """Wrap veRL 0.8 generation with bounded whole-prompt resampling.

    Accepted groups are retained across at most three generation calls.  The
    wrapper returns an unchanged batch shape; if too few groups are available,
    the hash-checked trainer connection patch skips that optimizer update.
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
    original = manager.generate_sequences
    state = BoundedSamplingState(
        required_groups=required_groups,
        max_generation_batches=max_batches,
        max_consecutive_skips=max_skips,
    )

    def generate_sequences(batch: Any) -> Any:
        if batch.meta_info.get("validate", False):
            return original(batch)
        accepted: dict[Any, Any] = {}
        prompt_order = _ordered_unique(batch.non_tensor_batch.get("uid", ()))
        if len(prompt_order) != required_groups:
            raise ValueError(
                "bounded sampler requires exactly two prompt UID groups per update"
            )
        aggregate = Counter()
        last_output: Any | None = None
        for _ in range(max_batches):
            output = original(batch)
            last_output = output
            uids, rewards, invalid, reasons = _verl_sampling_signals(output)
            _, stats = select_reward_varying_groups(
                uids,
                rewards,
                sampling_invalid=invalid,
                sampling_invalid_reasons=reasons,
                expected_group_size=expected_group_size,
                tolerance=tolerance,
            )
            state.record_batch(stats)
            for name in (
                "constant_reward_group_count",
                "sampling_invalid_group_count",
                "incomplete_group_count",
            ):
                aggregate[name] += int(stats[name])
            for reason, count in stats["sampling_invalid_reason_counts"].items():
                aggregate[f"invalid_reason/{reason}"] += int(count)
            for group in stats["groups"]:
                uid = group["uid"]
                if group["kept"] and uid not in accepted:
                    indices = group["indices"]
                    if tuple(indices) != tuple(range(indices[0], indices[0] + expected_group_size)):
                        raise ValueError("veRL prompt-group rows must remain contiguous")
                    accepted[uid] = output.slice(indices[0], indices[0] + expected_group_size)
            if len(accepted) >= required_groups:
                from verl import DataProto

                # The trainer unions this output with its original repeated
                # prompt batch by row position.  A group accepted in batch 2
                # must therefore not move ahead of one accepted in batch 1.
                ordered = [accepted[uid] for uid in prompt_order if uid in accepted]
                if len(ordered) != required_groups:
                    raise ValueError("accepted groups do not match the input prompt UIDs")
                merged = DataProto.concat(ordered)
                merged.meta_info.update(output.meta_info)
                merged.meta_info["travel_dynamic_sampling"] = {
                    "sampled_batches": state.generation_batches,
                    "accepted_groups": len(accepted),
                    "api_failure_count": sum(
                        count
                        for key, count in aggregate.items()
                        if key.startswith("invalid_reason/")
                        and key != "invalid_reason/reward_invalid"
                    ),
                    "cost_tracking_available": False,
                    **dict(aggregate),
                }
                state.accepted_groups = len(accepted)
                state.finish_update()
                return merged
        assert last_output is not None
        state.accepted_groups = len(accepted)
        try:
            state.finish_update()
        except RuntimeError as exc:
            raise DynamicSamplingExhausted(str(exc)) from exc
        last_output.meta_info["travel_skip_update"] = True
        last_output.meta_info["travel_dynamic_sampling"] = {
            "sampled_batches": max_batches,
            "accepted_groups": len(accepted),
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
