"""Versioned production Actor policy shared by SFT, GRPO, and evaluation.

The runtime policy is part of the Actor-visible conversation.  The teacher
generation instruction is deliberately separate: it is appended only to a
Teacher request and is stripped before a trajectory is archived for Actor
training.  Prompt helpers are idempotent so callers can safely compose the
SFT, GRPO, and evaluation boundaries without repeated policy blocks.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any


ACTOR_RUNTIME_POLICY_VERSION = "actor-runtime-v1"
# A descriptive alias is useful to callers that use "Actor policy" as the
# version field name.  POLICY_VERSION remains a compatibility export for
# prompt-only callers; the Teacher state-machine version is intentionally kept
# in travel_grpo.training.teacher_policy.POLICY_VERSION.
ACTOR_POLICY_VERSION = ACTOR_RUNTIME_POLICY_VERSION
POLICY_VERSION = ACTOR_RUNTIME_POLICY_VERSION

ACTOR_RUNTIME_POLICY_MARKER = (
    f"[travel-grpo actor runtime policy | version={ACTOR_RUNTIME_POLICY_VERSION}]"
)
ACTOR_RUNTIME_POLICY_END_MARKER = "[end travel-grpo actor runtime policy]"
ACTOR_RUNTIME_POLICY = f"""{ACTOR_RUNTIME_POLICY_MARKER}
You are the production travel Actor. Follow the public control state and the visible conversation; never infer or expose hidden reward state.
- If the user has already answered a preference or explicitly has no preference, do not ask that preference again or repeat it with different wording.
- When the visible information is sufficient, or the public control state requires it, immediately issue one search for the current aspect.
- After a normal candidate list is visible, immediately issue answer for that aspect.
- An answer must contain exactly one option ID that is visible in the current candidate list; do not output explanations or hidden IDs.
- After the first search fallback, make at most one materially rewritten query retry. Do not repeat the same query.
- After a second fallback, stop searching that aspect and switch to the next aspect required by the public control state.
- Do not ask for extra services, preferences, or details without visible evidence that they are needed.
- Emit exactly one interact_with_env tool call per turn and obey the public control state over any conflicting habit.
{ACTOR_RUNTIME_POLICY_END_MARKER}"""

TEACHER_GENERATION_INSTRUCTION_VERSION = "teacher-generation-v1"
TEACHER_GENERATION_INSTRUCTION_MARKER = (
    "[travel-grpo teacher generation instruction | "
    f"version={TEACHER_GENERATION_INSTRUCTION_VERSION}]"
)
TEACHER_GENERATION_INSTRUCTION_END_MARKER = (
    "[end travel-grpo teacher generation instruction]"
)
TEACHER_GENERATION_INSTRUCTION = f"""{TEACHER_GENERATION_INSTRUCTION_MARKER}
Teacher-only generation constraints (never include this block in Actor training or evaluation messages):
- Follow the deterministic Teacher controller's current phase, aspect, and preference field.
- Emit one valid interact_with_env call per turn; keep the operational thought to at most 200 characters.
- Use the controller's one-field elicitation, search-repair, and answer constraints exactly.
- These constraints control data generation and validation only; they are not runtime Actor policy.
{TEACHER_GENERATION_INSTRUCTION_END_MARKER}"""

# This is retained only so probes over pre-policy trajectories can remove the
# historical suffix without rewriting those artifacts.  New callers should
# use ACTOR_RUNTIME_POLICY and TEACHER_GENERATION_INSTRUCTION.
LEGACY_TEACHER_ACTOR_POLICY = """Teacher policy for strict UserBench trajectories:
- Emit exactly one interact_with_env call per turn. Keep thought to one short operational sentence of at most 200 characters.
- Follow the controller's current phase, aspect, and preference field exactly.
- Ask one concrete preference field per action; never ask vague "other preferences" questions or bundle fields.
- Search each travel aspect at most once, after its preferences are complete.
- Answer immediately after search with exactly one visible option ID for the current aspect.
- Never repeat an exact action, semantic preference field, search aspect, or answered aspect."""


def _copy_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    copied = copy.deepcopy(list(messages))
    if not copied or copied[0].get("role") != "system":
        raise ValueError("Actor prompt must begin with a system message")
    if not isinstance(copied[0].get("content"), str):
        raise ValueError("Actor system message must contain text")
    return copied


def _remove_blocks(content: str, blocks: Sequence[str]) -> str:
    result = content
    for block in blocks:
        while block in result:
            result = result.replace(block, "")
    return result.strip()


def strip_teacher_generation_instruction(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Remove Teacher-only request text while preserving the runtime policy."""

    result = _copy_messages(messages)
    content = str(result[0]["content"])
    content = _remove_blocks(content, (TEACHER_GENERATION_INSTRUCTION,))
    result[0]["content"] = content
    return result


def strip_actor_runtime_policy(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return a prompt without current or historical policy suffixes.

    This is primarily for controlled A/B probes.  Production SFT, GRPO, and
    evaluation callers should use :func:`ensure_actor_runtime_policy`.
    """

    result = _copy_messages(messages)
    content = str(result[0]["content"])
    content = _remove_blocks(
        content,
        (
            ACTOR_RUNTIME_POLICY,
            LEGACY_TEACHER_ACTOR_POLICY,
            TEACHER_GENERATION_INSTRUCTION,
        ),
    )
    result[0]["content"] = content
    return result


def ensure_actor_runtime_policy(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Add exactly one current Actor policy block to a message sequence."""

    result = strip_actor_runtime_policy(messages)
    content = str(result[0]["content"]).rstrip()
    result[0]["content"] = f"{content}\n\n{ACTOR_RUNTIME_POLICY}"
    return result


def ensure_teacher_generation_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build a Teacher request with runtime policy plus Teacher-only controls."""

    result = ensure_actor_runtime_policy(messages)
    content = str(result[0]["content"]).rstrip()
    result[0]["content"] = f"{content}\n\n{TEACHER_GENERATION_INSTRUCTION}"
    return result


__all__ = [
    "ACTOR_POLICY_VERSION",
    "ACTOR_RUNTIME_POLICY",
    "ACTOR_RUNTIME_POLICY_END_MARKER",
    "ACTOR_RUNTIME_POLICY_MARKER",
    "ACTOR_RUNTIME_POLICY_VERSION",
    "LEGACY_TEACHER_ACTOR_POLICY",
    "POLICY_VERSION",
    "TEACHER_GENERATION_INSTRUCTION",
    "TEACHER_GENERATION_INSTRUCTION_END_MARKER",
    "TEACHER_GENERATION_INSTRUCTION_MARKER",
    "TEACHER_GENERATION_INSTRUCTION_VERSION",
    "ensure_actor_runtime_policy",
    "ensure_teacher_generation_messages",
    "strip_actor_runtime_policy",
    "strip_teacher_generation_instruction",
]
