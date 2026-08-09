"""Prompt contracts shared across Actor training and inference."""

from travel_grpo.prompts.actor_policy import (
    ACTOR_POLICY_VERSION,
    ACTOR_RUNTIME_POLICY,
    ACTOR_RUNTIME_POLICY_VERSION,
    POLICY_VERSION,
    TEACHER_GENERATION_INSTRUCTION,
    ensure_actor_runtime_policy,
    ensure_teacher_generation_messages,
    strip_actor_runtime_policy,
    strip_teacher_generation_instruction,
)

__all__ = [
    "ACTOR_POLICY_VERSION",
    "ACTOR_RUNTIME_POLICY",
    "ACTOR_RUNTIME_POLICY_VERSION",
    "POLICY_VERSION",
    "TEACHER_GENERATION_INSTRUCTION",
    "ensure_actor_runtime_policy",
    "ensure_teacher_generation_messages",
    "strip_actor_runtime_policy",
    "strip_teacher_generation_instruction",
]
