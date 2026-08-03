"""Actor and external teacher model runtime boundaries."""

from travel_grpo.models.openai_compatible import (
    OpenAICompatibleTeacherClient,
    TeacherApiError,
    TeacherProtocolError,
    TeacherRuntime,
    TeacherToolCall,
)

__all__ = [
    "OpenAICompatibleTeacherClient",
    "TeacherApiError",
    "TeacherProtocolError",
    "TeacherRuntime",
    "TeacherToolCall",
]
