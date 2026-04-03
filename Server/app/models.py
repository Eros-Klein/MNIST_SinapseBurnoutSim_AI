from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)
    details: dict[str, Any] | None = None


class ImageEntry(BaseModel):
    id: str
    created_at: datetime
    shape: tuple[int, int, int]
    evaluation: EvaluationResult
    preview_url: str


class ImageListResponse(BaseModel):
    offset: int
    limit: int
    count: int
    items: list[ImageEntry]
