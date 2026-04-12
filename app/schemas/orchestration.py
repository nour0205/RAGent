from typing import Literal

from pydantic import BaseModel, Field


class PlannerDecision(BaseModel):
    route: Literal[
        "concept_explanation",
        "source_recall",
        "exam_preparation",
        "unknown",
    ]
    targets: list[str] = Field(default_factory=list)