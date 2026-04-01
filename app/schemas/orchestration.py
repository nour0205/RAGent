from typing import Literal

from pydantic import BaseModel, Field


class PlannerDecision(BaseModel):
    route: Literal["single", "multi", "unknown"]
    targets: list[str] = Field(default_factory=list)