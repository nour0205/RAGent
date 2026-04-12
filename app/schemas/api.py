from pydantic import BaseModel, Field
from typing import Literal

class QuestionRequest(BaseModel):
    question: str
    document_id: str | None = None
    document_ids: list[str] | None = None
    owner: str | None = None


class SourceItem(BaseModel):
    document_id: str
    chunk_index: int | None = None
    text: str
    retrieval_type: str | None = None
    hybrid_score: float | None = None
    rerank_score: float | None = None


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    route: str | None = None
    study_hint: str | None = None


class IngestRequest(BaseModel):
    text: str
    document_id: str
    source: str | None = None
    owner: str | None = None
    document_type: Literal["lecture_note"] | None = None
    course: str | None = None
    topic_tags: list[str] = Field(default_factory=list)


class RagAnswerResult(BaseModel):
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)