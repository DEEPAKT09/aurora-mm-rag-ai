from typing import Literal

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    document_id: str
    source_name: str
    file_path: str
    content_hash: str


class ContentChunk(BaseModel):
    chunk_id: str
    document_id: str
    page_number: int = Field(ge=1)
    content_type: Literal["text", "table", "image", "page_image"]
    text: str = ""
    asset_path: str | None = None


class EvidenceRecord(BaseModel):
    evidence_id: str
    document_id: str
    source_name: str
    page_number: int = Field(ge=1)
    content_type: Literal["text", "table", "image", "page_image"]
    content: str = ""
    asset_path: str | None = None
    relevance_score: float | None = None


class Citation(BaseModel):
    document_id: str
    source_name: str
    page_number: int = Field(ge=1)
    evidence_id: str


class RAGAnswer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    insufficient_evidence: bool = False