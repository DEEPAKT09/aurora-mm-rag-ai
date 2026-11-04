from pydantic import BaseModel, Field, field_validator
from multimodal_rag.app.config import settings

# 1. Request model for asking questions
class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=settings.max_question_characters,
        description="Question submitted to the multimodal RAG system",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Question cannot contain only whitespace.")

        return value


# 2. Citation response model
class CitationResponse(BaseModel):
    source: str
    page: int | None = None
    evidence_id: str


# 3. Response model for the RAG answer
class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationResponse] = Field(default_factory=list)
    insufficient_evidence: bool


# 4. Request model for creating a document record
class DocumentCreateRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    description: str = Field(default="", max_length=1000)

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Filename cannot be empty.")

        return value


# 5. Response model for a document record
class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    description: str = ""

# 6. Response model for PDF uploads
class UploadResponse(BaseModel):
    filename: str
    document_id: str
    status: str
    text_chunks: int = 0
    image_records: int = 0
    warnings: list[str] = Field(default_factory=list)