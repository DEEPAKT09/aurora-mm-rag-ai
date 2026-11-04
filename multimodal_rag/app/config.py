import os
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root: aurora_ai/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):

    # ---------------------------------------------------------
    # Application
    # ---------------------------------------------------------

    app_name: str = "Multimodal KnowledgeBase AI"
    environment: str = "development"

    # FastAPI authentication key.
    # This is NOT the Gemini API key.
    api_key: str = ""

    # ---------------------------------------------------------
    # Gemini models
    # ---------------------------------------------------------

    answer_model: str = "gemini-3.5-flash-lite"

    multimodal_embedding_model: str = "gemini-embedding-2"

    embedding_dimensions: int = Field(
        default=768,
        gt=0,
    )

    # ---------------------------------------------------------
    # Timeouts
    # ---------------------------------------------------------

    model_request_timeout_seconds: int = Field(
        default=60,
        gt=0,
    )

    api_request_timeout_seconds: int = Field(
        default=180,
        gt=0,
    )

    # ---------------------------------------------------------
    # Storage
    # ---------------------------------------------------------

    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"

    processed_dir: Path = PROJECT_ROOT / "data" / "processed"

    index_dir: Path = PROJECT_ROOT / "data" / "indexes"

    metadata_dir: Path = PROJECT_ROOT / "data" / "metadata"

    # ---------------------------------------------------------
    # Vector collections
    # ---------------------------------------------------------

    text_collection_name: str = "knowledgebase_text"

    image_collection_name: str = "knowledgebase_images"

    # ---------------------------------------------------------
    # Upload security
    # ---------------------------------------------------------

    max_upload_size_mb: int = Field(
        default=20,
        gt=0,
    )

    allowed_upload_extensions: tuple[str, ...] = (".pdf",)

    # ---------------------------------------------------------
    # PDF processing limits
    # ---------------------------------------------------------

    # Maximum number of pages accepted for processing.
    max_pdf_pages: int = Field(
        default=100,
        ge=1,
    )

    # Maximum number of embedded images extracted from one PDF.
    max_embedded_images_per_pdf: int = Field(
        default=50,
        ge=0,
    )

    # Resolution used when rendering complete PDF pages.
    page_render_dpi: int = Field(
        default=100,
        ge=50,
        le=150,
    )

    # ---------------------------------------------------------
    # Retrieval limits
    # ---------------------------------------------------------

    text_retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
    )

    image_retrieval_top_k: int = Field(
        default=3,
        ge=1,
        le=50,
    )

    max_fused_evidence: int = Field(
        default=8,
        ge=1,
        le=100,
    )

    # ---------------------------------------------------------
    # Runtime resource limits
    # ---------------------------------------------------------

    max_concurrent_model_calls: int = Field(
        default=4,
        ge=1,
        le=100,
    )

    # ---------------------------------------------------------
    # Optional response cache
    # ---------------------------------------------------------

    response_cache_enabled: bool = False

    response_cache_ttl_seconds: int = Field(
        default=300,
        gt=0,
    )

    response_cache_max_entries: int = Field(
        default=500,
        ge=1,
    )

    # ---------------------------------------------------------
    # Cost controls
    # ---------------------------------------------------------

    max_model_calls_per_request: int = Field(
        default=6,
        ge=1,
    )

    max_input_characters: int = Field(
        default=12000,
        ge=100,
    )

    max_question_characters: int = Field(
        default=2000,
        ge=1,
    )

    # Model pricing: USD per 1 million tokens.
    model_input_cost_per_million_tokens: float = Field(
        default=0.0,
        ge=0,
    )

    model_output_cost_per_million_tokens: float = Field(
        default=0.0,
        ge=0,
    )

    # ---------------------------------------------------------
    # CORS
    # ---------------------------------------------------------

    cors_allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # ---------------------------------------------------------
    # Pydantic settings configuration
    # ---------------------------------------------------------

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )

    # ---------------------------------------------------------
    # Environment validation
    # ---------------------------------------------------------

    @model_validator(mode="after")
    def validate_production_settings(self):

        if self.environment.lower() not in {
            "development",
            "testing",
            "production",
        }:
            raise ValueError(
                "ENVIRONMENT must be development, testing, or production."
            )

        if self.environment.lower() == "production":

            if not self.api_key:
                raise ValueError(
                    "API_KEY must be configured in production."
                )

            if self.api_key.lower() in {
                "changeme",
                "your-api-key",
                "secret",
            }:
                raise ValueError(
                    "Replace the placeholder API_KEY before production."
                )

            if "*" in self.cors_allowed_origins:
                raise ValueError(
                    "Wildcard CORS origins are not allowed in production."
                )

        return self


settings = Settings()