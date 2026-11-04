import hashlib
import logging
import os
import shutil
import sqlite3

from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from multimodal_rag.app.config import PROJECT_ROOT, settings
from multimodal_rag.app.ingestion.pdf_loader import (
    extract_pdf,
    calculate_file_hash,
)


load_dotenv()

api_key=(
    os.getenv("GOOGLE_API_KEY")
)

logger = logging.getLogger(__name__)



class IndexManager:

    def __init__(self):

        # -----------------------------------------------------
        # 1. Prepare persistent storage directories.
        # -----------------------------------------------------

        settings.upload_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        settings.processed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        settings.index_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        settings.metadata_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # -----------------------------------------------------
        # 2. Configure text embeddings.
        # -----------------------------------------------------

        self.text_embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=api_key,
        )

        # -----------------------------------------------------
        # 3. Configure multimodal embeddings.
        # -----------------------------------------------------

        self.genai_client = genai.Client()

        self.image_embedding_model = (
            settings.multimodal_embedding_model
        )

        self.image_dimensions = settings.embedding_dimensions

        # -----------------------------------------------------
        # 4. Configure text chunking.
        # -----------------------------------------------------

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
        )

        # -----------------------------------------------------
        # 5. Configure persistent text vector store.
        # -----------------------------------------------------

        self.text_store = Chroma(
            collection_name=settings.text_collection_name,
            embedding_function=self.text_embeddings,
            persist_directory=str(
                settings.index_dir / "text"
            ),
        )

        # -----------------------------------------------------
        # 6. Configure persistent image vector store.
        # -----------------------------------------------------

        import chromadb

        self.image_client = chromadb.PersistentClient(
            path=str(
                settings.index_dir / "images"
            )
        )

        self.image_collection = (
            self.image_client.get_or_create_collection(
                name=settings.image_collection_name,
            )
        )

        # -----------------------------------------------------
        # 7. Configure the document registry.
        # -----------------------------------------------------

        self.registry_path = (
            settings.metadata_dir
            / "documents.sqlite3"
        )

        self._initialize_registry()

    # =========================================================
    # DOCUMENT REGISTRY
    # =========================================================

    def _initialize_registry(self):
        """
        Create the SQLite registry.

        source_path is the primary key, allowing different uploaded
        copies of a document to be tracked independently.
        """

        with sqlite3.connect(
            self.registry_path,
            timeout=30,
        ) as connection:

            connection.execute(
                "PRAGMA journal_mode=WAL"
            )

            connection.execute(
                "PRAGMA busy_timeout=30000"
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS indexed_documents (
                    source_path TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    indexed_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _get_registered_document(
        self,
        source_path: str,
    ):
        """Return the registry entry for a source path."""

        with sqlite3.connect(
            self.registry_path,
            timeout=30,
        ) as connection:

            row = connection.execute(
                """
                SELECT document_id, content_hash
                FROM indexed_documents
                WHERE source_path = ?
                """,
                (source_path,),
            ).fetchone()

        return row

    def _is_document_referenced(
        self,
        document_id: str,
    ) -> bool:
        """
        Check whether any registered source still references
        a particular document ID.
        """

        with sqlite3.connect(
            self.registry_path,
            timeout=30,
        ) as connection:

            row = connection.execute(
                """
                SELECT 1
                FROM indexed_documents
                WHERE document_id = ?
                LIMIT 1
                """,
                (document_id,),
            ).fetchone()

        return row is not None

    def _record_document(
        self,
        source_path: str,
        document_id: str,
        content_hash: str,
    ):
        """Insert or update a document's registry entry."""

        with sqlite3.connect(
            self.registry_path,
            timeout=30,
        ) as connection:

            connection.execute(
                """
                INSERT INTO indexed_documents (
                    source_path,
                    document_id,
                    content_hash,
                    indexed_at
                )
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)

                ON CONFLICT(source_path) DO UPDATE SET
                    document_id = excluded.document_id,
                    content_hash = excluded.content_hash,
                    indexed_at = CURRENT_TIMESTAMP
                """,
                (
                    source_path,
                    document_id,
                    content_hash,
                ),
            )

    # =========================================================
    # DOCUMENT ID GENERATION
    # =========================================================

    @staticmethod
    def _generate_document_id(
        content_hash: str,
        namespace_id: str | None = None,
    ) -> str:
        """
        Generate a document ID.

        Without namespace_id:
            Retains the original content-hash-based ID.

        With namespace_id:
            Generates a session-specific document ID.

        The namespace itself is hashed rather than stored in
        the document ID.
        """

        if namespace_id is None:

            return f"doc_{content_hash[:16]}"

        if not isinstance(namespace_id, str) or not namespace_id.strip():

            raise ValueError(
                "namespace_id must be a non-empty string."
            )

        namespace_hash = hashlib.sha256(
            namespace_id.encode("utf-8")
        ).hexdigest()

        combined_value = (
            f"{namespace_hash}:{content_hash}"
        )

        document_hash = hashlib.sha256(
            combined_value.encode("utf-8")
        ).hexdigest()

        return f"doc_{document_hash[:32]}"

    # =========================================================
    # PROCESSED ASSET CLEANUP
    # =========================================================

    def _delete_processed_assets(
        self,
        document_id: str,
    ) -> None:
        """
        Remove processed assets only when no registry entry
        references the document.
        """

        if self._is_document_referenced(document_id):

            logger.info(
                "Skipping asset cleanup; document is still referenced: %s",
                document_id,
            )

            return

        asset_dir = (
            settings.processed_dir / document_id
        ).resolve()

        processed_root = settings.processed_dir.resolve()

        if not asset_dir.is_relative_to(processed_root):

            raise ValueError(
                "Processed asset path is outside the configured directory."
            )

        if asset_dir.exists():

            shutil.rmtree(asset_dir)

            logger.info(
                "Removed processed assets for document_id=%s",
                document_id,
            )

    def _cleanup_failed_indexing(
        self,
        document_id: str,
    ) -> None:
        """
        Remove partial vectors and assets after an indexing failure.

        Existing registered documents are not deleted.
        """

        if self._is_document_referenced(document_id):

            logger.warning(
                "Skipping rollback because document is already registered: %s",
                document_id,
            )

            return

        try:

            self._delete_document_vectors(document_id)

        except Exception:

            logger.exception(
                "Failed to remove partial vectors for document_id=%s",
                document_id,
            )

        try:

            self._delete_processed_assets(document_id)

        except Exception:

            logger.exception(
                "Failed to remove partial assets for document_id=%s",
                document_id,
            )

    # =========================================================
    # IMAGE EMBEDDING
    # =========================================================

    def _embed_image(
        self,
        image_path: str,
    ) -> list[float]:
        """
        Generate a multimodal embedding for one image.

        Images are embedded one at a time to limit memory usage.
        """

        path = Path(image_path)

        image_bytes = path.read_bytes()

        extension = path.suffix.lower()

        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }

        mime_type = mime_types.get(extension)

        if mime_type is None:

            raise ValueError(
                f"Unsupported image extension: {extension}"
            )

        response = self.genai_client.models.embed_content(
            model=self.image_embedding_model,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                )
            ],
            config=types.EmbedContentConfig(
                output_dimensionality=self.image_dimensions,
            ),
        )

        if not response.embeddings:

            raise RuntimeError(
                f"No embedding was returned for image: {path.name}"
            )

        return response.embeddings[0].values

    # =========================================================
    # VECTOR DELETION
    # =========================================================

    def _delete_document_vectors(
        self,
        document_id: str,
    ):
        """Delete text and image vectors belonging to one document."""

        self.text_store.delete(
            where={
                "document_id": document_id,
            }
        )

        self.image_collection.delete(
            where={
                "document_id": document_id,
            }
        )

    # =========================================================
    # PDF INDEXING
    # =========================================================

    def index_pdf(
        self,
        pdf_path: str | Path,
        namespace_id: str | None = None,
    ) -> dict:
        """
        Index a PDF.

        namespace_id is optional.

        Aurora Streamlit will pass its session ID here to ensure
        separate sessions receive separate document IDs.

        Standalone MM-RAG can continue calling index_pdf(pdf_path)
        without changing its existing behavior.
        """

        pdf_path = Path(pdf_path).resolve()

        if not pdf_path.is_file():

            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":

            raise ValueError(
                "The supplied file must have a .pdf extension."
            )

        source_path = str(pdf_path)

        # -----------------------------------------------------
        # 1. Calculate the current file hash.
        # -----------------------------------------------------

        current_hash = calculate_file_hash(pdf_path)

        # -----------------------------------------------------
        # 2. Generate the appropriate document ID.
        # -----------------------------------------------------

        expected_document_id = self._generate_document_id(
            content_hash=current_hash,
            namespace_id=namespace_id,
        )

        # -----------------------------------------------------
        # 3. Check whether this exact source is already indexed.
        # -----------------------------------------------------

        registered = self._get_registered_document(
            source_path
        )

        if (
            registered
            and registered[0] == expected_document_id
            and registered[1] == current_hash
        ):

            return {
                "status": "skipped",
                "reason": "The PDF has not changed.",
                "document_id": registered[0],
            }

        document_id = expected_document_id

        indexing_completed = False

        try:

            # -------------------------------------------------
            # 4. Extract PDF content.
            # -------------------------------------------------

            result = extract_pdf(
                pdf_path,
                document_id_override=document_id,
            )

            document = result["document"]

            chunks = result["chunks"]

            # -------------------------------------------------
            # 5. Prepare text and table records.
            # -------------------------------------------------

            text_records = [
                chunk
                for chunk in chunks
                if (
                    chunk.content_type in {"text", "table"}
                    and chunk.text.strip()
                )
            ]

            text_documents = []

            for chunk in text_records:

                langchain_document = Document(
                    page_content=chunk.text,
                    metadata={
                        "chunk_id": chunk.chunk_id,
                        "document_id": document_id,
                        "source_name": document.source_name,
                        "source_path": source_path,
                        "page_number": chunk.page_number,
                        "content_type": chunk.content_type,
                    },
                )

                text_documents.append(
                    langchain_document
                )

            # -------------------------------------------------
            # 6. Split text and table content.
            # -------------------------------------------------

            text_chunks = self.splitter.split_documents(
                text_documents
            )

            # Generate deterministic IDs for the split chunks.
            for chunk_number, chunk in enumerate(text_chunks):

                chunk.metadata["chunk_id"] = (
                    f"{document_id}_text_{chunk_number:06d}"
                )

            # -------------------------------------------------
            # 7. Prepare image records.
            # -------------------------------------------------

            image_records = [
                chunk
                for chunk in chunks
                if (
                    chunk.content_type in {"image", "page_image"}
                    and chunk.asset_path
                )
            ]

            # -------------------------------------------------
            # 8. Store text vectors.
            # -------------------------------------------------

            if text_chunks:

                self.text_store.add_documents(
                    documents=text_chunks,
                    ids=[
                        chunk.metadata["chunk_id"]
                        for chunk in text_chunks
                    ],
                )

            # -------------------------------------------------
            # 9. Generate and store image vectors.
            # -------------------------------------------------

            for image_record in image_records:

                image_path = image_record.asset_path

                vector = self._embed_image(
                    image_path
                )

                self.image_collection.upsert(
                    ids=[
                        image_record.chunk_id
                    ],
                    embeddings=[
                        vector
                    ],
                    documents=[
                        image_path
                    ],
                    metadatas=[
                        {
                            "chunk_id": image_record.chunk_id,
                            "document_id": document_id,
                            "source_name": document.source_name,
                            "source_path": source_path,
                            "page_number": image_record.page_number,
                            "content_type": image_record.content_type,
                            "asset_path": image_path,
                        }
                    ],
                )

            # -------------------------------------------------
            # 10. Register only after vector indexing succeeds.
            # -------------------------------------------------

            self._record_document(
                source_path=source_path,
                document_id=document_id,
                content_hash=current_hash,
            )

            indexing_completed = True

        except Exception:

            logger.exception(
                "Indexing failed for source_path=%s document_id=%s",
                source_path,
                document_id,
            )

            self._cleanup_failed_indexing(
                document_id
            )

            raise

        # -----------------------------------------------------
        # 11. Clean up a previous version if necessary.
        # -----------------------------------------------------

        if indexing_completed and registered:

            old_document_id = registered[0]

            if old_document_id != document_id:

                try:

                    if not self._is_document_referenced(
                        old_document_id
                    ):

                        self._delete_document_vectors(
                            old_document_id
                        )

                        self._delete_processed_assets(
                            old_document_id
                        )

                except Exception:

                    logger.exception(
                        "New version indexed, but old version "
                        "cleanup failed: %s",
                        old_document_id,
                    )

        return {
            "status": "indexed",
            "document_id": document_id,
            "text_chunks": len(text_chunks),
            "image_records": len(image_records),
            "warnings": result["warnings"],
        }

    # =========================================================
    # PDF DELETION
    # =========================================================

    def delete_pdf(
        self,
        pdf_path: str | Path,
    ) -> dict:
        """
        Remove a PDF's registry entry, vectors, and unreferenced assets.
        """

        source_path = str(
            Path(pdf_path).resolve()
        )

        registered = self._get_registered_document(
            source_path
        )

        if not registered:

            return {
                "status": "not_found",
                "source_path": source_path,
            }

        document_id = registered[0]

        # -----------------------------------------------------
        # 1. Remove this source from the registry.
        # -----------------------------------------------------

        with sqlite3.connect(
            self.registry_path,
            timeout=30,
        ) as connection:

            connection.execute(
                """
                DELETE FROM indexed_documents
                WHERE source_path = ?
                """,
                (source_path,),
            )

        # -----------------------------------------------------
        # 2. Delete vectors and assets if no other source uses ID.
        # -----------------------------------------------------

        if not self._is_document_referenced(document_id):

            self._delete_document_vectors(
                document_id
            )

            self._delete_processed_assets(
                document_id
            )

        return {
            "status": "deleted",
            "document_id": document_id,
        }


if __name__ == "__main__":

    manager = IndexManager()

    sample_pdf = (
        PROJECT_ROOT
        / "data"
        / "uploads"
        / "sample.pdf"
    )

    result = manager.index_pdf(
        sample_pdf
    )

    print("\nIndexing result:")

    print(result)