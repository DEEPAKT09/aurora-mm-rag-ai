from google.genai import types

from multimodal_rag.app.config import settings
from multimodal_rag.app.indexing.index_manager import IndexManager
from multimodal_rag.app.schemas import EvidenceRecord


class ImageRetriever:

    def __init__(self, index_manager: IndexManager):
        self.index_manager = index_manager
        self.image_collection = index_manager.image_collection
        self.genai_client = index_manager.genai_client
        self.embedding_model = index_manager.image_embedding_model
        self.dimensions = index_manager.image_dimensions

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[EvidenceRecord]:

        if not query or not query.strip():
            raise ValueError("The search query cannot be empty.")

        if document_ids is not None:
            document_ids = list(dict.fromkeys(document_ids))

            if not document_ids:
                return []

            if any(
                not isinstance(document_id, str)
                or not document_id.strip()
                for document_id in document_ids
            ):
                raise ValueError(
                    "document_ids must contain non-empty strings."
                )

        # Use the configured default unless the caller supplies k.
        if k is None:
            k = settings.image_retrieval_top_k

        if k < 1:
            raise ValueError("k must be at least 1.")

        # 1. Embed the text query in the multimodal embedding space.
        response = self.genai_client.models.embed_content(
            model=self.embedding_model,
            contents=[
                f"task: search for relevant images | query: {query}"
            ],
            config=types.EmbedContentConfig(
                output_dimensionality=self.dimensions
            ),
        )

        query_vector = response.embeddings[0].values

        # 2. Apply a document filter only when IDs were supplied.
        search_filter = None

        if document_ids is not None:
            search_filter = {
                "document_id": {
                    "$in": document_ids
                }
            }

        # 3. Search the persistent visual collection.
        results = self.image_collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where=search_filter,
            include=[
                "metadatas",
                "documents",
                "distances",
            ],
        )

        evidence_records = []

        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]
        distances = results["distances"][0]

        # 4. Convert visual matches to shared evidence records.
        for record_id, metadata, asset_path, distance in zip(
            ids,
            metadatas,
            documents,
            distances,
        ):
            evidence = EvidenceRecord(
                evidence_id=metadata["chunk_id"],
                document_id=metadata["document_id"],
                source_name=metadata["source_name"],
                page_number=metadata["page_number"],
                content_type=metadata["content_type"],
                content=(
                    "Visual asset retrieved for the question. "
                    f"Asset path: {asset_path}"
                ),
                asset_path=asset_path,
                relevance_score=1.0 / (
                    1.0 + max(float(distance), 0.0)
                ),
            )

            evidence_records.append(evidence)

        return evidence_records