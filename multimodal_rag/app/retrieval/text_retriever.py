from multimodal_rag.app.config import settings
from multimodal_rag.app.indexing.index_manager import IndexManager
from multimodal_rag.app.schemas import EvidenceRecord


class TextRetriever:

    def __init__(self, index_manager: IndexManager):
        self.index_manager = index_manager
        self.text_store = index_manager.text_store

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
            k = settings.text_retrieval_top_k

        if k < 1:
            raise ValueError("k must be at least 1.")

        # Apply a document filter only when document IDs were supplied.
        search_filter = None

        if document_ids is not None:
            search_filter = {
                "document_id": {
                    "$in": document_ids
                }
            }

        results = self.text_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=search_filter,
        )

        evidence_records = []

        for rank, (document, distance) in enumerate(
            results,
            start=1,
        ):
            metadata = document.metadata

            evidence = EvidenceRecord(
                evidence_id=metadata["chunk_id"],
                document_id=metadata["document_id"],
                source_name=metadata["source_name"],
                page_number=metadata["page_number"],
                content_type=metadata["content_type"],
                content=document.page_content,
                asset_path=None,
                relevance_score=1.0 / (
                    1.0 + max(float(distance), 0.0)
                ),
            )

            evidence_records.append(evidence)

        return evidence_records