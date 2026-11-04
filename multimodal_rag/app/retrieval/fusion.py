from multimodal_rag.app.config import settings
from multimodal_rag.app.indexing.index_manager import IndexManager
from multimodal_rag.app.retrieval.text_retriever import TextRetriever
from multimodal_rag.app.retrieval.image_retriever import ImageRetriever
from multimodal_rag.app.schemas import EvidenceRecord


class EvidenceFusion:

    def __init__(self, index_manager: IndexManager):
        self.text_retriever = TextRetriever(index_manager)
        self.image_retriever = ImageRetriever(index_manager)

    def retrieve(
        self,
        query: str,
        text_k: int | None = None,
        image_k: int | None = None,
        final_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[EvidenceRecord]:

        if not query or not query.strip():
            raise ValueError("The search query cannot be empty.")

        # An explicitly empty list means this session has no documents.
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

        # Read defaults from centralized configuration.
        text_k = (
            settings.text_retrieval_top_k
            if text_k is None else text_k
        )

        image_k = (
            settings.image_retrieval_top_k
            if image_k is None else image_k
        )

        final_k = (
            settings.max_fused_evidence
            if final_k is None else final_k
        )

        if min(text_k, image_k, final_k) < 1:
            raise ValueError(
                "All result counts must be at least 1."
            )

        # 1. Retrieve text evidence.
        text_results = self.text_retriever.retrieve(
            query=query,
            k=text_k,
            document_ids=document_ids,
        )

        # 2. Retrieve image evidence.
        image_results = self.image_retriever.retrieve(
            query=query,
            k=image_k,
            document_ids=document_ids,
        )

        # 3. Accumulate reciprocal-rank scores by evidence ID.
        rrf_constant = 60

        scores = {}
        evidence_by_id = {}

        for result_list in (
            text_results,
            image_results,
        ):
            for rank, evidence in enumerate(
                result_list,
                start=1,
            ):
                evidence_id = evidence.evidence_id

                scores[evidence_id] = (
                    scores.get(evidence_id, 0.0)
                    + 1.0 / (rrf_constant + rank)
                )

                evidence_by_id.setdefault(
                    evidence_id,
                    evidence,
                )

        # 4. Sort by combined RRF score.
        ranked_ids = sorted(
            scores,
            key=lambda evidence_id: scores[evidence_id],
            reverse=True,
        )

        # 5. Return the highest-ranked unique evidence records.
        fused_results = []

        for evidence_id in ranked_ids[:final_k]:
            evidence = evidence_by_id[evidence_id]

            evidence = evidence.model_copy(
                update={
                    "relevance_score": scores[evidence_id]
                }
            )

            fused_results.append(evidence)

        return fused_results