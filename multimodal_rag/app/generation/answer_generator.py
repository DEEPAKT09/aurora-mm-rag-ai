from pathlib import Path
import base64
import hashlib
import json
import mimetypes
import os
from multimodal_rag.app.usage_tracking import consume_model_call, record_model_call
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from multimodal_rag.app.config import settings
from multimodal_rag.app.schemas import EvidenceRecord, Citation, RAGAnswer
from multimodal_rag.app.generation.answer_verifier import AnswerVerifier
from multimodal_rag.app.runtime_controls import model_call_slot, response_cache


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


class GroundedAnswerGenerator:

    def __init__(self):
        self.model = ChatGoogleGenerativeAI(
            model=settings.answer_model,
            google_api_key=api_key,
            timeout=settings.model_request_timeout_seconds,
        ).with_structured_output(RAGAnswer, include_raw=True,)

        self.verifier = AnswerVerifier()

    def _format_evidence(
        self,
        evidence: list[EvidenceRecord],
    ) -> str:

        evidence_sections = []

        for rank, item in enumerate(evidence, start=1):
            section = (
                f"Evidence {rank}\n"
                f"Evidence ID: {item.evidence_id}\n"
                f"Source: {item.source_name}\n"
                f"Page: {item.page_number}\n"
                f"Type: {item.content_type}\n"
                f"Content: {item.content}\n"
            )

            if item.asset_path:
                section += f"Visual asset path: {item.asset_path}\n"

            evidence_sections.append(section)

        return "\n---\n".join(evidence_sections)

    def _image_content_parts(
        self,
        evidence: list[EvidenceRecord],
        max_images: int = 3,
    ) -> list[dict]:

        image_parts = []
        included_paths = set()

        for item in evidence:
            if not item.asset_path:
                continue

            image_path = Path(item.asset_path)

            if not image_path.is_file():
                continue

            resolved_path = str(image_path.resolve())

            if resolved_path in included_paths:
                continue

            mime_type, _ = mimetypes.guess_type(str(image_path))

            if not mime_type or not mime_type.startswith("image/"):
                continue

            image_bytes = image_path.read_bytes()
            encoded_image = base64.b64encode(
                image_bytes
            ).decode("utf-8")

            image_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            f"data:{mime_type};base64,{encoded_image}"
                        )
                    },
                }
            )

            included_paths.add(resolved_path)

            if len(image_parts) >= max_images:
                break

        return image_parts

    def _validate_citations(
        self,
        answer: RAGAnswer,
        evidence: list[EvidenceRecord],
    ) -> RAGAnswer:

        evidence_by_id = {
            item.evidence_id: item
            for item in evidence
        }

        verified_citations = []
        seen_evidence_ids = set()

        for citation in answer.citations:
            item = evidence_by_id.get(citation.evidence_id)

            if item is None:
                continue

            if citation.evidence_id in seen_evidence_ids:
                continue

            verified_citations.append(
                Citation(
                    document_id=item.document_id,
                    source_name=item.source_name,
                    page_number=item.page_number,
                    evidence_id=item.evidence_id,
                )
            )

            seen_evidence_ids.add(citation.evidence_id)

        return answer.model_copy(
            update={"citations": verified_citations}
        )

    def _cache_key(
        self,
        question: str,
        evidence: list[EvidenceRecord],
    ) -> str:
        """Build a stable key from the question and retrieved evidence."""

        evidence_data = []

        for item in evidence:
            asset_info = None

            if item.asset_path:
                asset_path = Path(item.asset_path)

                if asset_path.is_file():
                    stat = asset_path.stat()
                    asset_info = {
                        "path": str(asset_path.resolve()),
                        "size": stat.st_size,
                        "modified": stat.st_mtime_ns,
                    }

            evidence_data.append(
                {
                    "evidence_id": item.evidence_id,
                    "document_id": item.document_id,
                    "page_number": item.page_number,
                    "content_type": item.content_type,
                    "content": item.content,
                    "asset": asset_info,
                }
            )

        cache_data = {
            "model": settings.answer_model,
            "question": question.strip(),
            "evidence": evidence_data,
            "prompt_version": "grounded-answer-v1",
        }

        serialized = json.dumps(
            cache_data,
            sort_keys=True,
            ensure_ascii=False,
        )

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    def generate(
        self,
        question: str,
        evidence: list[EvidenceRecord],
    ) -> RAGAnswer:

        if not question.strip():
            raise ValueError("The question cannot be empty.")

        if not evidence:
            return RAGAnswer(
                question=question,
                answer=(
                    "I couldn't find sufficient evidence in the "
                    "available documents to answer this question."
                ),
                citations=[],
                insufficient_evidence=True,
            )

        # 1. Check whether response caching is enabled.
        cache_key = self._cache_key(question, evidence)

        if settings.response_cache_enabled:
            cached_answer = response_cache.get(cache_key)

            if cached_answer is not None:
                print("DEBUG: Returning cached answer")
                print("DEBUG: Cached answer:", cached_answer)
                return RAGAnswer.model_validate(cached_answer)

        # 2. Prepare evidence for the model.
        evidence_text = self._format_evidence(evidence)

        system_prompt = """
You are a careful assistant that answers questions using supplied evidence.

Rules:
1. Answer using only the supplied evidence and the provided visual assets.
2. Do not invent facts, percentages, dates, names, or explanations.
3. If the evidence does not adequately answer the question, say so.
4. Set insufficient_evidence to true when the evidence is inadequate.
5. Cite evidence by using the exact evidence_id values provided.
6. Only cite evidence that directly supports the answer.
7. Do not invent evidence IDs or citations.
8. Keep the answer clear and concise.
9. Treat text and images as evidence, but do not assume that an image
   proves something unless its contents support that conclusion.
"""

        user_content = [
            {
                "type": "text",
                "text": (
                    f"Question:\n{question}\n\n"
                    f"Retrieved evidence:\n{evidence_text}\n\n"
                    "Answer the question using the rules provided. "
                    "Return the result in the required structured format."
                ),
            }
        ]

        user_content.extend(
            self._image_content_parts(evidence)
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]

        # 3. Generate the structured answer with a concurrency limit.
        consume_model_call("answer_generation")
        print("DEBUG: Cache miss; generating a new answer")
        with model_call_slot():
            model_result = self.model.invoke(messages)

        # Support both LangChain's include_raw response and a plain RAGAnswer.
        if isinstance(model_result, dict) and "parsed" in model_result:
            answer = model_result["parsed"] 
            raw_response = model_result.get("raw")
        else:
            answer = model_result
            raw_response = None

        if answer is None:
            raise ValueError("The answer model returned an empty structured response.")

        # Token usage is available when LangChain returns the raw AIMessage.
        if raw_response is not None:
            record_model_call("answer_generation", raw_response)
        else:
            # Still count the model call when a test mock or alternate model
            # returns the structured answer directly.
            record_model_call("answer_generation", model_result)

        # 4. Validate citations against retrieved evidence.
        answer = self._validate_citations(answer, evidence)

        # 5. Verify factual grounding.
        verification = self.verifier.verify(
            answer=answer,
            evidence=evidence,
        )
        print("\n--- VERIFICATION DEBUG ---")
        print("Verifier object:", repr(self.verifier))
        print("Verification result:", repr(verification))
        print("Supported:", repr(verification.supported))
        print("Supported type:", type(verification.supported))

        if not verification.supported:
            print("Suppression Branch Executed")
            return answer.model_copy(
                update={
                    "answer": (
                        "I couldn't verify that the available evidence "
                        "fully supports an answer to this question."
                    ),
                    "citations": [],
                    "insufficient_evidence": True,
                }
            )

        # 6. Cache only successfully verified answers.
        if settings.response_cache_enabled:
            response_cache.set(
                cache_key,
                answer.model_dump(mode="json"),
            )

        return answer