import os
from multimodal_rag.app.usage_tracking import consume_model_call, record_model_call
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from multimodal_rag.app.config import settings
from multimodal_rag.app.schemas import EvidenceRecord, RAGAnswer
from multimodal_rag.app.runtime_controls import model_call_slot


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


class VerificationResult(BaseModel):
    supported: bool = Field(
        description="True only when the answer is supported by the evidence."
    )

    reason: str = Field(
        description="Brief explanation of the verification decision."
    )

    unsupported_claims: list[str] = Field(
        default_factory=list,
        description=(
            "Claims that are unsupported or contradicted by the evidence."
        ),
    )


class AnswerVerifier:

    def __init__(self):
        self.model = ChatGoogleGenerativeAI(
            model=settings.answer_model,
            google_api_key=api_key,
            timeout=settings.model_request_timeout_seconds,
        ).with_structured_output(VerificationResult,  include_raw=True,)

    def verify(
        self,
        answer: RAGAnswer,
        evidence: list[EvidenceRecord],
    ) -> VerificationResult:

        if not evidence:
            return VerificationResult(
                supported=False,
                reason="No evidence was supplied.",
                unsupported_claims=[answer.answer],
            )

        evidence_text = "\n\n".join(
            (
                f"Evidence ID: {item.evidence_id}\n"
                f"Source: {item.source_name}\n"
                f"Page: {item.page_number}\n"
                f"Type: {item.content_type}\n"
                f"Content: {item.content}"
            )
            for item in evidence
        )

        system_prompt = """
You are a strict evidence-grounding verifier.

Determine whether the proposed answer is supported by the supplied evidence.

Rules:
1. Check factual claims, numbers, percentages, names, dates, and comparisons.
2. Mark the answer unsupported if a material claim contradicts the evidence.
3. Mark the answer unsupported if it makes material factual claims that
   cannot be established from the evidence.
4. Do not treat a citation's existence as proof that its claim is correct.
5. Do not use outside knowledge to fill gaps.
6. Minor wording differences and faithful summaries are acceptable.
7. If the evidence is ambiguous or insufficient, be conservative.
8. Return unsupported_claims describing the specific problems.
"""

        user_prompt = (
            f"Proposed answer:\n{answer.answer}\n\n"
            f"Insufficient-evidence flag: {answer.insufficient_evidence}\n\n"
            f"Citations:\n"
            f"{[citation.model_dump() for citation in answer.citations]}\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            "Is the proposed answer supported by the supplied evidence?"
        )

        # Limit concurrent verification requests as wel        consume_model_call("answer_verification")l.
        consume_model_call("answer_verification")

        with model_call_slot():
            model_result = self.model.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )

        result = model_result["parsed"]
        raw_response = model_result["raw"]

        if result is None:
            raise ValueError(
                "The model did not return a valid verification result."
            )
        record_model_call("answer_verification", raw_response)

        return result