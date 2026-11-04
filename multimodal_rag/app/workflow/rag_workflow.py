from threading import Event
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from multimodal_rag.app.indexing.index_manager import IndexManager
from multimodal_rag.app.retrieval.fusion import EvidenceFusion
from multimodal_rag.app.generation.answer_generator import GroundedAnswerGenerator
from multimodal_rag.app.schemas import EvidenceRecord, RAGAnswer


class RAGState(TypedDict):
    question: str
    document_ids: list[str] | None
    evidence: list[EvidenceRecord]
    answer: RAGAnswer | None
    error: str | None
    retrieval_retries: int
    generation_retries: int


class MultimodalRAGWorkflow:

    def __init__(self):

        self.index_manager = IndexManager()
        self.fusion = EvidenceFusion(self.index_manager)
        self.generator = GroundedAnswerGenerator()

        workflow = StateGraph(RAGState)

        workflow.add_node(
            "retrieve",
            self.retrieve_node,
        )

        workflow.add_node(
            "generate",
            self.generate_node,
        )

        workflow.add_node(
            "failure",
            self.failure_node,
        )

        workflow.add_edge(
            START,
            "retrieve",
        )

        workflow.add_conditional_edges(
            "retrieve",
            self.route_after_retrieval,
            {
                "generate": "generate",
                "retry": "retrieve",
                "failure": "failure",
            },
        )

        workflow.add_conditional_edges(
            "generate",
            self.route_after_generation,
            {
                "retry": "generate",
                "success": END,
                "failure": "failure",
            },
        )

        workflow.add_edge(
            "failure",
            END,
        )

        self.app = workflow.compile()

    def retrieve_node(
        self,
        state: RAGState,
    ) -> dict:

        try:
            evidence = self.fusion.retrieve(
                query=state["question"],
                document_ids=state["document_ids"],
            )

            return {
                "evidence": evidence,
                "error": None,
            }

        except Exception as exc:
            return {
                "error": f"Retrieval failed: {str(exc)}",
                "retrieval_retries": (
                    state["retrieval_retries"] + 1
                ),
            }

    def generate_node(
        self,
        state: RAGState,
    ) -> dict:

        try:
            answer = self.generator.generate(
                question=state["question"],
                evidence=state["evidence"],
            )

            return {
                "answer": answer,
                "error": None,
            }

        except Exception as exc:
            return {
                "error": f"Generation failed: {str(exc)}",
                "generation_retries": (
                    state["generation_retries"] + 1
                ),
            }

    def failure_node(
        self,
        state: RAGState,
    ) -> dict:

        error_message = (
            state.get("error")
            or "Unknown workflow error"
        )

        return {
            "answer": RAGAnswer(
                question=state["question"],
                answer=(
                    "I'm temporarily unable to complete this request "
                    "because a retrieval or generation service failed. "
                    "Please try again shortly."
                ),
                citations=[],
                insufficient_evidence=True,
            ),
            "error": error_message,
        }

    def route_after_retrieval(
        self,
        state: RAGState,
    ) -> str:

        print(
            "[DEBUG] Retrieval routing:",
            "error =", state.get("error"),
            "| retrieval_retries =",
            state["retrieval_retries"],
        )

        if state.get("error"):

            if state["retrieval_retries"] < 2:
                print("[DEBUG] Retrying retrieval.")
                return "retry"

            print(
                "[DEBUG] Retrieval retry limit reached."
            )

            return "failure"

        print(
            "[DEBUG] Retrieval succeeded. "
            "Proceeding to generation."
        )

        return "generate"

    def route_after_generation(
        self,
        state: RAGState,
    ) -> str:

        if state.get("error"):

            if state["generation_retries"] < 2:
                return "retry"

            return "failure"

        return "success"

    @staticmethod
    def _validate_document_ids(
        document_ids: list[str] | None,
    ) -> list[str] | None:

        if document_ids is None:
            return None

        if not isinstance(document_ids, list):
            raise TypeError(
                "document_ids must be a list of strings or None."
            )

        normalized_ids = list(
            dict.fromkeys(document_ids)
        )

        if any(
            not isinstance(document_id, str)
            or not document_id.strip()
            for document_id in normalized_ids
        ):
            raise ValueError(
                "document_ids must contain non-empty strings."
            )

        return normalized_ids

    def _create_initial_state(
        self,
        question: str,
        document_ids: list[str] | None = None,
    ) -> RAGState:

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                "The question cannot be empty."
            )

        validated_document_ids = (
            self._validate_document_ids(document_ids)
        )

        return {
            "question": question.strip(),
            "document_ids": validated_document_ids,
            "evidence": [],
            "answer": None,
            "error": None,
            "retrieval_retries": 0,
            "generation_retries": 0,
        }

    def run(
        self,
        question: str,
        document_ids: list[str] | None = None,
    ) -> RAGState:

        initial_state = self._create_initial_state(
            question=question,
            document_ids=document_ids,
        )

        return self.app.invoke(initial_state)

    def stream(
        self,
        question: str,
        cancel_event: Event | None = None,
        document_ids: list[str] | None = None,
    ):

        initial_state = self._create_initial_state(
            question=question,
            document_ids=document_ids,
        )

        for update in self.app.stream(
            initial_state,
            stream_mode="updates",
        ):

            yield update

            # Check cancellation before advancing to the next update.
            if (
                cancel_event is not None
                and cancel_event.is_set()
            ):
                break