from pathlib import Path
from uuid import uuid4
import asyncio
import json
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from multimodal_rag.app.lifecycle import operation_tracker
from fastapi import UploadFile, File
import logging
from multimodal_rag.app.config import settings
from threading import Event
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from multimodal_rag.app.config import PROJECT_ROOT
from api.schemas import UploadResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import secrets
from fastapi import Depends, Security
from fastapi.security import APIKeyHeader
from fastapi import FastAPI, Request, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
import logging
from multimodal_rag.app.security import (
    create_upload_path,
    get_max_upload_bytes,
    validate_pdf_metadata,
    validate_pdf_signature,
)
import time
from multimodal_rag.app.usage_tracking import (
    start_request_usage,
    finish_request_usage,
)
from starlette.types import ASGIApp, Receive, Scope, Send
from multimodal_rag.app.usage_tracking import (
    start_request_usage,
    finish_request_usage,
)
from multimodal_rag.app.observability import (
    request_context,
    measure_time,
)
from api.schemas import (
    AskRequest,
    AskResponse,
    CitationResponse,
    DocumentCreateRequest,
    DocumentResponse
)
from multimodal_rag.app.workflow.rag_workflow import MultimodalRAGWorkflow

logger = logging.getLogger(__name__)


SHUTDOWN_TIMEOUT_SECONDS = 30

# 1. Initialize the RAG workflow when the API starts
@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Initializing multimodal RAG workflow...")

    app.state.rag_workflow = None
    app.state.workflow_ready = False

    try:
        workflow = MultimodalRAGWorkflow()

        app.state.rag_workflow = workflow

        operation_tracker.start()
        app.state.workflow_ready = True

        logger.info(
            "Multimodal RAG workflow initialized successfully."
        )

        yield

    except Exception:
        logger.exception(
            "Multimodal RAG application lifecycle failed."
        )
        raise

    finally:
        # Immediately mark the application as not ready.
        app.state.workflow_ready = False

        logger.info(
            "Application shutdown started. Waiting for active operations..."
        )

        # Wait without blocking FastAPI's event loop.
        shutdown_completed = await asyncio.to_thread(
            operation_tracker.stop_accepting_and_wait,
            SHUTDOWN_TIMEOUT_SECONDS,
        )

        logger.info(
            "Shutdown wait completed=%s active_operations=%s",
            shutdown_completed,
            operation_tracker.active_operations,
        )

        if shutdown_completed:
            workflow = getattr(
                app.state,
                "rag_workflow",
                None,
            )

            if workflow is not None:
                try:
                    workflow.index_manager.genai_client.close()

                    logger.info(
                        "Google GenAI client closed successfully."
                    )

                except Exception:
                    logger.exception(
                        "Failed to close Google GenAI client."
                    )
        else:
            # Do not close a client that may still be in use.
            logger.error(
                "Shutdown timeout reached. Skipping Google GenAI "
                "client cleanup because operations are still active."
            )

        app.state.rag_workflow = None

        logger.info("Multimodal RAG API shutdown completed.")

# 2. Create the FastAPI application
is_production = settings.environment.lower() == "production"

app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
    debug=not is_production,
    docs_url=None if is_production else "/docs",
    redoc_url=None if is_production else "/redoc",
    openapi_url=None if is_production else "/openapi.json",
)



class RequestObservabilityMiddleware:
    """
    Attach a request ID to each HTTP request and log its duration.

    The request ID is also returned to the client in the
    X-Request-ID response header.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))

        # Reuse a supplied request ID, or generate a new one.
        supplied_id = headers.get(b"x-request-id")
        request_id = (
            supplied_id.decode("latin-1")
            if supplied_id
            else uuid4().hex
        )

        status_code = 500
        started = time.perf_counter()

        async def send_with_request_id(message):
            nonlocal status_code

            if message["type"] == "http.response.start":
                status_code = message["status"]

                response_headers = list(
                    message.get("headers", [])
                )

                response_headers.append(
                    (
                        b"x-request-id",
                        request_id.encode("latin-1"),
                    )
                )

                message["headers"] = response_headers

            await send(message)

        with request_context(request_id):
            try:
                with measure_time(
                    "http_request",
                    logger=logging.getLogger("rag.performance"),
                ):
                    await self.app(
                        scope,
                        receive,
                        send_with_request_id,
                    )
            finally:
                elapsed_ms = (
                    time.perf_counter() - started
                ) * 1000

                logger.info(
                    "http_request method=%s path=%s status=%s "
                    "elapsed_ms=%.2f request_id=%s",
                    scope.get("method"),
                    scope.get("path"),
                    status_code,
                    elapsed_ms,
                    request_id,
                )
# Initialize the rate limiter using the client's IP address.
limiter = Limiter(key_func=get_remote_address)

# Attach the limiter to the FastAPI application.
app.state.limiter = limiter

# Register the rate-limit exception handler.
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,
)

# CORS configuration
allowed_origins = settings.cors_allowed_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key"],
)
app.add_middleware(RequestObservabilityMiddleware)
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


async def verify_api_key(
    api_key: str | None = Security(api_key_header),
):
    expected_api_key = settings.api_key

    if not expected_api_key:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured.",
        )

    if (
        api_key is None
        or not secrets.compare_digest(api_key, expected_api_key)
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
        )

    return True

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):

    return JSONResponse(
        status_code=exc.status_code,
        headers=exc.headers,
        content={
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": str(exc.detail),
                "details": None,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):

    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "The request contains invalid data.",
                "details": jsonable_encoder(exc.errors()),
            }
        },
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(request: Request, exc: Exception):

    logger.exception(
        "Unhandled API exception on %s %s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected server error occurred.",
                "details": None,
            }
        },
    )

    
# Temporary in-memory document storage
documents = {}


# 3. Root endpoint
@app.get("/")
def home():
    return {
        "message": "Multimodal Knowledge Base API is running"
    }


# 4. Health-check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": "Multimodal RAG",
    }

@app.get("/ready")
def readiness_check(request: Request):

    workflow_ready = getattr(
        request.app.state,
        "workflow_ready",
        False,
    )

    workflow = getattr(
        request.app.state,
        "rag_workflow",
        None,
    )

    if (not workflow_ready or workflow is None
        or not operation_tracker.accepting_operations):
        raise HTTPException(
            status_code=503,
            detail="Multimodal RAG workflow is not ready.",
        )

    return {
        "status": "ready",
        "application": "Multimodal RAG",
        "workflow": "initialized",
        "active_operations": operation_tracker.active_operations,
    }

# 5. Ask-question endpoint connected to LangGraph
@app.post("/ask", response_model=AskResponse)
@limiter.limit("10/minute")
def ask_question(
    payload: AskRequest,
    request: Request,
    authenticated: bool = Depends(verify_api_key),
):

    if not operation_tracker.try_begin():
        raise HTTPException(
            status_code=503,
            detail="The application is shutting down. Please retry shortly.",
        )

    # Get the workflow initialized during application startup
    workflow = request.app.state.rag_workflow

    usage, usage_token, budget_token = start_request_usage()

    try:

        # Execute the complete multimodal RAG workflow
        with measure_time("rag_workflow_execution", logger):
            result = workflow.run(payload.question)

        # Extract the generated RAGAnswer object
        rag_answer = result.get("answer")

        if rag_answer is None:
            raise HTTPException(
                status_code=500,
                detail="The RAG workflow did not return an answer."
            )

        # Convert internal citations into API citation responses
        citations = [
            CitationResponse(
                source=citation.source_name,
                page=citation.page_number,
                evidence_id=citation.evidence_id
            )
            for citation in rag_answer.citations
        ]

        # Return the structured API response
        return AskResponse(
            question=rag_answer.question,
            answer=rag_answer.answer,
            citations=citations,
            insufficient_evidence=rag_answer.insufficient_evidence
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("RAG API request failed")

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the question."
        )
    
    finally:
        try:
            logger.info(
                "model_usage calls=%s input_tokens=%s output_tokens=%s "
                "total_tokens=%s estimated_cost_usd=%.8f calls_by_purpose=%s",
                usage.model_calls,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
                usage.estimated_cost_usd,
                usage.calls_by_purpose,
            )

            finish_request_usage(usage_token, budget_token)

        finally:
            operation_tracker.end()


# 6. Create document endpoint
@app.post("/documents", response_model=DocumentResponse)
def create_document(request: DocumentCreateRequest):

    document_id = f"doc_{len(documents) + 1:03d}"

    new_document = DocumentResponse(
        document_id=document_id,
        filename=request.filename,
        description=request.description
    )

    documents[document_id] = new_document

    return new_document


# 7. Retrieve all documents
@app.get("/documents", response_model=list[DocumentResponse])
def get_documents():

    return list(documents.values())


# 8. Upload and index a PDF document

import logging

logger = logging.getLogger(__name__)


# 8. Upload and index a PDF document
@app.post("/documents/upload", response_model=UploadResponse)
@limiter.limit("5/hour")
def upload_document(
    request: Request,
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_key),
):
    if not operation_tracker.try_begin():
        raise HTTPException(
            status_code=503,
            detail="The application is shutting down. Please retry shortly.",
        )

    saved_path = None
    indexing_succeeded = False

    try:
        # 1. Validate filename extension and declared MIME type.
        validate_pdf_metadata(file)

        # 2. Create a unique destination inside configured storage.
        saved_path = create_upload_path()

        total_size = 0
        first_bytes = b""
        max_file_size = get_max_upload_bytes()

        # 3. Stream the upload while enforcing the configured size limit.
        with saved_path.open("wb") as destination:
            while True:
                chunk = file.file.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > max_file_size:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "PDF exceeds the configured upload limit "
                            f"of {settings.max_upload_size_mb} MB."
                        ),
                    )

                if len(first_bytes) < 5:
                    needed = 5 - len(first_bytes)
                    first_bytes += chunk[:needed]

                destination.write(chunk)

        # 4. Validate the actual file signature.
        validate_pdf_signature(first_bytes)

        # 5. Reuse the index manager created during application startup.
        workflow = request.app.state.rag_workflow
        index_manager = workflow.index_manager

        # 6. Extract, embed and index the PDF.
        with measure_time("pdf_indexing", logger):
            indexing_result = index_manager.index_pdf(saved_path)

        indexing_succeeded = True

        # 7. Return the indexing result.
        return UploadResponse(
            filename=Path(file.filename or "document.pdf").name,
            document_id=indexing_result["document_id"],
            status=indexing_result["status"],
            text_chunks=indexing_result.get("text_chunks", 0),
            image_records=indexing_result.get("image_records", 0),
            warnings=indexing_result.get("warnings", []),
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception("PDF upload or indexing failed")

        raise HTTPException(
            status_code=500,
            detail="An error occurred while uploading or indexing the PDF.",
        )

    finally:
        try:
            # Delete incomplete uploads, but preserve successfully indexed files.
            if not indexing_succeeded and saved_path is not None:
                saved_path.unlink(missing_ok=True)

            file.file.close()

        finally:
            operation_tracker.end()


@app.post("/ask/stream")
@limiter.limit("5/minute")
async def stream_question(
    request: Request,
    payload: AskRequest,
    authenticated: bool = Depends(verify_api_key),
):
    cancel_event = Event()
    question = payload.question

    async def event_stream():
        # Register this request before accessing the workflow.
        if not operation_tracker.try_begin():
            yield (
                "event: error\n"
                'data: {"message": "The application is shutting down. Please retry shortly."}\n\n'
            )
            return

        usage = None
        usage_token = None
        budget_token = None

        try:
            workflow = request.app.state.rag_workflow

            usage, usage_token, budget_token = start_request_usage()

            yield (
                "event: status\n"
                'data: {"stage": "started", "message": "Question received"}\n\n'
            )

            # Create the synchronous LangGraph iterator.
            updates = iter(
                workflow.stream(
                    question,
                    cancel_event=cancel_event,
                )
            )

            while True:
                # Check for a disconnect before starting another worker call.
                if await request.is_disconnected():
                    logger.info("Streaming client disconnected.")
                    cancel_event.set()
                    break

                # Advance the synchronous iterator in a worker thread.
                worker_task = asyncio.create_task(
                    asyncio.to_thread(
                        next_stream_update,
                        updates,
                    )
                )

                try:
                    update = await asyncio.shield(worker_task)

                
                
                except asyncio.CancelledError:
                    logger.info(
                        "Streaming request cancelled. "
                        "Waiting for the in-flight workflow step to finish."
                    )

                    # Signal cancellation to the workflow.
                    cancel_event.set()

                    try:
                        # Wait for the in-flight worker task to finish.
                        # Shield prevents cancellation from propagating to that task.
                        await asyncio.shield(worker_task)

                    except Exception:
                        logger.exception(
                            "Workflow worker failed during stream cancellation."
                        )

                    finally:
                        # The worker has finished using the synchronous iterator.
                        # Close the generator to execute its finally blocks.
                        close_method = getattr(updates, "close", None)

                        if close_method is not None:
                            try:
                                await asyncio.to_thread(close_method)
                            except Exception:
                                logger.exception(
                                    "Failed to close workflow iterator during cancellation."
                                )

                    # Propagate cancellation after cleanup.
                    raise



                if update is _STREAM_END:
                    break

                # Each update maps a node name to its returned state changes.
                for node_name, node_output in update.items():

                    if node_name == "retrieve":
                        evidence = node_output.get("evidence")

                        if evidence is not None:
                            event_data = {
                                "stage": "retrieval_completed",
                                "evidence_count": len(evidence),
                            }
                        else:
                            event_data = {
                                "stage": "retrieval_attempt_completed",
                                "error": node_output.get("error"),
                                "retry_count": node_output.get(
                                    "retrieval_retries"
                                ),
                            }

                        yield (
                            "event: workflow_update\n"
                            f"data: {json.dumps(event_data)}\n\n"
                        )

                    elif node_name == "generate":
                        error = node_output.get("error")

                        event_data = {
                            "stage": (
                                "generation_attempt_failed"
                                if error
                                else "generation_completed"
                            ),
                            "error": error,
                            "retry_count": node_output.get(
                                "generation_retries"
                            ),
                        }

                        yield (
                            "event: workflow_update\n"
                            f"data: {json.dumps(event_data)}\n\n"
                        )

                        answer = node_output.get("answer")

                        if answer is not None:
                            citations = [
                                {
                                    "source": citation.source_name,
                                    "page": citation.page_number,
                                    "evidence_id": citation.evidence_id,
                                }
                                for citation in answer.citations
                            ]

                            event_data = {
                                "question": question,
                                "answer": answer.answer,
                                "citations": citations,
                                "insufficient_evidence": (
                                    answer.insufficient_evidence
                                ),
                            }

                            yield (
                                "event: completed\n"
                                f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                            )

                    elif node_name == "failure":
                        answer = node_output.get("answer")

                        if answer is not None:
                            event_data = {
                                "question": question,
                                "answer": answer.answer,
                                "citations": [
                                    {
                                        "source": citation.source_name,
                                        "page": citation.page_number,
                                        "evidence_id": citation.evidence_id,
                                    }
                                    for citation in answer.citations
                                ],
                                "insufficient_evidence": (
                                    answer.insufficient_evidence
                                ),
                            }

                            yield (
                                "event: completed\n"
                                f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                            )

            # Reached only when the iterator finishes normally.
            yield (
                "event: stream_finished\n"
                'data: {"message": "Workflow execution finished"}\n\n'
            )

        except asyncio.CancelledError:
            # Preserve cancellation after the worker has been handled.
            logger.info("Streaming request cancellation completed.")
            raise

        except Exception:
            logger.exception("Streaming RAG request failed")

            event_data = {
                "message": "An error occurred while processing the question."
            }

            yield (
                "event: error\n"
                f"data: {json.dumps(event_data)}\n\n"
            )

        finally:
            cancel_event.set()

            try:
                if usage is not None:
                    logger.info(
                        "stream_model_usage calls=%s input_tokens=%s "
                        "output_tokens=%s total_tokens=%s "
                        "estimated_cost_usd=%.8f calls_by_purpose=%s",
                        usage.model_calls,
                        usage.prompt_tokens,
                        usage.completion_tokens,
                        usage.total_tokens,
                        usage.estimated_cost_usd,
                        usage.calls_by_purpose,
                    )

                if usage_token is not None and budget_token is not None:
                    finish_request_usage(usage_token, budget_token)

            finally:
                operation_tracker.end()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


_STREAM_END = object()


def next_stream_update(iterator):
    """
    Advance the synchronous workflow iterator safely.

    Return a sentinel instead of allowing StopIteration
    to escape into an asyncio Future.
    """
    try:
        return next(iterator)
    except StopIteration:
        return _STREAM_END