import hashlib
import threading
import uuid

from pathlib import Path

import streamlit as st

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

from chatbot_backend import (
    chatbot,
    detect_train_query,
)

from multimodal_rag.app.config import settings

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Aurora AI",
    page_icon="aurora-favicon.svg",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown(
    r"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * {
        font-family: 'Inter', sans-serif;
        box-sizing: border-box;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: #0B1120 !important;
    }

    [data-testid="stAppViewContainer"] {
        overflow-x: hidden;
        position: relative;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    .block-container {
        max-width: 1200px !important;
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
    }

    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        width: 700px;
        height: 700px;
        top: -250px;
        left: -250px;
        background: radial-gradient(
            circle,
            rgba(37, 99, 235, 0.28) 0%,
            rgba(37, 99, 235, 0.12) 30%,
            transparent 70%
        );
        filter: blur(70px);
        pointer-events: none;
        z-index: 0;
    }

    [data-testid="stAppViewContainer"]::after {
        content: "";
        position: fixed;
        width: 800px;
        height: 800px;
        right: -300px;
        bottom: -350px;
        background: radial-gradient(
            circle,
            rgba(124, 58, 237, 0.25) 0%,
            rgba(139, 92, 246, 0.12) 35%,
            transparent 70%
        );
        filter: blur(90px);
        pointer-events: none;
        z-index: 0;
    }

    .aurora-center {
        position: fixed;
        width: 500px;
        height: 500px;
        top: 35%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: radial-gradient(
            circle,
            rgba(6, 182, 212, 0.09) 0%,
            rgba(59, 130, 246, 0.07) 30%,
            transparent 70%
        );
        filter: blur(80px);
        pointer-events: none;
        z-index: 0;
    }

    .main-content {
        position: relative;
        z-index: 2;
    }

    .brand {
        text-align: center;
        margin-top: 20px;
        margin-bottom: 18px;
    }

    .brand-icon {
        display: inline-flex;
        width: 52px;
        height: 52px;
        align-items: center;
        justify-content: center;
        border-radius: 16px;
        background: linear-gradient(135deg, #2563EB, #7C3AED);
        box-shadow:
            0 0 30px rgba(59, 130, 246, 0.35),
            0 0 60px rgba(124, 58, 237, 0.18);
        color: white;
        font-size: 25px;
        font-weight: 800;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 15px;
        border: 1px solid rgba(96, 165, 250, 0.22);
        border-radius: 999px;
        background: rgba(30, 41, 59, 0.55);
        backdrop-filter: blur(12px);
        color: #CBD5E1;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.4px;
        box-shadow:
            inset 0 1px rgba(255, 255, 255, 0.04),
            0 8px 30px rgba(0, 0, 0, 0.15);
    }

    .badge-spark {
        color: #60A5FA;
        font-size: 14px;
    }

    .hero {
        text-align: center;
        margin-top: 22px;
        margin-bottom: 38px;
    }

    .hero h1 {
        margin: 0 auto;
        max-width: 950px;
        font-size: clamp(42px, 6vw, 78px);
        line-height: 1.02;
        font-weight: 800;
        letter-spacing: -3px;
        background: linear-gradient(
            100deg,
            #FFFFFF 5%,
            #BFDBFE 35%,
            #60A5FA 60%,
            #A78BFA 85%,
            #C4B5FD 100%
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 50px rgba(96, 165, 250, 0.08);
    }

    .hero p {
        max-width: 680px;
        margin: 22px auto 0;
        color: #94A3B8;
        font-size: 17px;
        line-height: 1.7;
    }

    .status-wrapper {
        display: flex;
        justify-content: center;
        margin-top: 24px;
    }

    .online-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: #94A3B8;
        font-size: 12px;
        font-weight: 500;
        padding: 7px 13px;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.10);
        backdrop-filter: blur(10px);
    }

    .online-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #22C55E;
        box-shadow:
            0 0 7px #22C55E,
            0 0 15px rgba(34, 197, 94, 0.55);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% {
            opacity: 1;
            transform: scale(1);
        }

        50% {
            opacity: 0.55;
            transform: scale(0.85);
        }
    }

    .ai-card {
        max-width: 900px;
        margin: 0 auto;
        padding: 28px;
        border-radius: 26px;
        background: linear-gradient(
            145deg,
            rgba(15, 23, 42, 0.92),
            rgba(15, 23, 42, 0.72)
        );
        border: 1px solid rgba(148, 163, 184, 0.12);
        box-shadow:
            0 25px 80px rgba(0, 0, 0, 0.40),
            inset 0 1px rgba(255, 255, 255, 0.035);
        backdrop-filter: blur(20px);
        position: relative;
    }

    .ai-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 12%;
        right: 12%;
        height: 1px;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(96, 165, 250, 0.65),
            rgba(167, 139, 250, 0.65),
            transparent
        );
        opacity: 0.65;
    }

    .card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 24px;
    }

    .card-title {
        display: flex;
        align-items: center;
        gap: 12px;
        color: #F8FAFC;
        font-size: 15px;
        font-weight: 700;
    }

    .card-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 11px;
        background: linear-gradient(
            135deg,
            rgba(37, 99, 235, 0.25),
            rgba(124, 58, 237, 0.25)
        );
        border: 1px solid rgba(96, 165, 250, 0.16);
        color: #93C5FD;
    }

    .card-label {
        color: #64748B;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    [data-testid="stTextArea"] {
        margin-bottom: 10px;
    }

    [data-testid="stTextArea"] textarea {
        background: rgba(2, 6, 23, 0.72) !important;
        color: #E2E8F0 !important;
        border: 1px solid rgba(100, 116, 139, 0.22) !important;
        border-radius: 17px !important;
        padding: 18px !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
        resize: vertical !important;
        min-height: 130px !important;
        transition:
            border 0.25s ease,
            box-shadow 0.25s ease,
            background 0.25s ease !important;
    }

    [data-testid="stTextArea"] textarea:focus {
        border-color: rgba(96, 165, 250, 0.65) !important;
        background: rgba(2, 6, 23, 0.88) !important;
        box-shadow:
            0 0 0 3px rgba(59, 130, 246, 0.08),
            0 0 30px rgba(59, 130, 246, 0.08) !important;
    }

    [data-testid="stTextArea"] textarea::placeholder {
        color: #475569 !important;
    }

    .stButton > button {
        width: 100%;
        min-height: 50px;
        border: none !important;
        border-radius: 14px !important;
        background: linear-gradient(
            100deg,
            #2563EB 0%,
            #4F46E5 50%,
            #7C3AED 100%
        ) !important;
        color: white !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        letter-spacing: 0.2px;
        box-shadow:
            0 10px 30px rgba(59, 130, 246, 0.20),
            inset 0 1px rgba(255, 255, 255, 0.14) !important;
        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease,
            filter 0.2s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        filter: brightness(1.08);
        box-shadow:
            0 14px 38px rgba(59, 130, 246, 0.30),
            0 0 35px rgba(124, 58, 237, 0.16),
            inset 0 1px rgba(255, 255, 255, 0.18) !important;
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    .response-card {
        max-width: 900px;
        margin: 22px auto 0;
        padding: 24px 26px;
        border-radius: 22px;
        background: linear-gradient(
            145deg,
            rgba(15, 23, 42, 0.90),
            rgba(15, 23, 42, 0.65)
        );
        border: 1px solid rgba(96, 165, 250, 0.12);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
        backdrop-filter: blur(18px);
    }

    .response-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 16px;
        color: #93C5FD;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .response-icon {
        width: 28px;
        height: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 9px;
        background: linear-gradient(
            135deg,
            rgba(37, 99, 235, 0.30),
            rgba(124, 58, 237, 0.30)
        );
        color: #BFDBFE;
    }

    .response-text {
        color: #CBD5E1;
        font-size: 15px;
        line-height: 1.75;
        white-space: pre-wrap;
    }

    .footer {
        text-align: center;
        margin-top: 34px;
        color: #475569;
        font-size: 11px;
        letter-spacing: 0.3px;
    }

    .footer span {
        color: #64748B;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 1rem !important;
        }

        .hero h1 {
            font-size: 42px;
            letter-spacing: -1.8px;
        }

        .hero p {
            font-size: 14px;
            padding: 0 10px;
        }

        .ai-card {
            padding: 20px;
            border-radius: 21px;
        }

        .response-card {
            padding: 20px;
        }

        .brand-icon {
            width: 46px;
            height: 46px;
        }
    }

    .chat-section-header {
        max-width: 900px;
        margin: 35px auto 20px auto;
        padding: 0 5px;
    }

    .chat-section-title {
        color: #F8FAFC;
        font-size: 19px;
        font-weight: 700;
        letter-spacing: -0.3px;
    }

    .chat-section-description {
        margin-top: 5px;
        color: #64748B;
        font-size: 12px;
    }

    [data-testid="stChatMessage"] {
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
        margin-bottom: 12px;
        padding: 15px 18px !important;
        border-radius: 18px;
        background: linear-gradient(
            145deg,
            rgba(15, 23, 42, 0.72),
            rgba(15, 23, 42, 0.48)
        ) !important;
        border: 1px solid rgba(148, 163, 184, 0.09);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.16);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
    }

    [data-testid="stChatMessage"] p {
        color: #CBD5E1 !important;
        font-size: 15px !important;
        line-height: 1.75 !important;
    }

    [data-testid="stChatInput"] {
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
        padding-bottom: 10px;
    }

    [data-testid="stChatInput"] > div {
        background: rgba(2, 6, 23, 0.90) !important;
        border: 1px solid rgba(96, 165, 250, 0.22) !important;
        border-radius: 18px !important;
        box-shadow:
            0 10px 40px rgba(0, 0, 0, 0.30),
            0 0 25px rgba(59, 130, 246, 0.05);
    }

    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: #E2E8F0 !important;
        border: none !important;
        font-size: 15px !important;
        line-height: 1.5 !important;
        padding: 15px 18px !important;
    }

    [data-testid="stChatInput"] > div:focus-within {
        border: 1px solid rgba(96, 165, 250, 0.60) !important;
        box-shadow:
            0 0 0 3px rgba(59, 130, 246, 0.08),
            0 0 35px rgba(59, 130, 246, 0.12) !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #475569 !important;
    }

    [data-testid="stChatInput"] button {
        color: #93C5FD !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stChatInput"] button:hover {
        color: #C4B5FD !important;
        transform: scale(1.08);
    }

    [data-testid="stSpinner"] {
        color: #93C5FD !important;
    }

    @media (max-width: 768px) {
        .chat-section-header {
            margin-top: 25px;
        }

        [data-testid="stChatMessage"] {
            border-radius: 15px;
            padding: 12px 14px !important;
        }

        [data-testid="stChatMessage"] p {
            font-size: 14px !important;
        }

        [data-testid="stChatInput"] {
            width: 100%;
        }
    }

    /* ======================================================
   CHAT INPUT BOTTOM CONTAINER
   ====================================================== */

[data-testid="stBottom"] {
    background:
        linear-gradient(
            180deg,
            rgba(11, 17, 32, 0.00) 0%,
            rgba(11, 17, 32, 0.96) 25%,
            #0B1120 100%
        ) !important;
}

[data-testid="stBottom"] > div {
    background: transparent !important;
}

[data-testid="stBottomBlockContainer"] {
    background: transparent !important;
    border-top: 1px solid rgba(148, 163, 184, 0.06);
}

/* Preserve the existing input box styling */
[data-testid="stChatInput"] {
    background: transparent !important;
}

[data-testid="stChatInput"] > div {
    background: rgba(2, 6, 23, 0.90) !important;
}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. CACHED MULTIMODAL RAG RUNTIME
# ============================================================

@st.cache_resource
def get_rag_runtime():
    """
    Create one shared RAG workflow for the Streamlit process.

    The workflow owns the IndexManager, Chroma clients,
    embedding clients, and answer generator.

    The lock prevents simultaneous indexing and retrieval
    operations from competing for resources.
    """
      # Import the heavier RAG components only when required.
    from multimodal_rag.app.workflow.rag_workflow import (
        MultimodalRAGWorkflow,
    )

    return {
        "workflow": MultimodalRAGWorkflow(),
        "lock": threading.RLock(),
    }


# ============================================================
# 4. SESSION INITIALIZATION
# ============================================================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": (
                "Nice to meet you! How can I help you today?"
            ),
        }
    ]


if "thread_id" not in st.session_state:

    st.session_state.thread_id = str(
        uuid.uuid4()
    )


if "rag_session_id" not in st.session_state:

    st.session_state.rag_session_id = str(
        uuid.uuid4()
    )


if "indexed_documents" not in st.session_state:

    st.session_state.indexed_documents = []


if "uploaded_hashes" not in st.session_state:

    st.session_state.uploaded_hashes = {}


# ============================================================
# 5. LANGGRAPH CONFIGURATION
# ============================================================

config = {
    "configurable": {
        "thread_id": st.session_state.thread_id,
    }
}


# ============================================================
# 6. PDF UPLOAD AND INDEXING
# ============================================================

def process_uploaded_pdf(uploaded_file):
    """
    Validate and index one uploaded PDF.

    The generated path is unique and stored under the current
    Streamlit session's upload directory.
    """

    filename = uploaded_file.name

    # --------------------------------------------------------
    # A. Validate extension.
    # --------------------------------------------------------

    if Path(filename).suffix.lower() != ".pdf":

        raise ValueError(
            f"{filename}: only PDF files are supported."
        )

    # --------------------------------------------------------
    # B. Validate file size.
    # --------------------------------------------------------

    file_buffer = uploaded_file.getbuffer()

    file_size = len(file_buffer)

    max_size = (
        settings.max_upload_size_mb
        * 1024
        * 1024
    )

    if file_size == 0:

        raise ValueError(
            f"{filename}: the uploaded file is empty."
        )

    if file_size > max_size:

        raise ValueError(
            f"{filename}: file size exceeds "
            f"{settings.max_upload_size_mb} MB."
        )

    # --------------------------------------------------------
    # C. Validate PDF signature.
    # --------------------------------------------------------

    if bytes(file_buffer[:5]) != b"%PDF-":

        raise ValueError(
            f"{filename}: the file does not have a valid PDF signature."
        )

    # --------------------------------------------------------
    # D. Calculate content hash.
    # --------------------------------------------------------

    content_hash = hashlib.sha256(
        file_buffer
    ).hexdigest()

    # --------------------------------------------------------
    # E. Skip duplicate uploads in the same session.
    # --------------------------------------------------------

    existing_document_id = (
        st.session_state.uploaded_hashes.get(
            content_hash
        )
    )

    if existing_document_id:

        return {
            "status": "skipped",
            "filename": filename,
            "document_id": existing_document_id,
            "message": (
                f"{filename} is already indexed in this session."
            ),
        }

    # --------------------------------------------------------
    # F. Create a session-specific upload directory.
    # --------------------------------------------------------

    upload_root = settings.upload_dir.resolve()

    session_directory = (
        upload_root
        / st.session_state.rag_session_id
    ).resolve()

    if not session_directory.is_relative_to(upload_root):

        raise ValueError(
            "Invalid upload directory."
        )

    session_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Use a generated filename instead of the supplied filename.
    # This avoids path traversal and filename collisions.

    destination = (
        session_directory
        / f"{uuid.uuid4().hex}.pdf"
    ).resolve()

    if not destination.is_relative_to(session_directory):

        raise ValueError(
            "Invalid PDF destination."
        )

    # --------------------------------------------------------
    # G. Save the PDF.
    # --------------------------------------------------------

    with destination.open("wb") as output_file:

        output_file.write(file_buffer)

    # --------------------------------------------------------
    # H. Index the PDF using the session namespace.
    # --------------------------------------------------------

    runtime = get_rag_runtime()

    try:

        with runtime["lock"]:

            result = runtime["workflow"].index_manager.index_pdf(
                pdf_path=destination,
                namespace_id=st.session_state.rag_session_id,
            )

        document_id = result["document_id"]

        document_information = {
            "document_id": document_id,
            "source_name": filename,
            "source_path": str(destination),
            "content_hash": content_hash,
        }

        st.session_state.indexed_documents.append(
            document_information
        )

        st.session_state.uploaded_hashes[
            content_hash
        ] = document_id

        return {
            "status": result["status"],
            "filename": filename,
            "document_id": document_id,
            "text_chunks": result.get("text_chunks", 0),
            "image_records": result.get("image_records", 0),
            "warnings": result.get("warnings", []),
            "message": (
                f"{filename} has been indexed successfully."
            ),
        }

    except Exception:

        # Remove the uploaded file if indexing fails.
        destination.unlink(
            missing_ok=True
        )

        raise


# ============================================================
# 7. NORMAL GEMINI / RAILRADAR STREAMING
# ============================================================

def stream_chatbot_response(
    user_prompt: str,
    response_placeholder,
):
    """
    Use the existing LangGraph chatbot.

    This preserves:
        - Gemini streaming
        - RailRadar routing
        - LangGraph thread memory
    """

    full_response = ""

    stream = chatbot.stream(
        {
            "messages": [
                HumanMessage(
                    content=user_prompt
                )
            ]
        },
        config=config,
        stream_mode="messages",
    )

    # --------------------------------------------------------
    # Process streamed model chunks.
    # --------------------------------------------------------

    for chunk, metadata in stream:

        chunk_content = chunk.content

        if isinstance(chunk_content, str):

            token = chunk_content

        elif isinstance(chunk_content, list):

            text_parts = []

            for item in chunk_content:

                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                ):

                    text_parts.append(
                        item.get("text", "")
                    )

            token = "".join(text_parts)

        else:

            token = ""

        if token:

            full_response += token

            response_placeholder.markdown(
                full_response + "▌"
            )

    # --------------------------------------------------------
    # Fallback for direct node responses.
    # --------------------------------------------------------

    if not full_response.strip():

        state_snapshot = chatbot.get_state(
            config
        )

        final_messages = (
            state_snapshot.values.get(
                "messages",
                [],
            )
        )

        final_ai_message = next(
            (
                message
                for message in reversed(final_messages)
                if isinstance(message, AIMessage)
            ),
            None,
        )

        if final_ai_message is not None:

            content = final_ai_message.content

            if isinstance(content, str):

                full_response = content

            elif isinstance(content, list):

                text_parts = []

                for item in content:

                    if (
                        isinstance(item, dict)
                        and item.get("type") == "text"
                    ):

                        text_parts.append(
                            item.get("text", "")
                        )

                full_response = "".join(text_parts)

    if not full_response.strip():

        full_response = (
            "Aurora AI could not generate a response. "
            "Please try again."
        )

    response_placeholder.markdown(
        full_response
    )

    return full_response


# ============================================================
# 8. MULTIMODAL RAG RESPONSE
# ============================================================

def run_document_question(
    question: str,
) -> str:
    """
    Run the multimodal RAG workflow using only document IDs
    indexed in the current Streamlit session.
    """

    document_ids = [
        item["document_id"]
        for item in st.session_state.indexed_documents
    ]

    if not document_ids:

        return (
            "There are no indexed documents in this session. "
            "Please upload a PDF first."
        )

    runtime = get_rag_runtime()

    with runtime["lock"]:

        result = runtime["workflow"].run(
            question=question,
            document_ids=document_ids,
        )

    answer_object = result.get("answer")

    if answer_object is None:

        return (
            "I couldn't generate an answer from the "
            "uploaded documents. Please try again."
        )

    response = answer_object.answer

    # --------------------------------------------------------
    # Add verified citations.
    # --------------------------------------------------------

    citations = answer_object.citations

    if citations:

        citation_lines = [
            "\n\n**Sources**"
        ]

        for citation in citations:

            safe_filename = (
                citation.source_name.replace("`", "'")
            )

            citation_lines.append(
                f"- `{safe_filename}`, "
                f"page {citation.page_number}"
            )

        response += "\n".join(
            citation_lines
        )

    return response


# ============================================================
# 9. AURORA APPLICATION CONTENT
# ============================================================

st.markdown(
    '<div class="main-content">',
    unsafe_allow_html=True,
)


# ============================================================
# BRAND
# ============================================================

st.markdown(
    """
    <div class="brand">
        <div class="brand-icon">✦</div>
        <div class="header">Aurora AI</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HERO BADGE
# ============================================================

st.markdown(
    """
    <div style="text-align:center;">
        <div class="hero-badge">
            <span class="badge-spark">✦</span>
            NEXT-GENERATION AI ASSISTANT
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>
            Multimodal RAG System with RailRadar API.
        </h1>
        <p>
            AI workspace designed to ask about railway services and able to answer questions from the uploaded PDFs.
        </p>
        <div class="status-wrapper">
            <div class="online-status">
                <span class="online-dot"></span>
                AI system online
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CHAT HEADER
# ============================================================

st.markdown(
    """
    <div class="chat-section-header">
              <div class="chat-section-description"> <span class="badge-spark">✦</span>
            Your intelligent conversational workspace
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CURRENT SESSION DOCUMENT STATUS
# ============================================================

if st.session_state.indexed_documents:

    document_count = len(
        st.session_state.indexed_documents
    )

    st.caption(
        f"📄 {document_count} PDF document(s) indexed "
        "in this conversation."
    )


# ============================================================
# DISPLAY EXISTING CONVERSATION
# ============================================================

for message in st.session_state.chat_history:

    role = message["role"]

    content = message["content"]

    with st.chat_message(role):

        st.markdown(content)


# ============================================================
# CHAT INPUT WITH PDF ATTACHMENTS
# ============================================================

submitted = st.chat_input(
    "Ask Aurora anything...",
    accept_file="multiple",
    file_type=["pdf"],
    max_upload_size=settings.max_upload_size_mb,
)


# ============================================================
# PROCESS NEW MESSAGE
# ============================================================

if submitted:

    user_prompt = (
        submitted.text or ""
    ).strip()

    uploaded_files = (
        submitted.files or []
    )

    upload_results = []

    upload_errors = []

    # --------------------------------------------------------
    # A. PROCESS ATTACHED PDFs FIRST.
    # --------------------------------------------------------

    if uploaded_files:

        with st.spinner(
            "Aurora is processing your PDF attachment(s)..."
        ):

            for uploaded_file in uploaded_files:

                try:

                    upload_result = process_uploaded_pdf(
                        uploaded_file
                    )

                    upload_results.append(
                        upload_result
                    )

                except Exception as exc:

                    upload_errors.append(
                        f"{uploaded_file.name}: {str(exc)}"
                    )

    # --------------------------------------------------------
    # B. PREPARE USER MESSAGE.
    # --------------------------------------------------------

    user_display = user_prompt

    if uploaded_files:

        attachment_names = "\n".join(
            f"- 📎 {file.name}"
            for file in uploaded_files
        )

        if user_display:

            user_display += "\n\n"

        user_display += (
            "**Attachments:**\n"
            + attachment_names
        )

    if not user_display:

        user_display = (
            "📎 PDF attachment(s) submitted."
        )

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_display,
        }
    )

    with st.chat_message("user"):

        st.markdown(user_display)

    # --------------------------------------------------------
    # C. GENERATE ASSISTANT RESPONSE.
    # --------------------------------------------------------

        # --------------------------------------------------------
    # C. GENERATE ASSISTANT RESPONSE.
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        full_response = ""

        try:

            # ------------------------------------------------
            # Upload-only submission.
            # ------------------------------------------------

            if not user_prompt:

                messages = []

                for result in upload_results:

                    messages.append(
                        f"✅ {result['message']}"
                    )

                    if result.get("text_chunks") is not None:

                        messages.append(
                            f"   Text chunks: {result['text_chunks']}"
                        )

                        messages.append(
                            f"   Image records: {result['image_records']}"
                        )

                    if result.get("warnings"):

                        for warning in result["warnings"]:

                            messages.append(
                                f"   Warning: {warning}"
                            )

                for error in upload_errors:

                    messages.append(
                        f"⚠️ {error}"
                    )

                if not messages:

                    full_response = (
                        "No PDF was successfully indexed. "
                        "Please check the attachment and try again."
                    )

                else:

                    full_response = (
                        "### PDF processing results\n\n"
                        + "\n\n".join(messages)
                        + "\n\nYou can now ask questions about "
                          "the indexed document(s)."
                    )

            # ------------------------------------------------
            # A question was submitted.
            # ------------------------------------------------

            else:

                # RailRadar always takes priority.
                is_train_query, _, _ = detect_train_query(
                    user_prompt
                )

                with st.spinner(
                    "Aurora is thinking..."
                ):

                    if is_train_query:

                        full_response = stream_chatbot_response(
                            user_prompt,
                            response_placeholder,
                        )

                    elif st.session_state.indexed_documents:

                        response_placeholder.markdown(
                            "Searching your uploaded documents..."
                        )

                        full_response = run_document_question(
                            user_prompt
                        )

                    else:

                        full_response = stream_chatbot_response(
                            user_prompt,
                            response_placeholder,
                        )

            # ------------------------------------------------
            # DISPLAY UPLOAD RESULTS
            # ------------------------------------------------

            upload_notices = []

            for result in upload_results:

                upload_notices.append(
                    f"📄 {result['message']}"
                )

            for error in upload_errors:

                upload_notices.append(
                    f"⚠️ {error}"
                )

            if upload_notices:

                full_response = (
                    "\n\n".join(upload_notices)
                    + "\n\n"
                    + full_response
                )

            # ------------------------------------------------
            # DISPLAY AND SAVE FINAL RESPONSE
            # ------------------------------------------------

            response_placeholder.markdown(
                full_response
            )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": full_response,
                }
            )

        except Exception as exc:

            error_message = (
                "⚠️ **Unable to complete your request.**\n\n"
                "Please check your AI backend, API credentials, "
                "and network connection, then try again.\n\n"
                f"`{str(exc)}`"
            )

            response_placeholder.markdown(
                error_message
            )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )