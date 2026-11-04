import json
import re
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from railway_service import get_train_information


load_dotenv()

api_key= os.getenv("GOOGLE_API_KEY")

# ============================================================
# 1. GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key
)


# ============================================================
# 2. LANGGRAPH STATE
# ============================================================

class ChatState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages,
    ]


# ============================================================
# 3. TRAIN QUERY DETECTION
# ============================================================

def detect_train_query(message: str):
    """
    Determine whether a message is asking about a train.

    Returns:
        is_train_query: bool
        train_number: str | None
        info_type: str
    """

    message_lower = message.lower()

    train_keywords = [
        "train",
        "railway",
        "rail",
        "pnr",
        "next halt",
        "next station",
        "train status",
        "train location",
        "train details",
        "running status",
        "arrival",
        "departure",
    ]

    live_keywords = [
        "live",
        "status",
        "location",
        "delay",
        "next halt",
        "running",
        "where is",
    ]

    is_train_query = any(
        keyword in message_lower
        for keyword in train_keywords
    )

    train_number_match = re.search(
        r"\b\d{5}\b",
        message,
    )

    train_number = (
        train_number_match.group(0)
        if train_number_match
        else None
    )

    info_type = (
        "live"
        if any(
            keyword in message_lower
            for keyword in live_keywords
        )
        else "details"
    )

    return (
        is_train_query,
        train_number,
        info_type,
    )


# ============================================================
# 4. GEMINI RESPONSE GENERATION
# ============================================================

def stream_gemini_response(
    messages: list[BaseMessage],
) -> AIMessage:
    """
    Stream Gemini internally and combine the generated chunks
    into one AIMessage for the LangGraph state.
    """

    full_response = ""

    for chunk in llm.stream(messages):

        content = chunk.content

        if isinstance(content, str):

            full_response += content

        elif isinstance(content, list):

            for item in content:

                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                ):

                    full_response += item.get(
                        "text",
                        "",
                    )

    return AIMessage(
        content=full_response
    )


# ============================================================
# 5. CHAT NODE
# ============================================================

def chat_node(state: ChatState):

    messages = state["messages"]

    latest_human_message = next(
        (
            message
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        ),
        None,
    )

    if latest_human_message is None:

        return {
            "messages": [
                AIMessage(
                    content="Please enter a message."
                )
            ]
        }

    user_message = latest_human_message.content

    if not isinstance(user_message, str):

        return {
            "messages": [
                AIMessage(
                    content=(
                        "I couldn't process that message. "
                        "Please send your question as text."
                    )
                )
            ]
        }

    # --------------------------------------------------------
    # A. RAILRADAR ROUTING
    # --------------------------------------------------------

    is_train_query, train_number, info_type = (
        detect_train_query(user_message)
    )

    if is_train_query:

        if not train_number:

            return {
                "messages": [
                    AIMessage(
                        content=(
                            "Please provide the five-digit train "
                            "number so I can retrieve its details."
                        )
                    )
                ]
            }

        try:

            train_data = get_train_information(
                train_number=train_number,
                info_type=info_type,
            )

            if train_data is None:

                return {
                    "messages": [
                        AIMessage(
                            content=(
                                "I couldn't retrieve information "
                                f"for train {train_number}. "
                                "Please try again."
                            )
                        )
                    ]
                }

            train_data_json = json.dumps(
                train_data,
                indent=2,
                default=str,
            )

            rail_system_message = SystemMessage(
                content=(
                    "You are Aurora AI, answering a question "
                    "using RailRadar train information.\n\n"
                    "Use only the supplied RailRadar data.\n"
                    "Do not invent train timings, locations, "
                    "platforms, delays, or other details.\n"
                    "If the supplied data does not contain "
                    "the requested information, say so clearly.\n"
                    "Present the answer in a clear and readable format."
                )
            )

            rail_human_message = HumanMessage(
                content=(
                    f"User question:\n{user_message}\n\n"
                    f"RailRadar information:\n{train_data_json}"
                )
            )

            response = stream_gemini_response(
                [
                    rail_system_message,
                    rail_human_message,
                ]
            )

            return {
                "messages": [response]
            }

        except Exception as exc:

            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I encountered a problem while "
                            "retrieving RailRadar information. "
                            "Please try again shortly.\n\n"
                            f"Details: {str(exc)}"
                        )
                    )
                ]
            }

    # --------------------------------------------------------
    # B. NORMAL GEMINI CONVERSATION
    # --------------------------------------------------------

    response = stream_gemini_response(
        messages
    )

    return {
        "messages": [response]
    }


# ============================================================
# 6. BUILD LANGGRAPH WORKFLOW
# ============================================================

graph = StateGraph(ChatState)

graph.add_node(
    "chat_node",
    chat_node,
)

graph.add_edge(
    START,
    "chat_node",
)

graph.add_edge(
    "chat_node",
    END,
)


# ============================================================
# 7. CONVERSATION MEMORY
# ============================================================

checkpoint = MemorySaver()

chatbot = graph.compile(
    checkpointer=checkpoint,
)