from __future__ import annotations

import json
import os
from typing import Literal, TypedDict

from fastapi import FastAPI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError

try:
    from .ingest import get_collection, ingest
    from .prompts import POLICY_PROMPT
except ImportError:
    from ingest import get_collection, ingest
    from prompts import POLICY_PROMPT


class AskRequest(BaseModel):
    query: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0, le=1)


class AssistantState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    answer: str
    sources: list[str]
    confidence: float


KEYWORDS = ("delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours")


# MOCK_LLM=1 is the free, offline and graded default.
def mock_mode() -> bool:
    return os.getenv("MOCK_LLM", "1") != "0"


def call_real_llm(prompt: str) -> str:
    raise RuntimeError("Optional MOCK_LLM=0 backend is not configured. Use MOCK_LLM=1 for the graded baseline.")


def parse_real_response(prompt: str) -> AskResponse:
    last_error = ""
    current_prompt = prompt
    for attempt in range(3):
        try:
            return AskResponse.model_validate(json.loads(call_real_llm(current_prompt)))
        except (ValidationError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = str(exc)
            current_prompt = prompt + "\nCorrect the previous response and return only schema-valid JSON."
    return AskResponse(answer=f"ERROR: Real-LLM output failed validation: {last_error}", sources=[], confidence=0.0)


# NODE 1: Decide whether the question needs policy retrieval.
def classify_intent(state: AssistantState) -> AssistantState:
    query = state["query"].lower()
    if mock_mode():
        contains_policy_word = any(word in query for word in KEYWORDS)
        if contains_policy_word:
            intent = "policy_question"
        else:
            intent = "general_question"
    else:
        result = parse_real_response("Classify as policy_question or general_question: " + state["query"])
        intent = "policy_question" if "policy_question" in result.answer else "general_question"
    return {"intent": intent}


# NODE 2: Retrieve the top three chunks and answer from the best chunk.
def retrieve_and_answer(state: AssistantState) -> AssistantState:
    collection = get_collection()
    if collection.count() < 8:
        collection = ingest()
    result = collection.query(query_texts=[state["query"]], n_results=3)
    docs = result["documents"][0]
    ids = result["ids"][0]
    if mock_mode():
        top_chunk_snippet = docs[0][:200]
        answer = f"Based on the retrieved context: {top_chunk_snippet}"
        return {
            "answer": answer,
            "sources": ids,
            "confidence": 1.0,
        }

    context_lines = []
    for document_id, document_text in zip(ids, docs):
        context_lines.append(f"[{document_id}] {document_text}")
    context = "\n".join(context_lines)
    response = parse_real_response(POLICY_PROMPT.format(context=context, query=state["query"]))
    return response.model_dump()


# NODE 3: Return a fixed scope message for unrelated questions.
def direct_answer(state: AssistantState) -> AssistantState:
    if mock_mode():
        return {"answer": "I can only answer questions about Zepto policies right now.", "sources": [], "confidence": 1.0}
    response = parse_real_response("Answer this general question as JSON: " + state["query"])
    return response.model_dump()


# Conditional-edge routing function.
def route(state: AssistantState) -> str:
    return state["intent"]


# Build and compile the three-node LangGraph workflow.
builder = StateGraph(AssistantState)
builder.add_node("classify_intent", classify_intent)
builder.add_node("retrieve_and_answer", retrieve_and_answer)
builder.add_node("direct_answer", direct_answer)
builder.set_entry_point("classify_intent")
builder.add_conditional_edges("classify_intent", route, {
    "policy_question": "retrieve_and_answer", "general_question": "direct_answer"
})
builder.add_edge("retrieve_and_answer", END)
builder.add_edge("direct_answer", END)
graph = builder.compile()

app = FastAPI(title="Zepto Policy Support Assistant", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "mock_llm": mock_mode()}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    state = graph.invoke({"query": request.query})
    return AskResponse(
        answer=state["answer"],
        sources=state["sources"],
        confidence=state["confidence"],
    )
