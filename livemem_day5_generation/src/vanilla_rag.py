"""
Vanilla RAG: the simplest baseline. Retrieves top-k events by embedding
similarity, generates an answer from just those events, no status
tracking, no graph, no reranking.

Written so retrieve() and build_context() can be imported and reused by
Rerank RAG (Day 2: same retrieve, add a rerank step) and Contextual
Retrieval (Day 2: same generate, different chunk preprocessing) without
copy-pasting this whole file.
"""
import time
from typing import List, Tuple

from .data_loader import Scenario, Event
from .embeddings import top_k_indices
from .llm_client import chat
from . import config


def retrieve(scenario: Scenario, question: str, k: int = None) -> List[Event]:
    """
    Retrieves the top-k most similar events to the question, from this
    scenario's events ONLY (never across scenarios, per spec).
    """
    k = k or config.TOP_K
    events = scenario.events
    if not events:
        return []
    texts = [ev.text for ev in events]
    idxs = top_k_indices(question, texts, k=k)
    return [events[i] for i in idxs]


def build_context(retrieved_events: List[Event]) -> str:
    """Formats retrieved events into a plain context block for the prompt."""
    lines = []
    for ev in retrieved_events:
        lines.append(f"[{ev.event_id} | {ev.timestamp}] {ev.text}")
    return "\n".join(lines)


GENERATION_PROMPT_TEMPLATE = """You are answering a question using only the context below. \
The context comes from a timestamped event log. If the context does not contain \
enough information to answer, respond with exactly: "Insufficient evidence."

Context:
{context}

Question: {question}

Instructions:
- Answer concisely, in one or two sentences.
- After your answer, on a new line, write "Evidence: " followed by a comma-separated \
list of the event IDs (e.g. E004) you used to answer.
- If you cannot answer from the context, respond with exactly "Insufficient evidence." \
and no evidence line.
"""


def generate_answer(question: str, retrieved_events: List[Event]) -> Tuple[str, List[str], int, int]:
    """
    Calls the LLM to generate an answer from retrieved events.
    Returns (answer_text, evidence_event_ids, input_tokens, output_tokens).
    """
    context = build_context(retrieved_events)
    prompt = GENERATION_PROMPT_TEMPLATE.format(context=context, question=question)

    result = chat(messages=[{"role": "user", "content": prompt}], max_tokens=600)
    raw_text = result["text"].strip()

    # Parse out the "Evidence: ..." line if present
    answer = raw_text
    evidence_ids = []
    if "Evidence:" in raw_text:
        parts = raw_text.split("Evidence:")
        answer = parts[0].strip()
        evidence_str = parts[1].strip()
        evidence_ids = [e.strip() for e in evidence_str.split(",") if e.strip()]

    return answer, evidence_ids, result["input_tokens"], result["output_tokens"]


def answer_question(scenario: Scenario, question_text: str, k: int = None) -> dict:
    """
    Full pipeline for one question: retrieve -> generate -> package result.
    Returns a dict matching the fields needed for the prediction JSONL schema
    (scenario_id/question_id are added by the caller, since this function
    doesn't know the question_id).
    """
    start = time.time()
    retrieved = retrieve(scenario, question_text, k=k)
    answer, evidence_ids, in_tok, out_tok = generate_answer(question_text, retrieved)
    latency_ms = int((time.time() - start) * 1000)

    retrieved_context_ids = [f"{scenario.scenario_id}::{ev.event_id}" for ev in retrieved]

    return {
        "method": "vanilla_rag",
        "answer": answer,
        "evidence_event_ids": evidence_ids,
        "retrieved_context_ids": retrieved_context_ids,
        "latency_ms": latency_ms,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }
