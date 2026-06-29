"""
Adaptive RAG: classifies each incoming question into a strategy bucket
BEFORE retrieving, then adjusts retrieval breadth (top-k) based on that
classification — rather than using a fixed k for every question like
Vanilla RAG does.

The classifier sees ONLY the question text, not the gold question_type
label from the dataset — it has to infer strategy the way a real
adaptive system would, from the query alone. This is the core
distinguishing feature of "Adaptive RAG" per the literature: a
query-complexity classifier routing to different retrieval strategies.

Reuses vanilla_rag.py's generate_answer() directly — only the
classify-then-retrieve step differs.
"""
import time
import json
import re
from typing import List, Tuple

from .data_loader import Scenario, Event
from .embeddings import top_k_indices
from .vanilla_rag import generate_answer
from .llm_client import chat
from . import config

# Retrieval breadth per strategy bucket. SIMPLE questions need just the
# one or two most relevant facts; COMPLEX questions (multi-hop, causal,
# comparative) benefit from a wider net since the answer may be spread
# across more than one event.
STRATEGY_TOP_K = {
    "SIMPLE": 2,
    "COMPLEX": 6,
}
DEFAULT_TOP_K = 4  # fallback if classification fails/is unparseable

CLASSIFY_QUERY_PROMPT = """Classify this question into exactly one category:

- SIMPLE: a direct factual lookup with one clear answer (e.g. "What is X?", "Where is Y?", \
yes/no questions)
- COMPLEX: requires connecting multiple facts, reasoning about cause/effect, comparing things, \
or tracing how something changed over time (e.g. "Why did X happen?", "What replaced Y?", \
multi-step reasoning)

Question: "{question}"

Respond with ONLY the single word SIMPLE or COMPLEX, nothing else.
"""


def classify_query(question: str) -> Tuple[str, int, int]:
    """
    Returns (strategy, input_tokens, output_tokens). Defaults to
    "COMPLEX" (the safer, wider-net fallback) if the response is
    unparseable, rather than crashing or silently picking the narrowest
    strategy.
    """
    prompt = CLASSIFY_QUERY_PROMPT.format(question=question)
    result = chat(messages=[{"role": "user", "content": prompt}], max_tokens=400)

    raw = (result["text"] or "").strip().upper()
    if "SIMPLE" in raw:
        strategy = "SIMPLE"
    elif "COMPLEX" in raw:
        strategy = "COMPLEX"
    else:
        print(f"[warn] could not parse query classification, defaulting to COMPLEX. Raw: {raw!r}")
        strategy = "COMPLEX"

    return strategy, result["input_tokens"], result["output_tokens"]


def retrieve(scenario: Scenario, question: str, strategy: str) -> List[Event]:
    """Same embedding-similarity retrieval as Vanilla RAG, just with k chosen by strategy."""
    k = STRATEGY_TOP_K.get(strategy, DEFAULT_TOP_K)
    events = scenario.events
    if not events:
        return []
    texts = [ev.text for ev in events]
    idxs = top_k_indices(question, texts, k=min(k, len(events)))
    return [events[i] for i in idxs]


def answer_question(scenario: Scenario, question_text: str) -> dict:
    """
    Mirrors vanilla_rag.answer_question's structure, with the classify
    step inserted first. Token cost from classification is added to the
    total, same accounting pattern as Rerank RAG's extra step.
    """
    start = time.time()
    strategy, class_in_tok, class_out_tok = classify_query(question_text)
    retrieved = retrieve(scenario, question_text, strategy)
    answer, evidence_ids, gen_in_tok, gen_out_tok = generate_answer(question_text, retrieved)
    latency_ms = int((time.time() - start) * 1000)

    retrieved_context_ids = [f"{scenario.scenario_id}::{ev.event_id}" for ev in retrieved]

    return {
        "method": "adaptive_rag",
        "answer": answer,
        "evidence_event_ids": evidence_ids,
        "retrieved_context_ids": retrieved_context_ids,
        "latency_ms": latency_ms,
        "input_tokens": class_in_tok + gen_in_tok,
        "output_tokens": class_out_tok + gen_out_tok,
        "_strategy_used": strategy,  # extra diagnostic field, not part of required schema, useful for error analysis later
    }