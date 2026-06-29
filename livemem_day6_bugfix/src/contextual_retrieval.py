"""
Contextual Retrieval: identical to Vanilla RAG's retrieve-then-generate
pipeline, except each event's text is prefixed with brief scenario-level
context BEFORE being embedded. The idea (from Anthropic's "Contextual
Retrieval" technique): an isolated chunk can be ambiguous on its own;
prepending what it's about gives the embedding model more to work with,
improving retrieval precision — without needing any extra LLM calls at
query time, since the context is prepended once, at indexing time.

Reuses vanilla_rag.py's generate_answer() directly — only retrieve()
differs, so any difference in results is attributable to the contextual
embedding step, not to unrelated generation differences.
"""
from typing import List, Tuple

from .data_loader import Scenario, Event
from .embeddings import embed_texts, cosine_similarity
from .vanilla_rag import generate_answer
from . import config

import numpy as np


def _build_contextualized_text(scenario: Scenario, event: Event) -> str:
    """
    Prepends scenario-level context to the event's raw text. This is the
    one thing that differs from Vanilla RAG's plain embedding of event.text.

    Context used: the scenario's title/domain (short, cheap, no LLM call
    needed) plus the event's position in time relative to the scenario,
    since "this is an early decision" vs "this is a later update" is
    exactly the kind of ambiguity-resolving context this technique is
    meant to add.
    """
    position = "early" if event in scenario.events[:len(scenario.events)//2] else "later"
    context_prefix = f"[{scenario.title} ({scenario.domain}) — {position} event, type: {event.event_type}] "
    return context_prefix + event.text


def retrieve(scenario: Scenario, question: str, k: int = None) -> List[Event]:
    """
    Same retrieval mechanics as vanilla_rag.retrieve(), but embeds
    CONTEXTUALIZED text for each event instead of raw event.text.
    """
    k = k or config.TOP_K
    events = scenario.events
    if not events:
        return []

    contextualized_texts = [_build_contextualized_text(scenario, ev) for ev in events]

    query_vec = embed_texts([question])[0]
    doc_vecs = embed_texts(contextualized_texts)
    sims = cosine_similarity(query_vec, doc_vecs)

    ranked_idx = np.argsort(-sims)[:k]
    return [events[i] for i in ranked_idx]


def answer_question(scenario: Scenario, question_text: str, k: int = None) -> dict:
    """
    Mirrors vanilla_rag.answer_question's structure exactly. Generation
    still uses the event's PLAIN text (not the contextualized version) —
    the context prefix is purely a retrieval-time aid, not something that
    should leak into what the LLM sees when answering, since that would
    conflate "better retrieval" with "different/more context at
    generation time" and muddy the comparison.
    """
    import time
    start = time.time()
    retrieved = retrieve(scenario, question_text, k=k)
    answer, evidence_ids, in_tok, out_tok = generate_answer(question_text, retrieved)
    latency_ms = int((time.time() - start) * 1000)

    retrieved_context_ids = [f"{scenario.scenario_id}::{ev.event_id}" for ev in retrieved]

    return {
        "method": "contextual_retrieval",
        "answer": answer,
        "evidence_event_ids": evidence_ids,
        "retrieved_context_ids": retrieved_context_ids,
        "latency_ms": latency_ms,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }