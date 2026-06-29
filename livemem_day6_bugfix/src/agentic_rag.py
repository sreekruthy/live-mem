"""
Agentic RAG: unlike every other baseline (single retrieve-then-generate
pass), this gives the model a bounded multi-step loop:
  1. Retrieve once, attempt to answer.
  2. The model itself judges whether its evidence was sufficient.
  3. If NOT sufficient, it proposes a refined search query, the system
     retrieves again with that query, and the model gets ONE more
     attempt with the combined (original + refined) evidence.
  4. After round 2, it must answer (or abstain) regardless — capped at
     2 rounds so cost/time stay bounded and comparable to other
     baselines, rather than an open-ended agentic loop.

This is the "plans its own approach, decides whether to investigate
further" behavior that distinguishes Agentic RAG from the others, per
the literature reviewed early in this project (Self-RAG/Agentic RAG:
"plans approach, decides investigation strategy, takes action").
"""
import json
import re
import time
from typing import List, Tuple

from .data_loader import Scenario, Event
from .embeddings import top_k_indices
from .llm_client import chat
from . import config

MAX_ROUNDS = 2

ATTEMPT_PROMPT_TEMPLATE = """You are answering a question using retrieved context. You may decide \
the context is insufficient and request ONE more targeted search before giving a final answer.

Context retrieved so far:
{context}

Question: {question}

Respond with ONLY valid JSON in this exact format, no other text:
{{"sufficient": true or false, "answer": "<your best answer using current context, or null if \
sufficient is false>", "evidence_event_ids": ["E001", ...], "refined_query": "<a more specific \
search query to find what's missing, or null if sufficient is true>"}}

If the context is sufficient, set sufficient=true, provide your answer and evidence, and set \
refined_query to null. If the context is NOT sufficient to answer confidently, set \
sufficient=false, set answer to null, and provide a refined_query describing specifically what \
additional information you need.
"""

FINAL_PROMPT_TEMPLATE = """You are answering a question using retrieved context. This is your \
FINAL attempt — you must answer now using whatever context is available, or state that there \
is insufficient evidence.

Context retrieved so far (including a follow-up search):
{context}

Question: {question}

Instructions:
- If the context allows a confident answer, answer concisely.
- If it still does not, respond with exactly: "Insufficient evidence."
- After your answer, write "Evidence: " followed by the event IDs used.
"""


def _parse_attempt_response(raw_text: str) -> dict:
    """
    Defensive parsing, same fallback pattern as other modules. On
    failure, defaults to "insufficient, no refined query" — which the
    caller treats as "give up, go straight to final answer" rather than
    crashing or looping unpredictably.
    """
    default = {"sufficient": False, "answer": None, "evidence_event_ids": [], "refined_query": None}
    if raw_text is None:
        return default
    text = re.sub(r"^```(json)?", "", raw_text.strip()).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        parsed = json.loads(text)
        # Fill in any missing keys with defaults rather than trusting the model fully
        return {**default, **parsed}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return {**default, **parsed}
            except json.JSONDecodeError:
                pass
    print(f"[warn] could not parse agentic attempt response, treating as insufficient with no "
          f"refined query. Raw: {raw_text!r}")
    return default


def _build_context(events: List[Event]) -> str:
    return "\n".join(f"[{ev.event_id}] {ev.text}" for ev in events)


def _retrieve(scenario: Scenario, query: str, k: int, exclude_ids: set = None) -> List[Event]:
    """Same embedding-similarity retrieval as Vanilla RAG, optionally excluding
    events already retrieved in a prior round, so round 2 surfaces NEW evidence
    rather than just re-finding the same top-k."""
    exclude_ids = exclude_ids or set()
    candidates = [ev for ev in scenario.events if ev.event_id not in exclude_ids]
    if not candidates:
        return []
    texts = [ev.text for ev in candidates]
    idxs = top_k_indices(query, texts, k=min(k, len(candidates)))
    return [candidates[i] for i in idxs]


def answer_question(scenario: Scenario, question_text: str, k: int = None) -> dict:
    k = k or config.TOP_K
    start = time.time()
    total_in_tok, total_out_tok = 0, 0

    # Round 1
    retrieved = _retrieve(scenario, question_text, k)
    retrieved_ids = {ev.event_id for ev in retrieved}
    context = _build_context(retrieved)
    prompt = ATTEMPT_PROMPT_TEMPLATE.format(context=context, question=question_text)
    result = chat(messages=[{"role": "user", "content": prompt}], max_tokens=700)
    total_in_tok += result["input_tokens"]
    total_out_tok += result["output_tokens"]
    parsed = _parse_attempt_response(result["text"])

    rounds_used = 1
    final_answer = parsed.get("answer")
    final_evidence = parsed.get("evidence_event_ids") or []

    if not parsed.get("sufficient", False) and parsed.get("refined_query") and rounds_used < MAX_ROUNDS:
        # Round 2: retrieve again with the model's own refined query,
        # excluding what we already have so we surface NEW evidence.
        refined_query = parsed["refined_query"]
        more_retrieved = _retrieve(scenario, refined_query, k, exclude_ids=retrieved_ids)
        retrieved = retrieved + more_retrieved
        rounds_used += 1

        context = _build_context(retrieved)
        final_prompt = FINAL_PROMPT_TEMPLATE.format(context=context, question=question_text)
        final_result = chat(messages=[{"role": "user", "content": final_prompt}], max_tokens=700)
        total_in_tok += final_result["input_tokens"]
        total_out_tok += final_result["output_tokens"]

        raw_text = (final_result["text"] or "").strip()
        final_answer = raw_text
        final_evidence = []
        if "Evidence:" in raw_text:
            parts = raw_text.split("Evidence:")
            final_answer = parts[0].strip()
            final_evidence = [e.strip() for e in parts[1].strip().split(",") if e.strip()]

    elif not parsed.get("sufficient", False) and final_answer is None:
        # Insufficient on round 1, but no refined query given (or already
        # at MAX_ROUNDS) — don't loop, just abstain honestly.
        final_answer = "Insufficient evidence."
        final_evidence = []

    latency_ms = int((time.time() - start) * 1000)
    retrieved_context_ids = [f"{scenario.scenario_id}::{ev.event_id}" for ev in retrieved]

    return {
        "method": "agentic_rag",
        "answer": final_answer or "Insufficient evidence.",
        "evidence_event_ids": final_evidence,
        "retrieved_context_ids": retrieved_context_ids,
        "latency_ms": latency_ms,
        "input_tokens": total_in_tok,
        "output_tokens": total_out_tok,
        "_rounds_used": rounds_used,
    }