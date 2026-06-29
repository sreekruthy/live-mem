"""
Rerank RAG: retrieves a larger candidate pool by embedding similarity
(wider net than Vanilla RAG), then uses the LLM itself to re-score and
re-order those candidates for relevance before picking the final top-k
to pass to generation.

Reuses vanilla_rag.py's generate_answer(), build_context(), and prompt
template directly — only the retrieval step differs. This keeps the two
baselines comparable: any difference in results is attributable to the
reranking step, not to unrelated differences in generation logic.
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

# How many candidates to pull via embedding similarity BEFORE reranking.
# Wider than the final TOP_K so the reranker has real candidates to choose
# from, not just re-confirming what similarity already picked.
CANDIDATE_POOL_SIZE = 5

RERANK_PROMPT_TEMPLATE = """You are ranking which pieces of context are most relevant to a question.

Question: {question}

Candidates (numbered):
{candidates}

Rank the candidates by relevance to the question, MOST relevant first. Respond ONLY with a \
JSON array of the candidate numbers in ranked order, e.g. [3, 1, 4, 2]. Include ALL candidate \
numbers exactly once.
"""


def _parse_ranking(raw_text: str, n_candidates: int) -> List[int]:
    """
    Defensive parsing of the rerank response. Falls back to original
    order (0..n-1) if parsing fails, rather than crashing — a degraded
    rerank (no reordering) is a safer failure mode than stopping the
    whole pipeline.
    """
    if raw_text is None:
        return list(range(n_candidates))
    text = raw_text.strip()
    match = re.search(r"\[[\d,\s]+\]", text)
    if not match:
        print(f"[warn] could not parse rerank response, using original order. Raw: {raw_text!r}")
        return list(range(n_candidates))
    try:
        ranked = json.loads(match.group(0))
        # Convert 1-indexed (as given to the model) to 0-indexed
        ranked_idx = [r - 1 for r in ranked if 1 <= r <= n_candidates]
        # Ensure every candidate appears exactly once; fall back if not
        if sorted(ranked_idx) != list(range(n_candidates)):
            print(f"[warn] rerank response missing/duplicate candidates, using original order. "
                  f"Parsed: {ranked_idx}")
            return list(range(n_candidates))
        return ranked_idx
    except (json.JSONDecodeError, TypeError):
        print(f"[warn] could not parse rerank response, using original order. Raw: {raw_text!r}")
        return list(range(n_candidates))


def rerank(question: str, candidates: List[Event]) -> Tuple[List[Event], int, int]:
    """
    Asks the LLM to rank candidates by relevance. Returns
    (reranked_events, input_tokens, output_tokens).
    """
    if len(candidates) <= 1:
        return candidates, 0, 0

    candidates_text = "\n".join(f"{i+1}. {ev.text}" for i, ev in enumerate(candidates))
    prompt = RERANK_PROMPT_TEMPLATE.format(question=question, candidates=candidates_text)

    result = chat(messages=[{"role": "user", "content": prompt}], max_tokens=700)
    ranked_idx = _parse_ranking(result["text"], len(candidates))
    reranked = [candidates[i] for i in ranked_idx]

    return reranked, result["input_tokens"], result["output_tokens"]


def retrieve(scenario: Scenario, question: str, k: int = None) -> Tuple[List[Event], int, int]:
    """
    Wider embedding-similarity pool -> LLM rerank -> top-k of the
    reranked order. Returns (final_events, rerank_input_tokens,
    rerank_output_tokens) so callers can log the extra token cost
    this step adds versus Vanilla RAG.
    """
    k = k or config.TOP_K
    events = scenario.events
    if not events:
        return [], 0, 0

    texts = [ev.text for ev in events]
    pool_size = min(CANDIDATE_POOL_SIZE, len(events))
    candidate_idxs = top_k_indices(question, texts, k=pool_size)
    candidates = [events[i] for i in candidate_idxs]

    reranked, rerank_in_tok, rerank_out_tok = rerank(question, candidates)
    final = reranked[:k]

    return final, rerank_in_tok, rerank_out_tok


def answer_question(scenario: Scenario, question_text: str, k: int = None) -> dict:
    """
    Mirrors vanilla_rag.answer_question's structure exactly, with the
    rerank step inserted and its token cost added to the total — so the
    prediction schema and token-cost accounting stay directly comparable.
    """
    start = time.time()
    retrieved, rerank_in_tok, rerank_out_tok = retrieve(scenario, question_text, k=k)
    answer, evidence_ids, gen_in_tok, gen_out_tok = generate_answer(question_text, retrieved)
    latency_ms = int((time.time() - start) * 1000)

    retrieved_context_ids = [f"{scenario.scenario_id}::{ev.event_id}" for ev in retrieved]

    return {
        "method": "rerank_rag",
        "answer": answer,
        "evidence_event_ids": evidence_ids,
        "retrieved_context_ids": retrieved_context_ids,
        "latency_ms": latency_ms,
        "input_tokens": rerank_in_tok + gen_in_tok,
        "output_tokens": rerank_out_tok + gen_out_tok,
    }