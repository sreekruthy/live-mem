"""
Day 5: generation for Living Memory v0.

Wires retrieval.py's retrieve() + build_context() into actual LLM calls,
producing predictions in the exact schema the benchmark spec requires.
Mirrors vanilla_rag.py's structure closely on purpose, so the two are
directly comparable (same answer-parsing logic, same output schema).
"""
import time
from typing import List, Tuple

from .data_loader import Scenario
from .memory_atoms import MemoryAtom
from .retrieval import retrieve, build_context, GENERATION_PROMPT_TEMPLATE
from .llm_client import chat
from . import config

ABSTAIN_TEXT = "Insufficient evidence."


def generate_answer(question: str, retrieved_atoms: List[MemoryAtom], all_atoms: List[MemoryAtom],
                     question_type: str) -> Tuple[str, List[str], int, int]:
    """
    Mirrors vanilla_rag.generate_answer's signature and parsing logic
    exactly, so both methods' predictions are produced the same way and
    differences in output are attributable to retrieval/memory behavior,
    not to differences in how answers get parsed.
    """
    context = build_context(retrieved_atoms, all_atoms, question_type)
    prompt = GENERATION_PROMPT_TEMPLATE.format(context=context, question=question)

    result = chat(messages=[{"role": "user", "content": prompt}], max_tokens=600)
    raw_text = result["text"].strip()

    answer = raw_text
    evidence_ids = []
    if "Evidence:" in raw_text:
        parts = raw_text.split("Evidence:")
        answer = parts[0].strip()
        evidence_str = parts[1].strip()
        evidence_ids = [e.strip() for e in evidence_str.split(",") if e.strip()]

    return answer, evidence_ids, result["input_tokens"], result["output_tokens"]


def answer_question(atoms: List[MemoryAtom], scenario: Scenario, question_text: str,
                     question_type: str, k: int = None) -> dict:
    """
    Full pipeline for one question: retrieve (status-aware) -> generate
    -> package result. Returns a dict matching the required prediction
    schema (scenario_id/question_id added by the caller).

    If retrieval signals abstention (nothing cleared the similarity bar),
    we skip the generation call entirely — same pattern as Vanilla RAG,
    saves a token cost and guarantees the exact abstention phrase.
    """
    start = time.time()
    retrieved, should_abstain = retrieve(atoms, question_text, question_type, k=k)

    if should_abstain:
        latency_ms = int((time.time() - start) * 1000)
        return {
            "method": "living_memory_v0",
            "answer": ABSTAIN_TEXT,
            "evidence_event_ids": [],
            "retrieved_context_ids": [],
            "latency_ms": latency_ms,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    answer, evidence_ids, in_tok, out_tok = generate_answer(
        question_text, retrieved, atoms, question_type
    )
    latency_ms = int((time.time() - start) * 1000)

    retrieved_context_ids = [
        f"{scenario.scenario_id}::{eid}" for a in retrieved for eid in a.evidence_event_ids
    ]

    return {
        "method": "living_memory_v0",
        "answer": answer,
        "evidence_event_ids": evidence_ids,
        "retrieved_context_ids": retrieved_context_ids,
        "latency_ms": latency_ms,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }
