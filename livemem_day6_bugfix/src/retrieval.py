"""
Day 4: retrieval logic for Living Memory v0.

This is where the method's mechanism actually changes system behavior,
not just internal bookkeeping. Different question types get different
retrieval treatment:

  - "current state" questions (latest_state, simple_recall, negative_recall,
    constraint_recall, list_recall, conditional_recall, causal_recall)
    -> retrieve ONLY status="active" memories. "superseded" and "inert"
    memories are excluded, so a stale fact can't be retrieved and
    presented as current even if it's textually similar to the question.

  - "history-aware" questions (historical_recall, supersession, provenance)
    -> retrieve active + superseded memories, WITH their supersession
    links surfaced, so the system can explain what replaced what.

  - multi_hop, disambiguation, abstention -> treated as current-state
    (active-only) for retrieval purposes; multi_hop questions in this
    benchmark ask about current reasoning chains, not history.

  - If nothing clears a minimum similarity bar at all, return an explicit
    abstain signal rather than forcing a retrieval that doesn't fit.
"""
from typing import List, Tuple
import numpy as np

from .memory_atoms import MemoryAtom, get_active_memories, get_memory_by_id
from .embeddings import embed_texts, cosine_similarity
from . import config

# Question types where retrieval should include superseded memories and
# their supersession links, because the question is explicitly about
# what changed, not just what's true now.
HISTORY_AWARE_TYPES = {"historical_recall", "supersession", "provenance"}

# Below this similarity, we don't trust the match enough to answer from
# it at all — used as one signal (not the only one) for abstention.
MIN_SIMILARITY_FOR_RETRIEVAL = 0.15


def _atoms_for_retrieval(atoms: List[MemoryAtom], question_type: str) -> List[MemoryAtom]:
    """
    Selects which atoms are even eligible for retrieval, based on
    question type. This is the core status-filtering mechanism — the
    one-sentence pitch of the whole method, implemented as a filter.
    """
    if question_type in HISTORY_AWARE_TYPES:
        # Active + superseded eligible. "inert" (hypothesis) memories are
        # still excluded here — a hypothesis was never confirmed, so it
        # has no place in either current-state or history-of-changes answers.
        return [a for a in atoms if a.status in ("active", "superseded", "uncertain")]
    else:
        # Current-state questions: only active (and uncertain, since an
        # uncertain memory hasn't been disproven, just not yet confirmed
        # as superseded — it's still the best available current answer).
        return [a for a in atoms if a.status in ("active", "uncertain")]


def retrieve(atoms: List[MemoryAtom], question: str, question_type: str, k: int = None) -> Tuple[List[MemoryAtom], bool]:
    """
    Returns (retrieved_atoms, should_abstain).

    should_abstain is True if nothing in the eligible pool clears the
    minimum similarity bar — the caller should skip generation entirely
    and return "Insufficient evidence." directly, same pattern as
    Vanilla RAG's abstention handling.
    """
    k = k or config.TOP_K
    eligible = _atoms_for_retrieval(atoms, question_type)

    if not eligible:
        return [], True

    texts = [a.content for a in eligible]
    query_vec = embed_texts([question])[0]
    doc_vecs = embed_texts(texts)
    sims = cosine_similarity(query_vec, doc_vecs)

    ranked_idx = np.argsort(-sims)[:k]
    top_sims = sims[ranked_idx]

    if top_sims[0] < MIN_SIMILARITY_FOR_RETRIEVAL:
        return [], True

    retrieved = [eligible[i] for i in ranked_idx]
    return retrieved, False


def build_context(retrieved_atoms: List[MemoryAtom], all_atoms: List[MemoryAtom], question_type: str) -> str:
    """
    Formats retrieved memories into a context block, labeling status and
    surfacing supersession/refinement links explicitly so the LLM can
    reason about "what replaced what" for history-aware questions.
    """
    lines = []
    for a in retrieved_atoms:
        label = a.status.upper()
        line = f"[{label} | {a.memory_id} | evidence: {','.join(a.evidence_event_ids)}] {a.content}"
        lines.append(line)

        if a.status == "superseded":
            # Find what superseded this one, so the model can cite it
            # directly without having to search the rest of the context.
            superseder = next((x for x in all_atoms if a.memory_id in x.supersedes_memory_ids), None)
            if superseder:
                lines.append(f"    -> superseded by [{superseder.memory_id}] {superseder.content}")

        if a.refines_memory_ids:
            for rid in a.refines_memory_ids:
                refined = get_memory_by_id(all_atoms, rid)
                if refined:
                    lines.append(f"    -> refines [{refined.memory_id}] {refined.content}")

    return "\n".join(lines)


GENERATION_PROMPT_TEMPLATE = """You are answering a question using a structured memory log. Each memory \
has a status: ACTIVE (currently valid), SUPERSEDED (replaced by a newer fact — shown with \
what replaced it), or UNCERTAIN (a possible update that hasn't been fully confirmed yet).

Memory log:
{context}

Question: {question}

Instructions:
- For questions about the CURRENT state, prefer ACTIVE memories. Do not use a SUPERSEDED \
memory as if it were still current.
- For questions about HISTORY or about what changed, you may reference SUPERSEDED memories \
explicitly, using the "superseded by" links shown.
- If a question asks WHICH EVENT proves something, or for supersession questions, cite the \
event IDs (e.g. "E004"), NOT the internal memory IDs (do not write "MEM_E004" in your answer \
text — use "E004" instead).
- For supersession questions specifically, mention BOTH the original event ID and the event \
ID that replaced it.
- If the memory log does not contain enough information to answer, respond with exactly: \
"Insufficient evidence."
- Answer concisely, in one or two sentences.
- After your answer, on a new line, write "Evidence: " followed by a comma-separated list \
of event IDs (e.g. E004) you used.
"""


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src import config
    from src.data_loader import load_scenarios
    from src.memory_atoms import extract_memory_atoms
    from src.contradiction_classifier import process_scenario_memories

    config.check_config()
    scenarios = load_scenarios(
        config.SCENARIO_DOCS_PATH, config.EVENTS_PATH, config.QUESTIONS_PATH
    )
    s001 = scenarios["S001"]
    atoms = extract_memory_atoms(s001)
    process_scenario_memories(atoms)

    print("Memory states after classification:")
    for a in atoms:
        print(f"  {a.memory_id} [{a.status}] {a.content[:50]}")

    print("\n--- Testing retrieval on a current-state question ---")
    q1 = "What backend framework is currently selected?"
    retrieved, abstain = retrieve(atoms, q1, "latest_state")
    print(f"Question: {q1}")
    print(f"Abstain: {abstain}")
    for a in retrieved:
        print(f"  retrieved: [{a.status}] {a.content[:60]}")

    print("\n--- Testing retrieval on a supersession question ---")
    q2 = "Which earlier framework decision was superseded?"
    retrieved2, abstain2 = retrieve(atoms, q2, "supersession")
    print(f"Question: {q2}")
    print(f"Abstain: {abstain2}")
    for a in retrieved2:
        print(f"  retrieved: [{a.status}] {a.content[:60]}")

    print("\n--- Context block for the supersession question ---")
    print(build_context(retrieved2, atoms, "supersession"))
