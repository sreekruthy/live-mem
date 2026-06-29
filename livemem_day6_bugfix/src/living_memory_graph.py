"""
Living Memory + Graph: combines Living Memory v0's status-tracking
(active/superseded/uncertain/inert, typed contradiction classification)
with GraphRAG's entity-relation structure and traversal.

Pipeline per scenario:
  1. Extract atomic memories (memory_atoms.extract_memory_atoms) - reused as-is
  2. Classify contradictions, assign status (contradiction_classifier.
     process_scenario_memories) - reused as-is, gives us active/superseded/
     uncertain/inert + supersedes/refines links
  3. NEW: extract entity-relation edges, but tagged onto the ALREADY-
     CLASSIFIED memory atoms (not raw events) - each edge inherits its
     source memory's status
  4. NEW retrieval: traverse by entity (GraphRAG-style: which entities
     does the question concern, follow their edges) but FILTER the
     candidate pool by status first (Living Memory v0-style) - so a
     superseded memory's edges are excluded from current-state
     questions, while history-aware questions can still traverse
     through them with status visible.

This is the key structural difference from plain GraphRAG: edges carry
status, so traversal can distinguish "this relationship is still valid"
from "this relationship existed but was superseded" - the exact
distinction plain GraphRAG could not make, which caused its
evidence-attribution errors (see FINDINGS.md).
"""
import json
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

from .data_loader import Scenario
from .memory_atoms import MemoryAtom, extract_memory_atoms
from .contradiction_classifier import process_scenario_memories
from .embeddings import top_k_indices
from .llm_client import chat
from . import config


@dataclass
class StatusEdge:
    """A graph edge that inherits the status of the memory atom that established it."""
    subject: str
    relation: str
    obj: str
    memory_id: str       # which classified memory atom this edge came from
    status: str          # inherited from that memory atom (active/superseded/uncertain/inert)
    evidence_event_ids: List[str] = field(default_factory=list)


@dataclass
class StatusGraph:
    entities: List[str] = field(default_factory=list)
    edges: List[StatusEdge] = field(default_factory=list)


EXTRACT_STATUS_GRAPH_PROMPT = """Extract entities and relations from these memories as a \
knowledge graph. Each memory already has a STATUS (active/superseded/uncertain/inert) - your \
job is just to extract the (subject, relation, object) triple for each memory, the status is \
handled separately.

Memories:
{memories_text}

Extract ONE triple per memory capturing its main fact. Keep entity/relation names short \
(1-3 words) and consistent. Respond ONLY with valid JSON, no other text:
{{"triples": [{{"memory_id": "MEM_E001", "subject": "...", "relation": "...", "object": "..."}}, ...]}}
"""


def _parse_status_graph_response(raw_text: str) -> List[dict]:
    """Same defensive parsing + truncation-salvage pattern as graph_rag.py."""
    if raw_text is None:
        return []
    text = re.sub(r"^```(json)?", "", raw_text.strip()).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text).get("triples", [])
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0)).get("triples", [])
        except json.JSONDecodeError:
            pass
    # Salvage complete triple objects from a truncated response
    triple_matches = re.findall(
        r'\{\s*"memory_id"\s*:\s*"[^"]*"\s*,\s*"subject"\s*:\s*"[^"]*"\s*,\s*'
        r'"relation"\s*:\s*"[^"]*"\s*,\s*"object"\s*:\s*"[^"]*"\s*\}',
        text,
    )
    salvaged = []
    for tm in triple_matches:
        try:
            salvaged.append(json.loads(tm))
        except json.JSONDecodeError:
            continue
    if salvaged:
        print(f"[warn] status graph extraction truncated; salvaged {len(salvaged)} triples.")
        return salvaged
    print(f"[warn] could not parse status graph extraction, returning empty graph. Raw: {raw_text!r}")
    return []


def build_status_graph(atoms: List[MemoryAtom]) -> Tuple[StatusGraph, int, int]:
    """
    Extracts entity-relation triples for already-classified memory atoms,
    tagging each resulting edge with that atom's status. One LLM call
    per scenario (same cost pattern as GraphRAG's build_graph).
    """
    memories_text = "\n".join(f"[{a.memory_id} | status: {a.status}] {a.content}" for a in atoms)
    prompt = EXTRACT_STATUS_GRAPH_PROMPT.format(memories_text=memories_text)

    result = chat(messages=[{"role": "user", "content": prompt}], max_tokens=2000)
    triples = _parse_status_graph_response(result["text"])

    atoms_by_id = {a.memory_id: a for a in atoms}
    graph = StatusGraph()
    seen_entities = set()

    for t in triples:
        mem_id = t.get("memory_id", "").strip()
        subj = t.get("subject", "").strip()
        obj = t.get("object", "").strip()
        rel = t.get("relation", "").strip()
        if not (mem_id and subj and obj and rel) or mem_id not in atoms_by_id:
            continue  # skip malformed or unmatched triples rather than crash
        atom = atoms_by_id[mem_id]
        graph.edges.append(StatusEdge(
            subject=subj, relation=rel, obj=obj,
            memory_id=mem_id, status=atom.status,
            evidence_event_ids=atom.evidence_event_ids,
        ))
        seen_entities.add(subj)
        seen_entities.add(obj)

    graph.entities = list(seen_entities)
    return graph, result["input_tokens"], result["output_tokens"]


# Same question-type categorization as retrieval.py, reused for consistency
HISTORY_AWARE_TYPES = {"historical_recall", "supersession", "provenance"}


def retrieve_via_status_graph(graph: StatusGraph, atoms: List[MemoryAtom],
                                question: str, question_type: str, k: int = None) -> List[StatusEdge]:
    """
    The key new mechanism: traverse by entity (GraphRAG-style) but
    filter the edge pool by status FIRST (Living Memory v0-style),
    before ranking by entity relevance. This is what lets the system
    distinguish "this relationship is current" from "this relationship
    was superseded" - exactly what plain GraphRAG could not do.
    """
    k = k or config.TOP_K

    if question_type in HISTORY_AWARE_TYPES:
        eligible_edges = [e for e in graph.edges if e.status in ("active", "superseded", "uncertain")]
    else:
        eligible_edges = [e for e in graph.edges if e.status in ("active", "uncertain")]

    if not eligible_edges or not graph.entities:
        return []

    # Match against full relation TEXT (subject+relation+object), not bare
    # entity names alone. Bare entity-name matching caused severe
    # degeneracy here (6 of 8 questions in one test scenario collapsed to
    # the identical retrieved set, regardless of what was asked) — status
    # filtering already narrows the eligible pool, so a few entities
    # dominate matches even more than in plain GraphRAG. Matching richer
    # relation text, while keeping k unchanged (NOT widening it — that
    # combination caused worse problems in the GraphRAG investigation),
    # isolates this one change.
    edge_texts = [f"{e.subject} {e.relation} {e.obj}" for e in eligible_edges]
    edge_idxs = top_k_indices(question, edge_texts, k=min(k, len(eligible_edges)))
    matched = [eligible_edges[i] for i in edge_idxs]

    # Deduplicate by memory_id, preserve order
    seen_mem_ids = set()
    ordered = []
    for e in matched:
        if e.memory_id not in seen_mem_ids:
            seen_mem_ids.add(e.memory_id)
            ordered.append(e)

    return ordered


def _build_context(edges: List[StatusEdge], atoms_by_id: Dict[str, MemoryAtom]) -> str:
    """Shows status AND the matching relation, combining both prior baselines' context strategies."""
    lines = []
    for e in edges:
        atom = atoms_by_id.get(e.memory_id)
        content = atom.content if atom else "(unknown)"
        lines.append(f"[{e.status.upper()} | {e.memory_id} | relation: \"{e.subject} {e.relation} {e.obj}\" "
                     f"| evidence: {','.join(e.evidence_event_ids)}] {content}")
    return "\n".join(lines)


PROMPT_TEMPLATE = """You are answering a question using facts from a status-aware knowledge \
graph. Each fact shows its STATUS (ACTIVE=currently valid, SUPERSEDED=replaced by something \
newer, UNCERTAIN=a possible update not yet confirmed) and the graph RELATION that matched it.

Facts:
{context}

Question: {question}

Instructions:
- Prefer ACTIVE facts for current-state questions. Do not present a SUPERSEDED fact as current.
- For supersession questions, identify both the superseded fact and what replaced it.
- If facts are insufficient, respond with exactly: "Insufficient evidence."
- Answer as briefly as possible while remaining accurate.
- When citing evidence, use the raw event ID (e.g. "E004"), NOT the internal memory ID — do \
NOT write "MEM_E004" anywhere in your answer or evidence line, use "E004" instead.
- After your answer, write "Evidence: " followed by the event IDs used.
"""


def answer_question(scenario: Scenario, question_text: str, question_type: str,
                     atoms: List[MemoryAtom], graph: StatusGraph) -> dict:
    start = time.time()
    atoms_by_id = {a.memory_id: a for a in atoms}

    edges = retrieve_via_status_graph(graph, atoms, question_text, question_type)

    if not edges:
        latency_ms = int((time.time() - start) * 1000)
        return {
            "method": "living_memory_graph",
            "answer": "Insufficient evidence.",
            "evidence_event_ids": [],
            "retrieved_context_ids": [],
            "latency_ms": latency_ms,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    context = _build_context(edges, atoms_by_id)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question_text)
    result = chat(messages=[{"role": "user", "content": prompt}], max_tokens=700)
    raw_text = (result["text"] or "").strip()

    answer = raw_text
    evidence_ids = []
    if "Evidence:" in raw_text:
        parts = raw_text.split("Evidence:")
        answer = parts[0].strip()
        evidence_ids = [e.strip() for e in parts[1].strip().split(",") if e.strip()]

    latency_ms = int((time.time() - start) * 1000)
    retrieved_context_ids = [f"{scenario.scenario_id}::{eid}" for e in edges for eid in e.evidence_event_ids]

    return {
        "method": "living_memory_graph",
        "answer": answer,
        "evidence_event_ids": evidence_ids,
        "retrieved_context_ids": retrieved_context_ids,
        "latency_ms": latency_ms,
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
    }