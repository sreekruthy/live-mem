"""
GraphRAG: builds a simple entity-relation graph from the scenario's
events, then retrieves by identifying which entities a question is
about and traversing to their connected facts — rather than pure
text-similarity-to-chunks like Vanilla RAG.

Per the benchmark spec: GraphRAG gets graph STRUCTURE but deliberately
NO status tracking and NO supersession logic (that's what distinguishes
it from Living Memory v0). This makes it the cleanest possible contrast:
same "facts connected in a graph" idea, but with no concept of which
facts are still valid. If GraphRAG still produces stale answers despite
having structure, that's evidence that graph structure alone isn't
what solves staleness — status-awareness is.
"""
import json
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

from .data_loader import Scenario, Event
from .embeddings import top_k_indices
from .vanilla_rag import generate_answer
from .llm_client import chat
from . import config


@dataclass
class GraphEdge:
    """One relation: subject entity -- relation --> object entity, tied to evidence."""
    subject: str
    relation: str
    obj: str
    evidence_event_ids: List[str]


@dataclass
class SceneGraph:
    entities: List[str] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)


EXTRACT_GRAPH_PROMPT = """Extract entities and relations from this event log as a knowledge graph.

Events:
{events_text}

Extract (subject, relation, object) triples capturing the meaningful facts. IMPORTANT: when a \
later event changes, replaces, or rejects something from an earlier event, extract a triple \
connecting them directly (e.g. subject="FastAPI", relation="replaces", object="Flask") — do \
NOT just describe each event in isolation, capture the RELATIONSHIP between events when one \
exists. Keep entity names short (1-3 words) and consistent. Keep relation names short (1-3 \
words, e.g. "replaces", "rejected", "requires"). Do not add commentary or extra fields.

Respond ONLY with valid JSON, no other text before or after:
{{"triples": [{{"subject": "...", "relation": "...", "object": "...", "evidence_event_ids": ["E001"]}}, ...]}}
"""


def _parse_graph_response(raw_text: str) -> List[dict]:
    """
    Defensive parsing. If the full JSON is truncated (a real, recurring
    issue with this model on longer extractions), salvage whatever
    complete {...} triple objects appear before the cutoff rather than
    discarding the entire response — a partial graph is more useful than
    an empty one, and we already saw truncation can happen even at a
    fairly generous max_tokens.
    """
    if raw_text is None:
        return []
    text = re.sub(r"^```(json)?", "", raw_text.strip()).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        parsed = json.loads(text)
        return parsed.get("triples", [])
    except json.JSONDecodeError:
        pass

    # Try finding a complete top-level object first
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0)).get("triples", [])
        except json.JSONDecodeError:
            pass

    # Salvage: find every COMPLETE {...} triple object, even if the
    # surrounding array/object never got closed.
    triple_matches = re.findall(
        r'\{\s*"subject"\s*:\s*"[^"]*"\s*,\s*"relation"\s*:\s*"[^"]*"\s*,\s*'
        r'"object"\s*:\s*"[^"]*"\s*,\s*"evidence_event_ids"\s*:\s*\[[^\]]*\]\s*\}',
        text,
    )
    salvaged = []
    for tm in triple_matches:
        try:
            salvaged.append(json.loads(tm))
        except json.JSONDecodeError:
            continue
    if salvaged:
        print(f"[warn] graph extraction response was truncated; salvaged {len(salvaged)} "
              f"complete triples out of a longer response.")
        return salvaged

    print(f"[warn] could not parse graph extraction response, returning empty graph. Raw: {raw_text!r}")
    return []


def build_graph(scenario: Scenario) -> Tuple[SceneGraph, int, int]:
    """
    One LLM call per scenario to extract the full entity-relation graph.
    Returns (graph, input_tokens, output_tokens).
    """
    events_text = "\n".join(f"[{ev.event_id}] {ev.text}" for ev in scenario.events)
    prompt = EXTRACT_GRAPH_PROMPT.format(events_text=events_text)

    result = chat(messages=[{"role": "user", "content": prompt}], max_tokens=2000)
    triples = _parse_graph_response(result["text"])

    graph = SceneGraph()
    seen_entities = set()
    for t in triples:
        subj = t.get("subject", "").strip()
        obj = t.get("object", "").strip()
        rel = t.get("relation", "").strip()
        ev_ids = t.get("evidence_event_ids", [])
        if not subj or not obj or not rel:
            continue  # skip malformed triples rather than crash
        graph.edges.append(GraphEdge(subject=subj, relation=rel, obj=obj, evidence_event_ids=ev_ids))
        seen_entities.add(subj)
        seen_entities.add(obj)
    graph.entities = list(seen_entities)

    return graph, result["input_tokens"], result["output_tokens"]


GRAPHRAG_PROMPT_TEMPLATE = """You are answering a question using facts retrieved via a knowledge \
graph. Each fact below shows the event it came from AND the graph relationship that matched it \
to your question — use the relationship to understand WHY each fact is relevant, and to \
correctly identify which event corresponds to the ORIGINAL/earlier fact versus a LATER \
change, if the question asks about timing or what replaced what.

Retrieved facts (with matching graph relationships):
{context}

Question: {question}

Instructions:
- If the question asks about something INITIAL/ORIGINAL/EARLIER, cite the event whose \
relationship establishes the original fact, not a later change to it.
- If the question asks about supersession/replacement, cite BOTH the original event and the \
event that replaced it.
- If the facts do not contain enough information, respond with exactly: "Insufficient evidence."
- Answer as briefly as possible while remaining accurate.
- After your answer, write "Evidence: " followed by the event IDs you used.
"""


def _build_graph_context(retrieved: List[Tuple[Event, str]]) -> str:
    """
    Formats retrieved events WITH the relation that matched them, so
    generation has a basis for distinguishing "this is the original
    fact" from "this is a later change" — rather than seeing
    undifferentiated event text and guessing at attribution.
    """
    lines = []
    for ev, matched_relation in retrieved:
        lines.append(f"[{ev.event_id} | matched via relation: \"{matched_relation}\"] {ev.text}")
    return "\n".join(lines)


def retrieve_via_graph(scenario: Scenario, graph: SceneGraph, question: str, k: int = None) -> List[Tuple[Event, str]]:
    """
    Entity-NAME matching at k=3, with simple dedup-and-truncate (no
    reranking before truncation). A reranking step was tried here and
    tested clean in isolation, but on the FULL 144-question run it
    collapsed supersession accuracy to 0/15 (from 1/15) while only
    marginally moving other metrics — net negative for the metric that
    matters most to this paper's argument. Reverted. See FINDINGS.md
    for the full investigation across 5 distinct tuning attempts.
    """
    k = k or config.TOP_K

    if not graph.entities or not graph.edges:
        texts = [ev.text for ev in scenario.events]
        if not texts:
            return []
        idxs = top_k_indices(question, texts, k=min(k, len(texts)))
        return [(scenario.events[i], "(no graph available)") for i in idxs]

    entity_idxs = top_k_indices(question, graph.entities, k=min(3, len(graph.entities)))
    relevant_entities = {graph.entities[i] for i in entity_idxs}

    matched: List[Tuple[str, str]] = []
    for edge in graph.edges:
        if edge.subject in relevant_entities or edge.obj in relevant_entities:
            relation_desc = f"{edge.subject} {edge.relation} {edge.obj}"
            for eid in edge.evidence_event_ids:
                matched.append((eid, relation_desc))

    seen = set()
    ordered_unique: List[Tuple[str, str]] = []
    for eid, rel_desc in matched:
        key = (eid, rel_desc)
        if key not in seen:
            seen.add(key)
            ordered_unique.append((eid, rel_desc))

    events_by_id = {ev.event_id: ev for ev in scenario.events}
    retrieved = [(events_by_id[eid], rel_desc) for eid, rel_desc in ordered_unique if eid in events_by_id]

    if not retrieved:
        texts = [ev.text for ev in scenario.events]
        idxs = top_k_indices(question, texts, k=min(k, len(texts)))
        return [(scenario.events[i], "(no matching relation found)") for i in idxs]

    return retrieved[:k] if len(retrieved) > k else retrieved


def _generate_answer_with_graph_context(question: str, retrieved: List[Tuple[Event, str]]) -> Tuple[str, List[str], int, int]:
    context = _build_graph_context(retrieved)
    prompt = GRAPHRAG_PROMPT_TEMPLATE.format(context=context, question=question)
    result = chat(messages=[{"role": "user", "content": prompt}], max_tokens=700)
    raw_text = (result["text"] or "").strip()

    answer = raw_text
    evidence_ids = []
    if "Evidence:" in raw_text:
        parts = raw_text.split("Evidence:")
        answer = parts[0].strip()
        evidence_ids = [e.strip() for e in parts[1].strip().split(",") if e.strip()]

    return answer, evidence_ids, result["input_tokens"], result["output_tokens"]


def answer_question(scenario: Scenario, question_text: str, graph: SceneGraph,
                     graph_in_tok: int = 0, graph_out_tok: int = 0) -> dict:
    """
    graph/graph_in_tok/graph_out_tok are passed in (built once per
    scenario by the caller) rather than rebuilt per question — same
    "classify/extract once, reuse for all questions" pattern as Living
    Memory v0's classification step.
    """
    start = time.time()
    retrieved = retrieve_via_graph(scenario, graph, question_text)
    answer, evidence_ids, gen_in_tok, gen_out_tok = _generate_answer_with_graph_context(question_text, retrieved)
    latency_ms = int((time.time() - start) * 1000)

    retrieved_context_ids = [f"{scenario.scenario_id}::{ev.event_id}" for ev, _ in retrieved]

    return {
        "method": "graph_rag",
        "answer": answer,
        "evidence_event_ids": evidence_ids,
        "retrieved_context_ids": retrieved_context_ids,
        "latency_ms": latency_ms,
        "input_tokens": gen_in_tok,  # graph extraction tokens logged separately by caller (once per scenario, not per question)
        "output_tokens": gen_out_tok,
    }