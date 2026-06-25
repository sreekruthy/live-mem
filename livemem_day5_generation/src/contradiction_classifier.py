"""
Day 3: the core novel mechanism of Living Memory v0.

For each new memory that might contradict/update something, this asks
the LLM to classify the relationship into one of 4 types, then applies
DIFFERENT confidence-update logic per type:

  CORRECTION   -> fast, near-immediate trust (old fact was simply wrong)
  STATE_CHANGE -> slower, dampened trust (situation evolved over time)
  REFINEMENT   -> no supersession at all; both stay active, linked as
                  "refines" (old fact still true, just incomplete)
  HYPOTHESIS   -> inert; stored and tagged, but never changes status or
                  confidence of anything (it's a possible future change,
                  not a confirmed one)

This is the full conceptual 4-type taxonomy from the paper's framing.
CORRECTION and STATE_CHANGE get full confidence dynamics this week;
REFINEMENT and HYPOTHESIS get simple-but-real handling (see module
docstring discussion in the project's planning notes) — deepening
their dynamics is explicitly future work, not implemented here.
"""
import json
import re
from typing import List, Optional

from .memory_atoms import MemoryAtom, get_active_memories, get_memory_by_id, EVENT_TYPES_THAT_MAY_SUPERSEDE
from .llm_client import chat
from . import config

# Confidence threshold above which a STATE_CHANGE or CORRECTION actually
# commits the status flip. Stated explicitly here (and in the paper) so
# it's a documented design choice, not a hidden magic number.
SUPERSESSION_THRESHOLD = 0.7

# STATE_CHANGE confidence is dampened relative to the LLM's raw judgment,
# reflecting that "the situation changed" claims deserve more scrutiny
# than explicit "that was wrong" corrections before being fully trusted.
STATE_CHANGE_DAMPENER = 0.85


CLASSIFICATION_PROMPT_TEMPLATE = """You are analyzing whether a new statement changes any existing \
facts in a knowledge base. Read the new statement and the list of existing active facts below.

Existing active facts:
{existing_facts}

New statement: "{new_text}"

Does the new statement relate to any of the existing facts above? If yes, classify the \
relationship into EXACTLY ONE of these four types:

- CORRECTION: the old fact was simply wrong or false (e.g. a past decision is described as \
a mistake, or factually incorrect information is being fixed)
- STATE_CHANGE: the old fact was true at the time, but circumstances have since changed \
(e.g. a decision was valid before but has now been replaced due to new requirements)
- REFINEMENT: the old fact is still true and is NOT being replaced — this new statement just \
adds more detail or specificity to it
- HYPOTHESIS: this describes a possible future change under consideration, not a confirmed \
fact yet (e.g. "the team is discussing whether to...")

If the new statement does not relate to any existing fact, respond with relation: "none".

Respond ONLY with valid JSON in exactly this format, no other text:
{{"relation": "CORRECTION" | "STATE_CHANGE" | "REFINEMENT" | "HYPOTHESIS" | "none", \
"related_memory_id": "<memory_id or null>", "confidence": <float 0.0 to 1.0>}}
"""


def _format_existing_facts(active_memories: List[MemoryAtom]) -> str:
    if not active_memories:
        return "(none yet)"
    lines = [f"- [{m.memory_id}] {m.content}" for m in active_memories]
    return "\n".join(lines)


def _parse_classification_response(raw_text: str) -> Optional[dict]:
    """
    Defensive JSON parsing: small/fast free-tier models occasionally wrap
    JSON in markdown fences or add stray text. We try a couple of fallback
    strategies before giving up, so one malformed response doesn't crash
    a whole batch run.
    """
    if raw_text is None:
        return None
    text = raw_text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: find the first {...} block in the text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def classify_and_link(new_atom: MemoryAtom, prior_atoms: List[MemoryAtom]) -> dict:
    """
    Calls the LLM once to classify new_atom's relationship to prior active
    memories. Returns a dict with the parsed classification, defaulting
    to "none" if the call fails or returns unparseable output (logged,
    not crashed — see _parse_classification_response).
    """
    active = get_active_memories(prior_atoms)
    prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(
        existing_facts=_format_existing_facts(active),
        new_text=new_atom.content,
    )

    result = chat(messages=[{"role": "user", "content": prompt}], max_tokens=600)
    parsed = _parse_classification_response(result["text"])

    if parsed is None:
        print(f"[warn] could not parse classification for {new_atom.memory_id}, "
              f"defaulting to 'none'. Raw response: {result['text']!r}")
        return {"relation": "none", "related_memory_id": None, "confidence": 0.0,
                "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"]}

    parsed["input_tokens"] = result["input_tokens"]
    parsed["output_tokens"] = result["output_tokens"]
    return parsed


def apply_classification(new_atom: MemoryAtom, prior_atoms: List[MemoryAtom], classification: dict) -> None:
    """
    Mutates new_atom and the relevant prior atom in-place based on the
    classification, applying DIFFERENT logic per type. This is the
    actual novel mechanism: type determines how confidence/status update.
    """
    relation = classification.get("relation", "none")
    related_id = classification.get("related_memory_id")
    raw_confidence = float(classification.get("confidence", 0.0) or 0.0)

    if relation == "none" or not related_id:
        new_atom.status = "active"
        return

    old_atom = get_memory_by_id(prior_atoms, related_id)
    if old_atom is None:
        # Model referenced a memory_id that doesn't exist — treat as no relation
        # rather than crashing. Worth logging for error analysis.
        print(f"[warn] classification referenced unknown memory_id '{related_id}' "
              f"for {new_atom.memory_id}; treating as unrelated.")
        new_atom.status = "active"
        return

    new_atom.contradiction_type = relation.lower()

    if relation == "CORRECTION":
        # Fast trust: use the model's raw confidence directly, no dampening.
        confidence = raw_confidence
        new_atom.supersession_confidence = confidence
        if confidence >= SUPERSESSION_THRESHOLD:
            old_atom.status = "superseded"
            new_atom.status = "active"
            new_atom.supersedes_memory_ids.append(old_atom.memory_id)
        else:
            old_atom.status = "uncertain"
            new_atom.status = "active"

    elif relation == "STATE_CHANGE":
        # Slower trust: dampen the model's raw confidence before comparing
        # to threshold, reflecting that "circumstances changed" claims
        # deserve more scrutiny than explicit corrections.
        confidence = raw_confidence * STATE_CHANGE_DAMPENER
        new_atom.supersession_confidence = confidence
        if confidence >= SUPERSESSION_THRESHOLD:
            old_atom.status = "superseded"
            new_atom.status = "active"
            new_atom.supersedes_memory_ids.append(old_atom.memory_id)
        else:
            old_atom.status = "uncertain"
            new_atom.status = "active"

    elif relation == "REFINEMENT":
        # No supersession at all. Both memories remain active; we just
        # record the link so retrieval can surface them together.
        old_atom.status = "active"
        new_atom.status = "active"
        new_atom.refines_memory_ids.append(old_atom.memory_id)

    elif relation == "HYPOTHESIS":
        # Inert by design: stored and tagged, but never changes anything
        # else's status or confidence. It's a possible future change,
        # not a confirmed fact.
        old_atom.status = "active"  # explicitly untouched
        new_atom.status = "inert"

    else:
        # Unrecognized label from the model — treat conservatively as
        # unrelated rather than guessing.
        new_atom.status = "active"


def process_scenario_memories(atoms: List[MemoryAtom]) -> dict:
    """
    Runs classification over all atoms in a scenario, in timestamp order,
    mutating their status/links as it goes. Returns token-cost totals so
    callers can log this against the spec's required token-cost metric.

    Only atoms whose event_type is in EVENT_TYPES_THAT_MAY_SUPERSEDE get
    a classification call — confirmation/requirement events are assumed
    not to contradict anything, saving API calls (see memory_atoms.py).
    """
    total_input_tokens = 0
    total_output_tokens = 0
    classifications_log = []

    for i, atom in enumerate(atoms):
        if atom.event_type not in EVENT_TYPES_THAT_MAY_SUPERSEDE:
            continue  # leave as default "active", no LLM call needed

        prior_atoms = atoms[:i]  # only memories that existed before this one, in time
        if not prior_atoms:
            continue  # nothing to possibly contradict yet

        classification = classify_and_link(atom, prior_atoms)
        apply_classification(atom, prior_atoms, classification)

        total_input_tokens += classification.get("input_tokens", 0)
        total_output_tokens += classification.get("output_tokens", 0)
        classifications_log.append({
            "memory_id": atom.memory_id,
            "relation": classification.get("relation"),
            "related_memory_id": classification.get("related_memory_id"),
            "confidence": classification.get("confidence"),
            "final_status": atom.status,
        })

    return {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "classifications": classifications_log,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src import config
    from src.data_loader import load_scenarios
    from src.memory_atoms import extract_memory_atoms

    config.check_config()
    scenarios = load_scenarios(
        config.SCENARIO_DOCS_PATH, config.EVENTS_PATH, config.QUESTIONS_PATH
    )
    s001 = scenarios["S001"]
    atoms = extract_memory_atoms(s001)

    print("Running classification on S001 (this will make a few LLM calls)...")
    result = process_scenario_memories(atoms)

    print(f"\nToken cost: {result['input_tokens']} in, {result['output_tokens']} out")
    print("\nClassifications:")
    for c in result["classifications"]:
        print(f"  {c}")

    print("\nFinal memory states:")
    for a in atoms:
        print(f"  {a.memory_id} [{a.status}] type={a.contradiction_type} "
              f"supersedes={a.supersedes_memory_ids} refines={a.refines_memory_ids} "
              f"conf={a.supersession_confidence}")
