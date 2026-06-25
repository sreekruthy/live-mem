"""
Day 2: Atomic memory extraction.

Walks a scenario's events in timestamp order and creates one MemoryAtom
per event. This is pure parsing — no LLM calls — so it costs nothing
against your Cerebras rate limit. Classification (Day 3) is a separate
pass over these atoms.

Why one atom per event, not one per "fact": event_type tags already
mark decision/update/confirmation/rejected_idea, which is a reasonable
proxy for "this event introduces or changes a fact." Splitting further
would need NLP we're explicitly skipping for week 1 (see config note
on EVENT_TYPES_THAT_MAY_SUPERSEDE).
"""
from dataclasses import dataclass, field
from typing import List, Optional, Literal
from collections import defaultdict

from .data_loader import Scenario, Event

Status = Literal["active", "superseded", "uncertain", "inert"]
ContradictionType = Literal["correction", "state_change", "refinement", "hypothesis"]

# Event types that might plausibly supersede a prior memory. Confirmation
# and requirement events rarely contradict anything, so we skip running
# the (costly) classification LLM call on them entirely — saves API calls
# without losing real signal, since these types essentially never replace
# an existing fact in this benchmark's design.
EVENT_TYPES_THAT_MAY_SUPERSEDE = {"update", "decision", "rejected_idea"}


@dataclass
class MemoryAtom:
    memory_id: str
    scenario_id: str
    content: str
    evidence_event_ids: List[str]
    event_type: str
    timestamp: str
    status: Status = "active"

    # Filled in during Day 3 classification, empty until then
    contradiction_type: Optional[ContradictionType] = None
    supersedes_memory_ids: List[str] = field(default_factory=list)
    refines_memory_ids: List[str] = field(default_factory=list)
    supersession_confidence: Optional[float] = None


def extract_memory_atoms(scenario: Scenario) -> List[MemoryAtom]:
    """
    One atom per event, in timestamp order (events are already sorted by
    data_loader.load_events). Status starts as "active" for everything;
    Day 3's classification step is what may flip some to "superseded"
    or "uncertain", or mark links for "refines".
    """
    atoms = []
    for ev in scenario.events:
        atoms.append(MemoryAtom(
            memory_id=f"MEM_{ev.event_id}",
            scenario_id=ev.scenario_id,
            content=ev.text,
            evidence_event_ids=[ev.event_id],
            event_type=ev.event_type,
            timestamp=ev.timestamp,
        ))
    return atoms


def get_active_memories(atoms: List[MemoryAtom]) -> List[MemoryAtom]:
    """Memories currently considered valid/current."""
    return [a for a in atoms if a.status == "active"]


def get_memory_by_id(atoms: List[MemoryAtom], memory_id: str) -> Optional[MemoryAtom]:
    for a in atoms:
        if a.memory_id == memory_id:
            return a
    return None


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src import config
    from src.data_loader import load_scenarios

    scenarios = load_scenarios(
        config.SCENARIO_DOCS_PATH, config.EVENTS_PATH, config.QUESTIONS_PATH
    )
    s001 = scenarios["S001"]
    atoms = extract_memory_atoms(s001)
    print(f"Extracted {len(atoms)} memory atoms for S001:")
    for a in atoms:
        print(f"  {a.memory_id} [{a.event_type}] {a.content[:60]}...")
