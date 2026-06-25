"""
Parses the benchmark's raw JSONL files into clean Python objects, grouped
by scenario_id throughout. Every other system (Vanilla RAG, Rerank RAG,
Living Memory v0, etc.) imports from this file instead of re-parsing the
raw data itself.

Why scenario-grouping matters: the spec requires "index each scenario
separately. Do not retrieve across different scenarios." Grouping by
scenario_id here, at the source, makes that constraint structural rather
than something every downstream script has to remember to enforce.
"""
import json
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict, Optional


@dataclass
class Event:
    event_id: str
    scenario_id: str
    timestamp: str
    actor: str
    event_type: str
    text: str


@dataclass
class Question:
    scenario_id: str
    question_id: str
    question: str
    question_type: str
    gold_answer: str
    gold_evidence_event_ids: List[str]
    requires_abstention: bool
    stale_trap: bool
    evaluation_notes: str = ""


@dataclass
class Scenario:
    scenario_id: str
    title: str
    domain: str
    document: str  # raw document text, used by chunk-based RAG baselines
    events: List[Event] = field(default_factory=list)
    questions: List[Question] = field(default_factory=list)


def _load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_event_line(raw_text: str, scenario_id: str) -> List[Event]:
    """
    Fallback parser: some scenario documents embed events inline as
    "[E004 | 2026-01-08T10:10:00 | Aria | update] text..." lines.
    Not used if events.jsonl already has clean structured rows — kept
    here only as a safety net. Prefer load_events() below.
    """
    events = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if not line.startswith("["):
            continue
        try:
            header, text = line.split("]", 1)
            header = header.strip("[]")
            event_id, timestamp, actor, event_type = [p.strip() for p in header.split("|")]
            events.append(Event(
                event_id=event_id,
                scenario_id=scenario_id,
                timestamp=timestamp,
                actor=actor,
                event_type=event_type,
                text=text.strip(),
            ))
        except ValueError:
            continue  # skip malformed lines rather than crashing the whole load
    return events


def load_events(path: str) -> Dict[str, List[Event]]:
    """Returns events grouped by scenario_id, sorted by timestamp within each scenario."""
    rows = _load_jsonl(path)
    by_scenario: Dict[str, List[Event]] = defaultdict(list)

    for row in rows:
        ev = Event(
            event_id=row["event_id"],
            scenario_id=row["scenario_id"],
            timestamp=row["timestamp"],
            actor=row.get("speaker", "unknown"),  # raw field is "speaker", not "actor"
            event_type=row.get("event_type", "unknown"),
            text=row["text"],
        )
        by_scenario[ev.scenario_id].append(ev)

    for scenario_id in by_scenario:
        by_scenario[scenario_id].sort(key=lambda e: e.timestamp)

    return dict(by_scenario)


def load_questions(path: str) -> Dict[str, List[Question]]:
    """Returns questions grouped by scenario_id."""
    rows = _load_jsonl(path)
    by_scenario: Dict[str, List[Question]] = defaultdict(list)

    for row in rows:
        q = Question(
            scenario_id=row["scenario_id"],
            question_id=row["question_id"],
            question=row["question"],
            question_type=row["question_type"],
            gold_answer=row["gold_answer"],
            gold_evidence_event_ids=row.get("gold_evidence_event_ids", []),
            requires_abstention=row.get("requires_abstention", False),
            stale_trap=row.get("stale_trap", False),
            evaluation_notes=row.get("evaluation_notes", ""),
        )
        by_scenario[q.scenario_id].append(q)

    return dict(by_scenario)


def load_scenarios(
    scenario_docs_path: str,
    events_path: str,
    questions_path: str,
) -> Dict[str, Scenario]:
    """
    The main entry point. Returns a dict of scenario_id -> Scenario,
    each fully populated with its events and questions.
    """
    doc_rows = _load_jsonl(scenario_docs_path)
    events_by_scenario = load_events(events_path)
    questions_by_scenario = load_questions(questions_path)

    scenarios = {}
    for row in doc_rows:
        sid = row["scenario_id"]
        scenarios[sid] = Scenario(
            scenario_id=sid,
            title=row.get("title", ""),
            domain=row.get("domain", ""),
            document=row.get("document", ""),
            events=events_by_scenario.get(sid, []),
            questions=questions_by_scenario.get(sid, []),
        )
    return scenarios


def get_event_by_id(scenario: Scenario, event_id: str) -> Optional[Event]:
    """Small helper used constantly when building evidence/context references."""
    for ev in scenario.events:
        if ev.event_id == event_id:
            return ev
    return None


if __name__ == "__main__":
    # Quick self-test: run `python3 src/data_loader.py` to sanity-check parsing.
    import sys
    sys.path.insert(0, ".")
    from src import config

    scenarios = load_scenarios(
        config.SCENARIO_DOCS_PATH, config.EVENTS_PATH, config.QUESTIONS_PATH
    )
    print(f"Loaded {len(scenarios)} scenarios.")
    total_events = sum(len(s.events) for s in scenarios.values())
    total_questions = sum(len(s.questions) for s in scenarios.values())
    print(f"Total events: {total_events}, total questions: {total_questions}")

    sample = scenarios["S001"]
    print(f"\nScenario S001: {sample.title} ({sample.domain})")
    print(f"  Events: {len(sample.events)}, Questions: {len(sample.questions)}")
    print(f"  First event: {sample.events[0]}")
    print(f"  First question: {sample.questions[0]}")
