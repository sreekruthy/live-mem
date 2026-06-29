"""
Entry point: runs GraphRAG across all scenarios/questions. Builds the
entity-relation graph ONCE per scenario (one LLM call), then answers
all of that scenario's questions using it — same pattern as
run_living_memory.py's classify-once-per-scenario approach.

Resumable: skips scenarios whose questions are already fully done.
Note: resume granularity here is per-SCENARIO, not per-question, since
the graph build needs to happen before any of that scenario's questions
can be answered — if a scenario is partially done, this re-does the
whole scenario (cheap: just one extra graph-extraction call wasted,
not a big cost).

Usage:
    python3 run_graph_rag.py
    python3 run_graph_rag.py --limit 3   # process up to 3 scenarios this run
    python3 run_graph_rag.py --restart
"""
import argparse
import json
import os
import sys
from tqdm import tqdm

from src import config
from src.data_loader import load_scenarios
from src.graph_rag import build_graph, answer_question


def load_completed_ids(path: str) -> set:
    completed = set()
    if not os.path.exists(path):
        return completed
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                completed.add(json.loads(line)["question_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Process at most this many scenarios this run.")
    parser.add_argument("--output", type=str, default=f"{config.OUTPUTS_DIR}/graph_rag_predictions.jsonl")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    config.check_config()

    if args.restart and os.path.exists(args.output):
        os.remove(args.output)
        print(f"--restart given: deleted existing {args.output}")

    already_done = load_completed_ids(args.output)
    if already_done:
        print(f"Found {len(already_done)} already-completed questions — will skip fully-done scenarios.")

    print("Loading scenarios...")
    scenarios = load_scenarios(
        config.SCENARIO_DOCS_PATH, config.EVENTS_PATH, config.QUESTIONS_PATH
    )

    scenarios_to_process = []
    for scenario in scenarios.values():
        question_ids = {q.question_id for q in scenario.questions}
        if question_ids and question_ids.issubset(already_done):
            continue  # this whole scenario is already done
        scenarios_to_process.append(scenario)

    if args.limit:
        scenarios_to_process = scenarios_to_process[: args.limit]

    if not scenarios_to_process:
        print("Nothing left to do — all scenarios already completed.")
        return

    print(f"Running GraphRAG on {len(scenarios_to_process)} remaining scenarios "
          f"({sum(len(s.questions) for s in scenarios_to_process)} questions)...")
    print("(1 extraction call per scenario + 1 generation call per question.)\n")

    errors = []
    total_extraction_in, total_extraction_out = 0, 0
    graph_size_log = []

    with open(args.output, "a", encoding="utf-8") as f:
        for scenario in tqdm(scenarios_to_process, desc="scenarios"):
            try:
                graph, ext_in, ext_out = build_graph(scenario)
                total_extraction_in += ext_in
                total_extraction_out += ext_out
                n_entities, n_edges, n_events = len(graph.entities), len(graph.edges), len(scenario.events)
                graph_size_log.append((scenario.scenario_id, n_entities, n_edges, n_events))
                print(f"  {scenario.scenario_id}: {n_events} events -> {n_entities} entities, {n_edges} edges")
            except Exception as e:
                errors.append({"scenario_id": scenario.scenario_id, "stage": "graph_build", "error": str(e)})
                print(f"\n[ERROR] graph build failed for {scenario.scenario_id}: {e}", file=sys.stderr)
                continue  # skip this scenario's questions if graph build itself failed

            for question in scenario.questions:
                if question.question_id in already_done:
                    continue
                try:
                    result = answer_question(scenario, question.question, graph)
                    result["scenario_id"] = scenario.scenario_id
                    result["question_id"] = question.question_id
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                except Exception as e:
                    errors.append({"question_id": question.question_id, "stage": "generation", "error": str(e)})
                    print(f"\n[ERROR] {question.question_id}: {e}", file=sys.stderr)

    print(f"\nDone this run. Graph extraction token cost: {total_extraction_in} in, {total_extraction_out} out")
    if errors:
        print(f"WARNING: {len(errors)} items failed this run:")
        for e in errors:
            print(f"  - {e}")
        print("Run this script again to retry — completed scenarios/questions are skipped.")


if __name__ == "__main__":
    main()