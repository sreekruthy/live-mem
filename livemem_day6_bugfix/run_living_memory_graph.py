"""
Entry point: runs Living Memory + Graph across all scenarios/questions.
Per scenario: extract atoms -> classify (status) -> build status graph
-> answer all questions. Resumable at scenario granularity, same pattern
as run_graph_rag.py.

Usage:
    python3 run_living_memory_graph.py
    python3 run_living_memory_graph.py --limit 3
    python3 run_living_memory_graph.py --restart
"""
import argparse
import json
import os
import sys
from tqdm import tqdm

from src import config
from src.data_loader import load_scenarios
from src.memory_atoms import extract_memory_atoms
from src.contradiction_classifier import process_scenario_memories
from src.living_memory_graph import build_status_graph, answer_question


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
    parser.add_argument("--output", type=str, default=f"{config.OUTPUTS_DIR}/living_memory_graph_predictions.jsonl")
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
            continue
        scenarios_to_process.append(scenario)

    if args.limit:
        scenarios_to_process = scenarios_to_process[: args.limit]

    if not scenarios_to_process:
        print("Nothing left to do — all scenarios already completed.")
        return

    print(f"Running Living Memory + Graph on {len(scenarios_to_process)} remaining scenarios "
          f"({sum(len(s.questions) for s in scenarios_to_process)} questions)...")
    print("(1 classification pass + 1 graph extraction + 1 generation call per question.)\n")

    errors = []
    total_class_in, total_class_out = 0, 0
    total_graph_in, total_graph_out = 0, 0

    with open(args.output, "a", encoding="utf-8") as f:
        for scenario in tqdm(scenarios_to_process, desc="scenarios"):
            try:
                atoms = extract_memory_atoms(scenario)
                class_result = process_scenario_memories(atoms)
                total_class_in += class_result["input_tokens"]
                total_class_out += class_result["output_tokens"]

                graph, graph_in, graph_out = build_status_graph(atoms)
                total_graph_in += graph_in
                total_graph_out += graph_out
            except Exception as e:
                errors.append({"scenario_id": scenario.scenario_id, "stage": "setup", "error": str(e)})
                print(f"\n[ERROR] setup failed for {scenario.scenario_id}: {e}", file=sys.stderr)
                continue

            for question in scenario.questions:
                if question.question_id in already_done:
                    continue
                try:
                    result = answer_question(scenario, question.question, question.question_type, atoms, graph)
                    result["scenario_id"] = scenario.scenario_id
                    result["question_id"] = question.question_id
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                except Exception as e:
                    errors.append({"question_id": question.question_id, "stage": "generation", "error": str(e)})
                    print(f"\n[ERROR] {question.question_id}: {e}", file=sys.stderr)

    print(f"\nDone this run.")
    print(f"Classification tokens: {total_class_in} in, {total_class_out} out")
    print(f"Graph extraction tokens: {total_graph_in} in, {total_graph_out} out")
    if errors:
        print(f"WARNING: {len(errors)} items failed this run:")
        for e in errors:
            print(f"  - {e}")
        print("Run this script again to retry — completed scenarios/questions are skipped.")


if __name__ == "__main__":
    main()