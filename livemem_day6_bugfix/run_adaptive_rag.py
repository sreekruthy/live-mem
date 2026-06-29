"""
Entry point: runs Adaptive RAG across all scenarios/questions.
Resumable, same pattern as run_rerank_rag.py and run_contextual_retrieval.py.

Usage:
    python3 run_adaptive_rag.py
    python3 run_adaptive_rag.py --limit 40
    python3 run_adaptive_rag.py --restart
"""
import argparse
import json
import os
import sys
from collections import Counter
from tqdm import tqdm

from src import config
from src.data_loader import load_scenarios
from src.adaptive_rag import answer_question


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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=f"{config.OUTPUTS_DIR}/adaptive_rag_predictions.jsonl")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    config.check_config()

    if args.restart and os.path.exists(args.output):
        os.remove(args.output)
        print(f"--restart given: deleted existing {args.output}")

    already_done = load_completed_ids(args.output)
    if already_done:
        print(f"Found {len(already_done)} already-completed questions — skipping those.")

    print("Loading scenarios...")
    scenarios = load_scenarios(
        config.SCENARIO_DOCS_PATH, config.EVENTS_PATH, config.QUESTIONS_PATH
    )

    work_items = []
    for scenario in scenarios.values():
        for q in scenario.questions:
            if q.question_id in already_done:
                continue
            work_items.append((scenario, q))

    if args.limit:
        work_items = work_items[: args.limit]

    if not work_items:
        print("Nothing left to do — all questions already completed.")
        return

    print(f"Running Adaptive RAG on {len(work_items)} remaining questions...")
    print("(2 LLM calls per question — classify + generate — similar cost to Rerank RAG.)\n")

    errors = []
    strategy_counts = Counter()
    with open(args.output, "a", encoding="utf-8") as f:
        for scenario, question in tqdm(work_items):
            try:
                result = answer_question(scenario, question.question)
                result["scenario_id"] = scenario.scenario_id
                result["question_id"] = question.question_id
                strategy_counts[result.get("_strategy_used", "?")] += 1
                f.write(json.dumps(result) + "\n")
                f.flush()
                os.fsync(f.fileno())
            except Exception as e:
                errors.append({"question_id": question.question_id, "error": str(e)})
                print(f"\n[ERROR] {question.question_id}: {e}", file=sys.stderr)

    total_done = len(already_done) + len(work_items) - len(errors)
    print(f"\nDone this run. {total_done} total predictions now in {args.output}")
    print(f"Strategy distribution this run: {dict(strategy_counts)}")
    if errors:
        print(f"WARNING: {len(errors)} questions failed this run (will retry next run):")
        for e in errors:
            print(f"  - {e['question_id']}: {e['error']}")
        print("Just run this script again to retry the failed ones.")


if __name__ == "__main__":
    main()