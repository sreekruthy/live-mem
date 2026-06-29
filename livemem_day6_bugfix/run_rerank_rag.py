"""
Entry point: runs Rerank RAG across all scenarios/questions and writes
a prediction JSONL file in the same schema as run_vanilla_rag.py.

RESUMABLE: if the output file already has predictions for some
questions (from a previous run that got interrupted, hit an hourly
quota wall, etc.), those are skipped and only the remaining questions
are processed. Results are written incrementally (one line at a time,
flushed immediately), so killing the process partway through never
loses already-completed work.

Usage:
    python3 run_rerank_rag.py            # run on everything not already done
    python3 run_rerank_rag.py --limit 40 # process up to 40 NEW questions this run
    python3 run_rerank_rag.py --restart  # ignore existing output, start fresh
"""
import argparse
import json
import os
import sys
from tqdm import tqdm

from src import config
from src.data_loader import load_scenarios
from src.rerank_rag import answer_question


def load_completed_ids(path: str) -> set:
    """Returns the set of question_ids already present in the output file."""
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
                continue  # skip any corrupted/partial line rather than crashing
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Process at most this many NEW (not-yet-done) questions this run.")
    parser.add_argument("--output", type=str, default=f"{config.OUTPUTS_DIR}/rerank_rag_predictions.jsonl")
    parser.add_argument("--restart", action="store_true",
                         help="Ignore any existing output file and start completely fresh.")
    args = parser.parse_args()

    config.check_config()

    if args.restart and os.path.exists(args.output):
        os.remove(args.output)
        print(f"--restart given: deleted existing {args.output}")

    already_done = load_completed_ids(args.output)
    if already_done:
        print(f"Found {len(already_done)} already-completed questions in {args.output} — skipping those.")

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

    print(f"Running Rerank RAG on {len(work_items)} remaining questions...")
    print("(Each question makes 2 LLM calls — rerank + generate.)\n")

    errors = []
    # Open in append mode and write incrementally, flushing after each
    # question, so progress is never lost even if the process is killed
    # or hits an unrecoverable error mid-run.
    with open(args.output, "a", encoding="utf-8") as f:
        for scenario, question in tqdm(work_items):
            try:
                result = answer_question(scenario, question.question)
                result["scenario_id"] = scenario.scenario_id
                result["question_id"] = question.question_id
                f.write(json.dumps(result) + "\n")
                f.flush()
                os.fsync(f.fileno())
            except Exception as e:
                errors.append({"question_id": question.question_id, "error": str(e)})
                print(f"\n[ERROR] {question.question_id}: {e}", file=sys.stderr)

    total_done = len(already_done) + len(work_items) - len(errors)
    print(f"\nDone this run. {total_done} total predictions now in {args.output}")
    if errors:
        print(f"WARNING: {len(errors)} questions failed this run (not written, will retry next run):")
        for e in errors:
            print(f"  - {e['question_id']}: {e['error']}")
        print("Just run this script again to retry the failed ones (already-done ones are skipped).")


if __name__ == "__main__":
    main()