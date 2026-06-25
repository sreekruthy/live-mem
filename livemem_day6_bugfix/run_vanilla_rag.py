"""
Entry point: runs Vanilla RAG across all scenarios/questions and writes
a prediction JSONL file matching the exact schema the spec requires.

Usage:
    python3 run_vanilla_rag.py            # run on everything
    python3 run_vanilla_rag.py --limit 5  # run on just 5 questions (use this first!)
"""
import argparse
import json
import sys
from tqdm import tqdm

from src import config
from src.data_loader import load_scenarios
from src.vanilla_rag import answer_question


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process this many questions total (for a quick test run)."
    )
    parser.add_argument(
        "--output", type=str, default=f"{config.OUTPUTS_DIR}/vanilla_rag_predictions.jsonl",
    )
    args = parser.parse_args()

    config.check_config()

    print("Loading scenarios...")
    scenarios = load_scenarios(
        config.SCENARIO_DOCS_PATH, config.EVENTS_PATH, config.QUESTIONS_PATH
    )

    # Build a flat list of (scenario, question) pairs to process
    work_items = []
    for scenario in scenarios.values():
        for q in scenario.questions:
            work_items.append((scenario, q))

    if args.limit:
        work_items = work_items[: args.limit]

    print(f"Running Vanilla RAG on {len(work_items)} questions...")
    print(f"At ~25 requests/min, this will take roughly {len(work_items) / 25:.1f} minutes.\n")

    results = []
    errors = []
    for scenario, question in tqdm(work_items):
        try:
            result = answer_question(scenario, question.question)
            result["scenario_id"] = scenario.scenario_id
            result["question_id"] = question.question_id
            results.append(result)
        except Exception as e:
            # Don't let one bad call kill the whole run — log it and move on.
            # This matters a lot given the rate-limit risk discussed earlier.
            errors.append({"question_id": question.question_id, "error": str(e)})
            print(f"\n[ERROR] {question.question_id}: {e}", file=sys.stderr)

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nDone. Wrote {len(results)} predictions to {args.output}")
    if errors:
        print(f"WARNING: {len(errors)} questions failed and were skipped:")
        for e in errors:
            print(f"  - {e['question_id']}: {e['error']}")
        print("Re-run with --limit on just these question_ids if needed, or investigate the errors above.")


if __name__ == "__main__":
    main()
