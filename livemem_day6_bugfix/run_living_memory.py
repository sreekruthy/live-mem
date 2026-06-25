"""
Entry point: runs Living Memory v0 across all scenarios/questions and
writes a prediction JSONL file in the same schema as run_vanilla_rag.py,
so the two are directly comparable by evaluate_predictions.py.

Processing order per scenario: extract atoms -> classify (Day 2-3) ->
answer every question for that scenario using the classified memory
state (Day 4-5). Classification happens ONCE per scenario, then all of
that scenario's questions reuse the same classified atoms — we don't
re-classify per question, since the memory state doesn't change between
questions about the same scenario.

Usage:
    python3 run_living_memory.py            # run on everything
    python3 run_living_memory.py --limit 5  # quick test on 5 questions
"""
import argparse
import json
import sys
from tqdm import tqdm

from src import config
from src.data_loader import load_scenarios
from src.memory_atoms import extract_memory_atoms
from src.contradiction_classifier import process_scenario_memories
from src.living_memory import answer_question


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process this many questions total (for a quick test run)."
    )
    parser.add_argument(
        "--output", type=str, default=f"{config.OUTPUTS_DIR}/living_memory_v0_predictions.jsonl",
    )
    args = parser.parse_args()

    config.check_config()

    print("Loading scenarios...")
    scenarios = load_scenarios(
        config.SCENARIO_DOCS_PATH, config.EVENTS_PATH, config.QUESTIONS_PATH
    )

    total_questions = sum(len(s.questions) for s in scenarios.values())
    if args.limit:
        total_questions = min(total_questions, args.limit)
    print(f"Running Living Memory v0 on up to {total_questions} questions across "
          f"{len(scenarios)} scenarios...")
    print(f"(Classification + generation calls combined — expect this to take "
          f"longer than Vanilla RAG did, given the extra classification pass.)\n")

    results = []
    errors = []
    classification_token_log = []
    questions_processed = 0

    for scenario in scenarios.values():
        if args.limit and questions_processed >= args.limit:
            break

        # Classify once per scenario, reuse for all its questions
        # Individual memory classification failures are now handled inside
        # process_scenario_memories (defaults to "none" and continues), so
        # this outer except is a safety net for truly unexpected errors —
        # it should rarely trigger now.
        atoms = extract_memory_atoms(scenario)
        try:
            class_result = process_scenario_memories(atoms)
            classification_token_log.append({
                "scenario_id": scenario.scenario_id,
                "input_tokens": class_result["input_tokens"],
                "output_tokens": class_result["output_tokens"],
            })
        except Exception as e:
            errors.append({"scenario_id": scenario.scenario_id, "stage": "classification", "error": str(e)})
            print(f"\n[ERROR] classification failed for {scenario.scenario_id}: {e}", file=sys.stderr)
            continue  # skip this scenario's questions entirely if classification itself failed

        for question in tqdm(scenario.questions, desc=scenario.scenario_id):
            if args.limit and questions_processed >= args.limit:
                break
            try:
                result = answer_question(atoms, scenario, question.question, question.question_type)
                result["scenario_id"] = scenario.scenario_id
                result["question_id"] = question.question_id
                results.append(result)
            except Exception as e:
                errors.append({"question_id": question.question_id, "stage": "generation", "error": str(e)})
                print(f"\n[ERROR] {question.question_id}: {e}", file=sys.stderr)
            questions_processed += 1

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    total_class_in = sum(c["input_tokens"] for c in classification_token_log)
    total_class_out = sum(c["output_tokens"] for c in classification_token_log)

    print(f"\nDone. Wrote {len(results)} predictions to {args.output}")
    print(f"Classification token cost (separate from generation): "
          f"{total_class_in} in, {total_class_out} out")
    if errors:
        print(f"\nWARNING: {len(errors)} items failed:")
        for e in errors:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
