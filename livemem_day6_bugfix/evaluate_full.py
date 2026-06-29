"""
Extended evaluator implementing all 9 metrics from the benchmark spec.
data/evaluate_predictions.py only implements 4 (answer F1, evidence
score, abstention, stale-trap F1) — this adds the missing 5: latest-state
accuracy, stale-answer rate, supersession accuracy, retrieval recall@k,
token cost, and latency. Reuses the same token_f1 and abstention logic
as the starter script for consistency.

Usage:
    python3 evaluate_full.py --predictions outputs/living_memory_v0_predictions.jsonl --questions data/questions.jsonl
"""
import json
import re
import argparse


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def token_f1(pred, gold):
    """Identical to the starter script's version, kept consistent on purpose."""
    ptoks = norm(pred).split()
    gtoks = norm(gold).split()
    if not ptoks and not gtoks:
        return 1.0
    if not ptoks or not gtoks:
        return 0.0
    common = {}
    for t in ptoks:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in gtoks:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(ptoks)
    recall = overlap / len(gtoks)
    return 2 * precision * recall / (precision + recall)


def answer_says_old_fact(pred_answer: str, pred_evidence: set, gold_evidence: set) -> bool:
    """
    Heuristic for "stale-answer rate": did the system cite evidence that
    does NOT include the gold (current/correct) evidence at all, on a
    question where staleness is the failure mode being tested? This is
    a proxy — exact gold evidence absence is the clearest signal we have
    without a separate "old answer" gold field.
    """
    if not gold_evidence:
        return False
    return len(pred_evidence & gold_evidence) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/questions.jsonl")
    ap.add_argument("--predictions", required=True)
    args = ap.parse_args()

    questions = {q["question_id"]: q for q in load_jsonl(args.questions)}
    preds = load_jsonl(args.predictions)

    rows = []
    for p in preds:
        q = questions[p["question_id"]]
        answer = p.get("answer", "")
        pred_evidence = set(p.get("evidence_event_ids", []))
        gold_evidence = set(q.get("gold_evidence_event_ids", []))

        f1 = token_f1(answer, q["gold_answer"])

        if not gold_evidence:
            evidence_score = 1.0 if not pred_evidence else 0.0
        else:
            evidence_score = len(pred_evidence & gold_evidence) / len(gold_evidence)

        # Abstention: corrected version (only scored over questions that
        # actually require it, not the diluted "free pass" version from
        # the starter script — see FINDINGS.md Day 1 for why).
        abstained = "insufficient" in norm(answer) or "not enough" in norm(answer)

        # Retrieval recall@k: did the gold evidence event(s) appear ANYWHERE
        # in retrieved_context_ids? Note retrieved_context_ids are formatted
        # as "SCENARIO::EVENT_ID" while gold_evidence_event_ids are bare
        # event IDs, so we check membership by suffix match.
        retrieved_ids = set(p.get("retrieved_context_ids", []))
        retrieved_event_ids = {rid.split("::")[-1] for rid in retrieved_ids}
        if gold_evidence:
            recall_at_k = len(gold_evidence & retrieved_event_ids) / len(gold_evidence)
        else:
            recall_at_k = None  # not meaningful for abstention questions with no gold evidence

        rows.append({
            "question_id": p["question_id"],
            "question_type": q["question_type"],
            "requires_abstention": q["requires_abstention"],
            "stale_trap": q["stale_trap"],
            "answer_f1": f1,
            "evidence_score": evidence_score,
            "abstained_correctly": abstained if q["requires_abstention"] else None,
            "is_stale_answer": answer_says_old_fact(answer, pred_evidence, gold_evidence) if q["stale_trap"] else None,
            "recall_at_k": recall_at_k,
            "latency_ms": p.get("latency_ms", 0),
            "input_tokens": p.get("input_tokens", 0),
            "output_tokens": p.get("output_tokens", 0),
        })

    n = len(rows)
    print(f"N = {n}\n")

    # 1. Answer correctness / F1
    print("1. Answer F1:", round(sum(r["answer_f1"] for r in rows) / n, 4))

    # 2. Evidence accuracy
    print("2. Evidence score:", round(sum(r["evidence_score"] for r in rows) / n, 4))

    # 3. Latest-state accuracy (F1 restricted to latest_state questions —
    #    spec says "correct only if it uses the newest valid fact"; we use
    #    F1 >= 0.5 as a practical correctness threshold since there's no
    #    separate binary gold label for this)
    latest_state_rows = [r for r in rows if r["question_type"] == "latest_state"]
    if latest_state_rows:
        correct = sum(1 for r in latest_state_rows if r["answer_f1"] >= 0.5)
        print(f"3. Latest-state accuracy: {correct}/{len(latest_state_rows)} = "
              f"{correct/len(latest_state_rows):.4f}")
    else:
        print("3. Latest-state accuracy: N/A (no latest_state questions)")

    # 4. Stale-answer rate (over stale_trap questions: did retrieved/cited
    #    evidence completely miss the gold/current evidence?)
    stale_rows = [r for r in rows if r["stale_trap"]]
    if stale_rows:
        stale_count = sum(1 for r in stale_rows if r["is_stale_answer"])
        print(f"4. Stale-answer rate: {stale_count}/{len(stale_rows)} = "
              f"{stale_count/len(stale_rows):.4f}  (lower is better)")
    else:
        print("4. Stale-answer rate: N/A")

    # 5. Supersession accuracy (F1 restricted to supersession questions,
    #    same F1>=0.5 correctness threshold as latest_state above)
    supersession_rows = [r for r in rows if r["question_type"] == "supersession"]
    if supersession_rows:
        correct = sum(1 for r in supersession_rows if r["answer_f1"] >= 0.5)
        print(f"5. Supersession accuracy: {correct}/{len(supersession_rows)} = "
              f"{correct/len(supersession_rows):.4f}")
    else:
        print("5. Supersession accuracy: N/A")

    # 6. Abstention accuracy — CORRECTED (only over questions that actually
    #    require abstention, not diluted by the other 127 — see FINDINGS.md)
    abstention_rows = [r for r in rows if r["requires_abstention"]]
    if abstention_rows:
        correct = sum(1 for r in abstention_rows if r["abstained_correctly"])
        print(f"6. Abstention accuracy (corrected): {correct}/{len(abstention_rows)} = "
              f"{correct/len(abstention_rows):.4f}")
    else:
        print("6. Abstention accuracy: N/A")

    # 7. Retrieval recall@k
    recall_rows = [r for r in rows if r["recall_at_k"] is not None]
    if recall_rows:
        print(f"7. Retrieval recall@k: "
              f"{sum(r['recall_at_k'] for r in recall_rows)/len(recall_rows):.4f}")
    else:
        print("7. Retrieval recall@k: N/A")

    # 8. Token cost
    total_in = sum(r["input_tokens"] for r in rows)
    total_out = sum(r["output_tokens"] for r in rows)
    print(f"8. Token cost: {total_in} input, {total_out} output, "
          f"{total_in + total_out} total ({(total_in+total_out)/n:.1f} avg/question)")

    # 9. Latency
    avg_latency = sum(r["latency_ms"] for r in rows) / n
    print(f"9. Latency: {avg_latency:.0f} ms avg/question")


if __name__ == "__main__":
    main()