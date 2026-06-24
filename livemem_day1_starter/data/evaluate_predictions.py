import json, re, argparse
from pathlib import Path

def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def token_f1(pred, gold):
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="questions.jsonl")
    ap.add_argument("--predictions", required=True)
    args = ap.parse_args()

    questions = {q["question_id"]: q for q in load_jsonl(args.questions)}
    preds = load_jsonl(args.predictions)

    rows = []
    for p in preds:
        q = questions[p["question_id"]]
        f1 = token_f1(p.get("answer", ""), q["gold_answer"])
        pred_evidence = set(p.get("evidence_event_ids", []))
        gold_evidence = set(q.get("gold_evidence_event_ids", []))
        if not gold_evidence:
            evidence_score = 1.0 if not pred_evidence else 0.0
        else:
            evidence_score = len(pred_evidence & gold_evidence) / len(gold_evidence)
        abstention_ok = (q["requires_abstention"] and ("insufficient" in norm(p.get("answer", "")) or "not enough" in norm(p.get("answer", "")))) or not q["requires_abstention"]
        rows.append({
            "question_id": p["question_id"],
            "question_type": q["question_type"],
            "answer_f1": f1,
            "evidence_score": evidence_score,
            "abstention_ok": float(abstention_ok),
            "stale_trap": q["stale_trap"]
        })

    n = len(rows)
    print("N", n)
    print("Answer F1", round(sum(r["answer_f1"] for r in rows) / n, 4))
    print("Evidence score", round(sum(r["evidence_score"] for r in rows) / n, 4))
    print("Abstention OK", round(sum(r["abstention_ok"] for r in rows) / n, 4))
    stale = [r for r in rows if r["stale_trap"]]
    if stale:
        print("Stale-trap Answer F1", round(sum(r["answer_f1"] for r in stale) / len(stale), 4))

if __name__ == "__main__":
    main()
