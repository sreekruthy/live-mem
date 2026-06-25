# Findings Log — Living Memory v0 vs. Vanilla RAG

Running log of results and analysis, written as we go rather than all at
the end. This file is meant to be lifted almost directly into the paper's
results/error-analysis sections later.

---

## Day 1 — Vanilla RAG baseline (144/144 questions, full run)

### Headline metrics (from evaluate_predictions.py, with one correction — see below)

| Metric | Reported by script | Corrected | Notes |
|---|---|---|---|
| Answer F1 | 0.3957 | (same) | — |
| Evidence score | 0.9317 | (same) | — |
| Abstention OK | 0.9792 | **0.8235** | See bug note below |
| Stale-trap Answer F1 | 0.3220 | (same) | **This is the key number to beat** |

### Bug found in evaluate_predictions.py's abstention metric

The script's `abstention_ok` formula gives an automatic pass to every
question where `requires_abstention == False` (127 of 144 questions),
regardless of what the system answered. This inflates the printed
abstention score. The TRUE abstention accuracy — computed only over the
17 questions that actually require abstention — is **14/17 = 0.8235**,
not 0.9792.

**Action for the team:** flag this to whoever owns the shared evaluator,
so everyone's papers report the corrected number, not the inflated one.

### Error examples — Vanilla RAG abstention failures (3 of 17)

All three failures share the same pattern: the model reasons to a
confident, contextually-grounded "No," but doesn't use the required
"insufficient evidence" framing. This is a calibration failure, not a
hallucination — the model retrieved correct context in all 3 cases.

1. **S003_Q008** — "Is there evidence that Stripe was selected instead of Razorpay?"
   - Answered: "No, there is no evidence that Stripe was selected instead of Razorpay."
   - Gold: "Insufficient evidence. The events only mention Razorpay."

2. **S006_Q008** — "Is there evidence that company tags are stored as JSON objects?"
   - Answered: "No, the context only indicates that the company field is stored as an array, not as JSON objects."
   - Gold: "Insufficient evidence. The events say the company field is an array."

3. **S014_Q008** — "Is there evidence that all bugs skip approval?"
   - Answered: "No, the context does not indicate that all bugs skip approval; the workflow routes bugs through the normal backlog and requires approvals for critical bugs."
   - Gold: "Insufficient evidence. The workflow applies after required approvals."

**Why this matters for Living Memory v0:** if our method's `uncertain`
status produces more appropriately-hedged answers on borderline cases
like these, that's a direct, demonstrable advantage tied to a real
mechanism (not a vague claim). Worth specifically re-checking these 3
question_ids once Living Memory v0 is built.

---

## Day 2-3 — Living Memory v0 (memory extraction + 4-way classification)

### Logic verification (done before any API calls, see contradiction_classifier.py tests)

All 4 contradiction types produce correctly differentiated behavior:
- CORRECTION: fast trust, no dampening
- STATE_CHANGE: dampened by 0.85x before threshold check — confirmed a
  borderline case (raw 0.8 -> dampened 0.68) correctly falls below the
  0.7 threshold and produces "uncertain" instead of "superseded"
- REFINEMENT: no supersession, both memories stay active, linked
- HYPOTHESIS: fully inert, never changes anything else's status

### [Day 2-3 results — S001 first real run, confirmed working]

**Token cost for S001 (3 classification calls):** 1302 input, 882 output tokens.
Extrapolating to all 18 scenarios (~6 events each, ~3 classifiable events
per scenario on average): roughly 54 classification calls total, ~23K
input / ~16K output tokens for the full classification pass. Well within
Cerebras free-tier daily budget.

**Confirmed working as designed — the core mechanism, on real data:**

- MEM_E004 (Flask -> FastAPI) classified as **STATE_CHANGE** relative to
  MEM_E001, raw confidence 0.97 -> dampened to **0.8245** (0.97 x 0.85
  dampener) -> clears the 0.7 threshold -> MEM_E001 correctly flips to
  `superseded`. This is the canonical example from the start of this
  project, now working end-to-end with real LLM classification.

- MEM_E006 (Django rejection) classified as **REFINEMENT** of MEM_E004,
  confidence 0.9 -> MEM_E004 correctly stayed `active` (no supersession),
  link recorded.

- MEM_E002 (Supabase for auth) correctly classified as **"none"** —
  doesn't contradict anything.

- MEM_E003 (requirement) and MEM_E005 (confirmation) correctly skipped
  by the event_type filter — neither event type is in
  EVENT_TYPES_THAT_MAY_SUPERSEDE, and neither plausibly contradicts an
  existing fact. Verified against raw event_type tags in events.jsonl.

**Quirk to watch across more scenarios:** E006 (Django rejected) being
classified as "REFINEMENT of the FastAPI decision" is debatable — it
reads more like an independent negative fact than something that adds
detail to the FastAPI decision specifically. Worth checking if the
model over-uses REFINEMENT for "related but not really refining" cases
once more scenarios are run. Not a bug, but a pattern to note in the
paper's limitations section if it recurs.

**Still to run:** remaining 17 scenarios, then move to Day 4 (retrieval).

### Full run across all 18 scenarios — 48 classification calls

**Distribution:**
| Type | Count | % |
|---|---|---|
| STATE_CHANGE | 21 | 43.8% |
| REFINEMENT | 22 | 45.8% |
| none (no relation found) | 5 | 10.4% |
| CORRECTION | 0 | 0% |
| HYPOTHESIS | 0 | 0% |

**Important limitation, confirmed by checking the raw source data directly:**
LiveMemBench-v0's 108 events contain ZERO instances of correction-style
language ("mistake", "wrong", "turned out", etc.) and ZERO instances of
hypothesis-style tentative language ("considering", "might", "possibly",
etc.) anywhere in the dataset. This means the benchmark, by construction,
only ever presents STATE_CHANGE-style and REFINEMENT-style updates.

**This is a benchmark coverage gap, not a bug in our classifier.** Our
4-type taxonomy and its branch logic are independently verified correct
via unit tests (see contradiction_classifier.py test results, all 4
branches confirmed to produce distinct, correct behavior on synthetic
inputs). But this specific synthetic dataset cannot exercise the
CORRECTION or HYPOTHESIS branches with real data.

**Write-up implication:** state this explicitly in the paper's
Limitations section. Something like: "We verify all four contradiction-
type branches in isolation via unit testing, but LiveMemBench-v0's
synthetic scenarios contain only state-change and refinement-style
updates; evaluating CORRECTION and HYPOTHESIS handling against real-world
data with genuine factual corrections and tentative/future-looking
statements is necessary future work."

**Closest borderline call in the whole run:** S009's MEM_E002, classified
STATE_CHANGE with raw confidence 0.85 -> dampened to 0.7225 -> barely
clears the 0.7 threshold. Worth using as a concrete "this is what the
dampening mechanism is for" example in the paper, and worth a manual
read of S009's actual events to sanity-check this borderline call makes
sense.

**Errors handled gracefully:** 3 scenarios (S003, S011, S016) hit
"[empty content]" retries (the gpt-oss-120b empty-response issue from
Day 1) but all succeeded after 1-3 retries — confirms the retry logic
added on Day 1 generalizes correctly to this new module.



---

## Day 4 — Status-aware retrieval (logic verified + first real test on S001)

### Logic verification (offline unit tests, before any API/embedding calls)

Confirmed with synthetic atoms (status=superseded/active/inert mix):
- current-state question types (latest_state, simple_recall, etc.)
  structurally exclude `superseded` and `inert` memories from the
  eligible retrieval pool, before similarity scoring even runs
- history-aware question types (historical_recall, supersession,
  provenance) include `superseded` memories, but still exclude `inert`
  (a hypothesis was never confirmed, so it has no place in a history-of-
  changes answer either)
- context builder correctly surfaces "-> superseded by [X]" and
  "-> refines [X]" links inline with the memory text

### Real test on S001 (first live run, classification + retrieval combined)

**Current-state question** ("What backend framework is currently
selected?") retrieved 4 active memories (MEM_E004, MEM_E005, MEM_E006,
MEM_E003). **MEM_E001 (the superseded Flask fact) did NOT appear in the
results**, despite being the most lexically similar single event to the
question's wording ("backend framework"). This is the core mechanism
working exactly as intended — status filtering removed it from
eligibility before similarity scoring ran, so a plain similarity-only
retriever's most likely mistake (retrieving Flask because it's textually
about "backend framework") is structurally prevented here.

**Supersession question** ("Which earlier framework decision was
superseded?") correctly retrieved MEM_E001, labeled SUPERSEDED, with the
link spelled out: "-> superseded by [MEM_E004] ...". Same memory item,
opposite (correct) behavior depending on question type — confirms the
type-aware filtering branches correctly.

**Observation to watch in Day 5+:** with only 5-6 active events in a
small scenario like S001, top_k=4 retrieval pulls in some lower-relevance
memories (e.g. the Django rejection event showed up for the "current
framework" question) just because there isn't much else to fill the
slots. Not wrong, but worth checking once generation is wired in whether
this dilutes answer quality — if so, lowering TOP_K for small scenarios
is a simple fix to note as a tuning consideration in the paper.


---

## Day 5-7 — [to be filled in as completed]

### Generation + full prediction run results

### Living Memory v0 vs Vanilla RAG — head-to-head metrics table

### 5-10 error examples (tagged: stale answer / wrong evidence / hallucination / failed-to-abstain)

### Honest limitations
