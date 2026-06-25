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

---

## Day 5 — First real generation test (S001, 8 questions)

### Result: 6/8 fully correct, 2 with specific fixable issues — no failures, no crashes

| Q | Type | Verdict |
|---|---|---|
| Q001 latest_state | Correct (FastAPI), extra irrelevant evidence ID included |
| Q002 historical_recall | Exact match |
| Q003 supersession | **Answer correct, but evidence missing E004** (only cited E001, gold wants both E001+E004) |
| Q004 multi_hop | Correct, but 57.7s latency outlier (others ran 0.7-6.8s) — watch if this recurs at scale |
| Q005 negative_recall | Correct but incomplete (same gap Vanilla RAG had — doesn't mention "small API surface, no ORM needed") |
| Q006 latest_state | Exact match |
| Q007 disambiguation | Correct |
| Q008 provenance | **Correct event, but answer text said "MEM_E004" instead of "E004"** — likely costs F1 points since gold is exactly "E004." |

### Two real bugs found and fixed in retrieval.py's prompt

1. **Supersession questions don't reliably cite both the old and new event
   ID**, even though both were correctly retrieved. Added explicit
   instruction: "For supersession questions specifically, mention BOTH
   the original event ID and the event ID that replaced it."

2. **Provenance-style answers echoed the internal "MEM_E004" memory_id
   format instead of the raw "E004" event_id** in the visible answer
   text (the Evidence: line was correct, only the prose answer was
   affected). Added explicit instruction not to use "MEM_" prefix in
   answer text. This was a prompt-wording gap, not a retrieval or
   classification bug — the right information was always present.

### Known pre-existing gap (carried over from Vanilla RAG, not new)

Q005 (Django negative_recall) still doesn't surface the *reason*
(API surface, ORM) even though E006's full text contains it and was
retrieved. Same limitation Vanilla RAG had on this exact question.
Worth checking after the prompt fix above whether this incidentally
improves too, or whether it needs separate attention.

**Action:** re-run S001's --limit 8 test after the prompt fix, confirm
Q003 and Q008 are now correct, before running the full 144-question set.

---

## Day 5/6 — Full 144-question run + a real, significant bug found and fixed

### Headline result (BEFORE fix)

| Metric | Vanilla RAG | Living Memory v0 (buggy) |
|---|---|---|
| Answer F1 | 0.3957 | 0.3919 |
| Evidence score | 0.9317 | 0.8576 |
| Abstention OK (corrected) | 0.8235 (14/17) | 0.8824 (15/17) |
| Stale-trap F1 | 0.3220 | **0.2559 (WORSE)** |

Stale-trap F1 — the single most important metric for this whole project
— got WORSE, not better. This needed real investigation rather than
acceptance, since it directly contradicts the method's core claim.

### Root cause, found via step-by-step tracing (S010_Q005)

Question: "What does backend own now?" Gold: "Only the API that reads
from Supabase" (the fact is in E004). Living Memory v0 answered
"Insufficient evidence" — a false abstention on a clearly answerable
question.

**Trace:** E004's embedding similarity to the question was 0.4543 (highest
of all 6 events in the scenario) — so retrieval was never the problem.
The actual bug: E004 had been incorrectly marked `superseded` during
classification.

**Why:** E006's text ("The old backend-owned ETL assignment is no longer
valid") is clearly about E001 (the ORIGINAL backend-owned assignment).
But `_format_existing_facts()` only showed the model ACTIVE memories —
and by the time E006 was classified, E001 had already been marked
superseded (correctly, by an earlier event) and was therefore HIDDEN
from the classification prompt. With E001 invisible, the model had to
guess among what it could see, and incorrectly targeted E004 (the
current, still-active fact) instead — wrongly superseding it.

**Confirmed via determinism test:** ran E006's exact classification call
3 times with the CORRECT prior state (E001 visible) -> got STATE_CHANGE
/ MEM_E001 / 0.97 confidence, identical all 3 times. The model itself
is reliable and correct GIVEN the right input — the bug was entirely in
what the pipeline showed it, not in the model's reasoning.

### The fix

`_format_existing_facts()` now shows ALL prior memories regardless of
status, each labeled (ACTIVE / SUPERSEDED / UNCERTAIN), instead of
filtering to active-only before the model ever sees them. The
classification prompt was updated to explain that a superseded fact may
still legitimately be referenced (e.g. a later event re-confirming it's
no longer valid) without that meaning it should be "re-superseded" in a
confusing way.

### Why this finding is actually good for the paper, not just a bug fix

This is a genuine, mechanism-level finding, not just a debugging note:
**hiding historical state from a classifier creates exactly the kind of
misattribution error our method is supposed to prevent, just one level
up.** Worth a dedicated paragraph in the paper — something like: "early
in development, we discovered that naively restricting classification
context to active memories causes the classifier to misattribute vague
references to already-superseded facts onto unrelated active memories,
itself producing a stale/incorrect-supersession error. This motivated
showing the full memory history, status-labeled, to the classification
step, while still restricting RETRIEVAL (not classification) by status."

**Action:** re-run the full 144 questions with the fix, get corrected
metrics, specifically re-check whether stale-trap F1 now actually beats
Vanilla RAG's 0.3220.
