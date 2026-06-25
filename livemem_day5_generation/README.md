# LiveMemBench-v0 — Day 1 Setup

## What's in this folder

```
livemem_project/
├── data/                          # benchmark files (copied from the zip you got)
│   ├── scenario_documents.jsonl
│   ├── events.jsonl
│   ├── questions.jsonl
│   ├── gold_memory_atoms.jsonl    # DO NOT use this in retrieval/generation
│   ├── sample_gold_predictions.jsonl
│   └── evaluate_predictions.py
├── src/
│   ├── config.py                  # API keys, model names, constants
│   ├── data_loader.py              # parses events.jsonl / questions.jsonl into clean objects
│   ├── llm_client.py                # Cerebras client with rate-limit handling + retries
│   ├── embeddings.py                # embedding model wrapper (for retrieval)
│   └── vanilla_rag.py               # Day 1's actual baseline system
├── outputs/                        # prediction JSONL files land here
├── run_vanilla_rag.py               # entry point script
└── requirements.txt
```

## Setup steps (do these first, in order)

1. **Get a Cerebras API key**: go to https://cloud.cerebras.ai, sign up with email (no credit card needed), generate a key from the dashboard.

2. **Get an embedding model.** You need something to turn text into vectors for similarity search. Two options:
   - **Free, runs locally, no API needed**: `sentence-transformers` (e.g. `all-MiniLM-L6-v2`) — recommended, since it doesn't eat into your Cerebras token budget at all and has no rate limit.
   - Alternative: use Cerebras/another provider's embedding endpoint if you prefer everything API-based — not recommended here since it adds unnecessary token cost for a free-tier budget.

   This setup uses **sentence-transformers locally** — it's free, fast, and removes embeddings entirely from your rate-limit math.

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt --break-system-packages
   ```

4. **Set your API key as an environment variable** (don't hardcode it in any file):
   ```bash
   export CEREBRAS_API_KEY="your-key-here"
   ```

5. **Test the setup:**
   ```bash
   python3 -c "from src.llm_client import test_connection; test_connection()"
   ```
   This should print a short response from the model. If it errors, check your API key and the error message — `src/llm_client.py` prints helpful diagnostics.

6. **Run Vanilla RAG on a small slice first** (don't run all 144 questions on your first try):
   ```bash
   python3 run_vanilla_rag.py --limit 5
   ```
   Check `outputs/vanilla_rag_predictions.jsonl` looks reasonable, THEN run the full set:
   ```bash
   python3 run_vanilla_rag.py
   ```

## Why things are structured this way

- **`data_loader.py` is shared infrastructure** — every system you build (Rerank, Adaptive, GraphRAG, Living Memory v0, etc.) will import from this file. Get this right once, reuse everywhere.
- **`llm_client.py` has rate-limit handling built in** (exponential backoff, automatic retry on 429) because Cerebras free tier WILL throttle you if you fire requests too fast — better to handle this once, centrally, than debug mysterious failures in every system you build later.
- **`vanilla_rag.py` is intentionally simple and modular** — its `retrieve()` and `generate()` functions are written so Rerank RAG and Contextual Retrieval (Day 2) can import and extend them instead of rewriting from scratch, per the build-order plan to save time.
- **Embeddings run locally**, not through an API, specifically to protect your Cerebras token/rate budget for the parts that actually need an LLM (generation, and later, supersession-type classification for Living Memory v0).

## A note on scenario isolation

The benchmark spec is explicit: **index and retrieve within each scenario separately — never retrieve across scenarios.** `data_loader.py` groups everything by `scenario_id` from the start so this constraint is structurally enforced, not something you have to remember to check later.
