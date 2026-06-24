"""
Central configuration. Every other script imports from here.
Keeps model names, paths, and rate-limit settings in ONE place so changing
the LLM provider or model later doesn't mean editing 7 different files.
"""
import os
from dotenv import load_dotenv
load_dotenv()

# ---- Cerebras API settings ----
# Get your key from https://cloud.cerebras.ai and set it as an environment
# variable. Never hardcode it in this file.
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

# Model choice: llama-3.3-70b is a good balance of quality and free-tier
# throughput as of mid-2026. If you hit persistent issues, llama-4-scout
# is the documented fallback on the same free tier.
LLM_MODEL = "gpt-oss-120b"

# ---- Rate limiting ----
# Cerebras free tier: ~30 requests/minute, 60K-100K tokens/minute,
# 1M tokens/day. We stay well under 30 RPM to leave headroom for retries.
MAX_REQUESTS_PER_MINUTE = 25
SECONDS_BETWEEN_REQUESTS = 60.0 / MAX_REQUESTS_PER_MINUTE  # ~2.4 sec
MAX_RETRIES = 4
RETRY_BASE_DELAY = 2.0  # seconds, doubles each retry (exponential backoff)

# ---- Embedding model (runs locally, not via API) ----
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ---- Retrieval settings ----
# Same top-k budget across all chunk-based methods, per the spec's
# "controlled settings" requirement.
TOP_K = 4

# ---- Paths ----
DATA_DIR = "data"
OUTPUTS_DIR = "outputs"

SCENARIO_DOCS_PATH = f"{DATA_DIR}/scenario_documents.jsonl"
EVENTS_PATH = f"{DATA_DIR}/events.jsonl"
QUESTIONS_PATH = f"{DATA_DIR}/questions.jsonl"
GOLD_ATOMS_PATH = f"{DATA_DIR}/gold_memory_atoms.jsonl"  # eval only, never for retrieval

# ---- Sanity check ----
def check_config():
    """Run this first to catch a missing API key before anything else runs."""
    if not CEREBRAS_API_KEY:
        raise RuntimeError(
            "CEREBRAS_API_KEY is not set. Run:\n"
            "  export CEREBRAS_API_KEY='your-key-here'\n"
            "before running any script."
        )
    print("Config OK. Using model:", LLM_MODEL)
