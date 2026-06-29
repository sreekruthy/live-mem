"""
Wraps the Cerebras API (OpenAI-compatible endpoint) with:
  - automatic pacing to stay under the free-tier rate limit
  - exponential backoff retry on 429 / transient errors
  - a test_connection() helper to verify setup before running anything real

Every system (Vanilla RAG, Living Memory v0, etc.) should call `chat()`
from this file instead of hitting the API directly, so rate-limit handling
lives in exactly one place.
"""
import time
from openai import OpenAI, RateLimitError, APIError

from . import config

_client = None
_last_call_time = 0.0


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        config.check_config()
        _client = OpenAI(
            api_key=config.CEREBRAS_API_KEY,
            base_url=config.CEREBRAS_BASE_URL,
        )
    return _client


def _pace_request():
    """
    Ensures we never fire requests faster than MAX_REQUESTS_PER_MINUTE.
    Simple sleep-based pacing — good enough for a sequential solo script;
    if you ever parallelize calls, this needs a proper token-bucket lock.
    """
    global _last_call_time
    elapsed = time.time() - _last_call_time
    wait = config.SECONDS_BETWEEN_REQUESTS - elapsed
    if wait > 0:
        time.sleep(wait)
    _last_call_time = time.time()


def chat(
    messages: list,
    model: str = None,
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> dict:
    """
    Sends a chat completion request with retry/backoff.
    Returns a dict: {"text": str, "input_tokens": int, "output_tokens": int}
    so callers can log token cost per the spec's required metrics.

    temperature=0.0 by default: for extraction/classification/eval tasks
    you generally want deterministic, repeatable outputs, not creative
    variation. Override per-call if a specific use case needs otherwise.
    """
    client = _get_client()
    model = model or config.LLM_MODEL

    last_error = None
    for attempt in range(config.MAX_RETRIES):
        _pace_request()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            choice = response.choices[0].message.content
            usage = response.usage
            if choice is None:
                # Some reasoning-style models (e.g. gpt-oss) can return empty
                # content if max_tokens was too low to finish their internal
                # reasoning before emitting the visible answer. Treat this
                # as a retryable condition rather than crashing downstream
                # code that expects a string.
                last_error = RuntimeError(
                    "Model returned empty content (likely max_tokens too low "
                    "for this reasoning model to finish). Retrying with same "
                    "params; consider raising max_tokens if this persists."
                )
                delay = config.RETRY_BASE_DELAY * (2 ** attempt)
                print(f"[empty content] retry {attempt + 1}/{config.MAX_RETRIES} in {delay:.1f}s...")
                time.sleep(delay)
                continue
            return {
                "text": choice,
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
            }
        except RateLimitError as e:
            last_error = e
            error_str = str(e)
            if "per hour" in error_str.lower() or "request_quota_exceeded" in error_str:
                delay = 300  # 5 minutes
                print(f"[hourly quota] retry {attempt + 1}/{config.MAX_RETRIES} in {delay}s "
                      f"(hourly limit hit, short backoff won't help)...")
            else:
                delay = config.RETRY_BASE_DELAY * (2 ** attempt)
                print(f"[rate limit] retry {attempt + 1}/{config.MAX_RETRIES} in {delay:.1f}s...")
            time.sleep(delay)
        except APIError as e:
            last_error = e
            delay = config.RETRY_BASE_DELAY * (2 ** attempt)
            print(f"[api error] {e}. retry {attempt + 1}/{config.MAX_RETRIES} in {delay:.1f}s...")
            time.sleep(delay)

    raise RuntimeError(
        f"Cerebras API call failed after {config.MAX_RETRIES} retries. "
        f"Last error: {last_error}"
    )


def test_connection():
    """Run this once after setting your API key to confirm everything works."""
    result = chat(
        messages=[{"role": "user", "content": "Reply with exactly the word: OK"}],
        max_tokens=50,
    )
    print("Cerebras connection OK.")
    print("Response:", result["text"])
    print(f"Tokens used: {result['input_tokens']} in, {result['output_tokens']} out")


if __name__ == "__main__":
    test_connection()
