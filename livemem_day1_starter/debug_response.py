"""
Diagnostic: prints the FULL raw response object from Cerebras, so we can
see exactly where the actual text is sitting when message.content is None.

Run this with: python3 debug_response.py
"""
from openai import OpenAI
from src import config

client = OpenAI(api_key=config.CEREBRAS_API_KEY, base_url=config.CEREBRAS_BASE_URL)

response = client.chat.completions.create(
    model=config.LLM_MODEL,
    messages=[{"role": "user", "content": "Reply with exactly the word: OK"}],
    max_tokens=10,
)

print("=" * 60)
print("FULL RAW RESPONSE OBJECT:")
print("=" * 60)
print(response.model_dump_json(indent=2))
