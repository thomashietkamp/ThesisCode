import os
import requests
import json
from typing import List, Dict, Tuple
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
_MODEL_ID = "qwen/qwen3-14b:free"  # OpenRouter model identifier
_BASE_URL = "https://openrouter.ai/api/v1"
_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not _API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is required")

_HEADERS = {
    "Authorization": f"Bearer {_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com",  # Optional: helps with rate limits
    "X-Title": "Qwen3 Committee"  # Optional: for tracking
}


# ── Simple chat (no thinking) ───────────────────────────────────────────────────
def chat_complete(prompt: str, model_id: str, temperature: float = 0.3) -> str:
    """
    Vanilla text-completion using OpenRouter. Returns the generated text after the prompt.
    """
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }

    response = requests.post(
        f"{_BASE_URL}/chat/completions",
        headers=_HEADERS,
        data=json.dumps(payload)
    )

    if response.status_code != 200:
        raise Exception(
            f"OpenRouter API error: {response.status_code} - {response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


# ── Chat with "thinking" split ─────────────────────────────────────────────────
def chat_with_thinking(
    messages: List[Dict],
    temperature: float = 0.3,
    model_id: str = _MODEL_ID,
) -> Tuple[str, str]:
    """
    Chat completion that returns both reasoning and final content.
    Qwen models provide a native "reasoning" field in their responses.

    Returns: (thinking_content, final_content).

    messages: list of {"role": "user"|"assistant", "content": "..."}
    """
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
    }

    response = requests.post(
        f"{_BASE_URL}/chat/completions",
        headers=_HEADERS,
        data=json.dumps(payload)
    )

    if response.status_code != 200:
        raise Exception(
            f"OpenRouter API error: {response.status_code} - {response.text}")

    data = response.json()
    message = data["choices"][0]["message"]

    # Extract thinking from the native reasoning field if available
    thinking = message.get("reasoning", "").strip()
    content = message["content"].strip()

    return thinking, content


# ── Alternative: Direct OpenAI-style chat completion ─────────────────────────────
def chat_messages(
    messages: List[Dict],
    temperature: float = 0.3,
    max_tokens: int = 1024
) -> str:
    """
    Direct chat completion with message history.

    messages: list of {"role": "user"|"assistant"|"system", "content": "..."}
    """
    payload = {
        "model": _MODEL_ID,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(
        f"{_BASE_URL}/chat/completions",
        headers=_HEADERS,
        data=json.dumps(payload)
    )

    if response.status_code != 200:
        raise Exception(
            f"OpenRouter API error: {response.status_code} - {response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


# ── Usage example ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        print("Testing OpenRouter API...")
        print(f"API Key present: {'Yes' if _API_KEY else 'No'}")
        print(
            f"API Key preview: {_API_KEY[:10]}..." if _API_KEY else "No API Key")

        # Test simple completion
        result = chat_complete("What is the capital of France?")
        print("Simple completion:", result)

        # Test thinking-based completion
        messages = [
            {"role": "user", "content": "Explain quantum computing in simple terms."}]
        thinking, content = chat_with_thinking(messages)
        print("\nThinking:", thinking)
        print("Final answer:", content)
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
