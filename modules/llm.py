"""Thin wrapper so the rest of the app doesn't care which LLM is used."""
import json
import os
import re
import time

import requests
from dotenv import load_dotenv

load_dotenv()


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "gemini").lower()


def extract_json(text: str):
    """Pull the first JSON object/array out of an LLM reply."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if not match:
        raise ValueError("LLM did not return valid JSON")
    return json.loads(match.group(1))


def _gemini(prompt: str, json_mode: bool) -> str:
    from google import genai

    key = os.getenv("GEMINI_API_KEY")
    if not key or key == "your_key_here":
        raise RuntimeError("GEMINI_API_KEY missing. Add it to your .env file.")
    client = genai.Client(api_key=key)
    config = {"response_mime_type": "application/json"} if json_mode else None
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    # Retry when Google is busy (503) or rate limits us (429)
    delays = [3, 6, 12, 20]
    for attempt in range(len(delays) + 1):
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=config)
            return resp.text
        except Exception as e:
            busy = any(c in str(e) for c in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
            if not busy or attempt == len(delays):
                raise
            time.sleep(delays[attempt])


def _ollama(prompt: str, json_mode: bool) -> str:
    payload = {
        "model": os.getenv("OLLAMA_MODEL", "llama3.2"),
        "prompt": prompt,
        "stream": False,
    }
    if json_mode:
        payload["format"] = "json"
    url = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/generate"
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["response"]


def generate(prompt: str, json_mode: bool = False):
    """Return text, or parsed JSON when json_mode=True."""
    raw = _ollama(prompt, json_mode) if provider() == "ollama" else _gemini(prompt, json_mode)
    return extract_json(raw) if json_mode else raw
