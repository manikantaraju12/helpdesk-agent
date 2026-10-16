"""
llm_client.py — OpenAI API Wrapper

Provides:
  - generate_response(): Main LLM call for helpdesk replies
  - classify_topic_llm(): Lightweight structured-output call for topic inference
  - Both functions propagate errors cleanly (no swallowing) to support
    req-10 graceful error handling in the API layer.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from openai import OpenAI, APIError, APIConnectionError, APITimeoutError, AuthenticationError

# ─────────────────────────────────────────────────────────────────────────────
# Client Initialization
# ─────────────────────────────────────────────────────────────────────────────

def _get_client() -> OpenAI:
    """
    Lazy OpenAI client factory.  Reads OPENAI_API_KEY from environment.
    Raises ValueError with a clear message if the key is missing.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError(
            "OPENAI_API_KEY is not set or is a placeholder. "
            "Copy .env.example to .env and set a real API key."
        )
    return OpenAI(api_key=api_key)


def _get_model() -> str:
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Main Generation Call
# ─────────────────────────────────────────────────────────────────────────────

def generate_response(messages: List[Dict[str, str]]) -> str:
    """
    Calls the OpenAI chat completions API and returns the assistant's reply.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
                  Must include the system prompt as the first message.

    Returns:
        The assistant's reply as a plain string.

    Raises:
        AuthenticationError: Invalid API key (triggers req-10 rollback).
        APIConnectionError: Network issue (triggers req-10 rollback).
        APITimeoutError: Request timed out (triggers req-10 rollback).
        APIError: Any other upstream error (triggers req-10 rollback).
        ValueError: Missing / placeholder API key.

    All exceptions are intentionally NOT caught here — they propagate to
    api.py which handles req-10 rollback and returns HTTP 500 to the client.
    """
    client = _get_client()
    model = _get_model()

    completion = client.chat.completions.create(
        model=model,
        messages=messages,       # type: ignore[arg-type]
        temperature=0.7,
        max_tokens=1024,
    )

    reply = completion.choices[0].message.content
    if reply is None:
        raise APIError(
            message="OpenAI returned an empty response",
            request=None,   # type: ignore[arg-type]
            body=None,
        )
    return reply.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Topic Classification via Structured Output  (req-5 enhancement)
# ─────────────────────────────────────────────────────────────────────────────

_TOPIC_SYSTEM_PROMPT = """You are a topic classifier for an IT helpdesk.
Given a user message, classify it into EXACTLY ONE of these categories:
- "general"          : greetings, thank you, unrelated small talk
- "ticket_inquiry"   : asking about a support ticket status or reference
- "wifi_issue"       : WiFi, network, internet connectivity problems
- "printing"         : printer, scanner, print queue issues
- "account_access"   : passwords, login, locked accounts, MFA
- "software_install" : installing or licensing software
- "hardware"         : physical device issues (laptop, keyboard, etc.)

Respond with a JSON object only: {"topic": "<category>"}
Do not include any other text."""


def classify_topic_llm(user_message: str) -> Optional[str]:
    """
    Uses a fast, cheap LLM call to classify the topic of a user message.
    Returns the topic string or None on failure (keyword fallback is used).

    This function SWALLOWS errors intentionally — it is an optional
    enhancement.  If it fails, the keyword classifier in context_engine.py
    is used as the authoritative fallback.
    """
    try:
        client = _get_client()
        model = _get_model()

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _TOPIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,      # deterministic for classification
            max_tokens=30,        # tiny — we only need {"topic": "..."}
            response_format={"type": "json_object"},
        )

        raw = completion.choices[0].message.content or "{}"
        data = json.loads(raw)
        topic = data.get("topic", "general")

        # Validate returned topic is in our allowed set
        allowed = {
            "general", "ticket_inquiry", "wifi_issue",
            "printing", "account_access", "software_install", "hardware",
        }
        return topic if topic in allowed else "general"

    except Exception:
        # Graceful degradation — keyword classifier takes over
        return None
