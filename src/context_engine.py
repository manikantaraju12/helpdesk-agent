"""
context_engine.py — Core Context Engineering Logic

Implements:
  - AgentContext: the structured state object (pinned + recent + active_topic)
    - Token counting via model-specific tiktoken BPE (not raw word counts)
  - FIFO eviction: removes oldest messages until total_recent_tokens ≤ TOKEN_BUDGET_LIMIT
  - Entity extraction: regex for IT-XXXX ticket IDs
  - Topic classification: keyword heuristics + LLM structured output
  - Context persistence: writes ./logs/context_{session_id}.json after every turn
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import tiktoken

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

LOGS_DIR = Path("logs")
TICKET_PATTERN = re.compile(r"\bIT-\d{4}\b")

# Select the BPE encoding for the configured model, with a known fallback.
try:
    _ENCODING = tiktoken.encoding_for_model(os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
except KeyError:
    _ENCODING = tiktoken.get_encoding("cl100k_base")

# Topic keyword maps — deterministic, zero-latency classification baseline
_TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "wifi_issue": [
        "wifi", "wi-fi", "wireless", "network", "internet", "connected",
        "connection", "dropping", "dropped", "hotspot", "ethernet", "lan",
        "ping", "vpn", "bandwidth", "router", "eduroam",
    ],
    "ticket_inquiry": [
        "ticket", "case", "issue number", "reference", "support request",
        "my ticket", "follow up", "status of", "it-",
    ],
    "printing": [
        "print", "printer", "printing", "scan", "scanner", "copier",
        "queue", "spooler", "ink", "toner",
    ],
    "account_access": [
        "password", "login", "log in", "sign in", "account", "locked",
        "reset", "authentication", "2fa", "mfa", "duo", "sso",
    ],
    "software_install": [
        "install", "software", "application", "app", "download", "license",
        "microsoft", "office", "zoom", "teams", "slack",
    ],
    "hardware": [
        "laptop", "computer", "keyboard", "mouse", "screen", "monitor",
        "charger", "battery", "ram", "disk", "storage", "broken",
    ],
    "general": [],  # fallback
}
_TICKET_RETURN_PHRASES = (
    "go back to my ticket",
    "back to my ticket",
    "original ticket",
    "ticket i mentioned",
    "status of the ticket",
    "ticket status",
)


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Message:
    """A single conversation message with pre-computed token count."""
    role: str          # "user" | "assistant" | "system"
    content: str
    tokens: int = field(default=0)

    def __post_init__(self) -> None:
        if self.tokens == 0:
            self.tokens = count_tokens(self.content)

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "tokens": self.tokens}

    def to_api_dict(self) -> dict:
        """Minimal dict for LLM API calls (no tokens field)."""
        return {"role": self.role, "content": self.content}


@dataclass
class AgentContext:
    """
    The full session context.  This is the single source of truth that is
    (a) updated in-memory on every turn and
    (b) persisted to ./logs/context_{session_id}.json for observability.
    """
    session_id: str
    active_topic: str = "general"
    pinned: Dict[str, Optional[str]] = field(default_factory=lambda: {"ticket_id": None})
    recent: List[Message] = field(default_factory=list)
    total_recent_tokens: int = 0

    def to_persistence_dict(self) -> dict:
        """
        Returns the exact schema required by req-2-context-persistence.
        Strips internal `tokens` field from each message for the clean public schema.
        """
        return {
            "session_id": self.session_id,
            "active_topic": self.active_topic,
            "pinned": self.pinned,
            "recent": [
                {"role": m.role, "content": m.content}
                for m in self.recent
            ],
            "total_recent_tokens": self.total_recent_tokens,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Token Utilities
# ─────────────────────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """
    Count tokens using the configured model's tiktoken BPE encoding.
    Raw word counts are not used.
    """
    return len(_ENCODING.encode(text))


def _sum_recent_tokens(recent: List[Message]) -> int:
    """Sum pre-computed token counts across the recent list."""
    return sum(m.tokens for m in recent)


# ─────────────────────────────────────────────────────────────────────────────
# FIFO Token Budget Eviction  (req-4)
# ─────────────────────────────────────────────────────────────────────────────

def apply_token_budget(recent: List[Message], max_tokens: int) -> List[Message]:
    """
    Enforces the TOKEN_BUDGET_LIMIT on the recent message list.

    Algorithm: FIFO — pop the oldest message (index 0) until the total
    token count of the remaining messages is within budget.

    Critical invariant:
      - Only `recent` is subject to eviction.
      - `pinned` facts and the system prompt are NOT counted here.
      - This preserves ticket IDs and other critical facts even when the
        conversation history that first introduced them has been evicted.

    Args:
        recent:     Current list of Message objects (mutable — returns new list).
        max_tokens: Maximum allowed total tokens (TOKEN_BUDGET_LIMIT env var).

    Returns:
        A new (possibly shorter) list of Message objects within budget.
    """
    pruned = list(recent)  # shallow copy; Messages are immutable after creation
    current_total = _sum_recent_tokens(pruned)

    while pruned and current_total > max_tokens:
        evicted = pruned.pop(0)       # FIFO: oldest first
        current_total -= evicted.tokens

    return pruned


# ─────────────────────────────────────────────────────────────────────────────
# Entity Extraction  (req-3)
# ─────────────────────────────────────────────────────────────────────────────

def extract_ticket_id(text: str) -> Optional[str]:
    """
    Deterministic regex extraction of IT-XXXX ticket IDs.
    Returns the first match or None.  No LLM call needed for this step.

    Pattern: word-boundary + IT- + exactly 4 digits + word-boundary
    Examples: "IT-9921", "IT-0001", "IT-8812"
    """
    match = TICKET_PATTERN.search(text)
    return match.group(0) if match else None


# ─────────────────────────────────────────────────────────────────────────────
# Topic Classification  (req-5)
# ─────────────────────────────────────────────────────────────────────────────

def classify_topic_keywords(text: str) -> str:
    """
    Fast, deterministic keyword-based topic classifier.
    Scans the lowercased message for each topic's keyword list.
    Returns the first matching topic, or "general" as fallback.

    This is the primary classifier — zero latency, no extra API cost.
    The LLM classifier in llm_client.py acts as an enhancement layer.
    """
    text_lower = text.lower()
    if any(phrase in text_lower for phrase in _TICKET_RETURN_PHRASES):
        return "ticket_inquiry"

    for topic, keywords in _TOPIC_KEYWORDS.items():
        if topic == "general":
            continue
        if any(kw in text_lower for kw in keywords):
            return topic
    return "general"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Construction  (req Phase-5)
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt(context: AgentContext) -> str:
    """
    Constructs the system prompt that gets injected into every LLM call.

    Injects:
      1. Agent persona + instructions
      2. Current active_topic (so the LLM knows what to focus on)
      3. All pinned facts (ticket ID, OS, etc.) — these are NEVER evicted

    The system prompt does NOT count toward TOKEN_BUDGET_LIMIT.
    """
    # Build the pinned facts block
    pinned_lines: List[str] = []
    for key, value in context.pinned.items():
        if value is not None:
            label = key.replace("_", " ").title()
            pinned_lines.append(f"  - {label}: {value}")

    pinned_block = (
        "## Known User Facts (Pinned — always preserved):\n" + "\n".join(pinned_lines)
        if pinned_lines
        else "## Known User Facts: None yet."
    )

    topic_display = context.active_topic.replace("_", " ").title()

    system_prompt = f"""You are a knowledgeable and empathetic IT Helpdesk Agent for a university campus.
Your role is to help students and faculty resolve technology issues efficiently and clearly.

## Current Conversation Topic: {topic_display}

{pinned_block}

## Guidelines:
- Always reference the user's ticket ID (if known from pinned facts) when discussing their support request.
- When helping with WiFi or network issues, provide step-by-step Mac/Windows specific troubleshooting.
- Be concise but thorough. Number your steps clearly.
- If the user returns to a previous topic, use your pinned facts to provide continuity.
- Do NOT say you cannot access ticket systems — acknowledge the ticket ID and say you are escalating or checking status.
- Maintain a professional, helpful tone at all times.
"""
    return system_prompt.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Context Persistence  (req-2)
# ─────────────────────────────────────────────────────────────────────────────

def persist_context(context: AgentContext) -> None:
    """
    Writes the session context to ./logs/context_{session_id}.json.

    Called after EVERY successful turn (req-2-context-persistence).
    The file is written atomically using a temp-then-rename pattern to
    prevent partial writes corrupting the log.

    File location: ./logs/context_{session_id}.json  (relative to CWD)
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"context_{context.session_id}.json"

    # Atomic write: write to temp file, then rename
    tmp_path = log_path.with_suffix(".json.tmp")
    try:
        data = context.to_persistence_dict()
        tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(log_path)
    except Exception:
        # Clean up partial temp file on failure
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def load_context_from_file(session_id: str) -> Optional[Dict]:
    """
    Loads a persisted context JSON from disk.  Returns None if not found.
    Used by GET /v1/context/{session_id} as a fallback when not in memory.
    """
    log_path = LOGS_DIR / f"context_{session_id}.json"
    if not log_path.exists():
        return None
    try:
        return json.loads(log_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Context Update Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def update_context_with_extraction(
    context: AgentContext,
    user_message: str,
    llm_topic: Optional[str] = None,
) -> None:
    """
    Mutates the context object in-place with extracted entities and inferred topic.

    Steps:
      1. Extract ticket ID via regex → update pinned["ticket_id"] if found
      2. Determine new active_topic (keyword classifier, optionally overridden by LLM)
      3. Update context.active_topic

    This is called BEFORE adding the message to recent[] and BEFORE LLM generation,
    so the system prompt already reflects the updated state for this turn.
    """
    # Step 1: Entity extraction — ticket ID (req-3)
    ticket_id = extract_ticket_id(user_message)
    if ticket_id:
        context.pinned["ticket_id"] = ticket_id

    # Explicit return-to-ticket intent takes precedence over incidental WiFi mentions.
    text_lower = user_message.lower()
    if any(phrase in text_lower for phrase in _TICKET_RETURN_PHRASES):
        context.active_topic = "ticket_inquiry"
        return

    # Step 2: Topic classification
    # LLM-provided topic (if available) takes precedence over keyword heuristic
    if llm_topic and llm_topic != "general":
        context.active_topic = llm_topic
    else:
        keyword_topic = classify_topic_keywords(user_message)
        # If keyword classifier found something specific, use it;
        # otherwise retain the current topic (don't flip back to general mid-conversation)
        if keyword_topic != "general":
            context.active_topic = keyword_topic
        elif context.active_topic == "general":
            # First message or no specific topic detected — keep general
            context.active_topic = "general"
        # If we already had a non-general topic and the new message is generic,
        # retain the existing topic (avoids thrashing)


def add_messages_and_evict(
    context: AgentContext,
    user_msg: Message,
    assistant_msg: Message,
    token_budget: int,
) -> None:
    """
    Appends user + assistant messages to recent[], then applies token budget eviction.

    Called ONLY after a successful LLM generation (req-10 graceful error handling).
    If the LLM call fails upstream, this function is never called, keeping recent[] clean.

    Args:
        context:        The session's AgentContext (mutated in-place).
        user_msg:       The user's Message object (with pre-computed tokens).
        assistant_msg:  The LLM's response as a Message object.
        token_budget:   TOKEN_BUDGET_LIMIT from environment.
    """
    context.recent.append(user_msg)
    context.recent.append(assistant_msg)

    # Apply FIFO eviction — only recent[] is subject to the budget
    context.recent = apply_token_budget(context.recent, token_budget)
    context.total_recent_tokens = _sum_recent_tokens(context.recent)
