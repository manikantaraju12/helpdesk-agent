"""
api.py — FastAPI Application

Exposes:
  POST /v1/chat             — Main conversational endpoint
  GET  /v1/context/{id}    — Session context inspection (req-9)
  GET  /health              — Container health check (req-7)

Context pipeline per turn:
  1. Hydrate session (in-memory store or create new)
  2. Extract entities (ticket ID via regex) + classify topic (keyword + LLM)
  3. Update context.pinned and context.active_topic
  4. Stage user message (do NOT commit yet — req-10 rollback safety)
  5. Build system prompt from updated context
  6. Assemble final message list (system + pruned recent + staged user msg)
  7. Call LLM — if it fails, raise 500 WITHOUT mutating context.recent
  8. On success: commit user + assistant to recent[], apply token eviction
  9. Persist context to ./logs/context_{session_id}.json
  10. Return ChatResponse
"""

from __future__ import annotations

import logging
import os
import re
from copy import deepcopy
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from src.context_engine import (
    AgentContext,
    Message,
    add_messages_and_evict,
    apply_token_budget,
    build_system_prompt,
    load_context_from_file,
    persist_context,
    update_context_with_extraction,
)
from src.llm_client import classify_topic_llm, generate_response

# ─────────────────────────────────────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("helpdesk-agent")

app = FastAPI(
    title="IT Helpdesk Agent",
    description=(
        "Context-Aware IT Helpdesk Agent with Token Budgets and Pinned Memory. "
        "Implements rolling-window token eviction, pinned facts, and dynamic topic tracking."
    ),
    version="1.0.0",
)

# In-memory session store: session_id -> AgentContext
# In production this would be Redis or a database.
session_store: Dict[str, AgentContext] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def get_token_budget() -> int:
    """
    Reads TOKEN_BUDGET_LIMIT from the environment.
    Defaults to 4000 if not set. Zero is allowed for deterministic eviction.
    """
    raw = os.environ.get("TOKEN_BUDGET_LIMIT", "4000")
    try:
        value = int(raw)
        if value < 0:
            raise ValueError("TOKEN_BUDGET_LIMIT must be non-negative")
        return value
    except ValueError:
        logger.warning("Invalid TOKEN_BUDGET_LIMIT=%r, defaulting to 4000", raw)
        return 4000


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    active_topic: str


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Hydrate or Create Session
# ─────────────────────────────────────────────────────────────────────────────

def _get_or_create_session(session_id: str) -> AgentContext:
    """Hydrates a session from memory or its persisted context file."""
    existing = session_store.get(session_id)
    if existing is not None:
        return existing

    saved = load_context_from_file(session_id)
    if saved is None:
        logger.info("Creating new session: %s", session_id)
        return AgentContext(session_id=session_id)

    context = AgentContext(
        session_id=session_id,
        active_topic=saved.get("active_topic", "general"),
        pinned=saved.get("pinned") or {"ticket_id": None},
        recent=[Message(role=item["role"], content=item["content"])
                for item in saved.get("recent", [])],
    )
    context.total_recent_tokens = sum(message.tokens for message in context.recent)
    return context


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Infra"])
async def health_check():
    """
    Health check endpoint required by req-7 (Docker health check).
    Returns 200 + {"status": "ok"} when the service is ready.
    """
    return {"status": "ok"}


@app.post("/v1/chat", response_model=ChatResponse, tags=["Chat"])
async def process_chat(req: ChatRequest) -> ChatResponse:
    """
    Main conversational endpoint (req-1).

    Implements the full context pipeline:
      extraction → topic update → prompt build → LLM → eviction → persist

    Graceful error handling (req-10):
      The user message is staged but NOT committed to context.recent until
      the LLM call succeeds.  On LLM failure, we raise HTTP 500 and the
      context state is unchanged (no one-sided messages in the log).
    """
    session_id = req.session_id.strip()
    user_text = req.message.strip()

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
        raise HTTPException(
            status_code=422,
            detail="session_id must be 1-128 letters, digits, underscores, or hyphens",
        )
    if not user_text:
        raise HTTPException(status_code=422, detail="message must not be empty")

    token_budget = get_token_budget()
    logger.info(
        "Turn start | session=%s | budget=%d | message=%.80s",
        session_id, token_budget, user_text,
    )

    # ── Step 1: Hydrate session ─────────────────────────────────────────────
    context = deepcopy(_get_or_create_session(session_id))

    # ── Step 2: Topic classification (optional LLM-enhanced) ────────────────
    # Run keyword + optional LLM classifier on the raw user message.
    # LLM call swallows its own errors; keyword fallback is always available.
    llm_topic: Optional[str] = None
    try:
        llm_topic = classify_topic_llm(user_text)
    except Exception as e:
        logger.warning("Topic LLM classification failed, using keyword fallback: %s", e)

    # ── Step 3: Entity extraction + context state update ────────────────────
    # Mutates context.pinned["ticket_id"] and context.active_topic
    update_context_with_extraction(context, user_text, llm_topic)
    logger.info(
        "Context updated | topic=%s | pinned=%s",
        context.active_topic, context.pinned,
    )

    # ── Step 4: Stage user message (pre-compute tokens) ──────────────────────
    # Staged here but committed to context.recent ONLY after successful LLM call.
    staged_user_msg = Message(role="user", content=user_text)

    # ── Step 5: Build system prompt from updated context ─────────────────────
    system_prompt = build_system_prompt(context)
    system_msg = {"role": "system", "content": system_prompt}

    # ── Step 6: Assemble final message list for LLM ──────────────────────────
    # Prune the prompt history before generation; the system prompt is excluded.
    history_budget = max(0, token_budget - staged_user_msg.tokens)
    prompt_history = apply_token_budget(context.recent, history_budget)
    final_messages = (
        [system_msg]
        + [m.to_api_dict() for m in prompt_history]
        + [staged_user_msg.to_api_dict()]
    )

    # ── Step 7: LLM Generation ───────────────────────────────────────────────
    # If this raises, we return 500 WITHOUT touching context.recent (req-10).
    try:
        assistant_text = generate_response(final_messages)
    except Exception as e:
        logger.error(
            "LLM generation failed for session=%s: %s — context NOT mutated",
            session_id, e,
        )
        # Re-raise as HTTP 500; context.recent is still clean
        raise HTTPException(
            status_code=500,
            detail=f"LLM generation failed: {type(e).__name__}: {e}",
        )

    # ── Step 8: Commit messages to context + apply token eviction ────────────
    # Only reached on LLM success.  Eviction happens here (req-4).
    assistant_msg = Message(role="assistant", content=assistant_text)
    add_messages_and_evict(context, staged_user_msg, assistant_msg, token_budget)

    logger.info(
        "After eviction | session=%s | recent_msgs=%d | total_tokens=%d",
        session_id, len(context.recent), context.total_recent_tokens,
    )

    # ── Step 9: Persist then publish the committed state ──────────────────────
    try:
        persist_context(context)
    except Exception as e:
        logger.exception("Context persistence failed for session=%s", session_id)
        raise HTTPException(
            status_code=500,
            detail="Context persistence failed; session state was not committed.",
        ) from e

    session_store[session_id] = context

    # ── Step 10: Return response ──────────────────────────────────────────────
    return ChatResponse(
        response=assistant_text,
        active_topic=context.active_topic,
    )


@app.get("/v1/context/{session_id}", tags=["Context"])
async def get_context(session_id: str):
    """
    Returns the current context state for a session (req-9).

    Lookup order:
      1. In-memory session_store (authoritative, always up-to-date)
      2. Persisted ./logs/context_{session_id}.json (fallback for cold reads)

    Returns:
      200 + context JSON if found.
      404 if the session has never been initialized.
    """
    # Check in-memory first
    if session_id in session_store:
        return JSONResponse(
            content=session_store[session_id].to_persistence_dict(),
            status_code=200,
        )

    # Fallback to persisted file
    data = load_context_from_file(session_id)
    if data is not None:
        return JSONResponse(content=data, status_code=200)

    # Session not found (req-9: return 404)
    raise HTTPException(
        status_code=404,
        detail=f"Session '{session_id}' not found. Send a POST /v1/chat first.",
    )
