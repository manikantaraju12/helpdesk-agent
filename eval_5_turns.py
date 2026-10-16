#!/usr/bin/env python3
"""
eval_5_turns.py — 5-Turn Scripted Evaluation for the IT Helpdesk Agent

Purpose:
    Proves that context pinning + token eviction work correctly across a
    multi-turn conversation with a topic switch.

    Critical contract:
      - Turn 2 introduces ticket IT-8812 → pinned["ticket_id"] = "IT-8812"
      - Turn 3 sends a long WiFi message that forces eviction of Turn 1-2 messages
      - Turn 5 asks about "the ticket I mentioned" — bot MUST know IT-8812
        even though Turn 2 was evicted from recent[]
      - Final context JSON must show pinned.ticket_id == "IT-8812"
      - Final LLM response must contain the string "IT-8812"

Exit codes:
    0 — All assertions passed
    1 — One or more assertions failed

Usage (from repo root, after server is running):
    python eval_5_turns.py

    To run with a custom base URL:
        BASE_URL=http://localhost:8000 python eval_5_turns.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
LOGS_DIR = Path("logs")

# Use a unique session ID per run so reruns don't interfere with each other
SESSION_ID = f"eval_{uuid.uuid4().hex[:8]}"

# The ticket ID the evaluation asserts is preserved
TICKET_ID = "IT-8812"
EXPECTED_TOKEN_BUDGET = int(os.environ.get("TOKEN_BUDGET_LIMIT", "100"))

# Turn-by-turn messages
TURNS = [
    # Turn 1: Greeting — expected topic: "general"
    "Hi, I need help.",

    # Turn 2: Introduce ticket ID — expected pinned["ticket_id"] = "IT-8812"
    f"I have an open support ticket, the ID is {TICKET_ID}. "
    "I submitted it two days ago about my email not syncing on Outlook.",

    # Turn 3: Long WiFi message — designed to trigger token eviction of earlier turns
    (
        "Actually, before we get to the ticket, something urgent just came up. "
        "I'm sitting in the main university library right now and my WiFi just dropped completely. "
        "I'm using a MacBook Pro running macOS Ventura 13.6. "
        "The network status bar shows the WiFi icon with a small exclamation mark and it says "
        "'connected without internet access'. "
        "I've already tried turning WiFi off and on, and I can see the eduroam network in the list "
        "but whenever I connect to it, it just says 'no internet connection'. "
        "This is really urgent because I have an assignment deadline in two hours. "
        "The library staff said the WiFi should be working fine for everyone else. "
        "Can you help me troubleshoot this step by step please?"
    ),

    # Turn 4: WiFi follow-up — expects detailed Mac steps
    (
        "I tried the steps you gave me. "
        "I forgot to mention I also have another device, a Windows 11 laptop. "
        "Can you give me the complete Mac-specific troubleshooting steps for the "
        "'connected without internet' eduroam issue? "
        "Please be as detailed as possible, including terminal commands if necessary."
    ),

    # Turn 5: Return to ticket — CRITICAL ASSERTION: agent must reference IT-8812
    (
        "Okay, I managed to fix the WiFi issue, thank you so much! "
        "Now let's go back to what I originally called about. "
        "Can you confirm the status of the ticket I mentioned at the beginning of our conversation? "
        "What was the ticket ID and what is the next step I should take?"
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Console Helpers
# ─────────────────────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def log_pass(msg: str) -> None:
    print(f"  {GREEN}✓ PASS{RESET} {msg}")


def log_fail(msg: str) -> None:
    print(f"  {RED}✗ FAIL{RESET} {msg}")


def log_info(msg: str) -> None:
    print(f"  {CYAN}ℹ{RESET} {msg}")


def log_section(title: str) -> None:
    print(f"\n{BOLD}{YELLOW}{'═' * 60}{RESET}")
    print(f"{BOLD}{YELLOW}  {title}{RESET}")
    print(f"{BOLD}{YELLOW}{'═' * 60}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# API Helpers
# ─────────────────────────────────────────────────────────────────────────────

def post_chat(client: httpx.Client, message: str) -> Dict[str, Any]:
    """Send a chat message and return the parsed JSON response."""
    resp = client.post(
        f"{BASE_URL}/v1/chat",
        json={"session_id": SESSION_ID, "message": message},
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()


def get_context(client: httpx.Client) -> Dict[str, Any]:
    """Retrieve the current context JSON for the evaluation session."""
    resp = client.get(f"{BASE_URL}/v1/context/{SESSION_ID}", timeout=30.0)
    resp.raise_for_status()
    return resp.json()


def read_context_file() -> Optional[Dict[str, Any]]:
    """
    Read the context JSON directly from disk as a secondary verification path.
    Returns None if the file doesn't exist.
    """
    path = LOGS_DIR / f"context_{SESSION_ID}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight Check
# ─────────────────────────────────────────────────────────────────────────────

def wait_for_server(client: httpx.Client, max_wait: int = 30) -> None:
    """Poll /health until the server responds or timeout is reached."""
    print(f"\nWaiting for server at {BASE_URL} ...")
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            resp = client.get(f"{BASE_URL}/health", timeout=5.0)
            if resp.status_code == 200:
                log_info(f"Server is healthy: {resp.json()}")
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(
        f"Server at {BASE_URL} did not become healthy within {max_wait}s. "
        "Is the service running?  Try: uvicorn src.api:app --port 8000"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation() -> int:
    """
    Execute all 5 turns and assert the required contracts.
    Returns 0 on full pass, 1 on any failure.
    """
    failures: list[str] = []
    responses: list[Dict[str, Any]] = []

    print(f"\n{BOLD}IT Helpdesk Agent — 5-Turn Evaluation{RESET}")
    print(f"Session ID  : {SESSION_ID}")
    print(f"Base URL    : {BASE_URL}")
    print(f"Ticket ID   : {TICKET_ID}")
    print(f"Expected Token Budget: {EXPECTED_TOKEN_BUDGET}")
    log_info("Start the API with TOKEN_BUDGET_LIMIT set to this value (default: 100).")

    with httpx.Client() as client:
        # ── Pre-flight ────────────────────────────────────────────────────────
        wait_for_server(client)

        # ── Execute 5 turns ───────────────────────────────────────────────────
        for i, message in enumerate(TURNS, start=1):
            log_section(f"Turn {i}")
            preview = message[:120].replace("\n", " ")
            log_info(f"User: {preview}{'...' if len(message) > 120 else ''}")

            try:
                result = post_chat(client, message)
                responses.append(result)
            except httpx.HTTPStatusError as e:
                err = f"Turn {i} HTTP error: {e.response.status_code} — {e.response.text[:200]}"
                log_fail(err)
                failures.append(err)
                responses.append({})
                continue
            except Exception as e:
                err = f"Turn {i} request failed: {e}"
                log_fail(err)
                failures.append(err)
                responses.append({})
                continue

            topic = result.get("active_topic", "")
            response_text = result.get("response", "")
            log_info(f"Topic     : {topic}")
            log_info(f"Response  : {response_text[:200]}{'...' if len(response_text) > 200 else ''}")

            # ── Per-turn assertions ───────────────────────────────────────────

            if i == 1:
                # Turn 1: Should be a general greeting topic
                if topic in ("general",):
                    log_pass(f"Turn 1 topic is 'general' (got: {topic!r})")
                else:
                    msg = f"Turn 1 topic expected 'general', got {topic!r}"
                    log_fail(msg)
                    failures.append(msg)

            elif i == 2:
                # Turn 2: Ticket ID should be pinned
                try:
                    ctx = get_context(client)
                    pinned_id = ctx.get("pinned", {}).get("ticket_id")
                    if pinned_id == TICKET_ID:
                        log_pass(f"Turn 2: pinned.ticket_id == '{TICKET_ID}'")
                    else:
                        msg = f"Turn 2: pinned.ticket_id expected '{TICKET_ID}', got '{pinned_id}'"
                        log_fail(msg)
                        failures.append(msg)

                    if topic == "ticket_inquiry":
                        log_pass(f"Turn 2 topic is 'ticket_inquiry'")
                    else:
                        msg = f"Turn 2 topic expected 'ticket_inquiry', got {topic!r}"
                        log_fail(msg)
                        failures.append(msg)
                except Exception as e:
                    msg = f"Turn 2 context check failed: {e}"
                    log_fail(msg)
                    failures.append(msg)

            elif i == 3:
                # Turn 3: Topic should shift to wifi_issue
                if "wifi" in topic.lower() or "network" in topic.lower():
                    log_pass(f"Turn 3 topic shifted to WiFi: {topic!r}")
                else:
                    msg = f"Turn 3 topic expected WiFi-related, got {topic!r}"
                    log_fail(msg)
                    failures.append(msg)

            elif i == 4:
                # Turn 4: Expect a detailed response (longer)
                if len(response_text) > 100:
                    log_pass(f"Turn 4 response is detailed ({len(response_text)} chars)")
                else:
                    msg = f"Turn 4 response is too short ({len(response_text)} chars)"
                    log_fail(msg)
                    failures.append(msg)

            elif i == 5:
                # ── CRITICAL ASSERTIONS for Turn 5 ───────────────────────────

                # A) Bot response must contain the ticket ID
                if TICKET_ID in response_text:
                    log_pass(f"Turn 5: LLM response contains '{TICKET_ID}'")
                else:
                    msg = f"Turn 5: LLM response does NOT contain '{TICKET_ID}'. Response: {response_text[:300]}"
                    log_fail(msg)
                    failures.append(msg)

        # ── Post-5-turns assertions ────────────────────────────────────────────
        log_section("Post-Conversation Context Assertions")

        # B) Check via API
        try:
            final_ctx = get_context(client)
            pinned_id = final_ctx.get("pinned", {}).get("ticket_id")
            recent_messages = final_ctx.get("recent", [])
            recent_contents = [m.get("content", "") for m in recent_messages]
            total_tokens = final_ctx.get("total_recent_tokens", -1)

            if pinned_id == TICKET_ID:
                log_pass(f"API: final pinned.ticket_id == '{TICKET_ID}' (persisted through eviction)")
            else:
                msg = f"API: final pinned.ticket_id == '{pinned_id}', expected '{TICKET_ID}'"
                log_fail(msg)
                failures.append(msg)

            log_info(f"API: total_recent_tokens = {total_tokens}")
            log_info(f"API: recent message count = {len(recent_contents)}")

            if total_tokens <= EXPECTED_TOKEN_BUDGET:
                log_pass(f"API: recent token total is within budget ({EXPECTED_TOKEN_BUDGET})")
            else:
                msg = (
                    f"API: total_recent_tokens={total_tokens} exceeds expected budget "
                    f"{EXPECTED_TOKEN_BUDGET}; configure the server with TOKEN_BUDGET_LIMIT"
                )
                log_fail(msg)
                failures.append(msg)

            # C) Check that earlier turns don't appear in recent (evicted)
            # Token budget forces Turn 1 and 2 out of the window
            turn1_in_recent = any(
                m.get("role") == "user" and m.get("content") == TURNS[0]
                for m in recent_messages
            )
            turn2_in_recent = any(
                m.get("role") == "user" and m.get("content") == TURNS[1]
                for m in recent_messages
            )

            if not turn1_in_recent:
                log_pass("Eviction: Turn 1 message correctly evicted from recent[]")
            else:
                msg = "Eviction failed: Turn 1 remains in recent[]"
                log_fail(msg)
                failures.append(msg)

            if not turn2_in_recent:
                log_pass("Eviction: Turn 2 message correctly evicted from recent[]")
            else:
                msg = "Eviction failed: Turn 2 ticket message remains in recent[]"
                log_fail(msg)
                failures.append(msg)

        except Exception as e:
            msg = f"Final context API check failed: {e}"
            log_fail(msg)
            failures.append(msg)

        # D) Verify context file on disk
        log_info("Checking context JSON file on disk ...")
        file_ctx = read_context_file()
        if file_ctx is not None:
            file_pinned = file_ctx.get("pinned", {}).get("ticket_id")
            if file_pinned == TICKET_ID:
                log_pass(f"Disk: context JSON pinned.ticket_id == '{TICKET_ID}'")
            else:
                msg = f"Disk: context JSON pinned.ticket_id == '{file_pinned}', expected '{TICKET_ID}'"
                log_fail(msg)
                failures.append(msg)

            # Schema validation (req-2)
            required_keys = {"session_id", "active_topic", "pinned", "recent", "total_recent_tokens"}
            missing_keys = required_keys - set(file_ctx.keys())
            if not missing_keys:
                log_pass("Disk: context JSON schema is valid (all required keys present)")
            else:
                msg = f"Disk: context JSON missing keys: {missing_keys}"
                log_fail(msg)
                failures.append(msg)
        else:
            msg = f"Context file not found at logs/context_{SESSION_ID}.json"
            log_fail(msg)
            failures.append(msg)

    # ── Summary ───────────────────────────────────────────────────────────────
    log_section("Evaluation Summary")

    if failures:
        print(f"\n{RED}{BOLD}FAILED — {len(failures)} assertion(s) did not pass:{RESET}")
        for f in failures:
            print(f"  {RED}• {f}{RESET}")
        print(f"\n{RED}Exit code: 1{RESET}\n")
        return 1
    else:
        print(f"\n{GREEN}{BOLD}ALL ASSERTIONS PASSED ✓{RESET}")
        print(
            f"\n{GREEN}The IT Helpdesk Agent correctly:{RESET}\n"
            f"  ✓ Pinned ticket {TICKET_ID} from Turn 2\n"
            f"  ✓ Switched topic to WiFi in Turn 3\n"
            f"  ✓ Returned ticket context in Turn 5 despite eviction\n"
            f"  ✓ Context JSON schema is valid on disk\n"
        )
        print(f"{GREEN}Exit code: 0{RESET}\n")
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        exit_code = run_evaluation()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Evaluation interrupted by user.{RESET}")
        exit_code = 1
    except Exception as e:
        print(f"\n{RED}Evaluation crashed unexpectedly:{RESET}")
        traceback.print_exc()
        exit_code = 1

    sys.exit(exit_code)
