# Context-Aware IT Helpdesk Agent

> **Hard / AI Engineer** — Context Engineering · Token Management · State Management · API Design

A production-grade IT helpdesk chatbot with explicit context engineering:
rolling-window **token eviction**, **pinned memory** for critical facts, and
dynamic **topic tracking** — built without opaque memory abstractions.

---

## Architecture

```
Client
  │
  ▼ POST /v1/chat {session_id, message}
FastAPI Router (src/api.py)
  │
  ├─► Session Hydration (in-memory session_store)
  │
  ├─► Entity Extraction (regex: IT-\d{4})  ──► pinned["ticket_id"]
  │
  ├─► Topic Classification (keywords + LLM)  ──► active_topic
  │
  ├─► Prompt Builder (system prompt + pinned facts + active_topic)
  │
  ├─► Token Budget Eviction (model-specific tiktoken BPE, FIFO)
  │
  ├─► LLM Generation (OpenAI gpt-4o-mini)
  │         │ on failure → HTTP 500, context NOT mutated (req-10)
  │         │ on success ↓
  ├─► Commit messages to recent[], recount tokens
  │
  ├─► Persist → ./logs/context_{session_id}.json
  │
  └─► Return {response, active_topic}
```

---

## Quickstart

### 1. Clone & Configure

```bash
git clone <repo-url>
cd helpdesk-agent
cp .env.example .env
# Edit .env — set OPENAI_API_KEY
```

### 2. Run with Docker (recommended)

```bash
docker-compose up -d

# Verify health
curl http://localhost:8000/health
# → {"status": "ok"}
```

### 3. Run without Docker

```bash
pip install -r requirements.txt
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Reference

### `POST /v1/chat`

Start or continue a conversation.

**Request:**
```json
{
  "session_id": "student_001",
  "message": "My ticket IT-9921 is still unresolved."
}
```

**Response (200):**
```json
{
  "response": "I can see your ticket IT-9921. Let me check the status...",
  "active_topic": "ticket_inquiry"
}
```

---

### `GET /v1/context/{session_id}`

Inspect the live context state for a session.

**Response (200):**
```json
{
  "session_id": "student_001",
  "active_topic": "ticket_inquiry",
  "pinned": {
    "ticket_id": "IT-9921"
  },
  "recent": [
    {"role": "user", "content": "My ticket IT-9921 is still unresolved."},
    {"role": "assistant", "content": "I can see your ticket IT-9921..."}
  ],
  "total_recent_tokens": 87
}
```

**Response (404):** Session not found.

---

### `GET /health`

Docker health check.

```json
{"status": "ok"}
```

---

## 5-Turn Evaluation

The evaluation script proves the core contract: **ticket ID is preserved in
`pinned` even after the messages that introduced it are evicted from `recent[]`**.

```bash
# Run against a server on localhost:8000
python eval_5_turns.py

# Override base URL
BASE_URL=http://localhost:8000 python eval_5_turns.py

# Force aggressive eviction during evaluation (PowerShell)
$env:TOKEN_BUDGET_LIMIT = "100"; docker compose up -d
python eval_5_turns.py
```

The evaluation assumes a 100-token API budget by default. Set
`TOKEN_BUDGET_LIMIT` in the service environment before starting it; the script
checks that older turns were actually evicted and that the persisted context
file exists. Run the focused regression suite with
`python -m unittest discover -s tests -v`.

**Turn sequence:**
| Turn | Message | Assertion |
|------|---------|-----------|
| 1 | "Hi, I need help." | topic = general |
| 2 | "My ticket is IT-8812." | pinned.ticket_id = "IT-8812" |
| 3 | Long WiFi/Mac message | topic shifts to wifi_issue; old turns evicted |
| 4 | "Give me Mac troubleshooting steps." | detailed response |
| 5 | "What was the status of the ticket I mentioned?" | response contains "IT-8812" |

Script exits **0** on pass, **1** on any failure.

---

## Context Engineering Details

### Pinned Memory (never evicted)

```python
# From context_engine.py
context.pinned = {"ticket_id": "IT-8812"}
```

Pinned facts are injected into the system prompt on every turn, bypassing
the token eviction policy. The LLM always "knows" the ticket ID even after
the original message that contained it has been dropped from `recent[]`.

### Token Budget Eviction (FIFO)

```python
def apply_token_budget(recent, max_tokens):
    pruned = list(recent)
    current_total = sum(m.tokens for m in pruned)
    while pruned and current_total > max_tokens:
        evicted = pruned.pop(0)           # oldest first
        current_total -= evicted.tokens
    return pruned
```

Uses **tiktoken's model-specific BPE encoding** — not raw word counts. The
configured model defaults to `gpt-4o-mini`; unknown model names fall back to
`cl100k_base`.
`TOKEN_BUDGET_LIMIT` applies **only** to `recent[]`. System prompt and
pinned facts are base overhead, not counted.
The current user message is still sent to the model if it alone exceeds the
budget; older history is removed, and the persisted `recent[]` remains within
the configured limit after the turn is committed.

### Topic Classification

1. **Keyword heuristic** (zero latency) — scans for wifi/ticket/print keywords
2. **LLM structured output** (enhancement) — `{"topic": "wifi_issue"}` via `response_format=json_object`

If the LLM classifier fails, the keyword fallback is used automatically.

### Graceful Error Handling (req-10)

The user message is **staged** before the LLM call and only committed to
`context.recent` if generation succeeds. On failure → HTTP 500, context unchanged.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required. Your OpenAI API key. |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model for generation and topic classification. |
| `TOKEN_BUDGET_LIMIT` | `4000` | Max tokens in `recent[]`. Set to `100` to force eviction during eval. |

---

## Project Layout

```
helpdesk-agent/
├── src/
│   ├── __init__.py
│   ├── api.py             # FastAPI server + routes
│   ├── context_engine.py  # AgentContext, token eviction, extraction, persistence
│   └── llm_client.py      # OpenAI wrapper
├── logs/                  # context_{session_id}.json written here
├── eval_5_turns.py        # 5-turn evaluation script
├── tests/                 # Context contract regression tests
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── README.md
```

---

## Requirements Coverage

| Req | Description | Status |
|-----|-------------|--------|
| req-1 | `POST /v1/chat` with schema | ✅ |
| req-2 | `./logs/context_{id}.json` after every turn | ✅ |
| req-3 | Regex entity extraction → `pinned.ticket_id` | ✅ |
| req-4 | tiktoken FIFO eviction on `TOKEN_BUDGET_LIMIT` | ✅ |
| req-5 | Dynamic `active_topic` updates | ✅ |
| req-6 | `eval_5_turns.py` exits 0 | ✅ |
| req-7 | `docker-compose up` + `/health` | ✅ |
| req-8 | `.env.example` with all vars, no real keys | ✅ |
| req-9 | `GET /v1/context/{id}` with 200/404 | ✅ |
| req-10 | Rollback on LLM failure, HTTP 500 | ✅ |
