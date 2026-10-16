"""Quick endpoint verification test."""
import httpx
import sys
import uuid

base = "http://localhost:8000"
session_id = f"verify_{uuid.uuid4().hex[:10]}"

# Test 1: Health check (req-7)
r = httpx.get(f"{base}/health")
print(f"Health: {r.status_code} {r.json()}")
assert r.status_code == 200 and r.json() == {"status": "ok"}, "Health FAILED"
print("  PASS: Health check returns 200 + {status: ok}")

# Test 2: POST /v1/chat — an invalid key should exercise req-10 rollback
r = httpx.post(
    f"{base}/v1/chat",
    json={"session_id": session_id, "message": "Hi I need help"},
    timeout=10,
)
print(f"Chat (bad key): {r.status_code}")
assert r.status_code in (200, 500), f"Unexpected status: {r.status_code}"
chat_status = r.status_code
if r.status_code == 500:
    print("  PASS: LLM failure returns 500 as expected")

# Test 3: GET /v1/context non-existent -> 404 (req-9)
r = httpx.get(f"{base}/v1/context/non_existent_123")
print(f"Context 404: {r.status_code}")
assert r.status_code == 404, f"Expected 404, got {r.status_code}"
print("  PASS: Unknown session returns 404")

# Test 4: Verify schema and, on failure, rollback (req-2 and req-10)
r = httpx.get(f"{base}/v1/context/{session_id}")
print(f"Context after chat: {r.status_code}")
if r.status_code == 200:
    ctx = r.json()
    # Schema check (req-2)
    required = {"session_id", "active_topic", "pinned", "recent", "total_recent_tokens"}
    missing = required - set(ctx.keys())
    assert not missing, f"Schema missing keys: {missing}"
    print("  PASS: req-2 schema — all required keys present")
    if chat_status == 500:
        assert not ctx["recent"], "req-10 FAILED: recent contains messages after LLM failure"
        print("  PASS: req-10 rollback — recent[] is empty after LLM failure")
elif r.status_code == 404:
    assert chat_status == 500, "Successful chat session unexpectedly missing"
    print("  PASS: Session not created on LLM failure (also valid for req-10)")
else:
    raise AssertionError(f"Unexpected context status: {r.status_code}")

print("\nALL ENDPOINT TESTS PASSED")
