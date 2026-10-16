import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from eval_5_turns import TICKET_ID, TURNS
from src import api
from src.context_engine import (
    AgentContext,
    Message,
    apply_token_budget,
    classify_topic_keywords,
    count_tokens,
    update_context_with_extraction,
)


class ContextContractsTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.logs_patch = patch("src.context_engine.LOGS_DIR", Path(self.temp_dir.name))
        self.logs_patch.start()
        api.session_store.clear()

    def tearDown(self):
        api.session_store.clear()
        self.logs_patch.stop()
        self.temp_dir.cleanup()

    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_generation_failure_does_not_commit_candidate_state(self):
        existing = AgentContext(
            session_id="existing",
            active_topic="general",
            recent=[Message(role="user", content="Earlier message")],
        )
        existing.total_recent_tokens = existing.recent[0].tokens
        api.session_store["existing"] = existing
        before = existing.to_persistence_dict()

        with patch("src.api.classify_topic_llm", return_value=None), patch(
            "src.api.generate_response", side_effect=RuntimeError("upstream down")
        ):
            response = await self.client.post(
                "/v1/chat",
                json={"session_id": "existing", "message": "Ticket IT-4921 has WiFi trouble"},
            )
            new_session_response = await self.client.post(
                "/v1/chat",
                json={"session_id": "new-session", "message": "Ticket IT-7777"},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(new_session_response.status_code, 500)
        self.assertEqual(existing.to_persistence_dict(), before)
        self.assertNotIn("new-session", api.session_store)
        self.assertFalse((Path(self.temp_dir.name) / "context_existing.json").exists())

        missing_context = await self.client.get("/v1/context/new-session")
        self.assertEqual(missing_context.status_code, 404)

    async def test_success_pins_ticket_and_persists_schema(self):
        with patch("src.api.classify_topic_llm", return_value=None), patch(
            "src.api.generate_response", return_value="I will review ticket IT-4921."
        ):
            response = await self.client.post(
                "/v1/chat",
                json={"session_id": "ticket-session", "message": "My ticket is IT-4921"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["active_topic"], "ticket_inquiry")
        context_path = Path(self.temp_dir.name) / "context_ticket-session.json"
        persisted = json.loads(context_path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["pinned"]["ticket_id"], "IT-4921")
        self.assertLessEqual(persisted["total_recent_tokens"], api.get_token_budget())
        self.assertEqual(
            set(persisted),
            {"session_id", "active_topic", "pinned", "recent", "total_recent_tokens"},
        )

    async def test_oversized_current_message_is_still_sent_to_model(self):
        message = "Please help with campus WiFi in the library " * 12
        captured = {}

        def generate(messages):
            captured["messages"] = messages
            return "I can help troubleshoot the connection."

        with patch("src.api.get_token_budget", return_value=10), patch(
            "src.api.classify_topic_llm", return_value=None
        ), patch("src.api.generate_response", side_effect=generate):
            response = await self.client.post(
                "/v1/chat",
                json={"session_id": "oversized", "message": message},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["messages"][-1], {"role": "user", "content": message.strip()})
        context = api.session_store["oversized"]
        self.assertLessEqual(context.total_recent_tokens, 10)

    async def test_persisted_session_is_hydrated_after_memory_is_cleared(self):
        captured = {}

        def generate(messages):
            captured["messages"] = messages
            return "The ticket remains recorded."

        with patch("src.api.classify_topic_llm", return_value=None), patch(
            "src.api.generate_response", side_effect=generate
        ):
            first = await self.client.post(
                "/v1/chat",
                json={"session_id": "hydrated", "message": "My ticket is IT-8812"},
            )
            api.session_store.clear()
            second = await self.client.post(
                "/v1/chat",
                json={"session_id": "hydrated", "message": "What was my ticket?"},
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertIn("Ticket Id: IT-8812", captured["messages"][0]["content"])
        self.assertIn("My ticket is IT-8812", [item["content"] for item in captured["messages"]])

    async def test_five_turn_evaluation_contract_without_external_llm(self):
        session_id = "offline-five-turns"

        def generate(messages):
            latest = messages[-1]["content"].lower()
            system_prompt = messages[0]["content"]
            if "ticket" in latest and TICKET_ID in system_prompt:
                response = f"I have the ticket ID {TICKET_ID}; I can follow up on it."
            elif "wifi" in latest or "eduroam" in latest:
                response = "Mac troubleshooting: " + "Check the network settings and reconnect. " * 45
            else:
                response = "Hello, I can help with your campus IT issue."
            return response

        topics = ["general", "ticket_inquiry", "wifi_issue", "wifi_issue", "ticket_inquiry"]
        with patch("src.api.get_token_budget", return_value=100), patch(
            "src.api.classify_topic_llm", return_value=None
        ), patch("src.api.generate_response", side_effect=generate):
            results = []
            for message in TURNS:
                response = await self.client.post(
                    "/v1/chat",
                    json={"session_id": session_id, "message": message},
                )
                self.assertEqual(response.status_code, 200)
                results.append(response.json())

        self.assertEqual([result["active_topic"] for result in results], topics)
        self.assertGreater(len(results[3]["response"]), 100)
        self.assertIn(TICKET_ID, results[-1]["response"])
        final_context = json.loads(
            (Path(self.temp_dir.name) / f"context_{session_id}.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(final_context["pinned"]["ticket_id"], TICKET_ID)
        self.assertLessEqual(final_context["total_recent_tokens"], 100)
        retained_user_messages = {
            message["content"]
            for message in final_context["recent"]
            if message["role"] == "user"
        }
        self.assertNotIn(TURNS[0], retained_user_messages)
        self.assertNotIn(TURNS[1], retained_user_messages)

    def test_ticket_return_topic_overrides_incidental_wifi_mention(self):
        message = "The WiFi is fixed; let's go back to my original ticket status."
        self.assertEqual(classify_topic_keywords(message), "ticket_inquiry")
        context = AgentContext(session_id="topic-return")
        update_context_with_extraction(context, message, llm_topic="wifi_issue")
        self.assertEqual(context.active_topic, "ticket_inquiry")

    def test_tiktoken_fifo_eviction_keeps_newest_message_within_budget(self):
        oldest = Message(role="user", content="alpha")
        newest = Message(role="user", content="beta")
        pruned = apply_token_budget([oldest, newest], count_tokens(newest.content))

        self.assertEqual(pruned, [newest])
        self.assertLessEqual(sum(message.tokens for message in pruned), newest.tokens)


if __name__ == "__main__":
    unittest.main()