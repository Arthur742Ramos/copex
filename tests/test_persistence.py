from __future__ import annotations

import json

import pytest

from copex import persistence
from copex.persistence import Message, PersistentSession, SessionStore


class TestSessionStore:
    def test_creates_base_directory(self, tmp_path):
        target = tmp_path / "sessions" / "nested"
        store = SessionStore(base_dir=target)
        assert store.base_dir == target
        assert target.is_dir()

    def test_session_path_sanitizes_invalid_and_reserved_names(self, tmp_path):
        store = SessionStore(base_dir=tmp_path)

        assert store._session_path("a/b:c*id").name == "a_b_c_id.json"
        assert store._session_path("CON").name.lower().startswith("_con")

    def test_save_load_round_trip_and_created_at_stability(self, tmp_path):
        store = SessionStore(base_dir=tmp_path)
        first_messages = [Message(role="user", content="hello")]

        store.save("session-1", first_messages, model="m1", reasoning_effort="low", metadata={"x": 1})
        first = store.load("session-1")

        store.save(
            "session-1",
            first_messages + [Message(role="assistant", content="world")],
            model="m1",
            reasoning_effort="low",
            metadata={"x": 2},
        )
        second = store.load("session-1")

        assert first is not None
        assert second is not None
        assert second.created_at == first.created_at
        assert second.updated_at >= first.updated_at
        assert len(second.messages) == 2
        assert second.metadata == {"x": 2}

    def test_load_missing_returns_none(self, tmp_path):
        store = SessionStore(base_dir=tmp_path)
        assert store.load("does-not-exist") is None

    def test_load_rejects_oversized_file(self, tmp_path, monkeypatch):
        store = SessionStore(base_dir=tmp_path)
        monkeypatch.setattr(persistence, "_MAX_SESSION_FILE_SIZE", 20)
        path = store._session_path("large")
        path.write_text("x" * 25, encoding="utf-8")

        with pytest.raises(ValueError, match="too large"):
            store.load("large")

    def test_load_rejects_invalid_json_with_clear_error(self, tmp_path):
        store = SessionStore(base_dir=tmp_path)
        path = store._session_path("bad-json")
        path.write_text("{not-json", encoding="utf-8")

        with pytest.raises(ValueError, match=r"Invalid session JSON in bad-json\.json:"):
            store.load("bad-json")

    def test_load_rejects_invalid_payload_with_clear_error(self, tmp_path):
        store = SessionStore(base_dir=tmp_path)
        path = store._session_path("bad-payload")
        path.write_text(
            json.dumps(
                {
                    "id": "bad-payload",
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T00:00:00",
                    "model": "m",
                    "reasoning_effort": "low",
                    "messages": "not-a-list",
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(
            ValueError,
            match=r"Invalid session payload in bad-payload\.json: messages must be a list",
        ):
            store.load("bad-payload")

    def test_list_sessions_skips_corruption_and_sorts_by_updated_at(self, tmp_path):
        store = SessionStore(base_dir=tmp_path)
        newer = {
            "id": "new",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-03T00:00:00",
            "model": "m",
            "reasoning_effort": "low",
            "messages": [{"role": "user", "content": "n", "timestamp": "t", "metadata": {}}],
        }
        older = {
            "id": "old",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-02T00:00:00",
            "model": "m",
            "reasoning_effort": "low",
            "messages": [{"role": "user", "content": "o", "timestamp": "t", "metadata": {}}],
        }
        store._session_path("new").write_text(json.dumps(newer), encoding="utf-8")
        store._session_path("old").write_text(json.dumps(older), encoding="utf-8")
        store._session_path("bad").write_text("{not-json", encoding="utf-8")

        sessions = store.list_sessions()

        assert [item["id"] for item in sessions] == ["new", "old"]

    def test_delete_and_export(self, tmp_path):
        store = SessionStore(base_dir=tmp_path)
        store.save("exp", [Message(role="user", content="hello")], model="m1", reasoning_effort="low")

        exported_json = store.export("exp", format="json")
        exported_md = store.export("exp", format="markdown")

        assert '"id": "exp"' in exported_json
        assert "# Session: exp" in exported_md
        assert store.delete("exp") is True
        assert store.delete("exp") is False


class TestPersistentSession:
    def test_loads_existing_session_on_init(self, tmp_path):
        store = SessionStore(base_dir=tmp_path)
        store.save(
            "existing",
            [Message(role="user", content="existing message")],
            model="model-x",
            reasoning_effort="medium",
            metadata={"project": "demo"},
        )

        session = PersistentSession("existing", store, auto_save=False)

        assert session.model == "model-x"
        assert session.reasoning_effort == "medium"
        assert len(session.messages) == 1
        assert session.metadata == {"project": "demo"}

    def test_add_assistant_message_saves_reasoning_metadata(self, tmp_path):
        store = SessionStore(base_dir=tmp_path)
        session = PersistentSession("auto", store, model="m", reasoning_effort="low", auto_save=True)

        session.add_assistant_message("answer", reasoning="step-by-step")
        loaded = store.load("auto")

        assert loaded is not None
        assert loaded.messages[-1].role == "assistant"
        assert loaded.messages[-1].metadata == {"reasoning": "step-by-step"}

    def test_clear_and_get_context(self, tmp_path):
        store = SessionStore(base_dir=tmp_path)
        session = PersistentSession("ctx", store, auto_save=False)
        session.add_user_message("u1")
        session.add_assistant_message("a1")

        assert session.get_context(max_messages=1) == [{"role": "assistant", "content": "a1"}]

        session.clear()
        assert session.messages == []
