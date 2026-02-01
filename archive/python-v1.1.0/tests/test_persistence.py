from __future__ import annotations

from pathlib import Path

import pytest

from copex.models import Model, ReasoningEffort
from copex.persistence import Message, PersistentSession, SessionData, SessionStore


def test_session_store_save_and_load(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    messages = [Message(role="user", content="Hello")]

    path = store.save("session-1", messages, Model.CLAUDE_OPUS_4_5, ReasoningEffort.XHIGH)

    assert path.exists()
    loaded = store.load("session-1")
    assert loaded is not None
    assert loaded.id == "session-1"
    assert loaded.model == Model.CLAUDE_OPUS_4_5.value
    assert loaded.reasoning_effort == ReasoningEffort.XHIGH.value
    assert loaded.messages[0].content == "Hello"


def test_session_store_path_sanitization(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    path = store._session_path("weird/id:with spaces")
    assert path.name == "weird_id_with_spaces.json"


def test_session_store_list_sessions_sorted(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    messages = [Message(role="user", content="Hi")]
    store.save("first", messages, Model.CLAUDE_OPUS_4_5, ReasoningEffort.XHIGH)
    store.save("second", messages, Model.CLAUDE_OPUS_4_5, ReasoningEffort.XHIGH)

    sessions = store.list_sessions()
    assert len(sessions) == 2
    assert sessions[0]["updated_at"] >= sessions[1]["updated_at"]


def test_session_store_export_formats(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    messages = [Message(role="user", content="Hello")]
    store.save("session-1", messages, Model.CLAUDE_OPUS_4_5, ReasoningEffort.XHIGH)

    json_export = store.export("session-1", format="json")
    assert '"id": "session-1"' in json_export

    markdown_export = store.export("session-1", format="markdown")
    assert "# Session: session-1" in markdown_export
    assert "Hello" in markdown_export


def test_session_store_delete_and_missing(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    messages = [Message(role="user", content="Hello")]
    store.save("session-1", messages, Model.CLAUDE_OPUS_4_5, ReasoningEffort.XHIGH)

    assert store.delete("session-1") is True
    assert store.load("session-1") is None
    assert store.delete("missing") is False


def test_persistent_session_auto_save(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    session = PersistentSession("persist-1", store, auto_save=True)

    session.add_user_message("Hello")

    loaded = store.load("persist-1")
    assert loaded is not None
    assert len(loaded.messages) == 1
    assert loaded.messages[0].content == "Hello"


def test_persistent_session_load_existing(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    messages = [Message(role="user", content="Hello")]
    store.save("persist-1", messages, Model.CLAUDE_OPUS_4_5, ReasoningEffort.XHIGH)

    session = PersistentSession("persist-1", store, auto_save=False)
    assert len(session.messages) == 1
    assert session.messages[0].content == "Hello"


def test_persistent_session_context_limit(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    session = PersistentSession("persist-1", store, auto_save=False)
    session.add_user_message("one")
    session.add_user_message("two")

    context = session.get_context(max_messages=1)
    assert context == [{"role": "user", "content": "two"}]
