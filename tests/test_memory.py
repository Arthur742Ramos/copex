from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from copex.cli import app
from copex.config import CopexConfig
from copex.memory import ProjectMemory, compose_memory_instructions, extract_learning_candidates


def test_project_memory_add_and_parse_entries(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)

    assert memory.add_entry("Always use type hints in Python code.", kind="manual")
    assert not memory.add_entry("Always use type hints in Python code.", kind="manual")

    entries = memory.parse_entries()
    assert len(entries) == 1
    assert entries[0].kind == "manual"
    assert "type hints" in entries[0].text
    assert memory.memory_path == tmp_path / ".copex" / "memory.md"


def test_build_prompt_context_reads_preferences_and_guidance(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)
    memory.add_entry("Use async/await for I/O code paths.", kind="pattern")

    memory.preferences_path.parent.mkdir(parents=True, exist_ok=True)
    memory.preferences_path.write_text(
        (
            "[preferences]\n"
            "preferred_model = \"gpt-5.3-codex\"\n"
            "reasoning_level = \"high\"\n"
            "coding_style = \"typed-python\"\n"
        ),
        encoding="utf-8",
    )
    (tmp_path / "CLAUDE.md").write_text("Always keep changes minimal and well-tested.", encoding="utf-8")

    context = memory.build_prompt_context()
    assert context is not None
    assert "## Project Memory" in context
    assert "## User Preferences (.copex/preferences.toml)" in context
    assert "preferred_model: gpt-5.3-codex" in context
    assert "## Existing Guidance Files" in context
    assert "CLAUDE.md" in context


def test_compaction_deduplicates_when_memory_too_large(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path, max_size_bytes=700)

    for i in range(120):
        memory.add_entry(
            (
                f"Architectural decision {i}: Use explicit module boundaries, "
                "typed interfaces, and deterministic test fixtures."
            ),
            kind="decision",
        )

    rendered = memory.read_memory()
    assert "## Summary" in rendered
    assert len(memory.parse_entries()) <= 20


def test_import_external_guidance_detects_known_files(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("Prefer clear architecture notes.", encoding="utf-8")
    (tmp_path / ".cursorrules").write_text("Use small, focused patches.", encoding="utf-8")

    memory = ProjectMemory(tmp_path)
    imported = memory.import_external_guidance()

    imported_names = {path.name for path in imported}
    assert imported_names == {"CLAUDE.md", ".cursorrules"}

    entries = memory.parse_entries()
    texts = [entry.text for entry in entries]
    assert any(text.startswith("CLAUDE.md:") for text in texts)
    assert any(text.startswith(".cursorrules:") for text in texts)


def test_extract_learning_candidates_captures_preference_decision_pattern() -> None:
    prompt = (
        "Please always use type hints in Python code and prefer dataclasses for config models. "
        "Avoid broad exception handlers and keep edits minimal."
    )
    response = (
        "Architectural decision: split orchestration and transport layers for clearer boundaries. "
        "Follow naming conventions and async patterns for I/O-heavy functions."
    )

    candidates = extract_learning_candidates(prompt, response)
    kinds = {kind for kind, _ in candidates}
    assert "preference" in kinds
    assert "decision" in kinds
    assert "pattern" in kinds


def test_compose_memory_instructions_merges_base_and_context(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)
    memory.add_entry("Use repository pattern for persistence adapters.", kind="decision")

    merged = compose_memory_instructions("Base instructions", root=tmp_path)
    assert merged is not None
    assert "Base instructions" in merged
    assert "Persistent Project Memory" in merged
    assert "repository pattern" in merged


def test_learn_from_session_persists_auto_captured_entries(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)
    prompt = "Please always use type hints and prefer dataclasses in Python code."
    response = "Architectural decision: keep domain models separate from API DTOs."

    added = memory.learn_from_session(prompt, response, mode="chat")
    assert added >= 1

    entries = memory.parse_entries()
    assert any(entry.kind in {"preference", "decision", "pattern"} for entry in entries)
    assert any(entry.text.startswith("[chat]") for entry in entries)


def test_session_options_include_memory_instructions(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path)
    memory.add_entry("Use snake_case for Python symbols.", kind="pattern")

    config = CopexConfig(working_directory=str(tmp_path))
    opts = config.to_session_options()

    instructions = opts.get("instructions")
    assert isinstance(instructions, str)
    assert "Persistent Project Memory" in instructions
    assert "snake_case" in instructions


def test_memory_cli_add_show_clear(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result_add = runner.invoke(app, ["memory", "add", "Always use type hints in Python code"])
    assert result_add.exit_code == 0
    assert "Added memory entry" in result_add.stdout

    result_show = runner.invoke(app, ["memory", "show"])
    assert result_show.exit_code == 0
    assert "Always use type hints in Python code" in result_show.stdout

    result_clear = runner.invoke(app, ["memory", "clear"])
    assert result_clear.exit_code == 0
    assert "Project memory reset" in result_clear.stdout
    assert "Always use type hints in Python code" not in (tmp_path / ".copex" / "memory.md").read_text(
        encoding="utf-8"
    )


def test_memory_cli_import(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "AGENTS.md").write_text("Prefer strict type checking and explicit error handling.", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["memory", "import"])

    assert result.exit_code == 0
    assert "AGENTS.md" in result.stdout

    entries = ProjectMemory(tmp_path).parse_entries()
    assert any(entry.text.startswith("AGENTS.md:") for entry in entries)
