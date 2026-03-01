"""Tests for structured edit parsing, application, undo, and verification."""

from __future__ import annotations

import subprocess
from pathlib import Path

from typer.testing import CliRunner

from copex.cli import app
from copex.edits import (
    EditFormat,
    apply_edit_text,
    list_undo_history,
    parse_structured_edits,
    run_verification,
    undo_last_edit_batch,
)


def test_parse_search_replace_blocks() -> None:
    text = """File: src/example.py
<<<<<<< SEARCH
print("old")
=======
print("new")
>>>>>>> REPLACE
"""
    operations = parse_structured_edits(text)
    assert len(operations) == 1
    op = operations[0]
    assert op.format is EditFormat.SEARCH_REPLACE
    assert op.file_path == "src/example.py"
    assert op.search == 'print("old")\n'
    assert op.replace == 'print("new")\n'


def test_parse_unified_diff_blocks() -> None:
    text = """--- a/src/example.py
+++ b/src/example.py
@@ -1,2 +1,2 @@
-print("old")
+print("new")
 print("same")
"""
    operations = parse_structured_edits(text)
    assert len(operations) == 1
    op = operations[0]
    assert op.format is EditFormat.UNIFIED_DIFF
    assert op.file_path == "src/example.py"
    assert len(op.hunks) == 1


def test_parse_whole_file_replacement_blocks() -> None:
    text = """src/example.py
```python
print("whole replacement")
```
"""
    operations = parse_structured_edits(text)
    assert len(operations) == 1
    op = operations[0]
    assert op.format is EditFormat.WHOLE_FILE
    assert op.file_path == "src/example.py"
    assert op.content == 'print("whole replacement")\n'


def test_fuzzy_search_matching_handles_indentation_differences(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.py"
    file_path.write_text(
        "def greet():\n"
        '    message = "hello"\n'
        "    return message\n",
        encoding="utf-8",
    )

    edit_text = """File: sample.py
<<<<<<< SEARCH
def greet():
  message = "hello"
  return message
=======
def greet():
    message = "hello, world"
    return message
>>>>>>> REPLACE
"""
    result = apply_edit_text(edit_text, root=tmp_path, verify=False)
    assert result.success
    assert result.failed_files == {}
    assert file_path.read_text(encoding="utf-8") == (
        "def greet():\n"
        '    message = "hello, world"\n'
        "    return message\n"
    )


def test_atomic_apply_is_per_file(tmp_path: Path) -> None:
    file_a = tmp_path / "a.py"
    file_b = tmp_path / "b.py"
    file_a.write_text("value = 1\n", encoding="utf-8")
    file_b.write_text("other = 1\n", encoding="utf-8")

    edit_text = """File: a.py
<<<<<<< SEARCH
value = 1
=======
value = 2
>>>>>>> REPLACE
<<<<<<< SEARCH
missing = 1
=======
missing = 2
>>>>>>> REPLACE
File: b.py
<<<<<<< SEARCH
other = 1
=======
other = 2
>>>>>>> REPLACE
"""
    result = apply_edit_text(edit_text, root=tmp_path, verify=False)
    assert "a.py" in result.failed_files
    assert "b.py" in result.applied_files
    assert file_a.read_text(encoding="utf-8") == "value = 1\n"
    assert file_b.read_text(encoding="utf-8") == "other = 2\n"


def test_undo_restores_previous_state_and_lists_history(tmp_path: Path) -> None:
    existing_file = tmp_path / "main.txt"
    created_file = tmp_path / "new.txt"
    existing_file.write_text("original\n", encoding="utf-8")

    edit_text = """main.txt
```
changed
```
new.txt
```
created
```
"""
    result = apply_edit_text(edit_text, root=tmp_path, verify=False)
    assert result.success
    assert result.undo_batch_id is not None
    assert existing_file.read_text(encoding="utf-8") == "changed\n"
    assert created_file.read_text(encoding="utf-8") == "created\n"

    history = list_undo_history(tmp_path)
    assert history
    assert history[0].batch_id == result.undo_batch_id
    assert history[0].file_count == 2

    undo_result = undo_last_edit_batch(tmp_path)
    assert undo_result.batch_id == result.undo_batch_id
    assert existing_file.read_text(encoding="utf-8") == "original\n"
    assert not created_file.exists()


def test_verification_reports_syntax_errors(tmp_path: Path) -> None:
    edit_text = """broken.py
```python
def oops(:
    return 1
```
"""
    result = apply_edit_text(edit_text, root=tmp_path, verify=True)
    assert result.verification is not None
    assert not result.verification.ok
    syntax_check = next(check for check in result.verification.checks if check.name == "syntax")
    assert syntax_check.ran
    assert not syntax_check.success
    assert "broken.py" in syntax_check.output
    assert "Verification failed" in result.verification_feedback()


def test_verification_runs_optional_lint_and_type_checks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    file_path = tmp_path / "ok.py"
    file_path.write_text("x = 1\n", encoding="utf-8")

    def fake_which(name: str) -> str | None:
        if name in {"ruff", "mypy"}:
            return f"/usr/bin/{name}"
        return None

    calls: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        cwd: Path,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[0] == "ruff":
            return subprocess.CompletedProcess(command, 1, stdout="ruff failed", stderr="")
        if command[0] == "mypy":
            return subprocess.CompletedProcess(command, 0, stdout="mypy passed", stderr="")
        raise AssertionError("Unexpected command")

    monkeypatch.setattr("copex.edits.shutil.which", fake_which)
    monkeypatch.setattr("copex.edits.subprocess.run", fake_run)

    report = run_verification([file_path], root=tmp_path)
    assert len(calls) == 2
    assert calls[0][0] == "ruff"
    assert calls[1][0] == "mypy"
    assert not report.ok
    assert not next(check for check in report.checks if check.name == "lint").success


def test_cli_edit_and_undo_commands() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        file_path = Path("demo.py")
        file_path.write_text("value = 1\n", encoding="utf-8")

        edit_text = """File: demo.py
<<<<<<< SEARCH
value = 1
=======
value = 2
>>>>>>> REPLACE
"""
        edit_result = runner.invoke(app, ["edit", "--no-verify"], input=edit_text)
        assert edit_result.exit_code == 0
        assert file_path.read_text(encoding="utf-8") == "value = 2\n"

        list_result = runner.invoke(app, ["undo", "--list"])
        assert list_result.exit_code == 0
        assert "files" in list_result.output

        undo_result = runner.invoke(app, ["undo"])
        assert undo_result.exit_code == 0
        assert file_path.read_text(encoding="utf-8") == "value = 1\n"
