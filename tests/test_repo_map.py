from __future__ import annotations

from typer.testing import CliRunner

from copex.cli import app
from copex.repo_map import RepoMap


def test_refresh_creates_cache_and_indexes_symbols(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "auth.py").write_text(
        "import json\n\n"
        "class AuthService:\n"
        "    def login(self, user):\n"
        "        return helper(user)\n\n"
        "def helper(user):\n"
        "    return user\n",
        encoding="utf-8",
    )

    repo_map = RepoMap(tmp_path)
    files = repo_map.refresh(force=True)

    assert "src/auth.py" in files
    entry = files["src/auth.py"]
    assert "AuthService" in entry.classes
    assert "login" in entry.methods
    assert "helper" in entry.functions
    assert (tmp_path / ".copex" / "repo_map.json").is_file()


def test_regex_fallback_when_tree_sitter_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr("copex.repo_map._get_ts_parser", None)
    (tmp_path / "app.py").write_text(
        "class Runner:\n"
        "    def start(self):\n"
        "        return 1\n\n"
        "def util():\n"
        "    return Runner()\n",
        encoding="utf-8",
    )

    repo_map = RepoMap(tmp_path)
    files = repo_map.refresh(force=True)
    entry = files["app.py"]

    assert entry.parser == "regex"
    assert "Runner" in entry.classes
    assert "start" in entry.methods
    assert "util" in entry.functions


def test_incremental_refresh_parses_only_git_changed_files(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")

    repo_map = RepoMap(tmp_path)
    repo_map.refresh(force=True)

    parsed: list[str] = []
    original_parse = repo_map._parse_file

    def tracking_parse(rel_path: str):
        parsed.append(rel_path)
        return original_parse(rel_path)

    monkeypatch.setattr(repo_map, "_parse_file", tracking_parse)
    monkeypatch.setattr(repo_map, "_git_changed_files", lambda: {"a.py"})
    repo_map.refresh(force=False)

    assert parsed == ["a.py"]


def test_rank_relevant_prioritizes_matching_file_and_symbols(tmp_path):
    (tmp_path / "auth.py").write_text(
        "def login_user(username):\n    return username\n",
        encoding="utf-8",
    )
    (tmp_path / "payments.py").write_text(
        "def capture_payment(amount):\n    return amount\n",
        encoding="utf-8",
    )

    repo_map = RepoMap(tmp_path)
    repo_map.refresh(force=True)
    ranked = repo_map.rank_relevant("fix login auth workflow", limit=2)

    assert ranked
    assert ranked[0].path == "auth.py"


def test_render_map_and_relevant_outputs(tmp_path):
    (tmp_path / "service.py").write_text(
        "class Service:\n"
        "    def run(self):\n"
        "        return helper()\n\n"
        "def helper():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    repo_map = RepoMap(tmp_path)
    repo_map.refresh(force=True)

    rendered = repo_map.render_map()
    relevant = repo_map.render_relevant("run service")

    assert "Repo Map" in rendered
    assert "service.py" in rendered
    assert "Relevant files for: run service" in relevant


def test_map_cli_command(tmp_path, monkeypatch):
    (tmp_path / "auth.py").write_text(
        "def login_user(username):\n    return username\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["map", "--relevant", "login auth"])

    assert result.exit_code == 0, result.output
    assert "auth.py" in result.output
