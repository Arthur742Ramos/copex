from __future__ import annotations

from pathlib import Path

import pytest

from copex.config import _parse_node_version


class TestParseNodeVersion:
    def test_standard_version(self):
        """Parse a standard Node version directory."""
        assert _parse_node_version(Path("v18.17.0")) == (18, 17, 0)

    def test_version_without_v_prefix(self):
        """Handle version dirs without v prefix."""
        assert _parse_node_version(Path("20.10.0")) == (20, 10, 0)

    def test_major_only(self):
        """Handle major-only version."""
        assert _parse_node_version(Path("v21")) == (21,)

    def test_major_minor(self):
        """Handle major.minor version."""
        assert _parse_node_version(Path("v18.17")) == (18, 17)

    def test_non_version_dir_returns_zero(self):
        """Non-version directories return (0,) so they sort last."""
        assert _parse_node_version(Path(".DS_Store")) == (0,)
        assert _parse_node_version(Path("lts")) == (0,)

    def test_sorting_semantic_order(self):
        """Sorting with _parse_node_version gives correct semantic order."""
        dirs = [Path("v9.11.2"), Path("v10.0.0"), Path("v20.1.0"), Path("v18.17.0")]
        sorted_dirs = sorted(dirs, key=_parse_node_version, reverse=True)
        names = [d.name for d in sorted_dirs]
        assert names == ["v20.1.0", "v18.17.0", "v10.0.0", "v9.11.2"]

    def test_sorting_beats_lexicographic(self):
        """The key case: v9.x should NOT sort above v10.x (which lexicographic does)."""
        dirs = [Path("v9.99.99"), Path("v10.0.0")]
        # Lexicographic (broken): v9 > v10 because '9' > '1'
        lex_sorted = sorted(dirs, reverse=True)
        assert lex_sorted[0].name == "v9.99.99"  # Wrong!

        # Semantic (fixed): v10 > v9
        sem_sorted = sorted(dirs, key=_parse_node_version, reverse=True)
        assert sem_sorted[0].name == "v10.0.0"  # Correct!

    def test_non_version_sorts_last(self):
        """Non-version dirs should sort after real versions."""
        dirs = [Path(".DS_Store"), Path("v18.0.0"), Path("lts")]
        sorted_dirs = sorted(dirs, key=_parse_node_version, reverse=True)
        assert sorted_dirs[0].name == "v18.0.0"
