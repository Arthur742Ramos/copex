"""Tests for skills auto-discovery."""

import tempfile
from pathlib import Path

import pytest

from copex.skills import SkillDiscovery, SkillInfo, get_skill_content, list_skills


class TestSkillInfo:
    """Tests for SkillInfo class."""

    def test_from_directory_valid_skill(self, tmp_path: Path) -> None:
        """Test parsing a valid skill directory."""
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            """---
name: my-skill
description: A test skill
---

# My Skill

Test content.
"""
        )

        info = SkillInfo.from_directory(skill_dir)
        assert info is not None
        assert info.name == "my-skill"
        assert info.description == "A test skill"
        assert info.path == skill_dir

    def test_from_directory_no_skill_file(self, tmp_path: Path) -> None:
        """Test directory without SKILL.md returns None."""
        skill_dir = tmp_path / "empty-skill"
        skill_dir.mkdir()

        info = SkillInfo.from_directory(skill_dir)
        assert info is None

    def test_from_directory_lowercase_skill_md(self, tmp_path: Path) -> None:
        """Test skill.md (lowercase) is also recognized."""
        skill_dir = tmp_path / "lower-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "skill.md"
        skill_file.write_text("# Skill content")

        info = SkillInfo.from_directory(skill_dir)
        assert info is not None
        assert info.name == "lower-skill"

    def test_from_directory_no_frontmatter(self, tmp_path: Path) -> None:
        """Test skill without frontmatter uses directory name."""
        skill_dir = tmp_path / "plain-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("# Just content\n\nNo frontmatter here.")

        info = SkillInfo.from_directory(skill_dir)
        assert info is not None
        assert info.name == "plain-skill"
        assert info.description is None


class TestSkillDiscovery:
    """Tests for SkillDiscovery class."""

    def test_discover_from_explicit_directory(self, tmp_path: Path) -> None:
        """Test discovering skills from explicit directories."""
        skills_root = tmp_path / "skills"
        skills_root.mkdir()

        skill1 = skills_root / "skill-one"
        skill1.mkdir()
        (skill1 / "SKILL.md").write_text("---\ndescription: First\n---\n# One")

        skill2 = skills_root / "skill-two"
        skill2.mkdir()
        (skill2 / "SKILL.md").write_text("---\ndescription: Second\n---\n# Two")

        discovery = SkillDiscovery(
            explicit_dirs=[skills_root],
            auto_discover=False,
        )

        skills = discovery.discover_skills()
        names = {s.name for s in skills}
        assert "skill-one" in names
        assert "skill-two" in names

    def test_discover_respects_disabled_skills(self, tmp_path: Path) -> None:
        """Test that disabled skills are excluded."""
        skills_root = tmp_path / "skills"
        skills_root.mkdir()

        for name in ["enabled", "disabled"]:
            skill = skills_root / name
            skill.mkdir()
            (skill / "SKILL.md").write_text(f"# {name}")

        discovery = SkillDiscovery(
            explicit_dirs=[skills_root],
            disabled_skills={"disabled"},
            auto_discover=False,
        )

        skills = discovery.discover_skills()
        names = {s.name for s in skills}
        assert "enabled" in names
        assert "disabled" not in names

    def test_get_skill_directories_for_sdk(self, tmp_path: Path) -> None:
        """Test getting skill directories as strings for SDK."""
        skills_root = tmp_path / "skills"
        skills_root.mkdir()

        discovery = SkillDiscovery(
            explicit_dirs=[skills_root],
            auto_discover=False,
        )

        dirs = discovery.get_skill_directories_for_sdk()
        assert str(skills_root) in dirs

    def test_auto_discover_false_skips_repo_and_user(self, tmp_path: Path) -> None:
        """Test auto_discover=False only uses explicit dirs."""
        discovery = SkillDiscovery(
            explicit_dirs=[],
            auto_discover=False,
        )

        dirs = discovery.discover_skill_directories()
        assert dirs == []


class TestListSkills:
    """Tests for list_skills function."""

    def test_list_skills_with_explicit_dir(self, tmp_path: Path) -> None:
        """Test listing skills from explicit directory."""
        skills_root = tmp_path / "skills"
        skills_root.mkdir()

        skill = skills_root / "test-skill"
        skill.mkdir()
        (skill / "SKILL.md").write_text("---\ndescription: Test\n---\n# Test")

        skills = list_skills(
            skill_directories=[str(skills_root)],
            auto_discover=False,
        )

        assert len(skills) == 1
        assert skills[0].name == "test-skill"


class TestGetSkillContent:
    """Tests for get_skill_content function."""

    def test_get_skill_content_found(self, tmp_path: Path) -> None:
        """Test getting content of existing skill."""
        skills_root = tmp_path / "skills"
        skills_root.mkdir()

        skill = skills_root / "my-skill"
        skill.mkdir()
        content = "---\nname: my-skill\n---\n# My Skill Content"
        (skill / "SKILL.md").write_text(content)

        result = get_skill_content("my-skill", skill_directories=[str(skills_root)])
        assert result == content

    def test_get_skill_content_not_found(self, tmp_path: Path) -> None:
        """Test getting content of non-existent skill returns None."""
        result = get_skill_content("nonexistent", skill_directories=[str(tmp_path)])
        assert result is None
