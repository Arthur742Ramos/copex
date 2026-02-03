"""Skills auto-discovery and management for Copex."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SkillInfo:
    """Information about a discovered skill."""

    name: str
    path: Path
    description: str | None = None
    source: str = "unknown"  # "repo", "user", "explicit"

    @classmethod
    def from_directory(cls, skill_dir: Path, source: str = "unknown") -> SkillInfo | None:
        """Parse skill info from a skill directory."""
        # Look for SKILL.md or skill.md
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            skill_file = skill_dir / "skill.md"
        if not skill_file.exists():
            return None

        name = skill_dir.name
        description = None

        # Try to parse frontmatter for description
        try:
            content = skill_file.read_text(encoding="utf-8")
            if content.startswith("---"):
                end_idx = content.find("---", 3)
                if end_idx != -1:
                    frontmatter = content[3:end_idx]
                    for line in frontmatter.split("\n"):
                        if line.startswith("description:"):
                            description = line.split(":", 1)[1].strip().strip("\"'")
                            break
                        if line.startswith("name:"):
                            name = line.split(":", 1)[1].strip().strip("\"'")
        except Exception:
            pass

        return cls(name=name, path=skill_dir, description=description, source=source)


@dataclass
class SkillDiscovery:
    """Handles skill auto-discovery from various locations."""

    repo_root: Path | None = None
    user_skills_dir: Path = field(default_factory=lambda: Path.home() / ".config" / "copex" / "skills")
    explicit_dirs: list[Path] = field(default_factory=list)
    disabled_skills: set[str] = field(default_factory=set)
    auto_discover: bool = True

    def find_git_root(self, start_dir: Path | None = None) -> Path | None:
        """Find the git repository root from start_dir or cwd."""
        if start_dir is None:
            start_dir = Path.cwd()

        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=start_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return Path(result.stdout.strip())
        except Exception:
            pass
        return None

    def discover_skill_directories(self) -> list[Path]:
        """Discover all skill directories.

        Returns list of directories containing skills (not individual skill dirs).
        """
        skill_dirs: list[Path] = []

        if self.auto_discover:
            # 1. Repo-level skills
            repo_root = self.repo_root or self.find_git_root()
            if repo_root:
                # .github/skills/
                github_skills = repo_root / ".github" / "skills"
                if github_skills.is_dir():
                    skill_dirs.append(github_skills)

                # .claude/skills/ (Claude Code compatibility)
                claude_skills = repo_root / ".claude" / "skills"
                if claude_skills.is_dir():
                    skill_dirs.append(claude_skills)

                # .copex/skills/
                copex_skills = repo_root / ".copex" / "skills"
                if copex_skills.is_dir():
                    skill_dirs.append(copex_skills)

            # 2. User-level skills (~/.config/copex/skills/)
            if self.user_skills_dir.is_dir():
                skill_dirs.append(self.user_skills_dir)

        # 3. Explicit directories (always added)
        for explicit_dir in self.explicit_dirs:
            if explicit_dir.is_dir() and explicit_dir not in skill_dirs:
                skill_dirs.append(explicit_dir)

        return skill_dirs

    def discover_skills(self) -> list[SkillInfo]:
        """Discover all available skills."""
        skills: list[SkillInfo] = []
        seen_names: set[str] = set()

        for skill_dir_root in self.discover_skill_directories():
            source = self._determine_source(skill_dir_root)

            # Each subdirectory in the skill_dir_root is a potential skill
            if not skill_dir_root.is_dir():
                continue

            for entry in skill_dir_root.iterdir():
                if not entry.is_dir():
                    continue
                if entry.name.startswith("."):
                    continue

                skill_info = SkillInfo.from_directory(entry, source=source)
                if skill_info and skill_info.name not in seen_names:
                    if skill_info.name not in self.disabled_skills:
                        skills.append(skill_info)
                        seen_names.add(skill_info.name)

        return skills

    def _determine_source(self, skill_dir: Path) -> str:
        """Determine the source type of a skill directory."""
        path_str = str(skill_dir)
        if ".github" in path_str or ".claude" in path_str or ".copex" in path_str:
            return "repo"
        if str(self.user_skills_dir) in path_str:
            return "user"
        return "explicit"

    def get_skill_directories_for_sdk(self) -> list[str]:
        """Get skill directories as strings for the SDK."""
        return [str(d.resolve()) for d in self.discover_skill_directories()]


def get_working_directory() -> str:
    """Get the current working directory for SDK context."""
    return str(Path.cwd())


def list_skills(
    skill_directories: list[str] | None = None,
    disabled_skills: list[str] | None = None,
    auto_discover: bool = True,
) -> list[SkillInfo]:
    """List all available skills.

    Args:
        skill_directories: Explicit skill directories to search
        disabled_skills: Skills to exclude
        auto_discover: Whether to auto-discover from repo and user dirs

    Returns:
        List of discovered skills
    """
    discovery = SkillDiscovery(
        explicit_dirs=[Path(d) for d in (skill_directories or [])],
        disabled_skills=set(disabled_skills or []),
        auto_discover=auto_discover,
    )
    return discovery.discover_skills()


def get_skill_content(skill_name: str, skill_directories: list[str] | None = None) -> str | None:
    """Get the content of a specific skill.

    Args:
        skill_name: Name of the skill to retrieve
        skill_directories: Explicit skill directories to search

    Returns:
        Skill content or None if not found
    """
    discovery = SkillDiscovery(
        explicit_dirs=[Path(d) for d in (skill_directories or [])],
    )

    for skill in discovery.discover_skills():
        if skill.name == skill_name:
            skill_file = skill.path / "SKILL.md"
            if not skill_file.exists():
                skill_file = skill.path / "skill.md"
            if skill_file.exists():
                return skill_file.read_text(encoding="utf-8")
    return None
