---
name: release
description: Release and publish Copex to PyPI. Use this when asked to release, publish, bump version, or create a new version.
---

# Release Process for Copex

Follow these steps to release a new version of Copex:

## 1. Run Tests First

```bash
python -m pytest tests/ -v
```

All tests must pass before releasing.

## 2. Bump Version

Update version in BOTH files (must stay in sync):

1. `pyproject.toml` → change `version = "X.Y.Z"`
2. `src/copex/cli.py` → change `__version__ = "X.Y.Z"`

### Version Guidelines
- **Patch** (0.3.1): Bug fixes, no API changes
- **Minor** (0.4.0): New features, backward compatible  
- **Major** (1.0.0): Breaking API changes

## 3. Commit Version Bump

```bash
git add -A
git commit -m "chore: bump version to X.Y.Z"
```

## 4. Create and Push Tag

```bash
git tag vX.Y.Z
git push
git push --tags
```

## 5. Create GitHub Release

This triggers the PyPI publish workflow.

```powershell
# PowerShell - clear conflicting env var if needed
$env:GH_TOKEN = $null

gh release create vX.Y.Z --title "vX.Y.Z" --notes "### Changes

- Description of changes"
```

## 6. Verify Publish

Check the workflow at: https://github.com/Arthur742Ramos/copex/actions/workflows/publish.yml

## Release Notes Format

```markdown
### Features
- New feature description

### Bug Fixes  
- Bug fix description

### Tests
- Test improvements
```

## Troubleshooting

If `gh` auth fails with "Bad credentials":
```powershell
$env:GH_TOKEN = $null
gh auth status
```

If PyPI publish fails:
- Verify `PYPI_API_TOKEN` secret is set
- Ensure version doesn't already exist on PyPI
