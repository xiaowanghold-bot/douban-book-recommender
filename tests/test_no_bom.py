"""Test that all source files are UTF-8 without BOM."""
import pathlib

ROOT = pathlib.Path(__file__).parent.parent


def test_no_bom_in_source_files():
    """Ensure no .py/.txt/.md file (outside data/) starts with UTF-8 BOM."""
    failed = []
    for ext in (".py", ".txt", ".md"):
        for p in ROOT.rglob(f"*{ext}"):
            # Exclude data/ directory
            if "data" in p.parts:
                continue
            # Exclude __pycache__, .git, venv, etc.
            if any(skip in p.parts for skip in ("__pycache__", ".git", "venv", ".eggs")):
                continue
            raw = p.read_bytes()
            if raw[:3] == b"\xef\xbb\xbf":
                failed.append(str(p.relative_to(ROOT)))
    assert not failed, f"Files with BOM: {failed}"
