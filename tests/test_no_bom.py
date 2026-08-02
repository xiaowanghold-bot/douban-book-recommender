"""Test that all source files are UTF-8 without BOM or control characters."""
import pathlib

ROOT = pathlib.Path(__file__).parent.parent

# Control chars to reject (allow TAB, LF, CR)
BAD_CONTROL = set(range(0x00, 0x20)) - {0x09, 0x0A, 0x0D}


def test_no_bom_in_source_files():
    """Ensure no .py/.txt/.md file (outside data/) starts with UTF-8 BOM."""
    failed = []
    for ext in (".py", ".txt", ".md"):
        for p in ROOT.rglob(f"*{ext}"):
            if "data" in p.parts:
                continue
            if any(skip in p.parts for skip in ("__pycache__", ".git", "venv", ".eggs")):
                continue
            raw = p.read_bytes()
            if raw[:3] == b"\xef\xbb\xbf":
                failed.append(str(p.relative_to(ROOT)))
    assert not failed, f"Files with BOM: {failed}"


def test_no_control_characters_in_source_files():
    """Ensure no .py/.txt/.md file (outside data/) contains control characters."""
    failed = []
    for ext in (".py", ".txt", ".md"):
        for p in ROOT.rglob(f"*{ext}"):
            if "data" in p.parts:
                continue
            if any(skip in p.parts for skip in ("__pycache__", ".git", "venv", ".eggs")):
                continue
            raw = p.read_bytes()
            for i, byte in enumerate(raw):
                if byte in BAD_CONTROL:
                    # Calculate line number
                    line_no = raw[:i].count(b"\n") + 1
                    failed.append(f"{p.relative_to(ROOT)}:{line_no}: byte 0x{byte:02x}")
    assert not failed, "Files with control characters:\n" + "\n".join(failed)
