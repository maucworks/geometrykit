# This file was generated with the assistance of an AI coding tool.
"""Boundary enforcement test (AGENTS.md → 'Repo boundary').

Invokes `tools/check_boundary.py`: tracked content + pending commit
messages must never reference check-out environments or plan files.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def test_repo_boundary():
    """AGENTS.md → 'Repo boundary': no check-out-references in tracked content."""
    proc = subprocess.run(
        [sys.executable, str(TOOLS / "check_boundary.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"repo-boundary violations:\n{proc.stdout}{proc.stderr}"
