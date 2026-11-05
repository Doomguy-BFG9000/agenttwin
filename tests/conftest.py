"""Test bootstrap: make the src layout importable without installing the package.

Tape fixtures live under build/ rather than the OS temp directory: the session
sandbox may not be able to write there, and build/ is gitignored anyway.
"""

from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

BUILD = Path(__file__).resolve().parents[1] / "build" / "pytest-tmp"


@pytest.fixture
def tape_dir():
    """An empty directory for one test, created and removed under build/."""
    path = BUILD / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        try:
            BUILD.rmdir()
        except OSError:
            pass
