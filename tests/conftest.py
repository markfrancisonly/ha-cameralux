"""Make the repo root importable so tests reach custom_components.*
regardless of how pytest is invoked (bare `pytest` does not add the CWD)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
