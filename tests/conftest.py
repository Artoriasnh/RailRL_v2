"""Pytest configuration — make src/railrl importable without install."""
import sys
from pathlib import Path

# Add src/ to path so `from railrl import ...` works
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
