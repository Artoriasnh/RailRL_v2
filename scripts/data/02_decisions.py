#!/usr/bin/env python3
"""Phase 1.1 decision-event runner.

Usage:
    python scripts/02_decisions.py            # full TD scan
    python scripts/02_decisions.py --nrows 5000000
"""
from __future__ import annotations
import sys
from railrl.cli import decisions_main

if __name__ == "__main__":
    sys.exit(decisions_main(sys.argv[1:]))
