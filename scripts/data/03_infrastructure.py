#!/usr/bin/env python3
"""Phase 1.1.5 infrastructure-graph runner.

Usage:
    python scripts/03_infrastructure.py
"""
from __future__ import annotations
import sys
from railrl.cli import infrastructure_main

if __name__ == "__main__":
    sys.exit(infrastructure_main(sys.argv[1:]))
