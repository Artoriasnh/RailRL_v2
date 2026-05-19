#!/usr/bin/env python3
"""Phase 1.1 inventory runner.

Usage:
    python scripts/01_inventory.py            # full TD scan
    python scripts/01_inventory.py --nrows 5000000  # sample 5M rows for sanity-check
"""
from __future__ import annotations
import sys
from railrl.cli import inventory_main

if __name__ == "__main__":
    sys.exit(inventory_main(sys.argv[1:]))
