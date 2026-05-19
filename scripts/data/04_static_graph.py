#!/usr/bin/env python3
"""Phase 2.3 (Iter 1) — static heterogeneous-graph backbone runner.

Usage:
    python scripts/p2_data_eng/04_static_graph.py

Produces parquet node/edge tables under outputs/p2_data_eng/static_graph/.
Depends on outputs from 01_inventory.py, 02_decisions.py, 03_infrastructure.py.
"""
from __future__ import annotations
import sys
from railrl.p2_data_eng.static_graph import run

if __name__ == "__main__":
    run()
    sys.exit(0)
