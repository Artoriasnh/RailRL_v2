"""railrl — Explainable offline + inverse RL for UK railway signaller decisions.

Project layout:
    src/railrl/                  importable package
      config.py                  paths and domain constants
      parsers.py                 route_id and headcode parsers
      data_io.py                 CSV → parquet caching, loaders
      cli.py                     argparse entry points
      p1_foundation/             Phase 1 — data foundation (Ch3 reused; README only)
      p2_data_eng/               Phase 2 — data engineering (active)
        inventory.py             ✓ TD + Movements inventory
        decisions.py             ✓ Decision-event extraction
        infrastructure.py        ✓ route_to_tc → infrastructure graph
      p3_modelling/              Phase 3 — modelling (placeholder)
      p4_xai/                    Phase 4 — interpretability (placeholder)
      p5_eval_deploy/            Phase 5 — evaluation & deployment (placeholder)
      p6_paper/                  Phase 6 — dissemination (placeholder)
"""
__version__ = "0.1.0"
