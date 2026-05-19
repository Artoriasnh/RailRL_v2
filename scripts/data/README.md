# Scripts — Phase 1 entry points

These are thin wrappers over `railrl.cli`. Run them in numeric order;
each step produces a discrete artefact under `../outputs/`.

| # | Script | What it does | Typical runtime (full data) |
|---|--------|--------------|------------------------------|
| 01 | `01_inventory.py` | Streaming pass over TD_data.csv + full Movements.csv → JSON statistics. | ≈ 1 min on a workstation; sample with `--nrows 5000000` for quick checks. |
| 02 | `02_decisions.py` | Extract every Panel_Request, parse route_id and headcode → parquet of decision events. | ≈ 1–2 min |
| 03 | `03_infrastructure.py` | Parse route_to_tc_all.csv → routes / tracks / signals inventories + graph stats JSON. | < 1 s |

After installing the package (`pip install -e .[dev]` from the parent dir),
the same commands are also available as console scripts on PATH:
`railrl-inventory`, `railrl-decisions`, `railrl-infrastructure`.

All accept `--show-paths` to verify which files will be read/written.
