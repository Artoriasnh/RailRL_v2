"""P2.5 §13.3 — finalize the rule base: write `outputs/rule_base/rules.parquet`.

The 19 rules were AI-drafted then **reviewed + approved per-rule by Hao** (2026-05-27;
`outputs/rule_base/rules_full_draft.md`). The machine-readable source of truth is
`railrl.data.rule_base.RULES`. This script flattens those to the spec §13.2 tabular schema
and writes parquet (only `user_approved=True` rows — all 19 here). It also embeds the
machine-readable `match`/`pref`/`kind` as JSON columns so `l4_rules` can round-trip exactly.

Run anywhere with pandas+pyarrow (no torch/GPU):
    python scripts/rules/03_finalize.py
Writes to outputs/rule_base/rules.parquet (+ rules.csv for eyeballing).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from railrl import config as C
from railrl.data import rule_base as RB


def flatten(rule: dict) -> dict:
    """RULES dict → spec §13.2 row (+ JSON-encoded match/pref/kind for round-trip)."""
    pref = rule.get("pref", {})
    plats = pref.get("preferred_platforms")
    return {
        "rule_id": rule["rule_id"],
        "source_section": rule["source_section"],
        "cond_origin": rule.get("cond_origin"),
        "cond_destination": rule.get("cond_destination"),
        "cond_train_class": rule.get("cond_train_class"),
        "cond_time_of_day": None,                       # none of our rules are time-gated
        "cond_other": rule.get("notes"),
        "preferred_route_id": pref.get("preferred_route_id"),
        "preferred_platform": (plats[0] if (plats and len(plats) == 1) else None),
        "preferred_platforms": json.dumps(plats) if plats else None,   # set form (extension)
        "non_preferred_alternatives": json.dumps(pref.get("non_preferred_route_ids") or []),
        "confidence": rule["confidence"],
        "user_approved": bool(rule.get("user_approved", False)),
        "kind": rule["kind"],
        "match_json": json.dumps(rule.get("match", {})),
        "pref_json": json.dumps(pref),
        "notes": rule.get("notes"),
    }


def main() -> int:
    import pandas as pd
    rows = [flatten(r) for r in RB.RULES if r.get("user_approved")]
    df = pd.DataFrame(rows)
    n_total = len(RB.RULES)
    n_approved = len(rows)
    print(f"rules: {n_approved}/{n_total} approved → writing")
    print(f"  hard(high): {sum(r['confidence']=='high' for r in rows)} | "
          f"soft(med): {sum(r['confidence']=='med' for r in rows)}")
    print(f"  kinds: {pd.Series([r['kind'] for r in rows]).value_counts().to_dict()}")

    out_dir = C.RULE_BASE_DIR if hasattr(C, "RULE_BASE_DIR") else (C.OUTPUTS_DIR / "rule_base")
    out_dir.mkdir(parents=True, exist_ok=True)
    pq_path = out_dir / "rules.parquet"
    csv_path = out_dir / "rules.csv"
    df.to_parquet(pq_path, index=False)
    df.to_csv(csv_path, index=False)
    print(f"→ wrote {pq_path}")
    print(f"→ wrote {csv_path}  (human-readable)")
    # sanity: reload + confirm count
    back = pd.read_parquet(pq_path)
    assert len(back) == n_approved, "round-trip count mismatch"
    print(f"round-trip OK: {len(back)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
