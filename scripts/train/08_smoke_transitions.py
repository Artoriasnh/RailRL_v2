"""Stage 4.7.2a smoke — (s,a,r,s',done) TransitionDataset + end-to-end CQL step.

Builds real transitions from the VAL split (smaller), verifies successor/done
correctness, then batches a few through the model + target net + cql_total +
backward — exercising the full training data path. Run on Windows:
    python scripts/train/08_smoke_transitions.py
"""
from __future__ import annotations
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from railrl import config as C
from railrl.encoders.input_pipeline import NormStats
from railrl.algorithms.transitions import TransitionDataset, transition_collate
from railrl.algorithms import losses as L
from railrl.model import RailRLModel


def main():
    import torch

    print("[1/4] Building TransitionDataset(split='val') ...")
    ds = TransitionDataset(C.SNAPSHOTS_V2_PARQUET, C.NORMALIZATION_STATS_JSON, split="val")
    n = len(ds)
    n_term = int(sum(ds._done))
    n_pass = len({pid for pid, _ in ds._meta})
    print(f"      {n:,} transitions | terminals(done=1)={n_term:,} | episodes={n_pass:,} "
          f"| missing_successor={ds.n_missing_successor}")
    assert n > 0, "empty val split"
    assert ds.n_missing_successor == 0, "successor missing — split is NOT episode-intact!"
    assert n_term == n_pass, f"terminals {n_term} != episodes {n_pass} (1 per episode, position-based)"
    print("      ✓ every episode has exactly one (position-based) terminal; no transition crosses a split")

    print("[2/4] Verifying successor relationship (sample) ...")
    checked = 0
    for i in range(min(n, 2000)):
        pid, pos = ds._meta[i]
        j = ds._succ[i]
        if ds._done[i] == 0.0:
            spid, spos = ds._meta[j]
            assert spid == pid and spos == pos + 1, \
                f"bad successor at {i}: ({pid},{pos})→({spid},{spos})"
            checked += 1
        else:
            assert j == i, "terminal successor must be self (dummy)"
    print(f"      ✓ {checked} non-terminal successors are same-pass position+1")

    print("[3/4] Collating a transition batch ...")
    items = [ds[i] for i in range(8)]
    batch_s, batch_sp, done = transition_collate(items)
    B = batch_s.num_graphs
    print(f"      batch_s graphs={B}  batch_sp graphs={batch_sp.num_graphs}  done={done.tolist()}")
    assert batch_sp.num_graphs == B and done.shape[0] == B

    print("[4/4] Model forward(s) + target(s') + cql_total + backward ...")
    stats = NormStats.load(C.NORMALIZATION_STATS_JSON)
    model = RailRLModel.build(stats)
    target = copy.deepcopy(model)
    for p in target.parameters():
        p.requires_grad = False

    model.train()
    out = model(batch_s)
    with torch.no_grad():
        out_next = target(batch_sp)

    a = batch_s.chosen_action_idx.view(B)
    r = batch_s.r_total.view(B)
    batch_dict = {
        "chosen_action_idx": a,
        "r_total": r,
        "done": done,
        "set_mask": a > 0,
        # time_bucket label not in data yet → all -1 ⇒ L_time = 0 (pluggable later)
        "time_bucket": torch.full((B,), -1, dtype=torch.long),
    }
    total, parts = L.cql_total(out, {"Q": out_next["Q"]}, batch_dict)
    total.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_tot = sum(1 for _ in model.parameters())
    print(f"      L_TD={parts['L_TD']:.4f}  L_cons={parts['L_cons']:.4f}  "
          f"L_CQL={parts['L_CQL']:.4f}  L_route={parts['L_route']:.4f}  "
          f"L_time={parts['L_time']:.4f}  L_total={parts['L_total']:.4f}")
    assert torch.isfinite(total), "L_total not finite"
    assert parts['L_time'].item() == 0.0, "L_time should be 0 (no labels yet)"
    assert n_grad > n_tot * 0.5, f"too few params got grad ({n_grad}/{n_tot})"
    print(f"      params with grad: {n_grad}/{n_tot}")

    print("\nPASS: transitions correct (s'/done), batch feeds model+target, "
          "cql_total backward flows through the network")


if __name__ == "__main__":
    main()
