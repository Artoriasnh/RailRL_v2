"""L5 — reward recovery via softmax / MaxEnt IRL (spec 05 §11).

Recover the signaller's EFFECTIVE reward weights w = (w_delay, w_throughput, w_headway,
w_wait): the weights under which the signaller's choices look (max-entropy) optimal.

The spec's literal MaxEnt-IRL ("Q*(s,a;w) via Bellman backup, solved per candidate w") is
intractable in our continuous high-dim state, so we use the standard FEATURE-MATCHING /
conditional-logit form on per-component action-VALUES:

    feature of action a in state s = Q_k(s,a)  (k=delay/throughput/headway/wait)
        — its long-run consequence on reward component k, from behavior-policy FQE.
    model:   P(a | s; w) = softmax_{a∈A(s)} ( Σ_k w_k · Q_k(s,a) )
    fit:     w* = argmax_w  Σ_decisions log P(a_signaller | s; w)  − λ‖w‖²    (CONVEX in w)

This is a conditional-logit MLE (choices over a per-decision candidate set) — convex,
identified (real-valued features pin the scale), solved by gradient ascent. Recovered w is
compared to the TRAINED reward weights (w_delay=1.0, w_thru=0.5, w_head=1.0, w_wait=0.3).

PURE NUMPY (no torch) — sandbox-tested. The Q_k(s,a) features come from the behavior-FQE
Q-table dumped by scripts/eval/08_fqe_behavior_qtable.py; the driver is scripts/eval/09_l5_irl.py.

Data layout (vectorised, segment form):
    X        : (M, K)  per-(decision,legal-action) component-Q rows, stacked over decisions
    offsets  : (D+1,)  X[offsets[d]:offsets[d+1]] = the legal actions of decision d
    chosen   : (D,)    GLOBAL row index (into X) of the signaller's chosen action per decision
    ep_id    : (D,)    episode id per decision (for cluster bootstrap)
"""
from __future__ import annotations
import numpy as np

COMPONENTS = ["delay", "throughput", "headway", "wait"]
TRAINED_W = {"delay": 1.0, "throughput": 0.5, "headway": 1.0, "wait": 0.3}


def _seg_index(offsets):
    """Row→decision index map (M,), and per-decision counts (D,)."""
    counts = np.diff(offsets)
    dec_of_row = np.repeat(np.arange(len(counts)), counts)
    return dec_of_row, counts


def _loglik_grad(w, X, offsets, chosen, dec_of_row, l2, dec_w=None):
    """Conditional-logit log-likelihood + gradient (segment-softmax over decisions).
    dec_w (D,): optional per-decision weight (for the weighted bootstrap; default all 1)."""
    D = len(offsets) - 1
    if dec_w is None:
        dec_w = np.ones(D)
    starts = offsets[:-1]                                   # decision start rows (contiguous)
    util = X @ w                                            # (M,)
    # per-decision logsumexp via FAST segment reductions (reduceat, C-level — NOT .at,
    # which is the unbuffered fallback and ~100× slower on big M).
    seg_max = np.maximum.reduceat(util, starts)            # (D,)
    e = np.exp(util - seg_max[dec_of_row])                  # (M,)
    Z = np.add.reduceat(e, starts)                          # (D,)
    p = e / Z[dec_of_row]                                   # softmax prob per row (M,)
    # weighted log-likelihood: Σ_d w_d[ util[chosen_d] − (seg_max_d + log Z_d) ]
    per_dec = util[chosen] - (seg_max + np.log(Z))
    ll = float((dec_w * per_dec).sum() - l2 * (w @ w))
    # gradient: Σ_d w_d ( X[chosen_d] − E_p[X]_d ) − 2λw
    EX = np.add.reduceat(p[:, None] * X, starts, axis=0)    # (D,K)
    grad = (dec_w[:, None] * X[chosen]).sum(0) - (dec_w[:, None] * EX).sum(0) - 2 * l2 * w
    return ll, grad


def _loglik_grad_hess(w, X, offsets, chosen, dec_of_row, l2, dec_w=None):
    """As _loglik_grad, plus H = −∇²ll = Σ_d w_d·Cov_p[X]_d + 2λI  (positive-definite),
    so a Newton ASCENT step solves H·Δw = grad (converges in ~10 iters; convex problem)."""
    D = len(offsets) - 1
    K = X.shape[1]
    if dec_w is None:
        dec_w = np.ones(D)
    starts = offsets[:-1]
    util = X @ w
    seg_max = np.maximum.reduceat(util, starts)
    e = np.exp(util - seg_max[dec_of_row])
    Z = np.add.reduceat(e, starts)
    p = e / Z[dec_of_row]
    per_dec = util[chosen] - (seg_max + np.log(Z))
    ll = float((dec_w * per_dec).sum() - l2 * (w @ w))
    EX = np.add.reduceat(p[:, None] * X, starts, axis=0)            # (D,K)
    grad = (dec_w[:, None] * X[chosen]).sum(0) - (dec_w[:, None] * EX).sum(0) - 2 * l2 * w
    # H = Σ_d w_d·Cov_p[X]_d + 2λI, computed WITHOUT materializing (M,K,K):
    #   Σ_d w_d Σ_a p_a x_a x_aᵀ = Xᵀ diag(w_dec·p) X ;  Σ_d w_d EX_d EX_dᵀ = (w·EX)ᵀ EX
    wp = dec_w[dec_of_row] * p                                     # (M,)
    H = X.T @ (wp[:, None] * X) - (dec_w[:, None] * EX).T @ EX + 2 * l2 * np.eye(K)
    return ll, grad, H


def maxent_irl(X, offsets, chosen, l2=1e-2, max_iter=600, lr=0.1, tol=1e-7,
               dec_w=None, w0=None, verbose=False):
    """Conditional-logit MaxEnt-IRL MLE → w (K,). Adam gradient ascent (convex objective).
    dec_w (D,): optional per-decision weight (weighted bootstrap). w0: warm-start init."""
    X = np.asarray(X, float); chosen = np.asarray(chosen, int)
    offsets = np.asarray(offsets, np.int64)
    dec_of_row, _ = _seg_index(offsets)
    K = X.shape[1]
    w = np.zeros(K) if w0 is None else np.asarray(w0, float).copy()
    # --- damped NEWTON (primary): convex → ~10 iters, no scipy needed, fast at scale ---
    try:
        prev = -np.inf
        for _ in range(min(max_iter, 60)):
            ll, g, H = _loglik_grad_hess(w, X, offsets, chosen, dec_of_row, l2, dec_w)
            step = np.linalg.solve(H, g)                    # ascent step: H Δw = g (H is PD)
            t = 1.0; advanced = False
            for _bt in range(30):                            # backtracking line search on ll
                lln, _ = _loglik_grad(w + t * step, X, offsets, chosen, dec_of_row, l2, dec_w)
                if lln >= ll - 1e-12:
                    w = w + t * step; advanced = True; break
                t *= 0.5
            if verbose:
                print(f"    [irl-newton] ll={ll:.5f} w={np.round(w,3)}")
            if (not advanced) or abs(lln - prev) < tol:
                break
            prev = lln
        return w
    except np.linalg.LinAlgError:
        pass
    # --- Adam ascent fallback (singular Hessian) ---
    w = np.zeros(K) if w0 is None else np.asarray(w0, float).copy()
    m = np.zeros(K); v = np.zeros(K); b1, b2, eps = 0.9, 0.999, 1e-8
    prev = -np.inf
    for it in range(1, max_iter + 1):
        ll, g = _loglik_grad(w, X, offsets, chosen, dec_of_row, l2, dec_w)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** it); vh = v / (1 - b2 ** it)
        w = w + lr * mh / (np.sqrt(vh) + eps)               # ASCENT (maximize ll)
        if abs(ll - prev) < tol and it > 20:
            break
        prev = ll
    return w


def normalize_w(w, mode="l1"):
    """Recovered w is identified up to nothing extra, but for COMPARING priorities we
    report a normalized version. l1: w/Σ|w| · K (so a flat policy → all 1); max: w/max|w|."""
    w = np.asarray(w, float)
    if mode == "l1":
        s = np.abs(w).sum()
        return w / s * len(w) if s > 0 else w
    if mode == "max":
        m = np.abs(w).max()
        return w / m if m > 0 else w
    return w


def bootstrap_irl(X, offsets, chosen, ep_id, n_resamples=1000, seed=0, **irl_kw):
    """Cluster-bootstrap by episode via the WEIGHTED-bootstrap trick (fast, no rebuild):
    resample E episodes with replacement → each decision's weight = its episode's draw count
    → weighted conditional-logit refit. Returns {'w','mean','std','ci_low','ci_high'}."""
    X = np.asarray(X, float); chosen = np.asarray(chosen, int); ep_id = np.asarray(ep_id)
    K = X.shape[1]
    w_point = maxent_irl(X, offsets, chosen, **irl_kw)
    uniq_ep, ep_inv = np.unique(ep_id, return_inverse=True)      # ep_inv: decision→episode idx
    E = len(uniq_ep)
    import time
    rng = np.random.default_rng(seed)
    boots = np.empty((n_resamples, K))
    t0 = time.time()
    step = max(1, n_resamples // 10)
    for b in range(n_resamples):
        counts = rng.multinomial(E, np.full(E, 1.0 / E))         # episodes drawn (w/ replacement)
        dec_w = counts[ep_inv].astype(float)                     # per-decision weight
        boots[b] = maxent_irl(X, offsets, chosen, dec_w=dec_w, w0=w_point, **irl_kw)  # warm-start
        if (b + 1) % step == 0:
            el = time.time() - t0
            print(f"    [bootstrap] {b+1}/{n_resamples} | {el:.0f}s | "
                  f"eta {el/(b+1)*(n_resamples-b-1):.0f}s", flush=True)
    return {
        "w": w_point,
        "mean": boots.mean(0), "std": boots.std(0, ddof=1),
        "ci_low": np.percentile(boots, 2.5, axis=0),
        "ci_high": np.percentile(boots, 97.5, axis=0),
    }
