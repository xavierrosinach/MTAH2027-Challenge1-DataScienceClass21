"""Learnt dead-ball evidence: is play stopped at this frame? Trained on OTHER games.

The default dead-ball evidence (context.py) is a logistic regression on the players' speed, fitted
on BAR-ATH itself with rule-made pseudo-labels, plus hand-set rules about where the ball is or was
last seen. This model looks at the same signals (and the rules' output), but its weights are
learnt from real stoppages: the event data of the public Metrica sample games, with their tracking
degraded to look like ours (validation/degrade.py). Never trained on BAR-ATH.

Trained by validation/train_dead.py; weights in a JSON file read by context.label.
"""

import json

import numpy as np

from .control import window_nanmean

FPS = 25
SPEED_WINDOWS = (("past5", -125, -26), ("past1", -25, -1), ("next1", 1, 25), ("next5", 26, 125))


def dead_features(f, rule_llr):
    """(names, X): per-frame features of the dead-ball model from context.frame_features."""
    seen = f["seen"].astype(float)
    near = np.minimum(f["d_h"], f["d_a"])
    near = np.where(np.isfinite(near), np.minimum(near, 30.0), 30.0)
    cols = {
        "spd_2s": np.clip(f["spd_2s"], 0, 5), "spd_6s": np.clip(f["spd_6s"], 0, 5),
        "seen": seen, "seen_4s": np.nan_to_num(f["seen_4s"], nan=0.0),
        "edge_last": np.clip(f["edge_last"], -5, 20) / 5,
        "edge_last_out": (f["edge_last"] < 0).astype(float),
        "edge_next": np.clip(f["edge_next"], -5, 20) / 5,
        "edge_next_out": (f["edge_next"] < 0).astype(float),
        "edge_pred": np.clip(f["edge_pred"], -10, 20) / 5,
        "t_since": np.log1p(np.minimum(f["t_since"], 60)),
        "t_until": np.log1p(np.minimum(f["t_until"], 60)),
        "ball_speed": np.clip(np.nan_to_num(f["ball_speed"], nan=0.0), 0, 30) / 10,
        "alone_s": np.clip(f["alone_s"], 0, 10) / 5,
        "near": near / 10,
        "rule_llr": np.clip(rule_llr, -5, 5),
    }
    for name, lo, hi in SPEED_WINDOWS:
        cols["spd_" + name] = np.nan_to_num(np.clip(window_nanmean(f["spd"], lo, hi), 0, 5),
                                            nan=1.0)
    for name, lo, hi in (("past2", -50, -1), ("next2", 1, 50)):
        cols["seen_" + name] = np.nan_to_num(window_nanmean(seen, lo, hi), nan=0.0)
    names = list(cols)
    return names, np.column_stack([cols[k] for k in names])


def fit(X, y, l2=1.0, iters=60, sw=None):
    """Plain L2 logistic regression with intercept (calibrated, not class-balanced)."""
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    sw = np.ones(len(y)) if sw is None else sw
    w = np.zeros(Z.shape[1])
    reg = np.r_[0, np.ones(len(w) - 1)]
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (sw * (p - y)) + l2 * reg * w
        Hs = (Z * (sw * p * (1 - p))[:, None]).T @ Z + l2 * np.diag(reg)
        step = np.linalg.solve(Hs, g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return mu, sd, w


def logit(model, X):
    mu, sd, w = model
    return np.column_stack([np.ones(len(X)), (X - mu) / sd]) @ w


def save(path, names, model, info):
    mu, sd, w = model
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"features": names, "info": info, "mu": mu.tolist(), "sd": sd.tolist(),
                   "w": w.tolist()}, handle, indent=1)


def load(path):
    with open(path, encoding="utf-8") as handle:
        m = json.load(handle)
    return m["features"], (np.array(m["mu"]), np.array(m["sd"]), np.array(m["w"]))


def dead_llr(f, rule_llr, path):
    """(N,) dead-vs-alive log-odds from the learnt model."""
    names, X = dead_features(f, rule_llr)
    want, model = load(path)
    if list(want) != list(names):
        raise ValueError("dead-ball model {} was trained on other features".format(path))
    return logit(model, X)
