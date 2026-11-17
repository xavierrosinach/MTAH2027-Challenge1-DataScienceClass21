"""Learnt H-vs-A evidence: which team controls the ball, trained on OTHER games.

With automatic tracking the ball sits a median ~2 m from the nearest player, so "who is nearest" is a weak, flickering signal. This model looks at more of the picture, over a few seconds around each frame:

- the ball against each team: how close its nearest and second-nearest players are, how many are within 5 and 10 m, how many are goal-side of the ball, where the ball is and where it moves;
- each team's shape (context.team_shape_features): the team with the ball stretches and pushes up;
- (`motion`) how the players move against the ball: closing speed, carries, pressing.

Each model is antisymmetric: swapping the teams flips the sign of its output, so there is no home bias. Two kinds are averaged (an ensemble): a one-hidden-layer network and a linear logit, each with one set of weights for frames with the ball seen and one for ball gaps.

The weights (models/*.json) were trained offline on 10 public games of other competitions (3 Metrica Sports sample games and 7 IDSSE games, CC BY 4.0) whose tracking was degraded to look like ours. BAR-ATH was never used for training. `run.py` only reads the weights.
"""

import json

import numpy as np

FPS = 25
WINDOWS = (("now", 0, 0), ("past1", -25, -1), ("next1", 1, 25), ("c2", -50, 50))
MOTION_WINDOWS = (("c05", -12, 12), ("past1", -25, -1), ("next1", 1, 25))
BALL_FEATURES = ("near1", "near2p5", "near6", "second2p5", "n5", "n10", "goalside")
SHAPE_FEATURES = ("centre", "spread_x", "spread_y", "speed", "front", "back", "advance")

def window_nanmean(x, lo, hi):
    """Mean of x over frames [t + lo, t + hi], ignoring NaN (NaN where there is nothing)."""

    ok = ~np.isnan(x)
    cv = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    cn = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    a, b = np.clip(i + lo, 0, len(x)), np.clip(i + hi + 1, 0, len(x))
    n = cn[b] - cn[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cv[b] - cv[a]) / np.maximum(n, 1), np.nan)

def _nearest(dist, team, t):
    """(distances to team t's players, index of its nearest, distance of its nearest)."""

    d = np.where(np.broadcast_to(team == t, dist.shape), dist, np.inf)
    j = np.argmin(d, axis=1)
    return d, j, d[np.arange(len(d)), j]

def _windowed(x, ok, windows, prefix):
    x = np.where(ok, x, np.nan)
    cols = [window_nanmean(x, lo, hi) for _, lo, hi in windows]
    return cols, ["{}_{}".format(prefix, w) for w, _, _ in windows]

def _velocity(x, scale, lag=4):
    """Finite-difference velocity in m/s of normalised positions (..., 2); NaN where unknown."""
    v = np.full(x.shape, np.nan)
    v[lag:] = (x[lag:] - x[:-lag]) * scale * FPS / lag
    return v

def _motion(tracking, ball, seen, dist, team):
    """How the players move against the ball: the nearest player's closing speed, whether the ball moves with him (a carry), his speed, and how many team-mates run at the ball (pressing)."""
    
    length, width = tracking["pitch"]
    scale = np.array([length, width])
    idx = np.arange(len(dist))
    pv = _velocity(tracking["players"], scale)
    bv = _velocity(ball, scale)
    dn = np.where(np.isinf(dist), np.nan, dist)
    closing = np.full(dist.shape, np.nan)
    closing[4:] = -(dn[4:] - dn[:-4]) * FPS / 4
    closing_ok = np.nan_to_num(closing, nan=-99.0)
    per_team = []
    for t in (0, 1):
        d, j, d1 = _nearest(dist, team, t)
        ok = seen & (d1 < 15)
        v1 = pv[idx, j]
        raw = {"close1": np.clip(closing[idx, j], -8, 8) / 4,
               "carry1": np.exp(-np.hypot(*(v1 - bv).T) / 3),
               "speed1": np.clip(np.hypot(*v1.T), 0, 10) / 5}
        cols, names = [], []
        for name, x in raw.items():
            c, nm = _windowed(x, ok, MOTION_WINDOWS, name)
            cols += c
            names += nm
        press = ((d < 12) & (closing_ok > 2)).sum(axis=1).astype(float)
        c, nm = _windowed(press, seen, MOTION_WINDOWS, "press")
        per_team.append((cols + c, names + nm))
    return (per_team[0][1],) + tuple(np.nan_to_num(np.column_stack(c), nan=0.0)
                                     for c, _ in per_team)

EXTRA = {"motion": _motion}

def team_features(tracking, ball, blocks, groups=("base",)):
    """(names, G_H, G_A, seen): per frame, the features of each team, (N, K) each.

    Ball features are NaN on frames without a ball before windowing; after it, NaN becomes 0 (equal for both teams, so it adds nothing to the difference).
    """
    from .context import home_attack_sign, team_shape_features

    if groups[0] != "base" or any(g not in EXTRA for g in groups[1:]):
        raise ValueError("control features must be 'base' followed by {}: {}".format(
            tuple(EXTRA), groups))
    length, width = tracking["pitch"]
    P = tracking["players"]
    team = tracking.get("team_frame", tracking["team"][None, :])
    n = len(P)
    seen = ~np.isnan(ball[:, 0])
    sign_h = home_attack_sign(tracking, blocks)

    dx = (P[..., 0] - ball[:, None, 0]) * length
    dy = (P[..., 1] - ball[:, None, 1]) * width
    dist = np.hypot(dx, dy)
    dist = np.where(np.isnan(dist), np.inf, dist)
    bx = (ball[:, 0] - 0.5) * length                      # ball x from the centre, m
    vx = np.full(n, np.nan)
    vx[4:] = (ball[4:, 0] - ball[:-4, 0]) * length * FPS / 4

    per_team = []
    for t in (0, 1):
        sgn = sign_h if t == 0 else -sign_h               # +1: team t attacks towards x = 1
        mine = np.broadcast_to(team == t, dist.shape)
        d = np.where(mine, dist, np.inf)
        two = np.partition(d, 1, axis=1)[:, :2]
        d1, d2 = np.minimum(two[:, 0], 40.0), np.minimum(two[:, 1], 40.0)
        fwd_p = (P[..., 0] - 0.5) * length * sgn[:, None]  # players' x towards team t's attack
        visible = mine & ~np.isnan(P[..., 0])
        with np.errstate(invalid="ignore"):
            behind = np.where(visible, fwd_p < (bx * sgn)[:, None], False).sum(axis=1)
        n_vis = np.maximum(1, visible.sum(axis=1))
        raw = {"near1": np.exp(-d1 / 1.0), "near2p5": np.exp(-d1 / 2.5),
               "near6": np.exp(-d1 / 6.0), "second2p5": np.exp(-d2 / 2.5),
               "n5": (d < 5).sum(axis=1).astype(float), "n10": (d < 10).sum(axis=1).astype(float),
               "goalside": behind / n_vis}
        cols, names = [], []
        for name in BALL_FEATURES:
            x = np.where(seen, raw[name], np.nan)
            for wname, lo, hi in WINDOWS:
                cols.append(window_nanmean(x, lo, hi) if (lo, hi) != (0, 0) else x)
                names.append("{}_{}".format(name, wname))
        # where the ball is and where it goes, seen from team t
        pos = np.where(seen, bx * sgn / (length / 2), np.nan)
        vel = np.where(seen, np.clip(vx * sgn, -20, 20) / 10, np.nan)
        for wname, lo, hi in WINDOWS:
            cols.append(window_nanmean(pos, lo, hi) if (lo, hi) != (0, 0) else pos)
            names.append("ballpos_" + wname)
        for wname, lo, hi in (("c05", -12, 12), ("past1", -25, -1), ("next1", 1, 25)):
            cols.append(window_nanmean(vel, lo, hi))
            names.append("ballvel_" + wname)
        per_team.append((names, np.nan_to_num(np.column_stack(cols), nan=0.0)))

    shape = team_shape_features(tracking, blocks, 2 * FPS)
    k = len(SHAPE_FEATURES)
    names = per_team[0][0] + ["shape_" + s for s in SHAPE_FEATURES]
    g_h = np.column_stack([per_team[0][1], shape[:, :k]])
    g_a = np.column_stack([per_team[1][1], shape[:, k:2 * k]])
    for group in groups[1:]:
        extra_names, e_h, e_a = EXTRA[group](tracking, ball, seen, dist, team)
        names = names + extra_names
        g_h = np.column_stack([g_h, e_h])
        g_a = np.column_stack([g_a, e_a])
    return names, g_h, g_a, seen

# THE TWO KINDS OF MODEL
def predict(model, D, S):
    """H-vs-A log-odds of either kind of model."""
    if model.get("type") == "mlp":
        Xd, Xs = D / model["sd_d"], (S - model["mu_s"]) / model["sd_s"]
        zp = np.tanh(np.hstack([Xd, Xs]) @ model["W1"] + model["b1"])
        zn = np.tanh(np.hstack([-Xd, Xs]) @ model["W1"] + model["b1"])
        return zp @ model["w2"] - zn @ model["w2"]
    return (D / model["scale"]) @ model["w"]

def load(path):
    """(feature names, {"seen": model, "gap": model}) from a model file."""
    with open(path, encoding="utf-8") as handle:
        m = json.load(handle)
    models = {k: {key: (np.array(x) if isinstance(x, list) else x) for key, x in v.items()}
              for k, v in m["models"].items()}
    return m["features"], models

def control_llr(tracking, ball, blocks, paths, groups=("base",)):
    """(N,) H-vs-A log-odds averaged over the model files in `paths`, and the ball-seen mask."""
    
    names, g_h, g_a, seen = team_features(tracking, ball, blocks, groups)
    D, S = g_h - g_a, g_h + g_a
    llr = np.zeros(len(D))
    for p in paths:
        want, models = load(p)
        if list(want) != list(names):
            raise ValueError("control model {} was trained on other features".format(p))
        llr += np.where(seen, predict(models["seen"], D, S), predict(models["gap"], D, S))
    return llr / len(paths), seen
