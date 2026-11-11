"""Learnt H-vs-A evidence: which team controls the ball, trained on OTHER games.

The rule-based evidence in context.py (nearest player within contact_m, completed passes) misses
most short possessions: with an automatic tracking the ball sits a median ~2 m from the nearest
player, so "who is nearest" is a weak, flickering signal. This model looks at more of the picture,
over a few seconds around each frame:

- the ball against each team: how close its nearest and second-nearest players are, how many are
  within 5 and 10 m, how many of them are goal-side of the ball, how far up the pitch the ball is
  and which way it moves (towards whose goal);
- each team's shape (context.team_shape_features): the team with the ball stretches and pushes up.

It is a conditional logit: each team gets a score w . g(team) from its own features, and
P(H) = sigmoid(score_H - score_A). Swapping the teams swaps the scores, so there is no home bias
and nothing about which team is H. Two models: one for frames with the ball seen, one for gaps.

Trained by validation/train_control.py on the public Metrica sample games (their event data gives
the possession; RULES.md allows human labels of other games), after degrading their tracking to
look like ours (validation/degrade.py). Never on BAR-ATH. The weights are stored in a JSON file
(models/control.json) so `python run.py` needs neither the Metrica data nor any training.
"""

import json
import warnings

import numpy as np

FPS = 25
# (name, offset from, offset to) in frames: the frame itself, the second before, the second after,
# and +/- 2 s. Windows ignore frames without a ball.
WINDOWS = (("now", 0, 0), ("past1", -25, -1), ("next1", 1, 25), ("c2", -50, 50))
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


def attack_sign(tracking, blocks):
    """+1 on frames where the home team attacks towards x = 1, -1 where towards x = 0."""
    P = tracking["players"]
    sign = np.ones(len(P))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, first, last in blocks:
            home_x = np.nanmean(P[first:last + 1, tracking["team"] == 0, 0])
            sign[first:last + 1] = 1 if home_x < 0.5 else -1
    return sign


def team_features(tracking, ball, blocks):
    """(names, G_H, G_A): per frame, the features of each team, (N, K) each.

    Ball features are NaN on frames without a ball before windowing; after it, NaN becomes 0
    (equal for both teams, so it adds nothing to the difference).
    """
    from .context import team_shape_features

    length, width = tracking["pitch"]
    P = tracking["players"]
    team = tracking.get("team_frame", tracking["team"][None, :])
    n = len(P)
    seen = ~np.isnan(ball[:, 0])
    sign_h = attack_sign(tracking, blocks)

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
        # where the ball is and where it goes, seen from team t (opposite sign for the other)
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
    return names, g_h, g_a, seen


def fit(X, y, l2=1.0, iters=60):
    """Conditional logit on X = G_H - G_A: logistic regression without intercept, made symmetric
    by adding every frame with the teams swapped (-X, 1 - y). Returns (scale, w)."""
    sd = X.std(axis=0) + 1e-9
    Z = np.concatenate([X / sd, -X / sd])
    t = np.concatenate([y, 1 - y]).astype(float)
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - t) + l2 * w
        Hs = (Z * (p * (1 - p))[:, None]).T @ Z + l2 * np.eye(len(w))
        step = np.linalg.solve(Hs, g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return sd, w


def logit(model, X):
    sd, w = model
    return (X / sd) @ w


def save(path, names, models, info):
    out = {"features": names, "info": info,
           "models": {k: {"scale": sd.tolist(), "w": w.tolist()} for k, (sd, w) in models.items()}}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1)


def load(path):
    with open(path, encoding="utf-8") as handle:
        m = json.load(handle)
    models = {k: (np.array(v["scale"]), np.array(v["w"])) for k, v in m["models"].items()}
    return m["features"], models


def control_llr(tracking, ball, blocks, path):
    """(N,) H-vs-A log-odds from the learnt model, and the ball-seen mask."""
    names, g_h, g_a, seen = team_features(tracking, ball, blocks)
    want, models = load(path)
    if list(want) != list(names):
        raise ValueError("control model {} was trained on other features".format(path))
    X = g_h - g_a
    llr = np.where(seen, logit(models["seen"], X), logit(models["gap"], X))
    return llr, seen
