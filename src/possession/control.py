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
# extra feature groups (context.control.features), each a function -> (names, E_H, E_A);
# registered at the end of the module
EXTRA = {}


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


def feature_groups(cfg):
    """The feature groups a config asks the control model for (context.control.features)."""
    ctl = cfg["context"].get("control") or {}
    groups = tuple(ctl.get("features") or ("base",))
    unknown = [g for g in groups if g not in GROUPS]
    if unknown or groups[0] != "base":
        raise ValueError("control features must start with 'base', known {}: {}".format(
            GROUPS, groups))
    return groups


def team_features(tracking, ball, blocks, groups=("base",)):
    """(names, G_H, G_A, seen): per frame, the features of each team, (N, K) each.

    Ball features are NaN on frames without a ball before windowing; after it, NaN becomes 0
    (equal for both teams, so it adds nothing to the difference). `groups` beyond "base" append
    more columns after the base ones (the base ones never change, so older models stay valid).
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
    for group in groups[1:]:
        extra_names, e_h, e_a = EXTRA[group](tracking, ball, blocks, seen, dist, team, sign_h)
        names = names + extra_names
        g_h = np.column_stack([g_h, e_h])
        g_a = np.column_stack([g_a, e_a])
    return names, g_h, g_a, seen


# --- extra feature groups -------------------------------------------------------------------

LONG_WINDOWS = (("past3", -75, -26), ("next3", 26, 75), ("c5", -125, 125))
MOTION_WINDOWS = (("c05", -12, 12), ("past1", -25, -1), ("next1", 1, 25))
TOUCH_WINDOWS = (("past2", -50, -1), ("next2", 1, 50), ("past5", -125, -51), ("next5", 51, 125))


def _nearest(dist, team, t):
    """(distances to team t's players, index of its nearest, distance of its nearest)."""
    d = np.where(np.broadcast_to(team == t, dist.shape), dist, np.inf)
    j = np.argmin(d, axis=1)
    return d, j, d[np.arange(len(d)), j]


def _windowed(x, ok, windows, prefix):
    x = np.where(ok, x, np.nan)
    cols = [window_nanmean(x, lo, hi) for _, lo, hi in windows]
    return cols, ["{}_{}".format(prefix, w) for w, _, _ in windows]


def _pack(per_team):
    names = per_team[0][1]
    return (names,) + tuple(np.nan_to_num(np.column_stack(c), nan=0.0) for c, _ in per_team)


def _velocity(x, scale, lag=4):
    """Finite-difference velocity in m/s of normalised positions (..., 2); NaN where unknown."""
    v = np.full(x.shape, np.nan)
    v[lag:] = (x[lag:] - x[:-lag]) * scale * FPS / lag
    return v


def _long(tracking, ball, blocks, seen, dist, team, sign_h):
    """The base ball features over longer windows: 3 s before / after and +/- 5 s."""
    length = tracking["pitch"][0]
    P = tracking["players"]
    bx = (ball[:, 0] - 0.5) * length
    per_team = []
    for t in (0, 1):
        sgn = sign_h if t == 0 else -sign_h
        d, _, d1 = _nearest(dist, team, t)
        d1 = np.minimum(d1, 40.0)
        visible = np.broadcast_to(team == t, dist.shape) & ~np.isnan(P[..., 0])
        with np.errstate(invalid="ignore"):
            fwd_p = (P[..., 0] - 0.5) * length * sgn[:, None]
            behind = np.where(visible, fwd_p < (bx * sgn)[:, None], False).sum(axis=1)
        raw = {"near2p5": np.exp(-d1 / 2.5), "near6": np.exp(-d1 / 6.0),
               "n5": (d < 5).sum(axis=1).astype(float),
               "goalside": behind / np.maximum(1, visible.sum(axis=1)),
               "ballpos": bx * sgn / (length / 2)}
        cols, names = [], []
        for name, x in raw.items():
            c, nm = _windowed(x, seen, LONG_WINDOWS, name)
            cols += c
            names += nm
        per_team.append((cols, names))
    return _pack(per_team)


def _motion(tracking, ball, blocks, seen, dist, team, sign_h):
    """How the players move against the ball: the nearest player's closing speed, whether the
    ball moves with him (a carry), his speed, and how many team-mates run at the ball (pressing)."""
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
    return _pack(per_team)


def _touch(tracking, ball, blocks, seen, dist, team, sign_h):
    """Who is in control around each frame: share of frames where the team's nearest player is
    within 2 m and nearer than the rival's, before and after, and how long since / until its last /
    next such frame (possession stays with the last team in control)."""
    n = len(dist)
    idx = np.arange(n)
    d1s = [_nearest(dist, team, t)[2] for t in (0, 1)]
    per_team = []
    for t in (0, 1):
        ctrl = seen & (d1s[t] < 2.0) & (d1s[t] < d1s[1 - t])
        cols, names = _windowed(ctrl.astype(float), seen, TOUCH_WINDOWS, "ctrl")
        last = np.maximum.accumulate(np.where(ctrl, idx, -10 ** 9))
        nxt = np.minimum.accumulate(np.where(ctrl, idx, 10 ** 9)[::-1])[::-1]
        since = np.minimum((idx - last) / FPS, 10.0)
        until = np.minimum((nxt - idx) / FPS, 10.0)
        cols += [np.exp(-since / 2), np.exp(-until / 2)]
        names += ["ctrl_since", "ctrl_until"]
        per_team.append((cols, names))
    return _pack(per_team)


SHORT_WINDOWS = (("c05", -12, 12), ("past05", -12, -1), ("next05", 1, 12))


def _short(tracking, ball, blocks, seen, dist, team, sign_h):
    """The proximity features over narrow windows (+/- 0.5 s, the half second before / after):
    the +/- 2 s averages that carry most of the base model drown a possession of one second."""
    per_team = []
    for t in (0, 1):
        d, _, d1 = _nearest(dist, team, t)
        two = np.partition(d, 1, axis=1)[:, :2]
        d1, d2 = np.minimum(two[:, 0], 40.0), np.minimum(two[:, 1], 40.0)
        raw = {"near1": np.exp(-d1 / 1.0), "near2p5": np.exp(-d1 / 2.5),
               "second2p5": np.exp(-d2 / 2.5), "n5": (d < 5).sum(axis=1).astype(float)}
        cols, names = [], []
        for name, x in raw.items():
            c, nm = _windowed(x, seen, SHORT_WINDOWS, name)
            cols += c
            names += nm
        per_team.append((cols, names))
    return _pack(per_team)


EXTRA.update({"long": _long, "motion": _motion, "touch": _touch, "short": _short})
GROUPS = ("base",) + tuple(EXTRA)


def fit(X, y, l2=1.0, iters=60, sw=None):
    """Conditional logit on X = G_H - G_A: logistic regression without intercept, made symmetric
    by adding every frame with the teams swapped (-X, 1 - y). Returns (scale, w).
    sw: optional per-frame sample weights (mean 1)."""
    sd = X.std(axis=0) + 1e-9
    Z = np.concatenate([X / sd, -X / sd])
    t = np.concatenate([y, 1 - y]).astype(float)
    w = np.zeros(Z.shape[1])
    sw2 = None if sw is None else np.concatenate([sw, sw])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        if sw2 is None:
            g = Z.T @ (p - t) + l2 * w
            Hs = (Z * (p * (1 - p))[:, None]).T @ Z + l2 * np.eye(len(w))
        else:
            g = Z.T @ (sw2 * (p - t)) + l2 * w
            Hs = (Z * (sw2 * p * (1 - p))[:, None]).T @ Z + l2 * np.eye(len(w))
        step = np.linalg.solve(Hs, g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return sd, w


def logit(model, X):
    sd, w = model
    return (X / sd) @ w



# --- non-linear control model: antisymmetric one-hidden-layer network -----------------------
#
# z(H vs A) = h([D, S]) - h([-D, S]),  D = G_H - G_A, S = G_H + G_A,  h = w2 . tanh(W1 x + b1)
# Swapping the teams maps D -> -D and leaves S, so z -> -z: no home bias, as in the logit.
# (D, S) together carry both teams' features, so h can model interactions (e.g. nearest
# distance only matters while the ball is slow) that a linear score cannot.


def _mlp_forward(p, Xd, Xs):
    zp = np.tanh(np.hstack([Xd, Xs]) @ p["W1"] + p["b1"])
    zn = np.tanh(np.hstack([-Xd, Xs]) @ p["W1"] + p["b1"])
    return zp @ p["w2"] - zn @ p["w2"], zp, zn


def fit_mlp(D, S, y, hidden=32, l2=1e-4, epochs=12, batch=2048, lr=2e-3, seed=0, sw=None):
    """Antisymmetric MLP, Adam on minibatches, fixed seed. Returns a model dict."""
    rng = np.random.default_rng(seed)
    sd_d, sd_s = D.std(axis=0) + 1e-9, S.std(axis=0) + 1e-9
    mu_s = S.mean(axis=0)
    Xd, Xs = D / sd_d, (S - mu_s) / sd_s
    k = Xd.shape[1] + Xs.shape[1]
    p = {"W1": rng.normal(0, 1 / np.sqrt(k), (k, hidden)), "b1": np.zeros(hidden),
         "w2": rng.normal(0, 1 / np.sqrt(hidden), hidden)}
    m = {key: np.zeros_like(v) for key, v in p.items()}
    v2 = {key: np.zeros_like(v) for key, v in p.items()}
    step = 0
    n = len(y)
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch):
            b = order[start:start + batch]
            xd, xs, yb = Xd[b], Xs[b], y[b]
            z, ap, an = _mlp_forward(p, xd, xs)
            g = (1 / (1 + np.exp(-z)) - yb) / len(b)               # dLoss/dz
            if sw is not None:
                g = g * sw[b]
            up, un = np.hstack([xd, xs]), np.hstack([-xd, xs])
            dp = g[:, None] * p["w2"] * (1 - ap ** 2)
            dn = g[:, None] * p["w2"] * (1 - an ** 2)
            grads = {"w2": ap.T @ g - an.T @ g + l2 * p["w2"],
                     "W1": up.T @ dp - un.T @ dn + l2 * p["W1"],
                     "b1": dp.sum(axis=0) - dn.sum(axis=0)}
            step += 1
            for key in p:
                m[key] = 0.9 * m[key] + 0.1 * grads[key]
                v2[key] = 0.999 * v2[key] + 0.001 * grads[key] ** 2
                mh = m[key] / (1 - 0.9 ** step)
                vh = v2[key] / (1 - 0.999 ** step)
                p[key] = p[key] - lr * mh / (np.sqrt(vh) + 1e-8)
    return {"type": "mlp", "sd_d": sd_d, "sd_s": sd_s, "mu_s": mu_s, **p}


def mlp_logit(model, D, S):
    z, _, _ = _mlp_forward(model, D / model["sd_d"], (S - model["mu_s"]) / model["sd_s"])
    return z


def predict(model, D, S):
    """H-vs-A log-odds of either kind of model: D = G_H - G_A, S = G_H + G_A.
    A list of models (networks trained with different seeds, context.control.mlp_seeds) gives
    the mean of their log-odds."""
    if isinstance(model, list):
        return np.mean([predict(m, D, S) for m in model], axis=0)
    if isinstance(model, dict):
        return mlp_logit(model, D, S)
    return logit(model, D)


def _to_json(model):
    if isinstance(model, list):
        return [_to_json(m) for m in model]
    if isinstance(model, dict):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in model.items()}
    sd, w = model
    return {"scale": sd.tolist(), "w": w.tolist()}


def _from_json(v):
    if isinstance(v, list):
        return [_from_json(x) for x in v]
    if v.get("type") == "mlp":
        return {k: (np.array(x) if isinstance(x, list) else x) for k, x in v.items()}
    return np.array(v["scale"]), np.array(v["w"])


def save(path, names, models, info):
    out = {"features": names, "info": info,
           "models": {k: _to_json(m) for k, m in models.items()}}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1)


def load(path):
    with open(path, encoding="utf-8") as handle:
        m = json.load(handle)
    return m["features"], {k: _from_json(v) for k, v in m["models"].items()}


def control_llr(tracking, ball, blocks, path, groups=("base",)):
    """(N,) H-vs-A log-odds from the learnt model, and the ball-seen mask.

    `path` may be a list of model files (same features): their log-odds are averaged. A file
    may hold several networks (context.control.mlp_seeds): predict() averages them first, so
    they count as one member of that average."""
    names, g_h, g_a, seen = team_features(tracking, ball, blocks, groups)
    D, S = g_h - g_a, g_h + g_a
    paths = [path] if isinstance(path, str) else list(path)
    llr = np.zeros(len(D))
    for p in paths:
        want, models = load(p)
        if list(want) != list(names):
            raise ValueError("control model {} was trained on other features".format(p))
        llr += np.where(seen, predict(models["seen"], D, S), predict(models["gap"], D, S))
    return llr / len(paths), seen
