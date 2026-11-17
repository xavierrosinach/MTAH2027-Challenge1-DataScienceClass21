"""Train the ball-control model (models/control_mlp.json, models/control_lin.json).

    python training/idsse.py download        # the IDSSE files -> ../external/idsse
    python training/idsse.py parse
    python training/train_control.py

Not needed to build the submission: run.py only reads the trained weights.

Training data (OTHER games only, never BAR-ATH; settings in training/training.yaml):
  - 3 Metrica Sports sample games: possession from their event data (human-collected events),
    built in training/gt_events.py;
  - 7 IDSSE games (Bassek et al. 2025, CC BY 4.0): ball status and possession flags of their own
    tracking (training/idsse.py).
Every live frame of every game is used, with its tracking degraded to look like ours
(training/degrade.py): two copies with the standard degradation and two with the `light` one.

Two models are written, each with weights for frames with the ball seen and for ball gaps:
  - control_mlp.json: an antisymmetric one-hidden-layer network (ball seen) + a linear model (gaps);
  - control_lin.json: linear models for both. run.py averages the two.
"""

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (HERE, os.path.join(REPO, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import degrade as deg  # noqa: E402
import gt_events  # noqa: E402
import idsse  # noqa: E402
import metrica  # noqa: E402
from possession import context, control  # noqa: E402
from possession.config import load_config  # noqa: E402


# --- data ---------------------------------------------------------------------------------------

def load_settings(path):
    with open(path, encoding="utf-8") as handle:
        s = yaml.safe_load(handle)
    for key in ("metrica_dir", "idsse_dir"):
        if not os.path.isabs(s[key]):
            s[key] = os.path.normpath(os.path.join(REPO, s[key]))
    return s


def merge(base, over):
    """Settings `base` with the keys of `over` replaced (recursively)."""
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_game(settings, game, cache_dir):
    """(tracking, ground truth (N,) H/A/D/X, graded blocks) of one training game."""
    if idsse.is_idsse(game):
        tr = idsse.load_tracking(settings["idsse_dir"], game)
        gt, blocks = idsse.ground_truth(settings["idsse_dir"], game)
        return tr, gt, blocks
    tr = metrica.load_tracking(settings["metrica_dir"], game, cache_dir=cache_dir)
    events = metrica.load_events(settings["metrica_dir"], game)
    gt, blocks = gt_events.ground_truth(events, tr["n_frames"])
    return tr, gt, blocks


def game_data(job):
    """Features (G_H - G_A, G_H + G_A), target and ball-seen mask on the live frames of one
    degraded copy of a game."""
    game, seed, settings, variant = job
    if variant:
        settings = merge(settings, settings["suite"][variant])
    cfg = load_config(os.path.join(REPO, "config.yaml"))
    tr, gt, blocks = load_game(settings, game, cfg["paths"]["cache_dir"])
    tr = deg.degrade(tr, gt, settings["degrade"], seed)
    ball = context.interpolate_ball(tr["ball"], cfg["ball_gaps"]["interpolate_max_frames"])
    tr = dict(tr)
    if cfg["context"]["keepers"]["enabled"]:
        tr["team_frame"], _ = context.keeper_teams(tr, blocks, cfg)
    groups = tuple(cfg["context"]["control"]["features"])
    names, g_h, g_a, seen = control.team_features(tr, ball, blocks, groups)
    live = np.isin(gt, ["H", "A"])
    # length (s) of the true possession each frame belongs to (frame weights)
    change = np.flatnonzero(gt[1:] != gt[:-1]) + 1
    lengths = np.diff(np.concatenate(([0], change, [len(gt)])))
    spell_s = np.repeat(lengths, lengths) / 25.0
    return {"names": names, "X": (g_h - g_a)[live], "S": (g_h + g_a)[live].astype(np.float32),
            "y": (gt[live] == "H").astype(float), "seen": seen[live], "spell_s": spell_s[live]}


# --- models -------------------------------------------------------------------------------------

def fit_logit(X, y, l2=1.0, iters=60, sw=None):
    """Conditional logit on X = G_H - G_A: logistic regression without intercept, made symmetric
    by adding every frame with the teams swapped (-X, 1 - y). sw: per-frame weights (mean 1)."""
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
    return {"scale": sd, "w": w}


def _mlp_forward(p, Xd, Xs):
    zp = np.tanh(np.hstack([Xd, Xs]) @ p["W1"] + p["b1"])
    zn = np.tanh(np.hstack([-Xd, Xs]) @ p["W1"] + p["b1"])
    return zp @ p["w2"] - zn @ p["w2"], zp, zn


def fit_mlp(D, S, y, hidden=32, l2=1e-4, epochs=12, batch=2048, lr=2e-3, seed=0, sw=None):
    """Antisymmetric network z = h([D, S]) - h([-D, S]), Adam on minibatches, fixed seed."""
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


def fit_models(parts, l2, power, mlp=None):
    """Weights for frames with the ball seen (network if `mlp`, else linear) and for gaps."""
    X = np.concatenate([p["X"] for p in parts])
    y = np.concatenate([p["y"] for p in parts])
    seen = np.concatenate([p["seen"] for p in parts])
    w = np.concatenate([p["spell_s"] for p in parts]) ** -power
    sw_seen, sw_gap = w[seen] / w[seen].mean(), w[~seen] / w[~seen].mean()
    gap = fit_logit(X[~seen], y[~seen], l2, sw=sw_gap)
    if mlp is None:
        return {"seen": fit_logit(X[seen], y[seen], l2, sw=sw_seen), "gap": gap}
    S = np.concatenate([p["S"] for p in parts]).astype(float)
    return {"seen": fit_mlp(X[seen], S[seen], y[seen], sw=sw_seen, **mlp), "gap": gap}


def save(path, names, models, info):
    out = {"features": names, "info": info,
           "models": {k: {key: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for key, v in m.items()} for k, m in models.items()}}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1)


def main():
    settings = load_settings(os.path.join(HERE, "training.yaml"))
    t = settings["train"]
    seeds, games = tuple(t["seeds"]), settings["games"]
    t0 = time.time()
    jobs = [(g, s, settings, None) for g in games for s in seeds]
    jobs += [(g, s, settings, v) for v in t["mix"] for g in games for s in seeds]
    with Pool(min(4, len(jobs))) as pool:
        parts = pool.map(game_data, jobs)
    names = parts[0]["names"]
    print("{} degraded copies, {} features per team ({:.0f} s)".format(
        len(parts), len(names), time.time() - t0))

    info = {"seeds": list(seeds), "l2": t["l2"], "degrade": settings["degrade"],
            "weight_power": t["weight_power"], "mix": list(t["mix"])}
    for out, mlp in ((t["out_mlp"], dict(t["mlp"])), (t["out_lin"], None)):
        models = fit_models(parts, t["l2"], t["weight_power"], mlp)
        meta = dict(info, mlp=mlp, games=games) if mlp else dict(info, games=games)
        save(os.path.join(REPO, out), names, models, meta)
        print("written {}".format(out))


if __name__ == "__main__":
    main()
