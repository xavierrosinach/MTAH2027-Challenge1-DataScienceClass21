"""Train the learnt control model (src/possession/control.py) on the Metrica sample games.

    python validation/train_control.py

Training data: every live frame (event-derived ground truth, validation/gt_events.py) of the
Metrica games, with their tracking degraded as in the bench's `degraded` condition, with seeds the
bench does not use. Writes
  - models/control.json: trained on all the games; the one run.py uses on BAR-ATH;
  - outputs/validation/control/<model name>/logo_<game>.json: trained without <game>. The bench uses it to
    label <game>, so no score of the bench comes from a game the model was trained on.
Prints, for each held-out game, per-frame accuracy on its live frames next to the nearest-player
rule (aggregate numbers only).
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench as B  # noqa: E402  (sets the paths)
import degrade as deg  # noqa: E402
import gt_events  # noqa: E402
import metrica  # noqa: E402
from possession import context, control  # noqa: E402
from possession.config import load_config  # noqa: E402

TRAIN_SEEDS = (100, 101)
EVAL_SEED = 0


def game_data(job):
    """(names, X, y, seen, nearest_h) on the live frames of one degraded game."""
    game, seed, bench, cfg_path = job
    cfg = load_config(cfg_path)
    tr = metrica.load_tracking(bench["metrica_dir"], game, cache_dir=cfg["paths"]["cache_dir"])
    gt, blocks = gt_events.ground_truth(metrica.load_events(bench["metrica_dir"], game),
                                        tr["n_frames"])
    tr = deg.degrade(tr, gt, bench["degrade"], seed)
    ball = context.interpolate_ball(tr["ball"], cfg["ball_gaps"]["interpolate_max_frames"])
    tr = dict(tr)
    if cfg["context"]["keepers"]["enabled"]:
        tr["team_frame"], _ = context.keeper_teams(tr, blocks, cfg)
    names, g_h, g_a, seen = control.team_features(tr, ball, blocks, control.feature_groups(cfg))
    f = context.frame_features(tr, ball, B.no_tags(), cfg)
    live = np.isin(gt, ["H", "A"])
    return {"game": game, "seed": seed, "names": names, "X": (g_h - g_a)[live],
            "S": (g_h + g_a)[live].astype(np.float32),
            "y": (gt[live] == "H").astype(float), "seen": seen[live],
            "d_h": f["d_h"][live], "d_a": f["d_a"][live]}


def fit_models(parts, l2, mlp=None):
    """Conditional logits for frames with and without the ball; with `mlp` (its settings), the
    model for frames with the ball is the antisymmetric network instead."""
    X = np.concatenate([p["X"] for p in parts])
    y = np.concatenate([p["y"] for p in parts])
    seen = np.concatenate([p["seen"] for p in parts])
    gap = control.fit(X[~seen], y[~seen], l2)
    if mlp is None:
        return {"seen": control.fit(X[seen], y[seen], l2), "gap": gap}
    S = np.concatenate([p["S"] for p in parts]).astype(float)
    return {"seen": control.fit_mlp(X[seen], S[seen], y[seen], **mlp), "gap": gap}


def evaluate(models, part, contact_m):
    X, y, seen, S = part["X"], part["y"], part["seen"], part["S"].astype(float)
    z = np.where(seen, control.predict(models["seen"], X, S), control.predict(models["gap"], X, S))
    p = np.clip(1 / (1 + np.exp(-z)), 1e-6, 1 - 1e-6)
    ok = (z > 0) == (y == 1)
    d_h, d_a = part["d_h"], part["d_a"]
    contact = seen & (np.minimum(d_h, d_a) < contact_m)
    rule_ok = (d_h < d_a) == (y == 1)
    return {"game": part["game"], "acc_seen": ok[seen].mean(), "acc_gap": ok[~seen].mean(),
            "logloss_seen": -np.mean(np.where(y == 1, np.log(p), np.log(1 - p))[seen]),
            "conf_share": np.mean(np.abs(z[seen]) > 2),
            "acc_conf": ok[seen & (np.abs(z) > 2)].mean(),
            "rule_cover": contact[seen].mean(), "rule_acc_contact": rule_ok[contact].mean(),
            "rule_acc_seen": rule_ok[seen].mean(), "model_acc_contact": ok[contact].mean()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bench", default=os.path.join(HERE, "bench.yaml"))
    parser.add_argument("--config", default=os.path.join(B.REPO, "config.yaml"))
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(TRAIN_SEEDS),
                        help="degradation seeds of the training copies")
    parser.add_argument("--no-save", action="store_true", help="only print the held-out scores")
    parser.add_argument("--mlp", nargs="*", default=None, metavar="KEY=VALUE",
                        help="override context.control.mlp settings (hidden=32 epochs=12 ...)")
    parser.add_argument("--out", default=None,
                        help="model file (default: context.control.model of the config)")
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    bench = B.load_bench(args.bench)
    cfg = load_config(args.config)
    if args.out is None:
        args.out = os.path.join(B.REPO, cfg["context"]["control"]["model"])
    print("features {} -> {}".format(list(control.feature_groups(cfg)), args.out))
    games = bench["games"]
    t0 = time.time()
    jobs = [(g, s, bench, args.config) for g in games for s in seeds + (EVAL_SEED,)]
    with Pool(min(args.workers, len(jobs))) as pool:
        parts = pool.map(game_data, jobs)
    names = parts[0]["names"]
    train = [p for p in parts if p["seed"] in seeds]
    print("features: {} per team, {} live frames per game x seed  ({:.0f} s)".format(
        len(names), int(np.mean([len(p["y"]) for p in parts])), time.time() - t0))

    logo_dir = B.logo_dir(cfg, args.out)
    os.makedirs(logo_dir, exist_ok=True)
    rows = []
    ctl = cfg["context"]["control"]
    mlp = None
    if ctl.get("model_type", "logit") == "mlp":
        mlp = dict(ctl.get("mlp") or {})
        for item in args.mlp or []:
            k, _, v = item.partition("=")
            mlp[k] = float(v) if "." in v or "e" in v else int(v)
        print("antisymmetric MLP for frames with the ball: {}".format(mlp))
    info = {"seeds": list(seeds), "l2": args.l2, "degrade": bench["degrade"]}
    if mlp is not None:
        info["mlp"] = mlp
    for g in games:
        models = fit_models([p for p in train if p["game"] != g], args.l2, mlp)
        if not args.no_save:
            control.save(os.path.join(logo_dir, "logo_{}.json".format(g)), names, models,
                         dict(info, games=[x for x in games if x != g]))
        held = [p for p in parts if p["game"] == g and p["seed"] == EVAL_SEED][0]
        rows.append(evaluate(models, held, cfg["context"]["ball"]["contact_m"]))
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print("\nheld-out game (trained on the other two), live frames:")
        print(pd.DataFrame(rows).set_index("game").to_string())

    if args.no_save:
        return
    models = fit_models(train, args.l2, mlp)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    control.save(args.out, names, models, dict(info, games=games))
    if mlp is not None:
        print("\nfull model -> {}".format(args.out))
        return
    w = pd.Series(models["seen"][1], index=names)
    print("\nfull model -> {}; largest |w| (seen): {}".format(
        args.out, ", ".join("{} {:+.2f}".format(k, v)
                            for k, v in w.reindex(w.abs().sort_values().index[::-1]).head(8).items())))


if __name__ == "__main__":
    main()
