"""Train the learnt dead-ball model (src/possession/dead.py) on the Metrica sample games.

    python validation/train_dead.py [--config config.yaml]

Training data: every graded frame of the Metrica games (dead = the event-derived ground truth says
D), tracking degraded as in the bench's `degraded` condition with seeds the bench does not use,
no Smart Tagging. Writes
  - context.dead_model.model of the config: trained on all the games; the one run.py uses;
  - outputs/validation/dead/<model name>/logo_<game>.json: trained without <game>, for the bench.
Prints, for each held-out game, aggregate scores of the learnt model next to the default dead-ball
evidence (pseudo-label speed model + rules): AUC, and accuracy at log-odds 0.
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
from possession import context, dead  # noqa: E402
from possession.config import load_config  # noqa: E402

TRAIN_SEEDS = (100, 101)
EVAL_SEED = 0


def default_dead_llr(f, cfg, graded):
    """The default D evidence of context.emissions: speed model on pseudo-labels + rules."""
    c = cfg["context"]
    X = context.design_matrix(f)
    y = context.pseudo_labels(f, None, cfg)
    train = graded & (y >= 0)
    p = np.clip(context.predict_logistic(context.fit_logistic(X[train], y[train].astype(float)), X),
                1e-3, 1 - 1e-3)
    llr = np.clip(np.log(p / (1 - p)), -c["dead_llr_clip"], c["dead_llr_clip"])
    return c["dead_weight"] * llr + context.ball_dead_llr(f, cfg)


def game_data(job):
    game, seed, bench, cfg_path, variant = job
    if variant:
        bench = B._merge(bench, bench["suite"][variant])
    cfg = load_config(cfg_path)
    tr = metrica.load_tracking(bench["metrica_dir"], game, cache_dir=cfg["paths"]["cache_dir"])
    gt, blocks = gt_events.ground_truth(metrica.load_events(bench["metrica_dir"], game),
                                        tr["n_frames"])
    tr = deg.degrade(tr, gt, bench["degrade"], seed)
    ball = context.interpolate_ball(tr["ball"], cfg["ball_gaps"]["interpolate_max_frames"])
    tr = dict(tr)
    if cfg["context"]["keepers"]["enabled"]:
        tr["team_frame"], _ = context.keeper_teams(tr, blocks, cfg)
    f = context.frame_features(tr, ball, B.no_tags(), cfg)
    graded = np.zeros(len(gt), dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    names, X = dead.dead_features(f, context.ball_dead_llr(f, cfg))
    return {"game": game, "seed": seed, "variant": variant, "names": names, "X": X[graded],
            "y": (gt[graded] == "D").astype(float),
            "default": default_dead_llr(f, cfg, graded)[graded]}


def auc(score, y):
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y == 1
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bench", default=os.path.join(HERE, "bench.yaml"))
    parser.add_argument("--config", default=os.path.join(B.REPO, "config.yaml"))
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--mix", nargs="*", default=None,
                        help="also train on copies degraded as these suite conditions (e.g. light)")
    args = parser.parse_args()
    bench = B.load_bench(args.bench)
    cfg = load_config(args.config)
    out = os.path.join(B.REPO, cfg["context"]["dead_model"]["model"])
    games = bench["games"]
    t0 = time.time()
    mix = list((cfg["context"]["dead_model"].get("train_mix") or []) if args.mix is None
               else args.mix)
    jobs = [(g, s, bench, args.config, None) for g in games for s in TRAIN_SEEDS + (EVAL_SEED,)]
    jobs += [(g, s, bench, args.config, v) for v in mix for g in games for s in TRAIN_SEEDS]
    if mix:
        print("training also on copies degraded as: {}".format(mix))
    with Pool(min(args.workers, len(jobs))) as pool:
        parts = pool.map(game_data, jobs)
    names = parts[0]["names"]
    train = [p for p in parts if p["seed"] in TRAIN_SEEDS]
    print("dead-ball features: {}  ({:.0f} s)".format(len(names), time.time() - t0))

    def fit(ps):
        return dead.fit(np.concatenate([p["X"] for p in ps]), np.concatenate([p["y"] for p in ps]),
                        args.l2)

    logo = B.logo_dir(cfg, out, "dead")
    os.makedirs(logo, exist_ok=True)
    info = {"seeds": list(TRAIN_SEEDS), "l2": args.l2, "degrade": bench["degrade"], "mix": mix}
    rows = []
    for g in games:
        model = fit([p for p in train if p["game"] != g])
        if not args.no_save:
            dead.save(os.path.join(logo, "logo_{}.json".format(g)), names, model,
                      dict(info, games=[x for x in games if x != g]))
        held = [p for p in parts if p["game"] == g and p["seed"] == EVAL_SEED
                and p["variant"] is None][0]
        z, y, d0 = dead.logit(model, held["X"]), held["y"], held["default"]
        rows.append({"game": g, "dead share": y.mean(), "AUC learnt": auc(z, y),
                     "AUC default": auc(d0, y), "acc learnt": np.mean((z > 0) == (y == 1)),
                     "acc default": np.mean((d0 > 0) == (y == 1))})
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print("\nheld-out game (trained on the other two), graded frames:")
        print(pd.DataFrame(rows).set_index("game").to_string())
    if args.no_save:
        return
    model = fit(train)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    dead.save(out, names, model, dict(info, games=games))
    w = pd.Series(model[2][1:], index=names)
    print("\nfull model -> {}; largest |w|: {}".format(out, ", ".join(
        "{} {:+.2f}".format(k, v) for k, v in w.reindex(w.abs().sort_values().index[::-1])
        .head(8).items())))


if __name__ == "__main__":
    main()
