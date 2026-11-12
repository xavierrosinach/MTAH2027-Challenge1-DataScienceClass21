"""Does the learnt dead-ball model see the same kind of signal on BAR-ATH as on the bench games?

Label-free: for every feature of the model (in units of its training scale) and for its output,
aggregate statistics on the graded frames of BAR-ATH next to the same statistics on the degraded
Metrica games. Only aggregates are printed, never a frame.

    python validation/check_dead.py --config outputs/validation/configs/dead1.yaml
"""

import argparse
import os
import sys

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


def features(tr, blocks, cfg):
    ball = context.interpolate_ball(tr["ball"], cfg["ball_gaps"]["interpolate_max_frames"])
    tr = dict(tr)
    tr["team_frame"], _ = context.keeper_teams(tr, blocks, cfg)
    f = context.frame_features(tr, ball, B.no_tags(), cfg)
    names, X = dead.dead_features(f, context.ball_dead_llr(f, cfg))
    graded = np.zeros(len(ball), bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    return names, X[graded]


def summary(z):
    flips = np.sum(np.sign(z[1:]) != np.sign(z[:-1]))
    return {"share llr>0 (dead)": np.mean(z > 0), "|llr|>2": np.mean(np.abs(z) > 2),
            "|llr| median": np.median(np.abs(z)), "sign flips/min": flips / (len(z) / 25 / 60)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bench", default=os.path.join(HERE, "bench.yaml"))
    parser.add_argument("--config", default=os.path.join(B.REPO, "config.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    bench = B.load_bench(args.bench)
    names, model = dead.load(os.path.join(B.REPO, cfg["context"]["dead_model"]["model"]))
    mu, sd, _ = model

    import games
    from possession.data import load_tracking
    _, Xb = features(load_tracking(cfg), games.half_video_blocks("BAR-ATH"), cfg)
    Xm = []
    for g in bench["games"]:
        tr = metrica.load_tracking(bench["metrica_dir"], g, cache_dir=cfg["paths"]["cache_dir"])
        gt, blocks = gt_events.ground_truth(metrica.load_events(bench["metrica_dir"], g),
                                            tr["n_frames"])
        Xm.append(features(deg.degrade(tr, gt, bench["degrade"], 0), blocks, cfg)[1])
    Xm = np.concatenate(Xm)

    rows = []
    for j, name in enumerate(names):
        a, b = (Xb[:, j] - mu[j]) / sd[j], (Xm[:, j] - mu[j]) / sd[j]
        rows.append({"feature": name, "BAR q10": np.percentile(a, 10), "MET q10": np.percentile(b, 10),
                     "BAR q50": np.median(a), "MET q50": np.median(b),
                     "BAR q90": np.percentile(a, 90), "MET q90": np.percentile(b, 90)})
    with pd.option_context("display.width", 200, "display.max_rows", 100,
                           "display.float_format", "{:.2f}".format):
        print("features (x - training mean) / training sd, graded frames:")
        print(pd.DataFrame(rows).set_index("feature").to_string())
        print("\nmodel output:")
        print(pd.DataFrame({"BAR-ATH": summary(dead.logit(model, Xb)),
                            "Metrica degraded": summary(dead.logit(model, Xm))}).to_string())


if __name__ == "__main__":
    main()
