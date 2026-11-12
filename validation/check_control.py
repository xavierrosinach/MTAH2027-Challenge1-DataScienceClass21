"""Does the learnt control model see the same kind of signal on BAR-ATH as on the bench games?

Label-free: for every feature of the model (H minus A, in units of its training scale) and for
its output, aggregate statistics on the graded frames of BAR-ATH next to the same statistics on
the degraded Metrica games. A feature whose distribution is very different on BAR-ATH (another
unit, another meaning) would make the weights learnt on Metrica wrong there. Only aggregates are
printed, never a frame.

    python validation/check_control.py
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
from possession import context, control  # noqa: E402
from possession.config import load_config  # noqa: E402


def features(tr, blocks, cfg):
    ball = context.interpolate_ball(tr["ball"], cfg["ball_gaps"]["interpolate_max_frames"])
    tr = dict(tr)
    tr["team_frame"], _ = context.keeper_teams(tr, blocks, cfg)
    names, g_h, g_a, seen = control.team_features(tr, ball, blocks, control.feature_groups(cfg))
    graded = np.zeros(len(ball), bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    return names, (g_h - g_a)[graded], seen[graded]


def summary(X, seen, models):
    z = np.where(seen, control.logit(models["seen"], X), control.logit(models["gap"], X))
    zs = z[seen]
    flips = np.sum(np.sign(zs[1:]) != np.sign(zs[:-1]))
    out = {"ball seen": seen.mean(), "|llr|>2 seen": np.mean(np.abs(zs) > 2),
           "|llr| median seen": np.median(np.abs(zs)), "|llr| median gap": np.median(np.abs(z[~seen])),
           "sign flips/min seen": flips / (len(zs) / 25 / 60), "share llr>0": np.mean(z > 0)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bench", default=os.path.join(HERE, "bench.yaml"))
    parser.add_argument("--config", default=os.path.join(B.REPO, "config.yaml"))
    parser.add_argument("--model", default=None,
                        help="model file (default: context.control.model of the config)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.model is None:
        args.model = os.path.join(B.REPO, cfg["context"]["control"]["model"])
    bench = B.load_bench(args.bench)
    names, models = control.load(args.model)
    sd = models["seen"][0]

    import games
    from possession.data import load_tracking
    _, Xb, sb = features(load_tracking(cfg), games.half_video_blocks("BAR-ATH"), cfg)
    Xm, sm = [], []
    for g in bench["games"]:
        tr = metrica.load_tracking(bench["metrica_dir"], g, cache_dir=cfg["paths"]["cache_dir"])
        gt, blocks = gt_events.ground_truth(metrica.load_events(bench["metrica_dir"], g),
                                            tr["n_frames"])
        tr = deg.degrade(tr, gt, bench["degrade"], 0)
        _, X, s = features(tr, blocks, cfg)
        Xm.append(X)
        sm.append(s)
    Xm, sm = np.concatenate(Xm), np.concatenate(sm)

    rows = []
    for j, name in enumerate(names):
        a, b = Xb[sb, j] / sd[j], Xm[sm, j] / sd[j]           # frames with ball (shape: all)
        if name.startswith("shape_"):
            a, b = Xb[:, j] / sd[j], Xm[:, j] / sd[j]
        qa, qb = np.percentile(np.abs(a), [50, 90]), np.percentile(np.abs(b), [50, 90])
        rows.append({"feature": name, "BAR |x| q50": qa[0], "MET |x| q50": qb[0],
                     "BAR |x| q90": qa[1], "MET |x| q90": qb[1],
                     "ratio q90": qa[1] / max(qb[1], 1e-9)})
    df = pd.DataFrame(rows).set_index("feature")
    with pd.option_context("display.width", 200, "display.max_rows", 100,
                           "display.float_format", "{:.3f}".format):
        print("features (H - A) / training scale, frames with ball (shape: all graded frames):")
        print(df.to_string())
        print("\nmodel output:")
        print(pd.DataFrame({"BAR-ATH": summary(Xb, sb, models),
                            "Metrica degraded": summary(Xm, sm, models)}).to_string())


if __name__ == "__main__":
    main()
