"""Calibrate the degradation of the bench games against BAR-ATH, without any label of BAR-ATH.

Two kinds of targets, neither of which is ground truth:
1. Label-free statistics of the ball and players, computed with the same code on BAR-ATH and on
   the (degraded) Metrica games: share of frames without ball, long gaps per 90 min, where they
   start, how fast the players move with and without a ball. Only aggregates are printed.
2. The full-match scores of the official baselines on BAR-ATH, published in starter/baseline.py.

    python validation/calibrate.py                      # BAR-ATH stats + the current bench.yaml
    python validation/calibrate.py --grid               # also a small grid of live-gap settings
    python validation/calibrate.py --bench validation/bench_dfl.yaml [--clean]   # IDSSE games
"""

import argparse
import copy
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench as B  # noqa: E402  (sets the paths)
import degrade as deg  # noqa: E402
from possession.config import load_config  # noqa: E402
from possession.context import edge_distance_m  # noqa: E402


def _runs(mask):
    m = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(m == 1), np.flatnonzero(m == -1)


def ball_stats(tr, blocks):
    """Aggregate, label-free statistics of one game's tracking."""
    ball, speed, pitch = tr["ball"], tr["speed"], tr["pitch"]
    graded = np.zeros(len(ball), bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    seen = ~np.isnan(ball[:, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spd = np.nanmean(speed, axis=1)
    edge = edge_distance_m(ball[:, 0], ball[:, 1], pitch)
    starts, ends = [], []
    for _, a, b in blocks:
        s, e = _runs(~seen[a:b + 1])
        starts += list(s + a)
        ends += list(e + a)
    starts, ends = np.array(starts, dtype=int), np.array(ends, dtype=int)
    long_ = ((ends - starts) >= 125) & (starts > 0)
    minutes = graded.sum() / 25 / 60
    s = graded & seen
    d = np.hypot(*np.moveaxis((tr["players"][s] - ball[s][:, None]) * np.array(pitch), -1, 0))
    near = np.where(np.isnan(d), np.inf, d).min(axis=1)
    codes = B.nearest_codes(tr)
    flicker = 0
    for _, a, b in blocks:
        x = codes[a:b + 1]
        x = x[x >= 0]
        flicker += int((x[1:] != x[:-1]).sum())
    bs = np.full(len(ball), np.nan)
    disp = (ball[4:] - ball[:-4]) * np.array(pitch)
    bs[4:] = np.hypot(disp[:, 0], disp[:, 1]) * 25 / 4
    bs = bs[s & ~np.isnan(bs)]
    return {"near_q10": np.percentile(near, 10), "near_q25": np.percentile(near, 25),
            "near_q50": np.percentile(near, 50), "flicker_90": flicker / minutes * 90,
            "ballspd_q50": np.percentile(bs, 50), "ballspd>6": (bs > 6).mean(),
            "missing": 1 - seen[graded].mean(),
            "long_gaps_90": long_.sum() / minutes * 90,
            "gap_start_line": float((edge[starts[long_] - 1] < 2).mean()),
            "gap_median_s": float(np.median(ends[long_] - starts[long_])) / 25,
            "spd_seen/gap": np.nanmean(spd[graded & seen]) / np.nanmean(spd[graded & ~seen])}


def bar_ath_stats(cfg):
    import games
    from possession.data import load_tracking
    return ball_stats(load_tracking(cfg), games.half_video_blocks("BAR-ATH"))


def bench_stats_and_baselines(bench, cfg, seed=0):
    import possession_metrics as pm
    smooth = B._official_smooth(cfg["paths"]["official_repo"])
    stats, scores = [], []
    for game in bench["games"]:
        tr, _, gt, blocks = B.load_game(bench, game, cfg["paths"]["cache_dir"])
        if bench["degrade"]["enabled"]:
            tr = deg.degrade(tr, gt, bench["degrade"], seed)
        stats.append(ball_stats(tr, blocks))
        codes = B.nearest_codes(tr)
        for name, window in B.BASELINES:
            scores.append({"model": name, **B.score(gt, smooth(codes, window), pm)})
    st = pd.DataFrame(stats).mean()
    sc = pd.DataFrame(scores).groupby("model")[["spellF1", "macroF1"]].mean()
    return st, sc


def baseline_error(sc):
    """Mean absolute gap to the published BAR-ATH baseline scores (spell and macro)."""
    err = [abs(sc.loc[m, "spellF1"] - s) + abs(sc.loc[m, "macroF1"] - f)
           for m, (f, s) in B.PUBLISHED.items()]
    return float(np.mean(err)) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bench", default=os.path.join(HERE, "bench.yaml"))
    parser.add_argument("--config", default=os.path.join(B.REPO, "config.yaml"))
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--clean", action="store_true",
                        help="statistics of the bench games without degradation")
    args = parser.parse_args()
    cfg = load_config(args.config)
    sys.path.insert(0, os.path.join(cfg["paths"]["official_repo"], "evaluation"))
    bench = B.load_bench(args.bench)
    if args.clean:
        bench["degrade"]["enabled"] = False

    pd.set_option("display.width", 200)
    print("BAR-ATH (label-free aggregates):")
    print(pd.Series(bar_ath_stats(cfg)).round(3).to_string())
    print("\npublished BAR-ATH baselines (macroF1, spellF1):", B.PUBLISHED)

    settings = [("bench.yaml", bench)]
    if args.grid:
        for sigma in (0.6, 1.0, 1.4):
            for corr in (0.2, 0.5, 1.0):
                b = copy.deepcopy(bench)
                b["degrade"]["enabled"] = True
                b["degrade"]["ball_noise"] = {"sigma_m": sigma, "corr_s": corr}
                settings.append(("ball noise {}m {}s".format(sigma, corr), b))
    rows = []
    for name, b in settings:
        st, sc = bench_stats_and_baselines(b, cfg)
        row = {"setting": name, **st.round(3).to_dict(),
               **{m.replace("baseline_", "") + "_spell": round(sc.loc[m, "spellF1"], 3)
                  for m in sc.index},
               **{m.replace("baseline_", "") + "_macro": round(sc.loc[m, "macroF1"], 3)
                  for m in sc.index},
               "base_err": round(baseline_error(sc), 4)}
        rows.append(row)
        print(pd.DataFrame([row]).to_string(index=False, header=len(rows) == 1), flush=True)


if __name__ == "__main__":
    main()
