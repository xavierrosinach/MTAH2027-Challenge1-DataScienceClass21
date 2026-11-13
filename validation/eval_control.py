"""Held-out quality of the control models of several configs, game by game: aggregates only.

    python validation/eval_control.py --configs ref=a.yaml new=b.yaml [--bench validation/bench_all.yaml]

Each game is scored with the model files the bench would use for it (bench.load_config_for_game:
the leave-one-game-out copy, or the full model when it was never trained on that game), on its
live frames with the tracking degraded as in `degraded` with the evaluation seed of
train_control.py. Prints, per config and game: log-loss and accuracy on frames with the ball for
each model file and for the average of their log-odds (what the labeller uses), and the share of
confident frames (|llr| > 2). Same features for every config (checked).
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
import train_control as T  # noqa: E402
from possession import control  # noqa: E402


def scores(z, y):
    p = np.clip(1 / (1 + np.exp(-z)), 1e-6, 1 - 1e-6)
    return {"logloss": float(-np.mean(np.where(y == 1, np.log(p), np.log(1 - p)))),
            "acc": float(np.mean((z > 0) == (y == 1))), "conf": float(np.mean(np.abs(z) > 2))}


def one_game(job):
    game, bench, configs = job
    part = T.game_data((game, T.EVAL_SEED, bench, configs[0][1]))
    seen = part["seen"]
    D, S, y = part["X"][seen], part["S"][seen].astype(float), part["y"][seen]
    rows = []
    for name, path in configs:
        cfg = B.load_config_for_game(path, game)
        ctl = cfg["context"]["control"]
        files = [ctl["model"]] + list(ctl.get("ensemble") or [])
        zs = []
        for f in files:
            names, models = control.load(f)
            if list(names) != list(part["names"]):
                raise ValueError("{}: other features".format(f))
            zs.append(control.predict(models["seen"], D, S))
        row = {"config": name, "game": game, "frames": int(len(y))}
        for f, z in zip(files, zs):
            kind = "mlp" if f == files[0] and ctl.get("model_type") == "mlp" else "lin"
            row.update({"{}_{}".format(kind, k): v for k, v in scores(z, y).items()})
        row.update({"avg_{}".format(k): v for k, v in scores(np.mean(zs, axis=0), y).items()})
        row["files"] = " + ".join(os.path.relpath(f, B.REPO) for f in files)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--bench", default=os.path.join(HERE, "bench_all.yaml"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    bench = B.load_bench(args.bench)
    configs = []
    for item in args.configs:
        name, _, path = item.rpartition("=")
        configs.append((name, os.path.abspath(path)))
    t0 = time.time()
    jobs = [(g, bench, configs) for g in bench["games"]]
    with Pool(min(args.workers, len(jobs))) as pool:
        rows = [r for rs in pool.map(one_game, jobs) for r in rs]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("held-out control models, live frames with the ball, degradation seed {}  ({:.0f} s)\n"
          .format(T.EVAL_SEED, time.time() - t0))
    print(df.drop(columns=["files"]).set_index(["game", "config"]).to_string())
    print("\nmodel files used:")
    for (g, c), f in df.set_index(["game", "config"])["files"].items():
        print("  {} {}: {}".format(g, c, f))
    metr = df["game"].str.startswith("Sample_Game")
    skc = df["game"].str.startswith("SKC_")
    for label, m in (("Metrica", metr), ("IDSSE", ~metr & ~skc), ("SkillCorner", skc)):
        if m.any():
            print("\nmean over {} games:".format(label))
            print(df[m].groupby("config").mean(numeric_only=True).drop(columns=["frames"])
                  .to_string())
    if len(configs) == 2:
        a, b = configs[0][0], configs[1][0]
        p = df.pivot(index="game", columns="config", values="avg_logloss")
        d = p[b] - p[a]
        print("\navg log-loss {} - {} per game: {}; better on {}/{} games".format(
            b, a, ", ".join("{} {:+.4f}".format(g, v) for g, v in d.items()),
            int((d < 0).sum()), len(d)))


if __name__ == "__main__":
    main()
