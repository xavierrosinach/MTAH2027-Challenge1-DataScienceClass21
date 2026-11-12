"""Paired comparison of two models in a bench CSV (bench.py --suite --out x.csv): aggregates only.

    python validation/compare.py x.csv [y.csv ...] --ref ref --new new

Rows of the CSV are model x condition x game x seed runs on the OTHER games (Metrica), never on
BAR-ATH. Prints, for `new` minus `ref` paired on (condition, game, seed): the mean spellF1 /
macroF1 / predicted possessions of each, the paired difference (mean, sd, standard error, share
of runs better / worse), the difference per condition, per (condition, game) cell (cells better
out of all) and the last game (game 3) per condition. Nothing per frame.
"""

import argparse

import numpy as np
import pandas as pd

KEY = ["condition", "game", "seed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--ref", default="ref")
    parser.add_argument("--new", default="new")
    args = parser.parse_args()
    df = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
    df = df[df["model"].isin([args.ref, args.new])]
    a = df[df["model"] == args.ref].set_index(KEY)
    b = df[df["model"] == args.new].set_index(KEY)
    if a.index.duplicated().any() or b.index.duplicated().any():
        raise SystemExit("duplicated (condition, game, seed) runs: pass each run once")
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    d = b[["spellF1", "macroF1", "n_pred"]] - a[["spellF1", "macroF1", "n_pred"]]
    conds = list(dict.fromkeys(df["condition"]))
    games = sorted(df["game"].unique())

    pd.set_option("display.width", 200)
    print("runs paired: {} ({} conditions x {} games x {} seeds)".format(
        len(d), len(conds), len(games), d.reset_index()["seed"].nunique()))

    means = pd.DataFrame({m: {"spellF1": x["spellF1"].mean(), "macroF1": x["macroF1"].mean(),
                              "poss": x["n_pred"].mean(), "gt_poss": x["n_gt"].mean()}
                          for m, x in ((args.ref, a), (args.new, b))}).T
    print("\nmeans over all paired runs:")
    print(means.to_string(float_format="{:.4f}".format))

    s = d["spellF1"]
    print("\npaired difference {} - {} (spellF1): mean {:+.4f}, sd {:.4f}, se {:.4f}, "
          "better {:.0%}, worse {:.0%}, equal {:.0%}".format(
              args.new, args.ref, s.mean(), s.std(), s.std() / np.sqrt(len(s)),
              np.mean(s > 0), np.mean(s < 0), np.mean(s == 0)))
    m = d["macroF1"]
    print("paired difference (macroF1): mean {:+.4f}, se {:.4f}, better {:.0%}".format(
        m.mean(), m.std() / np.sqrt(len(m)), np.mean(m > 0)))
    print("paired difference (possessions): mean {:+.1f}".format(d["n_pred"].mean()))

    by_cond = d.groupby(level="condition")[["spellF1", "macroF1", "n_pred"]].mean().loc[conds]
    by_cond["ref spell"] = a.groupby(level="condition")["spellF1"].mean().loc[conds]
    by_cond["new spell"] = b.groupby(level="condition")["spellF1"].mean().loc[conds]
    by_cond["runs better"] = (d["spellF1"] > 0).groupby(level="condition").mean().loc[conds]
    print("\nby condition (new - ref):")
    print(by_cond.to_string(float_format="{:+.4f}".format))

    cell = d.groupby(level=["condition", "game"])[["spellF1", "macroF1"]].mean()
    print("\ncells (condition, game) better: spellF1 {}/{}, macroF1 {}/{}".format(
        int((cell["spellF1"] > 0).sum()), len(cell), int((cell["macroF1"] > 0).sum()), len(cell)))
    grid = cell["spellF1"].unstack("game").loc[conds]
    grid.columns = ["G" + str(c).rsplit("_", 1)[-1] for c in grid.columns]
    print("spellF1 difference per cell:")
    print(grid.to_string(float_format="{:+.4f}".format))

    last = games[-1]
    g3 = d.xs(last, level="game").groupby(level="condition")["spellF1"].agg(["mean", "count"])
    print("\n{} (spellF1 new - ref) by condition:".format(last))
    print(g3.loc[conds].to_string(float_format="{:+.4f}".format))
    print("{} overall: {:+.4f}".format(last, d.xs(last, level="game")["spellF1"].mean()))


if __name__ == "__main__":
    main()
