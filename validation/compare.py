"""Paired comparison of two models in bench CSVs (bench.py --suite --out x.csv): aggregates only.

    python validation/compare.py x.csv [y.csv ...] --ref ref --new new [--game Sample_Game_3]
    python validation/compare.py x.csv --same-as other.csv:ref      # is ref here = ref there?

Rows of the CSVs are model x condition x game x seed runs on the OTHER games (Metrica, IDSSE),
never on BAR-ATH. Several CSVs are concatenated (e.g. the Metrica and the DFL bench = the 10-game
bench); a run must appear once. Prints, for `new` minus `ref` paired on (condition, game, seed):
the mean spellF1 / macroF1 / predicted possessions of each, the paired difference (mean, sd,
standard error, share of runs better / worse), the difference per condition, per (condition, game)
cell (cells better out of all), per game, per source when there are several (Metrica / IDSSE),
one game per condition (--game; default Sample_Game_3 if present, else the last game) and the
predicted possessions per condition. Nothing per frame.
"""

import argparse

import numpy as np
import pandas as pd

KEY = ["condition", "game", "seed"]


def short(game):
    """Sample_Game_3 -> G3, IDSSE_J03WMX -> WMX, SKC_1886347 -> S1886347."""
    if game.startswith("Sample_Game_"):
        return "G" + game.rsplit("_", 1)[-1]
    if game.startswith("IDSSE_"):
        return game[-3:]
    if game.startswith("SKC_"):
        return "S" + game[4:]
    return game


def source(game):
    if game.startswith("SKC_"):
        return "SkillCorner"
    return "IDSSE" if game.startswith("IDSSE_") else "Metrica"


def same_as(df, ref, spec):
    """Check that model `ref` of these CSVs equals MODEL of OTHER.csv run by run."""
    other, _, model = spec.rpartition(":")
    o = pd.read_csv(other)
    a = df[df["model"] == ref].set_index(KEY).sort_index()
    b = o[o["model"] == model].set_index(KEY).sort_index()
    common = a.index.intersection(b.index)
    diffs = {m: float(np.abs(a.loc[common, m] - b.loc[common, m]).max()) if len(common) else
             float("nan") for m in ("spellF1", "macroF1", "n_pred", "n_gt")}
    same = len(common) > 0 and all(v == 0 for v in diffs.values())
    print("{} here vs {} in {}: {} common runs (here {}, there {}); max |diff| {} -> {} on the "
          "common runs; {}".format(
              ref, model, other, len(common), len(a), len(b), diffs,
              "IDENTICAL" if same else "DIFFERENT",
              "same set of runs" if len(common) == len(a) == len(b) else "run sets differ"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--ref", default="ref")
    parser.add_argument("--new", default="new")
    parser.add_argument("--game", default=None,
                        help="game shown by condition (default Sample_Game_3, else the last)")
    parser.add_argument("--same-as", default=None, metavar="OTHER.csv:MODEL",
                        help="check that --ref here equals MODEL in OTHER.csv run by run")
    args = parser.parse_args()
    df = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
    if args.same_as:
        same_as(df, args.ref, args.same_as)
        if args.new not in set(df["model"]):
            return
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
    print("{}: {} vs {}".format(" + ".join(args.csv), args.new, args.ref))
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
    by_cond["se spell"] = (d["spellF1"].groupby(level="condition").std()
                           / np.sqrt(d["spellF1"].groupby(level="condition").size())).loc[conds]
    by_cond["ref spell"] = a.groupby(level="condition")["spellF1"].mean().loc[conds]
    by_cond["new spell"] = b.groupby(level="condition")["spellF1"].mean().loc[conds]
    by_cond["runs better"] = (d["spellF1"] > 0).groupby(level="condition").mean().loc[conds]
    print("\nby condition (new - ref):")
    print(by_cond.to_string(float_format="{:+.4f}".format))

    cell = d.groupby(level=["condition", "game"])[["spellF1", "macroF1"]].mean()
    print("\ncells (condition, game) better: spellF1 {}/{}, macroF1 {}/{}".format(
        int((cell["spellF1"] > 0).sum()), len(cell), int((cell["macroF1"] > 0).sum()), len(cell)))
    grid = cell["spellF1"].unstack("game").loc[conds]
    grid.columns = [short(c) for c in grid.columns]
    print("spellF1 difference per cell:")
    print(grid.to_string(float_format="{:+.4f}".format))

    by_game = pd.DataFrame({
        "ref spell": a.groupby(level="game")["spellF1"].mean(),
        "new spell": b.groupby(level="game")["spellF1"].mean(),
        "diff": d.groupby(level="game")["spellF1"].mean(),
        "macro diff": d.groupby(level="game")["macroF1"].mean(),
        "cells better": (cell["spellF1"] > 0).groupby(level="game").sum()})
    print("\nby game (new - ref):")
    print(by_game.to_string(formatters={"ref spell": "{:.4f}".format, "new spell": "{:.4f}".format,
                                        "diff": "{:+.4f}".format, "macro diff": "{:+.4f}".format}))

    sources = sorted({source(g) for g in games})
    if len(sources) > 1:
        rows = {}
        for src in sources:
            ds = d[[source(g) == src for g in d.index.get_level_values("game")]]
            cs = cell[[source(g) == src for g in cell.index.get_level_values("game")]]
            rows[src] = {"runs": len(ds), "spell diff": ds["spellF1"].mean(),
                         "se": ds["spellF1"].std() / np.sqrt(len(ds)),
                         "macro diff": ds["macroF1"].mean(),
                         "cells better": "{}/{}".format(int((cs["spellF1"] > 0).sum()), len(cs))}
        print("\nby source (new - ref):")
        print(pd.DataFrame(rows).T.to_string(float_format="{:+.4f}".format))

    focus = args.game or ("Sample_Game_3" if "Sample_Game_3" in games else games[-1])
    if focus in games:
        g3 = d.xs(focus, level="game").groupby(level="condition")["spellF1"].agg(["mean", "count"])
        print("\n{} (spellF1 new - ref) by condition:".format(focus))
        print(g3.loc[conds].to_string(float_format="{:+.4f}".format))
        print("{} overall: {:+.4f}".format(focus, d.xs(focus, level="game")["spellF1"].mean()))

    poss = pd.DataFrame({args.ref: a.groupby(level="condition")["n_pred"].mean(),
                         args.new: b.groupby(level="condition")["n_pred"].mean(),
                         "GT": a.groupby(level="condition")["n_gt"].mean()}).T[conds]
    print("\npredicted possessions per run, by condition:")
    print(poss.to_string(float_format="{:.1f}".format))


if __name__ == "__main__":
    main()
