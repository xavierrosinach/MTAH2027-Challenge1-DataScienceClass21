"""Sweep one labeller parameter on the bench suite, against the unmodified config.

    python validation/sweep.py context.transition_cost.team_switch 8 12 16
    python validation/sweep.py --set context.transition_cost.team_switch=12 \
        context.transition_cost.to_dead 6 10 14

For every value: spellF1 / macroF1 under each condition of the suite, the change against the
reference, and in how many (condition, game) cells it beats the reference. A value is only worth
keeping if it wins under every condition and on most cells; among those, prefer a round value in
the middle of a flat region over the single best number (3 games are few: picking the maximum of
a fine grid overfits the bench).
"""

import argparse
import copy
import os
import sys
import time

import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench as B  # noqa: E402


def _set(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    if keys[-1] not in node:
        raise KeyError("{} not in config".format(dotted))
    node[keys[-1]] = value


def write_variants(base_path, out_dir, fixed, param, values):
    with open(base_path, encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    os.makedirs(out_dir, exist_ok=True)
    configs = [("ref", os.path.abspath(base_path))]
    for v in values:
        cfg = copy.deepcopy(base)
        for k, fv in fixed:
            _set(cfg, k, fv)
        if param:
            _set(cfg, param, v)
        name = "{}={}".format(param.split(".")[-1], v) if param else "fixed"
        path = os.path.join(out_dir, name.replace("=", "_") + ".yaml")
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, allow_unicode=True, sort_keys=False)
        configs.append((name, path))
    return configs


def summarize(df):
    key = ["condition", "game"]
    cell = df.groupby(key + ["model"])["spellF1"].mean().unstack("model")
    ref = cell["ref"]
    rows = []
    for m in cell.columns:
        if m == "ref":
            continue
        delta = cell[m] - ref
        by_cond = delta.groupby(level="condition").mean()
        rows.append({"model": m, "d_spell_mean": delta.mean(), "d_spell_worst_cond": by_cond.min(),
                     "cells_won": "{}/{}".format(int((delta > 0).sum()), len(delta)),
                     "d_macro_mean": (df[df.model == m].groupby(key)["macroF1"].mean()
                                      - df[df.model == "ref"].groupby(key)["macroF1"].mean()).mean()})
    out = pd.DataFrame(rows).set_index("model")
    with pd.option_context("display.float_format", "{:+.4f}".format, "display.width", 200):
        print(out.to_string())


def parse_value(text):
    return yaml.safe_load(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("param", nargs="?", default=None)
    parser.add_argument("values", nargs="*", type=parse_value)
    parser.add_argument("--set", nargs="*", default=[], help="key=value applied to every variant")
    parser.add_argument("--config", default=os.path.join(B.REPO, "config.yaml"))
    parser.add_argument("--bench", default=os.path.join(HERE, "bench.yaml"))
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--out", default=os.path.join(B.REPO, "outputs", "validation"))
    args = parser.parse_args()

    fixed = [(k, parse_value(v)) for k, v in (s.split("=", 1) for s in args.set)]
    values = args.values or [None]
    configs = write_variants(args.config, os.path.join(args.out, "configs"), fixed,
                             args.param, values)
    bench = B.load_bench(args.bench)
    t0 = time.time()
    frames = []
    for cond, over in bench["suite"].items():
        d, _ = B.run_condition(B._merge(bench, over), configs, bench["games"], None, False,
                               False, args.workers)
        frames.append(d.assign(condition=cond))
    df = pd.concat(frames, ignore_index=True)
    stem = "sweep_{}".format((args.param or "fixed").split(".")[-1])
    df.to_csv(os.path.join(args.out, stem + ".csv"), index=False)
    print("sweep {} {} (fixed: {})  {:.0f} s\n".format(args.param, values, args.set,
                                                      time.time() - t0))
    B.print_suite(df)
    print()
    summarize(df)


if __name__ == "__main__":
    main()
