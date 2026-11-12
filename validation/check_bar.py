"""Label-free checks of one or more configs on BAR-ATH: what a change does to our own match.

    python validation/check_bar.py --configs ref=a.yaml new=b.yaml

Runs the labeller on BAR-ATH (no CSV written, NO public ground truth used or printed) and prints,
per config, aggregates over the graded frames only:
  - possessions: count and mean duration; dead-ball share and number of stoppages;
  - Smart Tagging concordance: share of SET PIECES tags (a restart the tagging saw) with a
    predicted stoppage between 2 s before the tag and its end; by restart type;
  - frames whose label differs from the first config.
A change that wins on the bench but makes these implausible (fewer stoppages than a match has,
set pieces without a stoppage) is not transferring to our tracking.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)

from possession.config import load_config  # noqa: E402


def stats(labels, tags, runs):
    graded = np.zeros(len(labels), dtype=bool)
    for a, b in runs:
        graded[a:b] = True
    spells, stops = [], 0
    for a, b in runs:
        seg = labels[a:b]
        cut = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        bounds = np.concatenate(([0], cut, [len(seg)]))
        for i in range(len(bounds) - 1):
            k = seg[bounds[i]]
            if k in ("H", "A"):
                spells.append((bounds[i + 1] - bounds[i]) / 25)
            else:
                stops += 1
    sp = tags[tags["code"] == "SET PIECES"]
    sp = sp[[graded[min(int(s), len(labels) - 1)] for s in sp["start_frame"]]]
    hit = np.array([np.any(labels[max(0, int(s) - 50):int(e) + 1] == "D")
                    for s, e in zip(sp["start_frame"], sp["end_frame"])])
    out = {"possessions": len(spells), "mean poss s": np.mean(spells),
           "dead share": np.mean(labels[graded] == "D"), "stoppages": stops,
           "set pieces with D": hit.mean() if len(hit) else np.nan, "set pieces": len(hit)}
    for t in sorted(sp["type"].unique()):
        m = (sp["type"] == t).to_numpy()
        out["  " + t] = "{}/{}".format(int(hit[m].sum()), int(m.sum()))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", nargs="+", default=[os.path.join(REPO, "config.yaml")])
    args = parser.parse_args()
    import run
    from possession.data import load_tracking

    cols, first = {}, None
    tracking = None
    for item in args.configs:
        name, _, path = item.rpartition("=")
        cfg = load_config(os.path.abspath(path))
        if tracking is None:
            tracking = load_tracking(cfg)
        labels, extra = run.possession_labels(tracking, cfg, verbose=False)
        runs = [(first_f, last_f + 1) for _, first_f, last_f in extra["blocks"]]
        col = stats(labels, extra["tags"], runs)
        if first is None:
            first = labels
        graded = np.concatenate([np.arange(a, b) for a, b in runs])
        col["frames changed vs first"] = np.mean(labels[graded] != first[graded])
        cols[name or os.path.basename(path)] = col
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(pd.DataFrame(cols).to_string())


if __name__ == "__main__":
    main()
