"""Does control.attack_sign get each team's attacking direction right? Label-free aggregates.

    python validation/check_direction.py [--bench validation/bench_all.yaml]

control.attack_sign (and context.team_shape_features) say the home team attacks towards x = 1 in a
block when its mean x over the block is < 0.5. A team that dominates pushes its centroid past the
half-way line, so that reading can flip. The robust reading is relative: the team whose mean x is
lower defends x = 0. Prints, per block, both teams' mean x (all their visible players, metadata
team) and whether the two readings agree; for BAR-ATH (no labels) and the bench games.
"""

import argparse
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench as B  # noqa: E402  (sets the paths)
from possession.config import load_config  # noqa: E402


def directions(tr, blocks):
    out = []
    P, team = tr["players"], tr["team"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for period, a, b in blocks:
            hx = float(np.nanmean(P[a:b + 1, team == 0, 0]))
            ax = float(np.nanmean(P[a:b + 1, team == 1, 0]))
            out.append((period, hx, ax, (hx < 0.5) == (hx < ax)))
    return out


def show(name, rows):
    print("{:16s} ".format(name) + "  ".join(
        "P{}: home {:.3f} away {:.3f} {}".format(p, h, a, "ok" if ok else "WRONG")
        for p, h, a, ok in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bench", default=os.path.join(HERE, "bench_all.yaml"))
    parser.add_argument("--config", default=os.path.join(B.REPO, "config.yaml"))
    parser.add_argument("--no-bar", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not args.no_bar:
        import games
        from possession.data import load_tracking
        show("BAR-ATH", directions(load_tracking(cfg), games.half_video_blocks("BAR-ATH")))
    bench = B.load_bench(args.bench)
    for g in bench["games"]:
        tr, _, _, blocks = B.load_game(bench, g, cfg["paths"]["cache_dir"])
        show(g, directions(tr, blocks))


if __name__ == "__main__":
    main()
