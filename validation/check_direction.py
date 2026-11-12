"""Does the attacking direction the labeller uses agree with where the teams stand? Label-free.

    python validation/check_direction.py [--bench validation/bench_all.yaml] [--no-bar]

The control model and the team-shape features read, per block, whether the home team attacks
towards x = 1 (context.home_attack_sign). Two readings exist:
  - absolute (until 12-nov): home mean x < 0.5;
  - relative (now): home mean x < away mean x (the team nearer x = 0 defends x = 0).
A team that dominates a half pushes its centroid past the half-way line, which fools the absolute
reading. Prints, per block, both teams' mean x (all their visible players, metadata team), the
absolute and relative readings, and what context.home_attack_sign returns ("code"), for BAR-ATH
(no labels) and the bench games. ok = the code's reading agrees with the relative one.
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
from possession.context import home_attack_sign  # noqa: E402


def directions(tr, blocks):
    out = []
    P, team = tr["players"], tr["team"]
    sign = home_attack_sign(tr, blocks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for period, a, b in blocks:
            hx = float(np.nanmean(P[a:b + 1, team == 0, 0]))
            ax = float(np.nanmean(P[a:b + 1, team == 1, 0]))
            absolute, relative = hx < 0.5, hx < ax
            code = bool(np.all(sign[a:b + 1] > 0))
            if not (np.all(sign[a:b + 1] > 0) or np.all(sign[a:b + 1] < 0)):
                raise ValueError("home_attack_sign is not constant over a block")
            out.append((period, hx, ax, absolute == relative, code == relative))
    return out


def show(name, rows):
    print("{:16s} ".format(name) + "  ".join(
        "P{}: home {:.3f} away {:.3f} absolute {} code {}".format(
            p, h, a, "agrees" if agree else "DIFFERS", "ok" if ok else "WRONG")
        for p, h, a, agree, ok in rows), flush=True)
    return all(ok for *_, ok in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bench", default=os.path.join(HERE, "bench_all.yaml"))
    parser.add_argument("--config", default=os.path.join(B.REPO, "config.yaml"))
    parser.add_argument("--no-bar", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ok = True
    if not args.no_bar:
        import games
        from possession.data import load_tracking
        ok &= show("BAR-ATH", directions(load_tracking(cfg), games.half_video_blocks("BAR-ATH")))
    bench = B.load_bench(args.bench)
    for g in bench["games"]:
        tr, _, _, blocks = B.load_game(bench, g, cfg["paths"]["cache_dir"])
        ok &= show(g, directions(tr, blocks))
    print("RESULT: {}".format("all blocks ok" if ok else "SOME BLOCK WRONG"))


if __name__ == "__main__":
    main()
