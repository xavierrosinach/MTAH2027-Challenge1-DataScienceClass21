"""Build the submission CSV (one H / A / D / X label per video frame) from the match data.

    python run.py                       # uses config.yaml
    python run.py --config other.yaml
"""

import argparse
import hashlib
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from possession.config import load_config  # noqa: E402

GAME = "BAR-ATH"

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    # imported after load_config, which puts the official games.py on the path
    import games
    from possession import context
    from possession.data import load_smart_tagging, load_tracking
    from possession.submission import mark_ungraded, validate, write_csv

    tracking = load_tracking(cfg, GAME)
    tags = load_smart_tagging(cfg, GAME)
    blocks = games.half_video_blocks(GAME)
    ball = context.interpolate_ball(tracking["ball"], cfg["ball_gaps"]["interpolate_max_frames"])

    # FAQ: every graded frame before the 2nd-half kick-off is D
    period = np.full(len(ball), "", dtype=object)
    second = [b for b in blocks if b[0] == 2][0]
    period[second[1]:cfg["context"]["second_half_kickoff_frame"]] = "2nd_half_pre_kickoff"

    labels = context.label(tracking, ball, tags, cfg, blocks, frame_period=period, verbose=cfg["output"]["verbose"])
    labels = mark_ungraded(labels, GAME)

    path = os.path.join(cfg["paths"]["output_dir"], "submissions", cfg["team_name"] + ".csv")
    write_csv(path, labels)
    counts = {k: int(np.sum(labels == k)) for k in "HADX"}
    print("written {}   H {H}  A {A}  D {D}  X {X}".format(path, **counts))
    with open(path, "rb") as handle:
        print("SHA256 {}".format(hashlib.sha256(handle.read()).hexdigest()))

    if cfg["output"]["validate"] and not validate(path, cfg["paths"]["official_repo"]):
        sys.exit("CSV failed the official validation")

if __name__ == "__main__":
    main()
