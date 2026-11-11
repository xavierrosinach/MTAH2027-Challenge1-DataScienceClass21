"""Build the submission CSV from the data and config.yaml (optionally, the review videos).

    python run.py
    python run.py --config other.yaml
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from possession.config import load_config  # noqa: E402


def possession_labels(tracking, cfg, verbose):
    """Labels H / A / D per frame (src/possession/context.py) and the labeller's extras."""
    import games
    from possession import context
    from possession.data import load_smart_tagging

    game = tracking["game"]
    blocks = games.half_video_blocks(game)
    ball = context.interpolate_ball(tracking["ball"], cfg["ball_gaps"]["interpolate_max_frames"])
    tags = load_smart_tagging(cfg, game)

    # period of each frame: only "2nd_half_pre_kickoff" is used (FAQ: always D)
    period = np.full(len(ball), "", dtype=object)
    second = [b for b in blocks if b[0] == 2][0]
    period[second[1]:cfg["context"]["second_half_kickoff_frame"]] = "2nd_half_pre_kickoff"

    labels, extra = context.label(tracking, ball, tags, cfg, blocks, frame_period=period,
                                  verbose=verbose)
    extra.update(tags=tags, blocks=blocks)
    return labels, extra


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    # after load_config: these need games.py on the path
    from possession.data import load_tracking
    from possession.submission import compare, mark_ungraded, validate, write_csv

    out = cfg["output"]
    tracking = load_tracking(cfg)
    labels, extra = possession_labels(tracking, cfg, verbose=out["verbose"])

    if out["diagnostics"]:
        from possession import context
        from possession.diagnostics import print_report, report
        passes = context.completed_passes(extra["features"], cfg)
        print_report(report(labels, extra["features"], extra["tags"], extra["blocks"],
                            extra["p_dead"], passes))

    if out["ungraded_as_x"]:
        labels = mark_ungraded(labels, tracking["game"])
    sub_dir = os.path.join(cfg["paths"]["output_dir"], "submissions")
    path = write_csv(os.path.join(sub_dir, "{}.csv".format(cfg["team_name"])), labels)
    counts = {k: int(np.sum(labels == k)) for k in "HADX"}
    print("written {}   H {H}  A {A}  D {D}  X {X}".format(path, **counts))

    official = cfg["paths"]["official_repo"]
    if out["validate"] and not validate(path, official):
        sys.exit("CSV failed validation")
    if out["compare_public"]:
        compare(sub_dir, os.path.join(cfg["paths"]["output_dir"], "scores"), official,
                bootstrap=out["bootstrap"])

    if cfg["video"]["enabled"]:
        from possession.video import render_blocks
        render_blocks(cfg, labels, tracking)


if __name__ == "__main__":
    main()
