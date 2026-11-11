"""Video of the public ground-truth window with the GT label and our label side by side.

    python create_video_gt.py
    python create_video_gt.py --config mi_variante.yaml

Runs the labeller with the given configuration (writes no CSV) and renders the 5 public minutes
to outputs/videos/GT_vs_<config>.mp4: the tactical video with the tracking, and top right two big
letters, GT and modelo, framed in yellow whenever they differ. One video per configuration, so
variants can be compared by eye.

The public window is 17 possessions. It is the reference for how each situation is labelled
(CHALLENGE.md); it is not a tuning set (RULES.md): use the video to understand the definition,
not to tweak parameters until the yellow frames disappear.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, HERE)

from possession.config import load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=os.path.join(HERE, "config.yaml"),
                        help="configuracion a usar (por defecto config.yaml)")
    parser.add_argument("--out", help="ruta del mp4 (por defecto outputs/videos/GT_vs_<config>.mp4)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    from possession.data import load_tracking
    from possession.video import make_video, video_tables
    from run import possession_labels

    tracking = load_tracking(cfg)
    labels, _ = possession_labels(tracking, cfg, verbose=cfg["output"]["verbose"])
    frames, players, shown = video_tables(cfg, labels, tracking, public_gt=True)

    sel = frames[frames["GT"] != "X"]                     # the public window
    gt, ours = sel["GT"].to_numpy(), sel["modelo"].to_numpy()
    differ = int((gt != ours).sum())
    print("ventana publica: frames {}-{} ({} frames), distintos del GT: {} ({:.1f} %)".format(
        sel.frame.min(), sel.frame.max(), len(sel), differ, 100 * differ / len(sel)))
    for k in "HAD":
        print("  {}: GT {:5d}   modelo {:5d}".format(k, int((gt == k).sum()), int((ours == k).sum())))

    stem = os.path.splitext(os.path.basename(args.config))[0]
    out = args.out or os.path.join(cfg["paths"]["output_dir"], "videos", "GT_vs_{}.mp4".format(stem))
    vc = cfg["video"]
    make_video(sel, players, cfg, out, mode=vc["mode"], scale=vc["scale"], minimap=vc["minimap"],
               label_col=shown, progress_every=0)


if __name__ == "__main__":
    main()
