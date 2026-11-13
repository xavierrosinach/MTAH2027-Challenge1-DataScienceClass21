"""Diagnóstico por frame: python xavier/diag.py first last [step]"""
import os, sys
import numpy as np, pandas as pd
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raíz del repo
sys.path.insert(0, os.path.join(repo, "src")); sys.path.insert(0, repo)
from possession.config import load_config
cfg = load_config(os.path.join(repo, os.environ.get("DIAG_CFG", "config.yaml")))
import games
from possession.data import load_tracking
from possession import context, control
from run import possession_labels

tr = load_tracking(cfg)
labels, ex = possession_labels(tr, cfg, verbose=False)
f, e = ex["features"], ex["emissions"]
blocks = ex["blocks"]
T, _ = context.keeper_teams(tr, blocks, cfg)
tr2 = dict(tr); tr2["team_frame"] = T
ball = context.interpolate_ball(tr["ball"], cfg["ball_gaps"]["interpolate_max_frames"])
ctl = cfg["context"]["control"]
paths = [os.path.join(repo, m) for m in [ctl["model"]] + list(ctl["ensemble"])]
llr, seen = control.control_llr(tr2, ball, blocks, paths, control.feature_groups(cfg))
gt = pd.read_csv(os.path.join(cfg["paths"]["official_repo"], "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))["label"].to_numpy()
L, W = tr["pitch"]
P = tr["players"]
raw_seen = ~np.isnan(tr["ball"][:, 0])

args = sys.argv[1:]
a, b = int(args[0]), int(args[1]); step = int(args[2]) if len(args) > 2 else 5
print("frame GT mod raw/int ball(m)        v   nearH(slot d)  nearA(slot d)  nH nA  ctl_llr  eH-eA   eD-max  pdead touch")
for t in range(a, b + 1, step):
    d = np.hypot((P[t, :, 0] - ball[t, 0]) * L, (P[t, :, 1] - ball[t, 1]) * W)
    d = np.where(np.isnan(d), np.inf, d)
    tm = T[t]
    ok = ~np.isnan(P[t, :, 0])
    dh = np.where(tm == 0, d, np.inf); da = np.where(tm == 1, d, np.inf)
    sh, sa = int(np.argmin(dh)), int(np.argmin(da))
    print("{} {}  {}   {}{}  ({:5.1f},{:5.1f}) {:5.1f}  H s{:<2d} {:5.1f}   A s{:<2d} {:5.1f}   {:2d} {:2d}  {:+6.2f}  {:+7.1f} {:+7.1f}  {:.2f}  {}".format(
        t, gt[t], labels[t], int(raw_seen[t]), int(f["seen"][t]), ball[t, 0] * L, ball[t, 1] * W,
        f["ball_speed"][t], sh + 1, dh[sh], sa + 1, da[sa], int((ok & (tm == 0)).sum()), int((ok & (tm == 1)).sum()),
        llr[t], e[t, 0] - e[t, 1], e[t, 2] - max(e[t, 0], e[t, 1]), ex["p_dead"][t], f["touch_team"][t]))
