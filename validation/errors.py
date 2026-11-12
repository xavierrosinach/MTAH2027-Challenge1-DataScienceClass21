"""Where a labeller loses spellF1 on the bench (other games, never BAR-ATH): aggregate numbers only.

    python validation/errors.py                                    # config.yaml, condition 'degraded'
    python validation/errors.py --configs ref=a.yaml new=b.yaml --condition degraded+coarseGT

For each config, over the bench games x seeds of one suite condition:
  - oracles: spellF1 if the stoppages were perfect (dead/alive from the ground truth, our teams),
    and if the team were perfect (our dead/alive, team from the ground truth where both are alive).
    The larger gap is where the work pays;
  - possessions found at IoU 0.5 by true duration, and predicted ones that are right by predicted
    duration;
  - why a true possession is missed (mostly given to the rival, mostly called dead, split, or
    merged with its neighbours / wrong edges) and why a predicted one is spurious;
  - hand-overs H<->A: share with a predicted one of the same kind within 1 / 2 / 5 s.
"""

import argparse
import copy
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench as B  # noqa: E402  (sets the paths)
from possession import context  # noqa: E402

FPS = 25
BUCKETS = ((0, 2), (2, 5), (5, 10), (10, 20), (20, 1e9))
BUCKET_NAMES = ["<2s", "2-5s", "5-10s", "10-20s", ">20s"]


def bucket(seconds):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= seconds < hi:
            return i
    return len(BUCKETS) - 1


def greedy_match(overlaps, n_gt, n_pred, threshold):
    """Same one-to-one greedy match as the official scorer, returning both sides' flags."""
    g, p = np.zeros(n_gt, dtype=bool), np.zeros(n_pred, dtype=bool)
    for iou, i, j in overlaps:
        if iou < threshold:
            break
        if g[i] or p[j]:
            continue
        g[i], p[j] = True, True
    return g, p


def dead_oracle(pred, gt, blocks):
    """Our teams, the ground truth's stoppages."""
    out = pred.copy()
    out[gt == "D"] = "D"
    for _, first, last in blocks:
        seg = out[first:last + 1]
        need = (gt[first:last + 1] != "D") & (seg == "D")
        if not need.any():
            continue
        team = pd.Series(np.where(seg == "D", None, seg)).ffill().bfill()
        fill = team.to_numpy(dtype=object)
        fill = np.where(pd.isna(fill), gt[first:last + 1], fill)
        seg[need] = fill[need]
    return out


def team_oracle(pred, gt):
    """Our stoppages, the ground truth's team wherever both say the ball is alive."""
    out = pred.copy()
    both = (pred != "D") & np.isin(gt, ["H", "A"])
    out[both] = gt[both]
    return out


def analyse(gt, pred, pm):
    pred = np.where(gt == "X", "X", pred)
    gc, pc = pm.label_codes(gt), pm.label_codes(pred)
    runs = pm.valid_runs(gc)
    gs, ps = pm.possession_spells(gc, runs), pm.possession_spells(pc, runs)
    ov = pm.spell_overlaps(gs, ps)
    gflag, pflag = greedy_match(ov, len(gs), len(ps), 0.5)
    out = {"spellF1": pm.spell_scores(gc, pc, runs)["mean_f1"], "n_gt": len(gs), "n_pred": len(ps)}

    for i, name in enumerate(BUCKET_NAMES):
        gi = [k for k, (a, b, _) in enumerate(gs) if bucket((b - a) / FPS) == i]
        pi = [k for k, (a, b, _) in enumerate(ps) if bucket((b - a) / FPS) == i]
        out["gt_n " + name] = len(gi)
        out["gt_found " + name] = gflag[gi].sum() if gi else 0
        out["pred_n " + name] = len(pi)
        out["pred_ok " + name] = pflag[pi].sum() if pi else 0

    codes = {"H": 0, "A": 1, "D": 2}
    for k, (a, b, team) in enumerate(gs):
        if gflag[k]:
            continue
        seg = pc[a:b]
        same, dead = np.mean(seg == team), np.mean(seg == codes["D"])
        other = 1 - same - dead
        if other >= 0.5:
            why = "miss_rival"
        elif dead >= 0.5:
            why = "miss_dead"
        elif same >= 0.5:
            n_same = sum(1 for (c, d, t) in ps if t == team and d > a and c < b)
            why = "miss_split" if n_same > 1 else "miss_merged_or_edges"
        else:
            why = "miss_mixed"
        out[why] = out.get(why, 0) + 1
    for k, (a, b, team) in enumerate(ps):
        if pflag[k]:
            continue
        seg = gc[a:b]
        same, dead = np.mean(seg == team), np.mean(seg == codes["D"])
        other = 1 - same - dead
        why = ("spur_rival" if other >= 0.5 else "spur_dead" if dead >= 0.5 else
               "spur_split_or_edges" if same >= 0.5 else "spur_mixed")
        out[why] = out.get(why, 0) + 1

    gt_tr = [(f, a, b) for f, a, b in pm.transitions(gc, runs) if {a, b} == {0, 1}]
    pr_tr = [(f, a, b) for f, a, b in pm.transitions(pc, runs) if {a, b} == {0, 1}]
    out["handovers"] = len(gt_tr)
    out["pred_handovers"] = len(pr_tr)
    for tol in (1, 2, 5):
        hit = 0
        for f, a, b in gt_tr:
            hit += any(abs(f2 - f) <= tol * FPS and (a2, b2) == (a, b) for f2, a2, b2 in pr_tr)
        out["handover_hit_{}s".format(tol)] = hit
    return out


def evidence_stats(gt, pred, ev, pm):
    """For true possessions by duration, found or missed at IoU 0.5: the mean per-frame team
    evidence towards the true team (clipped to +/- 5) and the share of frames where it points to
    the true team. Missed ones with evidence pointing right = the decoder drops them; pointing
    wrong = the evidence itself is wrong."""
    pred = np.where(gt == "X", "X", pred)
    gc, pc = pm.label_codes(gt), pm.label_codes(pred)
    runs = pm.valid_runs(gc)
    gs, ps = pm.possession_spells(gc, runs), pm.possession_spells(pc, runs)
    gflag, _ = greedy_match(pm.spell_overlaps(gs, ps), len(gs), len(ps), 0.5)
    out = {}
    for k, (a, b, team) in enumerate(gs):
        v = (1 if team == 0 else -1) * ev[a:b]
        key = "{} {}".format(BUCKET_NAMES[bucket((b - a) / FPS)], "found" if gflag[k] else "missed")
        out["ev_n " + key] = out.get("ev_n " + key, 0) + 1
        out["ev_mean " + key] = out.get("ev_mean " + key, 0) + np.mean(np.clip(v, -5, 5))
        out["ev_pos " + key] = out.get("ev_pos " + key, 0) + np.mean(v > 0)
    return out


def handover_mask(gt, blocks, half_width):
    """Frames within +/- half_width of a true direct H<->A hand-over."""
    mask = np.zeros(len(gt), dtype=bool)
    for _, first, last in blocks:
        g = gt[first:last + 1]
        idx = np.flatnonzero((g[1:] != g[:-1]) & np.isin(g[1:], ["H", "A"])
                             & np.isin(g[:-1], ["H", "A"])) + 1 + first
        for i in idx:
            mask[max(first, i - half_width):min(last + 1, i + half_width + 1)] = True
    return mask


def run_job(job):
    game, seed, bench, configs, switch_oracles = job
    cfgs = [(name, B.load_config_for_game(path, game)) for name, path in configs]
    pm = B.official_metrics(cfgs[0][1])
    tr, _, gt, blocks, tags = B.load_case(game, seed, bench, cfgs[0][1]["paths"]["cache_dir"])
    rows = []
    for name, cfg in cfgs:
        ball = context.interpolate_ball(tr["ball"], cfg["ball_gaps"]["interpolate_max_frames"])
        pred, diag = context.label(tr, ball, tags, cfg, blocks, frame_period=None, verbose=False)
        e = diag["emissions"]
        row = {"model": name, "game": game, "seed": seed, **analyse(gt, pred, pm),
               **evidence_stats(gt, pred, e[:, context.H] - e[:, context.A], pm)}
        row["oracle_dead"] = analyse(gt, dead_oracle(pred, gt, blocks), pm)["spellF1"]
        row["oracle_team"] = analyse(gt, team_oracle(pred, gt), pm)["spellF1"]
        # a perfect hand-over detector: cheap switches within +/- 0.5 s of a true H<->A hand-over
        # (only cheaper there / also dearer elsewhere)
        near = handover_mask(gt, blocks, 12)
        base = cfg["context"]["transition_cost"]["team_switch"]
        for key, other in ((("oracle_switch_low", base), ("oracle_switch_both", 2.5 * base))
                           if switch_oracles else ()):
            c2 = copy.deepcopy(cfg)
            c2["context"]["transition_cost"]["team_switch"] = np.where(near, 2.0, other)
            row[key] = analyse(gt, B.context_labels(tr, tags, c2, blocks), pm)["spellF1"]
        # frames where both say alive: wrong team with the ball seen / in a ball gap
        seen = ~np.isnan(tr["ball"][:, 0])
        both = (pred != "D") & np.isin(gt, ["H", "A"])
        wrong = both & (pred != gt)
        row["both_alive"] = both.sum()
        row["wrong_seen"], row["wrong_gap"] = (wrong & seen).sum(), (wrong & ~seen).sum()
        row["alive_seen"], row["alive_gap"] = (both & seen).sum(), (both & ~seen).sum()
        rows.append(row)
    return rows


def report(df):
    s = df.groupby("model").sum(numeric_only=True)
    m = df.groupby("model").mean(numeric_only=True)
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", "{:.3f}".format):
        print("spellF1 and oracles (mean over runs):")
        cols = [c for c in ("spellF1", "oracle_dead", "oracle_team", "oracle_switch_low",
                            "oracle_switch_both") if c in m.columns]
        print(m[cols].to_string())

        print("\nwrong team (frames both call alive): with the ball seen / in ball gaps")
        print(pd.DataFrame({
            "err_seen": s["wrong_seen"] / s["alive_seen"], "err_gap": s["wrong_gap"] / s["alive_gap"],
            "gap_share": s["alive_gap"] / s["both_alive"],
            "errors_in_gaps": s["wrong_gap"] / (s["wrong_seen"] + s["wrong_gap"])}).to_string())

        print("\npossessions per run: true / predicted")
        print((s[["n_gt", "n_pred", "handovers", "pred_handovers"]] / len(df["seed"].unique())
               / len(df["game"].unique())).to_string())

        rec = pd.DataFrame({b: s["gt_found " + b] / s["gt_n " + b] for b in BUCKET_NAMES})
        rec["share of true"] = ""
        print("\ntrue possessions found at IoU 0.5, by true duration (share of all true: {}):".format(
            ", ".join("{} {:.0%}".format(b, s["gt_n " + b].iloc[0] / s["n_gt"].iloc[0])
                      for b in BUCKET_NAMES)))
        print(rec.drop(columns="share of true").to_string())
        prec = pd.DataFrame({b: s["pred_ok " + b] / s["pred_n " + b].replace(0, np.nan)
                             for b in BUCKET_NAMES})
        n_runs = len(df) / df["model"].nunique()
        cnt = pd.DataFrame({b: s["pred_n " + b] / n_runs for b in BUCKET_NAMES})
        print("\npredicted possessions that match at IoU 0.5, by predicted duration:")
        print(prec.to_string())
        print("predicted possessions per run, by predicted duration:")
        print(cnt.to_string())

        rows = []
        for model in s.index:
            for bname in BUCKET_NAMES:
                for kind in ("found", "missed"):
                    key = "{} {}".format(bname, kind)
                    n = s.loc[model].get("ev_n " + key, 0)
                    if n:
                        rows.append({"model": model, "true duration": bname, "": kind,
                                     "n per run": n / n_runs,
                                     "mean evidence": s.loc[model]["ev_mean " + key] / n,
                                     "share frames right": s.loc[model]["ev_pos " + key] / n})
        print("\nteam evidence inside true possessions (towards the true team):")
        print(pd.DataFrame(rows).set_index(["model", "true duration", ""]).to_string())

        miss = [c for c in s.columns if c.startswith("miss_")]
        spur = [c for c in s.columns if c.startswith("spur_")]
        print("\nwhy true possessions are missed (share of all true possessions):")
        print((s[miss].div(s["n_gt"], axis=0)).to_string())
        print("\nwhy predicted possessions are spurious (share of all predicted):")
        print((s[spur].div(s["n_pred"], axis=0)).to_string())
        print("\nH<->A hand-overs with a predicted one of the same kind within:")
        print(pd.DataFrame({"{}s".format(t): s["handover_hit_{}s".format(t)] / s["handovers"]
                            for t in (1, 2, 5)}).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bench", default=os.path.join(HERE, "bench.yaml"))
    parser.add_argument("--configs", nargs="+", default=[os.path.join(B.REPO, "config.yaml")])
    parser.add_argument("--condition", default="degraded")
    parser.add_argument("--games", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", default=None, help="CSV with one row per model x game x seed")
    parser.add_argument("--switch-oracles", action="store_true",
                        help="also the perfect hand-over detector oracles (3x slower)")
    args = parser.parse_args()

    bench = B.load_bench(args.bench)
    bench = B._merge(bench, bench["suite"][args.condition])
    configs = []
    for item in args.configs:
        name, _, path = item.rpartition("=")
        path = os.path.abspath(path)
        configs.append((name or os.path.splitext(os.path.basename(path))[0], path))
    games = args.games or bench["games"]
    seeds = args.seeds or bench["degrade"]["seeds"]
    t0 = time.time()
    jobs = [(g, s, bench, configs, args.switch_oracles) for g in games for s in seeds]
    with Pool(min(args.workers, len(jobs))) as pool:
        rows = [r for rs in pool.map(run_job, jobs) for r in rs]
    df = pd.DataFrame(rows).fillna(0)
    print("errors: condition {}, {} games x {} seeds ({:.0f} s)\n".format(
        args.condition, len(games), len(seeds), time.time() - t0))
    report(df)
    if args.out:
        df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
