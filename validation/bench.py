"""Validation bench: run the labeller on OTHER games (the public Metrica sample matches) and score
it with the official scorer against a ground truth derived from their event data.

Why: the only ground truth we have for BAR-ATH is the public 5 minutes, and RULES.md forbids using
it to tune or to choose between versions (it is 17 possessions anyway). Every change to the
labeller is decided here instead, and the public window stays a smoke test.

    python validation/bench.py                                   # config.yaml + official baselines
    python validation/bench.py --configs config.yaml other.yaml  # versions side by side
    python validation/bench.py --clean                           # without degrading the tracking

Settings (data location, games, degradation, Smart Tagging stand-in) are in validation/bench.yaml.
"""

import argparse
import importlib.util
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (HERE, os.path.join(REPO, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import degrade as deg  # noqa: E402
import gt_events  # noqa: E402
import metrica  # noqa: E402
from possession.config import load_config  # noqa: E402

TAG_COLUMNS = ["code", "start_frame", "end_frame", "team", "type", "half", "tags"]
BASELINES = (("baseline_raw", 0), ("baseline_smooth25", 25), ("baseline_smooth50", 50),
             ("baseline_smooth125", 125))
# Full-match scores of the official baselines on BAR-ATH, published in starter/baseline.py. The
# bench is a good stand-in for our match if it reproduces their order and rough level.
PUBLISHED = {"baseline_raw": (0.6532, 0.0486), "baseline_smooth25": (0.6752, 0.1996),
             "baseline_smooth50": (0.6724, 0.2433), "baseline_smooth125": (0.6356, 0.2242)}


def load_bench(path):
    with open(path, encoding="utf-8") as handle:
        b = yaml.safe_load(handle)
    root = b["metrica_dir"]
    b["metrica_dir"] = root if os.path.isabs(root) else os.path.normpath(os.path.join(REPO, root))
    return b


# --- inputs the labeller expects ----------------------------------------------------------

def no_tags():
    return pd.DataFrame({c: pd.Series(dtype=object) for c in TAG_COLUMNS}).assign(
        start_frame=pd.Series(dtype=int), end_frame=pd.Series(dtype=int))


def synthetic_tags(events, blocks, cfg, seed):
    """A stand-in for the Smart Tagging: SET PIECES tags at the restarts of the event data.

    Starts lead_s before the restart (the real tags start a few seconds early), keeps each with
    probability `recall`. Kick-offs other than a period's first are tagged as after a goal.
    """
    rng = np.random.default_rng(seed)
    firsts = {first for _, first, _ in blocks}
    rows = []
    for e in events[gt_events.is_restart(events)].itertuples():
        if rng.random() > cfg["recall"]:
            continue
        kind = (e.subtype or "").split("-")[0].title() or "Goal Kick"
        if "KICK OFF" in e.subtype:
            kind = "Kick Off" if e.start in firsts else "Kick Off (After Goal)"
        if e.type == "PASS":
            kind = "Goal Kick"
        rows.append({"code": "SET PIECES", "start_frame": int(e.start - cfg["lead_s"] * 25),
                     "end_frame": int(e.start + 2 * 25), "team": e.team, "type": kind,
                     "half": str(e.period), "tags": {}})
    return pd.DataFrame(rows, columns=TAG_COLUMNS) if rows else no_tags()


# --- models -------------------------------------------------------------------------------

def _official_smooth(official_repo):
    spec = importlib.util.spec_from_file_location(
        "official_baseline", os.path.join(official_repo, "starter", "baseline.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.smooth


def nearest_codes(tracking):
    """Official baseline step 1: team of the player nearest to the seen ball, -1 without ball."""
    ball, players, team = tracking["ball"], tracking["players"], tracking["team"]
    scale = np.array(tracking["pitch"])
    d = np.hypot(*np.moveaxis((players - ball[:, None, :]) * scale, -1, 0))
    d = np.where(np.isnan(d), np.inf, d)
    best = np.argmin(d, axis=1)
    ok = np.isfinite(d[np.arange(len(d)), best])
    return np.where(ok, team[best], -1).astype(np.int8)


def context_labels(tracking, tags, cfg, blocks):
    from possession import context
    ball = context.interpolate_ball(tracking["ball"], cfg["ball_gaps"]["interpolate_max_frames"])
    labels, _ = context.label(tracking, ball, tags, cfg, blocks, frame_period=None, verbose=False)
    return labels


# --- scoring ------------------------------------------------------------------------------

def score(gt, pred, pm):
    pred = np.where(gt == "X", "X", pred)
    r = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred))
    return {"spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"],
            "n_gt": r["n_gt_spells"], "n_pred": r["n_pred_spells"],
            "dur": r["mean_spell_duration_s"], "gt_dur": r["gt_mean_spell_duration_s"],
            "alive_dead_f1": r["alive_dead_f1"], "team_acc": r["team_acc_gt_alive"]}


def logo_dir(cfg, model_path, kind="control"):
    """Where train_control.py / train_dead.py write the leave-one-game-out copies of a model: one
    folder per model file name, so versions with different features can be benched side by side."""
    stem = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(cfg["paths"]["output_dir"], "validation", kind, stem)


def load_config_for_game(path, game):
    """A labeller config whose learnt control model (if any) was trained WITHOUT this game."""
    cfg = load_config(path)
    ctl = cfg["context"].get("control") or {}
    if ctl.get("enabled"):
        logo = [os.path.join(logo_dir(cfg, m), "logo_{}.json".format(game))
                for m in [ctl["model"]] + list(ctl.get("ensemble") or [])]
        for m in logo:
            if not os.path.isfile(m):
                raise FileNotFoundError("run validation/train_control.py first: " + m)
        ctl["model"], ctl["ensemble"] = logo[0], logo[1:]
    dm = cfg["context"].get("dead_model") or {}
    if dm.get("enabled"):
        dm["model"] = os.path.join(logo_dir(cfg, dm["model"], "dead"), "logo_{}.json".format(game))
        if not os.path.isfile(dm["model"]):
            raise FileNotFoundError("run validation/train_dead.py first: " + dm["model"])
    return cfg


def load_case(game, seed, bench, cache_dir):
    """(tracking, events, gt, blocks, tags) of one game under one degradation seed."""
    tr = metrica.load_tracking(bench["metrica_dir"], game, cache_dir=cache_dir)
    events = metrica.load_events(bench["metrica_dir"], game)
    gt, blocks = gt_events.ground_truth(events, tr["n_frames"],
                                        (bench.get("gt") or {}).get("min_spell_s", 0.0))
    if seed is not None:
        tr = deg.degrade(tr, gt, bench["degrade"], seed)
    tags = (synthetic_tags(events, blocks, bench["synthetic_tags"], 0 if seed is None else seed)
            if bench["tags"] == "events" else no_tags())
    return tr, events, gt, blocks, tags


def official_metrics(cfg):
    sys.path.insert(0, os.path.join(cfg["paths"]["official_repo"], "evaluation"))
    import possession_metrics as pm
    return pm


def run_job(job):
    """One game x one degradation seed: every model on it. Runs in a worker process."""
    game, seed, bench, configs, baselines = job
    cfgs = [(name, load_config_for_game(path, game)) for name, path in configs]
    base_cfg = cfgs[0][1]
    pm = official_metrics(base_cfg)
    tr, events, gt, blocks, tags = load_case(game, seed, bench, base_cfg["paths"]["cache_dir"])

    rows = []
    if baselines:
        smooth = _official_smooth(base_cfg["paths"]["official_repo"])
        codes = nearest_codes(tr)
        for name, window in BASELINES:
            t0 = time.time()
            rows.append({"model": name, "game": game, "seed": seed,
                         **score(gt, smooth(codes, window), pm), "secs": time.time() - t0})
    for name, cfg in cfgs:
        t0 = time.time()
        labels = context_labels(tr, tags, cfg, blocks)
        rows.append({"model": name, "game": game, "seed": seed, **score(gt, labels, pm),
                     "secs": time.time() - t0})
    return rows


def print_table(df, show_published):
    short = {g: "G" + g.rsplit("_", 1)[-1] for g in df["game"].unique()}
    per_game = df.groupby(["model", "game"])["spellF1"].mean().unstack()
    per_game.columns = [short[c] + " spell" for c in per_game.columns]
    agg = df.groupby("model").agg(spellF1=("spellF1", "mean"), spell_sd=("spellF1", "std"),
                                  macroF1=("macroF1", "mean"), poss=("n_pred", "mean"),
                                  gt_poss=("n_gt", "mean"), dur=("dur", "mean"),
                                  aliveF1=("alive_dead_f1", "mean"), team_acc=("team_acc", "mean"),
                                  secs=("secs", "mean"))
    table = agg.join(per_game).sort_values("spellF1", ascending=False)
    if show_published:
        table["pub_spell"] = [PUBLISHED.get(m, (np.nan, np.nan))[1] for m in table.index]
        table["pub_macro"] = [PUBLISHED.get(m, (np.nan, np.nan))[0] for m in table.index]
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", "{:.4f}".format):
        print(table.drop(columns=["secs"]).to_string())
    n_runs = df.groupby("model").size().iloc[0]
    print("\n{} runs per model (games x seeds); spell_sd = std over runs. Published = BAR-ATH "
          "full match (starter/baseline.py).".format(n_runs))


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def run_condition(bench, configs, games, seeds_override, clean, baselines, workers):
    degrade_on = bench["degrade"]["enabled"] and not clean
    seeds = (seeds_override or bench["degrade"]["seeds"]) if degrade_on else [None]
    jobs = [(g, s, bench, configs, baselines) for g in games for s in seeds]
    if workers > 1 and len(jobs) > 1:
        with Pool(min(workers, len(jobs))) as pool:
            results = pool.map(run_job, jobs)
    else:
        results = [run_job(j) for j in jobs]
    return pd.DataFrame([r for rows in results for r in rows]), seeds


def print_suite(df):
    """One line per model: spellF1 and macroF1 under each condition, and their means."""
    spell = df.groupby(["model", "condition"])["spellF1"].mean().unstack()
    macro = df.groupby(["model", "condition"])["macroF1"].mean().unstack()
    poss = df.groupby(["model", "condition"])["n_pred"].mean().unstack()
    gt = df.groupby("condition")["n_gt"].mean()
    order = list(dict.fromkeys(df["condition"]))
    table = pd.concat({"spellF1": spell[order], "macroF1": macro[order], "poss": poss[order]},
                      axis=1)
    table[("spellF1", "MEAN")] = spell[order].mean(axis=1)
    table[("macroF1", "MEAN")] = macro[order].mean(axis=1)
    table = table.sort_values(("spellF1", "MEAN"), ascending=False)
    with pd.option_context("display.width", 220, "display.max_columns", 40,
                           "display.float_format", "{:.4f}".format):
        print(table.to_string())
    print("\nGT possessions per game: " + ", ".join(
        "{} {:.0f}".format(c, gt[c]) for c in order))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", action="store_true",
                        help="run every condition in bench.yaml 'suite' and print one summary")
    parser.add_argument("--bench", default=os.path.join(HERE, "bench.yaml"))
    parser.add_argument("--configs", nargs="+", default=[os.path.join(REPO, "config.yaml")],
                        help="labeller configs to compare (name=path or path)")
    parser.add_argument("--games", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--clean", action="store_true", help="no degradation of the tracking")
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", default=None, help="CSV with one row per model x game x seed")
    args = parser.parse_args()

    bench = load_bench(args.bench)
    configs = []
    for item in args.configs:
        name, _, path = item.rpartition("=")
        path = os.path.abspath(path)
        configs.append((name or os.path.splitext(os.path.basename(path))[0], path))
    games = args.games or bench["games"]
    t0 = time.time()
    if args.suite:
        frames = []
        for cond, over in bench["suite"].items():
            d, _ = run_condition(_merge(bench, over), configs, games, args.seeds, False,
                                 not args.no_baselines, args.workers)
            frames.append(d.assign(condition=cond))
        df = pd.concat(frames, ignore_index=True)
        print("bench suite: {} games, conditions {}  ({:.0f} s)\n".format(
            len(games), list(bench["suite"]), time.time() - t0))
        print_suite(df)
    else:
        df, seeds = run_condition(bench, configs, games, args.seeds, args.clean,
                                  not args.no_baselines, args.workers)
        print("bench: {} games x {} seeds, degrade={}, tags={}  ({:.0f} s)\n".format(
            len(games), len(seeds), bench["degrade"]["enabled"] and not args.clean,
            bench["tags"], time.time() - t0))
        print_table(df, show_published=not args.no_baselines)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
