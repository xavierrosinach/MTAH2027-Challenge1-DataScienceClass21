"""Build the joined dataset (ATD + Smart Tagging + Homography) and save it to cache/.

    python src/possession/data_proc.py [--config config.yaml]

Output (pickles in cache/):
    slots    one row per track slot (slot -> team)
    tags     one row per Smart Tagging instance
    frames   one row per video frame: time, ball, players summary, homography, active tags
    players  one row per observed slot per frame (long format)

Alignment (DATA.md): ATD frame_id - 1, Smart Tagging round(seconds * 25), homography as is.
"""

import argparse
import io
import os
import sys
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from possession.config import REPO_ROOT, load_config  # noqa: E402

GAME = "BAR-ATH"
KICKOFF = {1: 11185, 2: 113742}   # video frames (DATA.md, FAQ.md)
H_COLS = [f"h{r}{c}" for r in range(3) for c in range(3)]
TEAM_MAP = {"barcelona": "H", "atletic club": "A", "athletic club": "A", "n/a": "N/A"}
FILES = ("frames", "players", "tags", "slots")


def snake(s):
    return s.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")


def game_info(cfg):
    import games

    return {
        "paths": games.asset_paths(GAME, cfg["paths"]["data_dir"]),
        "n": games.source_frames("tactical", GAME),
        "fps": games.GAMES[GAME]["fps"],
        "pitch": games.GAMES[GAME]["pitch"],
        "halves": games.half_video_blocks(GAME),
    }


# --- raw sources

def load_slots(meta_path):
    """Team of each track slot. First team in the XML is home."""
    root = ET.parse(meta_path).getroot()
    home_id = [t.get("id") for t in root.iter("Team")][0]
    return pd.DataFrame([
        {"slot": i + 1, "player_id": p.get("id"), "name": p.findtext("Name"),
         "team": "H" if p.get("teamId") == home_id else "A"}
        for i, p in enumerate(root.iter("Player"))
    ])


def load_atd(raw_path, n_slots, n_frames):
    """Wide ATD: frame, frame_id, s{i}_x/y/v for every slot, ball_x/y/v."""
    with open(raw_path) as f:
        text = f.read().replace(":", ",").replace(";", ",")

    cols = (["frame_id"]
            + [f"s{i}_{c}" for i in range(1, n_slots + 1) for c in ("x", "y", "v")]
            + ["ball_x", "ball_y", "ball_v"])
    atd = pd.read_csv(io.StringIO(text), header=None, names=cols, dtype=np.float32)
    del text

    atd["frame_id"] = atd["frame_id"].astype(np.int64)
    atd = pd.concat([(atd["frame_id"] - 1).rename("frame"), atd], axis=1)   # frame_id starts at 1
    atd = atd.sort_values("frame").reset_index(drop=True)

    assert len(atd) == n_frames, len(atd)
    assert (atd["frame"].to_numpy() == np.arange(n_frames)).all(), "frames not contiguous"
    return atd


def load_homography(path, n_frames):
    homog = pd.read_csv(path)
    assert len(homog) == n_frames and (homog["frame"].to_numpy() == np.arange(n_frames)).all()
    return homog


def load_tags(path, fps):
    """One row per Smart Tagging instance, every tag group as a column. end_frame is exclusive."""
    rows = []
    for inst in ET.parse(path).iter("instance"):
        start_s, end_s = float(inst.findtext("start")), float(inst.findtext("end"))
        row = {"id": inst.findtext("ID"), "code": inst.findtext("code").strip().upper(),
               "start_s": start_s, "end_s": end_s,
               "start_frame": int(round(start_s * fps)),
               "end_frame": int(round(end_s * fps))}
        for lab in inst.findall("label"):
            row[snake(lab.findtext("group"))] = lab.findtext("text")
        rows.append(row)

    tags = pd.DataFrame(rows).sort_values("start_frame").reset_index(drop=True)
    tags["team_raw"] = tags["team"]
    tags["team"] = tags["team"].str.strip().str.lower().map(TEAM_MAP)
    tags["half"] = tags["half"].map({"1st Half": 1, "2nd Half": 2})
    tags["duration_s"] = tags["end_s"] - tags["start_s"]
    return tags


# --- joined tables

def player_arrays(atd, slots, pitch):
    """(frame x slot) arrays in metres: x, y, speed."""
    n_slots = len(slots)
    px = atd[[f"s{i}_x" for i in range(1, n_slots + 1)]].to_numpy(np.float64) * pitch[0]
    py = atd[[f"s{i}_y" for i in range(1, n_slots + 1)]].to_numpy(np.float64) * pitch[1]
    pv = atd[[f"s{i}_v" for i in range(1, n_slots + 1)]].to_numpy(np.float64)
    return px, py, pv


def project(H, x, y):
    """H (N,3,3), x/y (N,) normalised -> pixels."""
    p = np.einsum("nij,nj->ni", H, np.stack([x, y, np.ones_like(x)], axis=1))
    return p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]


def build_frames(atd, homog, tags, slots, info):
    n, fps, (L, W), halves = info["n"], info["fps"], info["pitch"], info["halves"]
    frames = pd.DataFrame({"frame": np.arange(n)})

    # time
    half = np.zeros(n, dtype=np.int8)
    for h, first, last in halves:
        half[first:last + 1] = h
    frames["half"] = half                     # 0 = not graded
    frames["graded"] = half > 0
    period = np.full(n, "pre_match", dtype=object)
    period[halves[0][1]:halves[0][2] + 1] = "1st_half"
    period[halves[0][2] + 1:halves[1][1]] = "half_time"
    period[halves[1][1]:KICKOFF[2]] = "2nd_half_pre_kickoff"   # graded, always D
    period[KICKOFF[2]:halves[1][2] + 1] = "2nd_half"
    period[halves[1][2] + 1:] = "post_match"
    frames["period"] = pd.Categorical(period)
    frames["time_s"] = frames["frame"] / fps
    match_s = np.full(n, np.nan)
    match_s[half == 1] = (frames["frame"][half == 1] - KICKOFF[1]) / fps
    match_s[half == 2] = (frames["frame"][half == 2] - KICKOFF[2]) / fps + 45 * 60
    frames["match_s"] = match_s
    frames["clock"] = [f"{int(s // 60):02d}:{int(s % 60):02d}" if s == s else "" for s in match_s]

    # ball
    frames["ball_x"], frames["ball_y"], frames["ball_v"] = atd["ball_x"], atd["ball_y"], atd["ball_v"]
    frames["ball_x_m"], frames["ball_y_m"] = atd["ball_x"] * L, atd["ball_y"] * W
    frames["ball_seen"] = atd["ball_x"].notna()
    frames["ball_in_pitch"] = frames["ball_x"].between(0, 1) & frames["ball_y"].between(0, 1)

    # players summary
    px, py, pv = player_arrays(atd, slots, (L, W))
    seen = ~np.isnan(px)
    is_h = (slots["team"] == "H").to_numpy()
    team_arr = slots["team"].to_numpy()

    frames["n_H"] = (seen & is_h).sum(axis=1)
    frames["n_A"] = (seen & ~is_h).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # frames with no players -> NaN
        frames["speed_H"] = np.nanmean(np.where(is_h, pv, np.nan), axis=1)
        frames["speed_A"] = np.nanmean(np.where(~is_h, pv, np.nan), axis=1)

    dist = np.hypot(px - frames["ball_x_m"].to_numpy()[:, None],
                    py - frames["ball_y_m"].to_numpy()[:, None])
    d = np.where(np.isnan(dist), np.inf, dist)
    order = np.argsort(d, axis=1)[:, :2]
    rows_ = np.arange(n)
    d1, d2 = d[rows_, order[:, 0]], d[rows_, order[:, 1]]

    frames["nearest_slot"] = np.where(np.isfinite(d1), order[:, 0] + 1, 0)
    frames["nearest_team"] = np.where(np.isfinite(d1), team_arr[order[:, 0]], "")
    frames["nearest_dist_m"] = np.where(np.isfinite(d1), d1, np.nan)
    frames["second_team"] = np.where(np.isfinite(d2), team_arr[order[:, 1]], "")
    frames["second_dist_m"] = np.where(np.isfinite(d2), d2, np.nan)
    frames["nearest_H_m"] = np.where(is_h, d, np.inf).min(axis=1)
    frames["nearest_A_m"] = np.where(~is_h, d, np.inf).min(axis=1)
    frames.loc[~np.isfinite(frames["nearest_H_m"]), "nearest_H_m"] = np.nan
    frames.loc[~np.isfinite(frames["nearest_A_m"]), "nearest_A_m"] = np.nan

    # homography
    for c in H_COLS:
        frames[c] = homog[c].to_numpy()
    frames["reproj_err"] = homog["reprojection_error"].to_numpy()
    frames["h_valid"] = homog["valid"].to_numpy().astype(bool)
    Hm = homog[H_COLS].to_numpy().reshape(-1, 3, 3)
    frames["ball_px"], frames["ball_py"] = project(Hm, frames["ball_x"].to_numpy(np.float64),
                                                   frames["ball_y"].to_numpy(np.float64))

    # Smart Tagging: one column per code with the team when active
    for code in sorted(tags["code"].unique()):
        values = np.full(n, "", dtype=object)
        for _, t in tags[tags["code"] == code].iterrows():
            values[t["start_frame"]:t["end_frame"]] = t["team"] if isinstance(t["team"], str) else "N/A"
        frames["st_" + snake(code)] = pd.Categorical(values)

    sp_type = np.full(n, "", dtype=object)
    for _, t in tags[tags["code"] == "SET PIECES"].iterrows():
        sp_type[t["start_frame"]:t["end_frame"]] = t["type"]
    frames["st_set_piece_type"] = pd.Categorical(sp_type)
    st_cols = [c for c in frames.columns if c.startswith("st_") and c != "st_set_piece_type"]
    frames["st_n_active"] = (frames[st_cols] != "").sum(axis=1)
    return frames


def build_players(atd, homog, slots, frames, info):
    """Long format: one row per observed slot per frame."""
    n, (L, W) = info["n"], info["pitch"]
    px, py, pv = player_arrays(atd, slots, (L, W))
    seen = ~np.isnan(px)
    team_arr = slots["team"].to_numpy()

    dist = np.hypot(px - frames["ball_x_m"].to_numpy()[:, None],
                    py - frames["ball_y_m"].to_numpy()[:, None])
    d = np.where(np.isnan(dist), np.inf, dist)
    rank = np.empty_like(d, dtype=np.int16)
    rank[np.arange(n)[:, None], np.argsort(d, axis=1)] = np.arange(1, len(slots) + 1, dtype=np.int16)

    f_idx, s_idx = np.nonzero(seen)
    x01, y01 = (px / L)[f_idx, s_idx], (py / W)[f_idx, s_idx]
    Hm = homog[H_COLS].to_numpy().reshape(-1, 3, 3)
    p = np.einsum("nij,nj->ni", Hm[f_idx], np.stack([x01, y01, np.ones_like(x01)], axis=1))

    players = pd.DataFrame({
        "frame": f_idx.astype(np.int32),
        "slot": (s_idx + 1).astype(np.int16),
        "team": pd.Categorical(team_arr[s_idx]),
        "x": x01.astype(np.float32), "y": y01.astype(np.float32),
        "x_m": px[f_idx, s_idx].astype(np.float32), "y_m": py[f_idx, s_idx].astype(np.float32),
        "v": pv[f_idx, s_idx].astype(np.float32),
        "dist_ball_m": dist[f_idx, s_idx].astype(np.float32),
        "rank_ball": np.where(np.isnan(dist[f_idx, s_idx]), 0, rank[f_idx, s_idx]).astype(np.int16),
        "px": (p[:, 0] / p[:, 2]).astype(np.float32), "py": (p[:, 1] / p[:, 2]).astype(np.float32),
    })
    return players.merge(frames[["frame", "half", "graded"]], on="frame", how="left")


# --- entry points

def build_all(cfg, save=True, verbose=True):
    """Read the raw data, build all tables and (optionally) save them to cache/."""
    log = print if verbose else (lambda *a, **k: None)
    info = game_info(cfg)
    paths = info["paths"]

    log("slots ...");    slots = load_slots(paths["atd_meta"])
    log("atd ...");      atd = load_atd(paths["atd_raw"], len(slots), info["n"])
    log("homography ..."); homog = load_homography(paths["homography"], info["n"])
    log("tags ...");     tags = load_tags(paths["smart_tagging"], info["fps"])
    log("frames ...");   frames = build_frames(atd, homog, tags, slots, info)
    log("players ...");  players = build_players(atd, homog, slots, frames, info)

    assert int(frames["graded"].sum()) == 158857
    out = {"frames": frames, "players": players, "tags": tags, "slots": slots}
    if save:
        cache = cfg["paths"]["cache_dir"]
        os.makedirs(cache, exist_ok=True)
        for name, df in out.items():
            df.to_pickle(os.path.join(cache, f"{name}.pkl"))
        log("saved to", cache)
    return out


def load_all(cfg, rebuild=False):
    """Load the tables from cache/, building them first if missing."""
    cache = cfg["paths"]["cache_dir"]
    if rebuild or not all(os.path.isfile(os.path.join(cache, f"{n}.pkl")) for n in FILES):
        return build_all(cfg)
    return {n: pd.read_pickle(os.path.join(cache, f"{n}.pkl")) for n in FILES}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=os.path.join(REPO_ROOT, "config.yaml"))
    args = parser.parse_args()

    out = build_all(load_config(args.config))
    for name, df in out.items():
        print(f"{name:8} {df.shape}")
