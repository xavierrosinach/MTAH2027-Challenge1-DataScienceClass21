"""Load tracking (ATD) and Smart Tagging.

ATD line: "<frame_id>:<x,y,v>;<x,y,v>;...:<ball x,y,v>", NaN if missing.
video frame = frame_id - 1. Coordinates are 0-1 (multiply by pitch for metres).
"""

import os
import xml.etree.ElementTree as ET
import numpy as np

HOME, AWAY = 0, 1

def team_of_each_slot(meta_path):
    """Team of each track slot: 0 home, 1 away (first team in the XML is home)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, got {}".format(meta_path, len(team_ids)))
    return np.array([HOME if p.get("teamId") == team_ids[0] else AWAY for p in root.iter("Player")], dtype=np.int8)

def _parse_raw(raw_path, n_frames, n_slots):
    ball = np.full((n_frames, 2), np.nan)
    players = np.full((n_frames, n_slots, 2), np.nan)
    speed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            for slot, group in enumerate(parts[1].split(";")):
                if group[0] != "N":
                    x, y, v = group.split(",")
                    players[frame, slot] = (float(x), float(y))
                    speed[frame, slot] = float(v)
            if parts[2][0] != "N":
                x, y = parts[2].split(",")[:2]
                ball[frame] = (float(x), float(y))
    return ball, players, speed


SMART_TAGGING_TEAMS = {"barcelona": "H", "atletic club": "A", "athletic club": "A"}

def load_smart_tagging(cfg, game="BAR-ATH"):
    """Smart Tagging as a DataFrame (code, start_frame, end_frame, team, type, half, tags)."""
    import games
    import pandas as pd

    (path,) = games.require(["smart_tagging"], game, cfg["paths"]["data_dir"])
    fps = games.GAMES[game]["fps"]
    rows = []
    for inst in ET.parse(path).iter("instance"):
        tags = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        start_s, end_s = float(inst.findtext("start")), float(inst.findtext("end"))
        rows.append({"code": (inst.findtext("code") or "").strip().upper(), 
                     "start_frame": int(round(start_s * fps)),
                     "end_frame": int(round(end_s * fps)),
                     "team": SMART_TAGGING_TEAMS.get((tags.get("Team") or "").strip().lower()),
                     "type": tags.get("Type"), 
                     "half": tags.get("Half"),
                     "tags": tags})
    return pd.DataFrame(rows).sort_values("start_frame").reset_index(drop=True)


def load_tracking(cfg, game="BAR-ATH"):
    """Returns dict: ball (N,2), players (N,S,2), speed (N,S), team (S,), pitch."""
    import games

    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], game, cfg["paths"]["data_dir"])
    n_frames = games.source_frames("tactical", game)
    team = team_of_each_slot(meta_path)

    cache_dir = cfg["paths"]["cache_dir"]
    cache_path = os.path.join(cache_dir, "{}_tactical_atd.npz".format(game))
    if os.path.isfile(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(raw_path):
        cached = np.load(cache_path)
        ball, players, speed = cached["ball"], cached["players"], cached["speed"]
    else:
        print("reading {} (1-2 min the first time)...".format(os.path.basename(raw_path)))
        ball, players, speed = _parse_raw(raw_path, n_frames, len(team))
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache_path, ball=ball, players=players, speed=speed)

    return {"ball": ball, "players": players, "speed": speed, "team": team,
            "pitch": games.GAMES[game]["pitch"], "n_frames": n_frames, "game": game}
