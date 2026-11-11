"""Load the public Metrica Sports sample games (other matches, not BAR-ATH) into the same arrays
that possession.data.load_tracking returns, plus their event data.

Source: https://github.com/metrica-sports/sample-data (anonymised; 105 x 68 m; 25 fps; tracking and
events synchronised). Games 1 and 2 come as CSV (one file per team), game 3 in the FIFA EPTS format
(tracking .txt + metadata .xml) with JSON events.

    git clone https://github.com/metrica-sports/sample-data ../external/metrica-sample-data

Frame f of the files (1-based) is index f - 1 of every array, as with the ATD.
"""

import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

GAMES = ("Sample_Game_1", "Sample_Game_2", "Sample_Game_3")
PITCH = (105.0, 68.0)
FPS = 25
HOME, AWAY = 0, 1


def game_dir(root, game):
    path = os.path.join(root, "data", game)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            "Metrica sample data not found at {}\n  git clone "
            "https://github.com/metrica-sports/sample-data {}".format(path, root))
    return path


# --- tracking -------------------------------------------------------------------------------

def _read_team_csv(path):
    """(frames, period, players (N,S,2), ball (N,2)) from one Metrica CSV tracking file."""
    raw = pd.read_csv(path, skiprows=3, header=None, low_memory=False)
    values = raw.to_numpy(dtype=float)
    period, frame = values[:, 0].astype(int), values[:, 1].astype(int)
    xy = values[:, 3:]
    players = xy[:, :-2].reshape(len(xy), -1, 2)
    ball = xy[:, -2:]
    return frame, period, players, ball


def _load_csv_game(folder, game):
    fh, ph, home, ball = _read_team_csv(os.path.join(folder, game + "_RawTrackingData_Home_Team.csv"))
    fa, _, away, _ = _read_team_csv(os.path.join(folder, game + "_RawTrackingData_Away_Team.csv"))
    if not np.array_equal(fh, fa) or not np.array_equal(fh, np.arange(1, len(fh) + 1)):
        raise ValueError("{}: home/away frames differ or are not 1..N".format(game))
    players = np.concatenate([home, away], axis=1)
    team = np.r_[np.full(home.shape[1], HOME), np.full(away.shape[1], AWAY)].astype(np.int8)
    return ball, players, team, ph


def _load_epts_game(folder, game):
    root = ET.parse(os.path.join(folder, game + "_metadata.xml")).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    team_of_player = {p.get("id"): (HOME if p.get("teamId") == team_ids[0] else AWAY)
                      for p in root.iter("Player")}
    player_of_channel = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    specs, slot_of_player = [], {}
    for spec in root.iter("DataFormatSpecification"):
        order = []
        for ref in spec.iter("PlayerChannelRef"):
            cid = ref.get("playerChannelId")
            if cid.endswith("_x"):
                pid = player_of_channel[cid]
                slot_of_player.setdefault(pid, len(slot_of_player))
                order.append(slot_of_player[pid])
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), np.array(order)))
    params = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}

    lines = open(os.path.join(folder, game + "_tracking.txt")).read().splitlines()
    n, S = len(lines), len(slot_of_player)
    ball = np.full((n, 2), np.nan)
    players = np.full((n, S, 2), np.nan)
    for line in lines:
        frame_s, pl, bl = line.split(":")
        frame = int(frame_s)
        order = next(o for a, b, o in specs if a <= frame <= b)
        xy = np.array(pl.replace(";", ",").split(","), dtype=float).reshape(-1, 2)
        players[frame - 1, order] = xy
        ball[frame - 1] = np.array(bl.split(","), dtype=float)
    team = np.array([team_of_player[p] for p in sorted(slot_of_player, key=slot_of_player.get)],
                    dtype=np.int8)

    period = np.zeros(n, dtype=int)
    for k, name in ((1, "first_half"), (2, "second_half")):
        a, b = params.get(name + "_start"), params.get(name + "_end")
        if a and b:
            period[int(a) - 1:int(b)] = k
    return ball, players, team, period


def player_speed(players, smooth=2):
    """Per slot speed (m/s) from positions smoothed over +/- smooth frames; NaN where unseen."""
    n = len(players)
    pos = players * np.array(PITCH)
    ok = ~np.isnan(pos[..., 0])
    c = np.concatenate([np.zeros((1,) + pos.shape[1:]), np.cumsum(np.nan_to_num(pos), axis=0)])
    cn = np.concatenate([np.zeros((1, pos.shape[1])), np.cumsum(ok, axis=0)])
    i = np.arange(n)
    lo, hi = np.maximum(0, i - smooth), np.minimum(n, i + smooth + 1)
    cnt = (cn[hi] - cn[lo])[..., None]
    with np.errstate(invalid="ignore", divide="ignore"):
        sm = np.where(cnt > 0, (c[hi] - c[lo]) / cnt, np.nan)
    sm[~ok] = np.nan
    v = np.full(ok.shape, np.nan, dtype=np.float32)
    d = sm[2:] - sm[:-2]
    v[1:-1] = np.hypot(d[..., 0], d[..., 1]) * FPS / 2
    return v


def load_tracking(root, game, cache_dir=None):
    """Same dict as possession.data.load_tracking, plus "period" (N,) 0 = outside both halves."""
    folder = game_dir(root, game)
    cache = os.path.join(cache_dir, "metrica_{}.npz".format(game)) if cache_dir else None
    if cache and os.path.isfile(cache):
        z = np.load(cache)
        ball, players, team, period = z["ball"], z["players"], z["team"], z["period"]
    else:
        loader = _load_epts_game if game == "Sample_Game_3" else _load_csv_game
        ball, players, team, period = loader(folder, game)
        if cache:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(cache, ball=ball, players=players, team=team, period=period)
    return {"ball": ball, "players": players, "speed": player_speed(players), "team": team,
            "pitch": PITCH, "n_frames": len(ball), "game": game, "period": period}


# --- events ---------------------------------------------------------------------------------

def _subtype_json(sub):
    if not sub:
        return ""
    if isinstance(sub, dict):
        return sub["name"]
    return "-".join(s["name"] for s in sub)


def load_events(root, game):
    """DataFrame: team ('H'/'A'/None), type, subtype, period, start, end (0-based frame indices)."""
    folder = game_dir(root, game)
    if game == "Sample_Game_3":
        data = json.load(open(os.path.join(folder, game + "_events.json")))
        meta = ET.parse(os.path.join(folder, game + "_metadata.xml")).getroot()
        home_id = [t.get("id") for t in meta.iter("Team")][0]
        rows = [{
            "team": None if not e.get("team") else ("H" if e["team"]["id"] == home_id else "A"),
            "type": e["type"]["name"], "subtype": _subtype_json(e.get("subtypes")),
            "period": e["period"], "start": e["start"]["frame"], "end": e["end"]["frame"],
        } for e in data["data"]]
        ev = pd.DataFrame(rows)
    else:
        raw = pd.read_csv(os.path.join(folder, game + "_RawEventsData.csv"))
        ev = pd.DataFrame({
            "team": raw["Team"].map({"Home": "H", "Away": "A"}),
            "type": raw["Type"].str.strip().str.upper(),
            "subtype": raw["Subtype"].fillna("").str.strip().str.upper(),
            "period": raw["Period"], "start": raw["Start Frame"], "end": raw["End Frame"],
        })
    ev["start"] = ev["start"].astype(int) - 1
    ev["end"] = np.maximum(ev["end"].fillna(0).astype(int) - 1, ev["start"])
    return ev.sort_values(["start", "end"], kind="stable").reset_index(drop=True)
