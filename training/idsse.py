"""IDSSE: seven professional matches (Bundesliga and 2. Bundesliga 2022/23) with DFL tracking and event data, as a second source of OTHER games for validation and training (never BAR-ATH).

Source: Bassek, Rein, Weber, Memmert (2025), "An integrated dataset of spatiotemporal and event data in elite soccer", Scientific Data. Figshare DOI 10.6084/m9.figshare.28196177, licence CC BY 4.0, public download without registration. Positions: TRACAB optical tracking, 25 Hz (Sportec Solutions for the DFL); events: official DFL event data (collected by human operators).

    python training/idsse.py download     # the 21 files -> ../external/idsse (never in the repo)
    python training/idsse.py parse        # tracking + events -> ../external/idsse/parsed

Games are named IDSSE_<match id> (e.g. IDSSE_J03WMX). `load_tracking` returns the same dict as training/metrica.load_tracking (0-1 coordinates, x along the length from the left goal line, y across from the top touch line; "period" 0 between the halves), `load_events` the same columns as training/metrica.load_events, and `ground_truth` the per-frame H/A/D/X labels and graded blocks built from the tracking's own BallStatus / BallPossession (the construction of the official ground truth of the challenge: dead where the tracking says dead, else the team it says has the ball). Only aggregates are printed.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import time
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_ROOT = os.path.normpath(os.path.join(REPO, "..", "external", "idsse"))
FPS = 25
HOME, AWAY = 0, 1
HALF_GAP = 10 * FPS          # NaN frames inserted between the halves (no window mixes two halves)
PARSE_VERSION = 1            # bump when the parsed arrays change

DOI = "10.6084/m9.figshare.28196177"
ARTICLE = 28196177
LICENCE = "CC BY 4.0"
CITATION = ("Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). An integrated dataset of "
            "spatiotemporal and event data in elite soccer. Scientific Data.")

# figshare file ids: matchinformation, events_raw, positions_raw_observed
FILES = {
    "J03WMX": (51643475, 51643493, 51643514),
    "J03WN1": (51643472, 51643496, 51643517),
    "J03WOH": (51643478, 51643499, 51643520),
    "J03WOY": (51643481, 51643502, 51643523),
    "J03WPY": (51643487, 51643505, 51643526),
    "J03WQQ": (51643484, 51643508, 51643529),
    "J03WR9": (51643490, 51643511, 51643532),
}
KINDS = ("matchinformation", "events", "positions")
PREFIX = {"matchinformation": "DFL_02_01_matchinformation", "events": "DFL_03_02_events_raw",
          "positions": "DFL_04_03_positions_raw_observed"}
GAMES = tuple("IDSSE_" + m for m in FILES)


def competition(match):
    return "DFL-COM-000001" if match in ("J03WMX", "J03WN1") else "DFL-COM-000002"


def file_name(match, kind):
    return "{}_{}_DFL-MAT-{}.xml".format(PREFIX[kind], competition(match), match)


def match_id(game):
    if not game.startswith("IDSSE_") or game[6:] not in FILES:
        raise ValueError("not an IDSSE game: {}".format(game))
    return game[6:]


def is_idsse(game):
    return game.startswith("IDSSE_")


def path_of(root, game, kind):
    path = os.path.join(root, file_name(match_id(game), kind))
    if not os.path.isfile(path):
        raise FileNotFoundError("IDSSE file missing: {}\n  python training/idsse.py download"
                                .format(path))
    return path


# --- download -------------------------------------------------------------------------------

def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "mtah2027-training/1.0 (urllib)"})
    return urllib.request.urlopen(req, timeout=timeout)


def figshare_files():
    """{file id: (name, size, md5)} from the figshare API (None if unreachable)."""
    out = {}
    try:
        for page in range(1, 10):
            url = ("https://api.figshare.com/v2/articles/{}/files?page={}&page_size=100"
                   .format(ARTICLE, page))
            with _get(url) as r:
                items = json.loads(r.read().decode("utf-8"))
            if not items:
                break
            for it in items:
                out[int(it["id"])] = (it["name"], int(it["size"]), it.get("computed_md5"))
    except Exception as exc:  # noqa: BLE001 - fall back to the names we know
        print("figshare API not reachable ({}); using the known names, no size check".format(exc))
        return None
    return out


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download(root):
    os.makedirs(root, exist_ok=True)
    meta = figshare_files()
    rows = []
    for match, ids in FILES.items():
        for kind, fid in zip(KINDS, ids):
            name = file_name(match, kind)
            size = md5 = None
            if meta is not None:
                if fid not in meta:
                    raise ValueError("file id {} not in figshare article {}".format(fid, ARTICLE))
                api_name, size, md5 = meta[fid]
                if api_name != name:
                    raise ValueError("file id {}: figshare name {} != expected {}".format(
                        fid, api_name, name))
            path = os.path.join(root, name)
            if os.path.isfile(path) and (size is None or os.path.getsize(path) == size):
                print("have  {} ({:.1f} MB)".format(name, os.path.getsize(path) / 1e6), flush=True)
            else:
                t0 = time.time()
                tmp = path + ".partial"
                with _get("https://ndownloader.figshare.com/files/{}".format(fid), 300) as r, \
                        open(tmp, "wb") as fh:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                got = os.path.getsize(tmp)
                if size is not None and got != size:
                    raise IOError("{}: got {} bytes, expected {}".format(name, got, size))
                if md5 and md5sum(tmp) != md5:
                    raise IOError("{}: md5 mismatch".format(name))
                os.replace(tmp, path)
                print("got   {} ({:.1f} MB, {:.0f} s)".format(name, got / 1e6, time.time() - t0),
                      flush=True)
            rows.append((name, fid, os.path.getsize(path), md5 or ""))
    with open(os.path.join(root, "SOURCE.txt"), "w", encoding="utf-8") as fh:
        fh.write("IDSSE: An integrated dataset of spatiotemporal and event data in elite soccer\n")
        fh.write(CITATION + "\n")
        fh.write("DOI: https://doi.org/{}\n".format(DOI))
        fh.write("Licence: {} (https://creativecommons.org/licenses/by/4.0/)\n".format(LICENCE))
        fh.write("Access: public figshare download, no registration "
                 "(https://ndownloader.figshare.com/files/<id>)\n")
        fh.write("Content: 7 matches, Bundesliga and 2. Bundesliga 2022/23; positions TRACAB 25 Hz "
                 "(Sportec Solutions / DFL), official DFL event data\n")
        fh.write("Downloaded: {} by training/idsse.py download (MTAH 2027, DataScienceClass21; "
                 "other matches, used for validation and training only)\n".format(
                     dt.datetime.now().strftime("%Y-%m-%d %H:%M")))
        fh.write("\nfile\tfigshare id\tbytes\tmd5\n")
        for row in rows:
            fh.write("\t".join(str(x) for x in row) + "\n")
    print("{} files in {}; SOURCE.txt written".format(len(rows), root))


# --- structure of the files (aggregates only) -------------------------------------------------

def _mask(value):
    """A value's shape without its content: digits -> d, letters -> a (for formats like T)."""
    return "".join("d" if c.isdigit() else ("a" if c.isalpha() else c) for c in value)


def match_info(root, game):
    """dict: pitch (x, y) in m, home and guest team ids, match title-free general attributes."""
    mi = ET.parse(path_of(root, game, "matchinformation")).getroot()
    general = mi.find(".//General")
    env = mi.find(".//Environment")
    teams = {t.get("TeamId"): t.get("Role") for t in mi.iter("Team")}
    home = general.get("HomeTeamId")
    guest = general.get("GuestTeamId")
    if teams.get(home) != "home" or teams.get(guest) != "guest":
        raise ValueError("{}: home/guest ids disagree with the team roles".format(game))
    people = {p.get("PersonId"): (HOME if t.get("TeamId") == home else AWAY)
              for t in mi.iter("Team") for p in t.iter("Player")}
    return {"pitch": (float(env.get("PitchX")), float(env.get("PitchY"))), "home": home,
            "guest": guest, "players": people, "competition": general.get("CompetitionName"),
            "season": general.get("Season")}


SECTIONS = ("firstHalf", "secondHalf", "firstHalfExtra", "secondHalfExtra")
RESTARTS = {"KickOff": ("SET PIECE", "KICK OFF"), "ThrowIn": ("SET PIECE", "THROW IN"),
            "FreeKick": ("SET PIECE", "FREE KICK"), "CornerKick": ("SET PIECE", "CORNER KICK"),
            "Penalty": ("SET PIECE", "PENALTY"), "GoalKick": ("PASS", "GOAL KICK")}


def _ms(stamp):
    """ISO time with offset -> milliseconds since the epoch (UTC)."""
    return int(round(dt.datetime.fromisoformat(stamp).timestamp() * 1000))


def parse_positions(path, info):
    """Stream the positions file. Returns (sections, persons):
    sections[name] = dict(N, x, y, status, poss, t_ms) of the ball, one entry per frame;
    persons[person id] = (team, [(section, N, x, y), ...]) for the players of both teams.
    Referees are skipped. Coordinates stay in metres (centre origin) here."""
    sections, persons = {}, {}
    cur = None
    buf = None
    parent = None
    for event, elem in ET.iterparse(path, events=("start", "end")):
        if event == "start":
            if elem.tag == "Positions":
                parent = elem
            elif elem.tag == "FrameSet":
                team = elem.get("TeamId")
                kind = ("ball" if team.lower() == "ball" else HOME if team == info["home"]
                        else AWAY if team == info["guest"] else None)
                cur = (elem.get("GameSection"), kind, elem.get("PersonId"))
                buf = {"N": [], "x": [], "y": [], "status": [], "poss": [], "t": []}
            continue
        if elem.tag == "Frame":
            if cur[1] is not None:
                a = elem.attrib
                buf["N"].append(int(a["N"]))
                buf["x"].append(float(a["X"]))
                buf["y"].append(float(a["Y"]))
                if cur[1] == "ball":
                    buf["status"].append(int(a.get("BallStatus", -1)))
                    buf["poss"].append(int(a.get("BallPossession", 0)))
                    buf["t"].append(a.get("T", ""))
            elem.clear()
        elif elem.tag == "FrameSet":
            section, kind, person = cur
            if kind == "ball":
                if section in sections:
                    raise ValueError("two ball FrameSets in {}".format(section))
                sections[section] = {
                    "N": np.array(buf["N"], dtype=np.int64), "x": np.array(buf["x"]),
                    "y": np.array(buf["y"]), "status": np.array(buf["status"], dtype=np.int8),
                    "poss": np.array(buf["poss"], dtype=np.int8),
                    "t_ms": np.array([_ms(t) if t else -1 for t in buf["t"]], dtype=np.int64)}
            elif kind is not None:
                entry = persons.setdefault(person, (kind, []))
                if entry[0] != kind:
                    raise ValueError("person {} in both teams".format(person))
                entry[1].append((section, np.array(buf["N"], dtype=np.int64),
                                 np.array(buf["x"]), np.array(buf["y"])))
            cur = buf = None
            elem.clear()
            if parent is not None:
                parent.clear()
    return sections, persons


def layout(sections):
    """Global frame index of each section: [(name, period, first index, N of the first frame,
    number of frames)], halves in order with HALF_GAP empty frames between them."""
    out, pos = [], 0
    for period, name in enumerate(SECTIONS, start=1):
        if name not in sections:
            continue
        N = sections[name]["N"]
        if np.any(np.diff(N) != 1):
            raise ValueError("{}: ball frame counter is not contiguous".format(name))
        if out:
            pos += HALF_GAP
        out.append((name, period, pos, int(N[0]), len(N)))
        pos += len(N)
    unknown = set(sections) - set(SECTIONS)
    if unknown:
        raise ValueError("unknown game sections: {}".format(unknown))
    return out, pos


def parse_events(path, info, lay, sections):
    """Event data -> columns of training/metrica.load_events (team H/A/None, type, subtype,
    period, start, end; 0-based frame indices of the global layout). Frames come from EventTime
    against the ball's T (the files rarely carry a frame). Events outside both halves are
    dropped; Delete events (no type, no team) too."""
    rows = []
    starts = {name: (first, sections[name]["t_ms"]) for name, _, first, _, _ in lay}
    period_of = {name: p for name, p, _, _, _ in lay}
    team_of = {info["home"]: "H", info["guest"]: "A"}
    for e in ET.parse(path).getroot().iter("Event"):
        kids = list(e)
        if not kids or kids[0].tag == "Delete" or not e.get("EventTime"):
            continue
        k = kids[0]
        t = _ms(e.get("EventTime"))
        where = None
        for name, (first, tms) in starts.items():
            if tms[0] - 2000 <= t <= tms[-1] + 2000:
                i = int(np.clip(np.searchsorted(tms, t), 0, len(tms) - 1))
                if i > 0 and abs(tms[i - 1] - t) < abs(tms[i] - t):
                    i -= 1
                where = (name, first + i)
                break
        if where is None:
            continue
        team = k.get("Team")
        if team is None and k.tag == "KickOff":
            play = k.find("Play")
            team = None if play is None else play.get("Team")
        if team is None:
            team = k.get("WinnerTeam") or k.get("TeamFouler")
        kind, sub = RESTARTS.get(k.tag, (k.tag.upper(), kids[0][0].tag.upper()
                                         if len(kids[0]) else ""))
        rows.append({"team": team_of.get(team), "type": kind, "subtype": sub,
                     "period": period_of[where[0]], "start": where[1], "end": where[1]})
    ev = pd.DataFrame(rows, columns=["team", "type", "subtype", "period", "start", "end"])
    ev = ev.sort_values(["start", "end"], kind="stable").reset_index(drop=True)
    # the first kick-off of each half is the start of its graded block (as with Metrica)
    for name, period, first, _, _ in lay:
        ko = ev.index[(ev["period"] == period) & (ev["subtype"] == "KICK OFF")]
        if len(ko) and abs(ev.at[ko[0], "start"] - first) <= 10 * FPS:
            ev.loc[ko[0], ["start", "end"]] = first
    return ev.sort_values(["start", "end"], kind="stable").reset_index(drop=True)


def parsed_path(root, game):
    return os.path.join(root, "parsed", "{}.npz".format(game))


def parse(root, game):
    """Parse one game and save its arrays under <root>/parsed (outside the repository)."""
    t0 = time.time()
    info = match_info(root, game)
    sections, persons = parse_positions(path_of(root, game, "positions"), info)
    lay, n = layout(sections)
    length, width = info["pitch"]
    offset = {name: (first, n0, cnt) for name, _, first, n0, cnt in lay}

    ball = np.full((n, 2), np.nan)
    status = np.full(n, -1, dtype=np.int8)
    poss = np.zeros(n, dtype=np.int8)
    period = np.zeros(n, dtype=np.int8)
    for name, p, first, _, cnt in lay:
        s = sections[name]
        ball[first:first + cnt, 0] = s["x"] / length + 0.5
        ball[first:first + cnt, 1] = 0.5 - s["y"] / width
        status[first:first + cnt] = s["status"]
        poss[first:first + cnt] = s["poss"]
        period[first:first + cnt] = p

    # one slot per person (substitutes included), home first, in order of first appearance
    def first_seen(item):
        team, chunks = item[1]
        return (team, min(offset[sec][0] + int(N[0]) - offset[sec][1] for sec, N, _, _ in chunks))
    order = sorted(persons.items(), key=first_seen)
    players = np.full((n, len(order), 2), np.nan)
    team = np.array([t for _, (t, _) in order], dtype=np.int8)
    for j, (_, (_, chunks)) in enumerate(order):
        for sec, N, x, y in chunks:
            first, n0, cnt = offset[sec]
            idx = first + N - n0
            ok = (N >= n0) & (N < n0 + cnt)
            players[idx[ok], j, 0] = x[ok] / length + 0.5
            players[idx[ok], j, 1] = 0.5 - y[ok] / width
    ev = parse_events(path_of(root, game, "events"), info, lay, sections)
    blocks = np.array([(p, first, first + cnt - 1) for _, p, first, _, cnt in lay], dtype=np.int64)

    out = parsed_path(root, game)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".{}.tmp.npz".format(os.getpid())
    np.savez(tmp, version=PARSE_VERSION, ball=ball, players=players, team=team, period=period,
             status=status, poss=poss, pitch=np.array(info["pitch"]), blocks=blocks,
             ev_team=ev["team"].fillna("").to_numpy(dtype="<U1"),
             ev_type=ev["type"].to_numpy(dtype=str), ev_subtype=ev["subtype"].to_numpy(dtype=str),
             ev_period=ev["period"].to_numpy(dtype=np.int64),
             ev_start=ev["start"].to_numpy(dtype=np.int64),
             ev_end=ev["end"].to_numpy(dtype=np.int64))
    os.replace(tmp, out)
    print("parsed {}: {} frames, {} slots ({} home), {} events, sections {}  ({:.0f} s)".format(
        game, n, len(team), int((team == HOME).sum()), len(ev), [x[0] for x in lay],
        time.time() - t0), flush=True)


def _parsed(root, game):
    path = parsed_path(root, game)
    if os.path.isfile(path):
        z = np.load(path)
        if int(z["version"]) == PARSE_VERSION:
            return z
    parse(root, game)
    return np.load(path)


# --- the interface of training/metrica.py -------------------------------------------------

def load_tracking(root, game, cache_dir=None):
    """Same dict as training/metrica.load_tracking (cache_dir unused: the parsed arrays live in
    <root>/parsed)."""
    from metrica import player_speed
    z = _parsed(root, game)
    pitch = tuple(float(v) for v in z["pitch"])
    players = z["players"]
    return {"ball": z["ball"], "players": players, "speed": player_speed(players, pitch=pitch),
            "team": z["team"], "pitch": pitch, "n_frames": len(z["ball"]), "game": game,
            "period": z["period"].astype(int)}


def load_events(root, game):
    """DataFrame like training/metrica.load_events: team ('H'/'A'/None), type, subtype,
    period, start, end (0-based frame indices). Restarts follow Metrica's coding (SET PIECE +
    KICK OFF / THROW IN / FREE KICK / CORNER KICK / PENALTY; goal kick = PASS + GOAL KICK)."""
    z = _parsed(root, game)
    team = z["ev_team"].astype(object)
    team[team == ""] = None
    return pd.DataFrame({"team": team, "type": z["ev_type"].astype(str),
                         "subtype": z["ev_subtype"].astype(str),
                         "period": z["ev_period"], "start": z["ev_start"], "end": z["ev_end"]})


def ground_truth(root, game, min_spell_s=0.0):
    """(labels (N,) 'H'/'A'/'D'/'X', blocks [(period, first, last)]) from the tracking's own
    BallStatus (0 dead) and BallPossession (1 home, 2 guest): the construction of the challenge's
    ground truth. Graded: every frame of each half. min_spell_s > 0: the coarse variant of
    training/gt_events.absorb_short_spells."""
    import gt_events
    z = _parsed(root, game)
    status, poss = z["status"], z["poss"]
    labels = np.full(len(status), "X", dtype="<U1")
    blocks = [(int(p), int(a), int(b)) for p, a, b in z["blocks"]]
    for _, a, b in blocks:
        s, q = status[a:b + 1], poss[a:b + 1]
        if np.any(s < 0) or np.any((s == 1) & ~np.isin(q, [1, 2])):
            raise ValueError("{}: ball frame without BallStatus / BallPossession".format(game))
        labels[a:b + 1] = np.where(s == 0, "D", np.where(q == 1, "H", "A"))
    if min_spell_s > 0:
        labels = gt_events.absorb_short_spells(labels, blocks, min_spell_s)
    return labels, blocks


# --- aggregate check of the ground truth --------------------------------------------------------


# --- command line -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["download", "parse"])
    parser.add_argument("--root", default=DEFAULT_ROOT)
    args = parser.parse_args()
    if args.command == "download":
        download(args.root)
    else:
        for g in GAMES:
            parse(args.root, g)


if __name__ == "__main__":
    main()
