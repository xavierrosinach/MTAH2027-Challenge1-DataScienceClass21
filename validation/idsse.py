"""IDSSE: seven professional matches (Bundesliga and 2. Bundesliga 2022/23) with DFL tracking and
event data, as a second source of OTHER games for the bench and for training (never BAR-ATH).

Source: Bassek, Rein, Weber, Memmert (2025), "An integrated dataset of spatiotemporal and event data
in elite soccer", Scientific Data. Figshare DOI 10.6084/m9.figshare.28196177, licence CC BY 4.0,
public download without registration. Positions: TRACAB optical tracking, 25 Hz (Sportec Solutions
for the DFL); events: official DFL event data (collected by human operators).

    python validation/idsse.py download     # the 21 files -> ../external/idsse (never in the repo)
    python validation/idsse.py inspect      # structure of the files: tags, attributes, value counts
    python validation/idsse.py parse        # tracking + events -> ../external/idsse/parsed
    python validation/idsse.py summary      # aggregate check of the ground truth (no rows printed)

Games are named IDSSE_<match id> (e.g. IDSSE_J03WMX). `load_tracking` returns the same dict as
validation/metrica.load_tracking (0-1 coordinates, x along the length from the left goal line, y
across from the top touch line; "period" 0 between the halves), `load_events` the same columns as
validation/metrica.load_events, and `ground_truth` the per-frame H/A/D/X labels and graded blocks
built from the tracking's own BallStatus / BallPossession (the construction of the official ground
truth of the challenge: dead where the tracking says dead, else the team it says has the ball).
Only aggregates are printed.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
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
        raise FileNotFoundError("IDSSE file missing: {}\n  python validation/idsse.py download"
                                .format(path))
    return path


# --- download -------------------------------------------------------------------------------

def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "mtah2027-bench/1.0 (urllib)"})
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
        fh.write("Downloaded: {} by validation/idsse.py download (MTAH 2027, DataScienceClass21; "
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


def inspect(root, game, positions=True):
    info = match_info(root, game)
    print("== {}: matchinformation: pitch {} m, {} listed players, competition {}, season {}"
          .format(game, info["pitch"], len(info["players"]), info["competition"], info["season"]))

    # events: tags under each Event, their attributes (names only)
    ev = ET.parse(path_of(root, game, "events")).getroot()
    print("events root <{}>, children: {}".format(ev.tag, dict(pd.Series(
        [c.tag for c in ev]).value_counts())))
    first, attrs, depth2 = [], {}, []
    event_attrs = set()
    for e in ev.iter("Event"):
        event_attrs |= set(e.attrib)
        kids = list(e)
        if kids:
            first.append(kids[0].tag)
            attrs.setdefault(kids[0].tag, set()).update(kids[0].attrib)
            depth2 += ["{}>{}".format(kids[0].tag, k.tag) for k in kids[0]]
    print("Event attributes: {}".format(sorted(event_attrs)))
    print("EventTime format: {}".format(sorted({_mask(e.get("EventTime", ""))
                                                for e in ev.iter("Event")})[:3]))
    print("first child of Event (count):")
    print(pd.Series(first).value_counts().to_string())
    print("attributes of each first child:")
    for tag, names in sorted(attrs.items()):
        print("  {}: {}".format(tag, sorted(names)))
    print("second level (count, top 40):")
    print(pd.Series(depth2).value_counts().head(40).to_string())
    events = list(ev.iter("Event"))
    for a in ("CalculatedFrame", "StartFrame", "EndFrame", "CalculatedTimestamp"):
        print("events with {}: {} of {}".format(a, sum(a in e.attrib for e in events), len(events)))
    both = [e for e in events if "CalculatedFrame" in e.attrib and "StartFrame" in e.attrib]
    if both:
        d = np.array([int(e.get("StartFrame")) - int(e.get("CalculatedFrame")) for e in both])
        print("StartFrame - CalculatedFrame: median {}, |d| > 25 on {} of {}".format(
            np.median(d), int((np.abs(d) > 25).sum()), len(d)))
    ids = pd.Series([e.get("EventId") for e in events])
    kinds = pd.Series([e[0].tag if len(e) else "" for e in events])
    dup = ids.duplicated(keep=False)
    print("EventIds repeated: {} events; their types: {}".format(
        int(dup.sum()), dict(kinds[dup].value_counts())))
    restart_tags = ("KickOff", "ThrowIn", "FreeKick", "CornerKick", "GoalKick", "Penalty")
    for tag in restart_tags:
        m = kinds == tag
        print("{}: {} events, with a frame attribute {}".format(
            tag, int(m.sum()), int(sum(("CalculatedFrame" in e.attrib or "StartFrame" in e.attrib)
                                       for e, k in zip(events, m) if k))))
    team_vals = pd.Series([k[0].get("Team") for k in (list(e) for e in ev.iter("Event")) if k])
    print("Team attribute of the first child: home {}, guest {}, missing {}, other {}".format(
        int((team_vals == info["home"]).sum()), int((team_vals == info["guest"]).sum()),
        int(team_vals.isna().sum()),
        int((~team_vals.isin([info["home"], info["guest"]]) & team_vals.notna()).sum())))
    if not positions:
        return

    # positions: framesets by section and kind, frame attributes, value counts
    t0 = time.time()
    sets, frame_attrs, values = [], {}, {}
    ranges = {}
    kind = section = None
    n_list = []
    for event, elem in ET.iterparse(path_of(root, game, "positions"), events=("start", "end")):
        if event == "start":
            if elem.tag == "FrameSet":
                team = elem.get("TeamId")
                kind = ("home" if team == info["home"] else "guest" if team == info["guest"]
                        else "other:" + str(team))
                section = elem.get("GameSection")
                sets.append((section, kind, sorted(elem.attrib)))
                n_list = []
            continue
        if elem.tag == "Frame":
            key = "ball" if kind.lower() == "other:ball" else kind
            frame_attrs.setdefault(key, set()).update(elem.attrib)
            n_list.append(int(elem.get("N")))
            for a in ("BallStatus", "BallPossession", "M"):
                if a in elem.attrib:
                    values.setdefault((key, a), {}).setdefault(elem.get(a), 0)
                    values[(key, a)][elem.get(a)] += 1
            for a in ("X", "Y", "Z", "S"):
                v = elem.get(a)
                if v is not None:
                    lo, hi = ranges.get((key, a), (np.inf, -np.inf))
                    f = float(v)
                    ranges[(key, a)] = (min(lo, f), max(hi, f))
            if "T" in elem.attrib and ("T", key) not in values:
                values[("T", key)] = {_mask(elem.get("T")): 1}
            elem.clear()
        elif elem.tag == "FrameSet":
            n = np.array(n_list)
            ranges.setdefault(("N", section, kind if "ball" not in kind.lower() else "ball"),
                              [])
            ranges[("N", section, kind if "ball" not in kind.lower() else "ball")].append(
                (int(n.min()), int(n.max()), len(n), int((np.diff(n) != 1).sum())))
            elem.clear()
    print("\npositions parsed in {:.0f} s; FrameSets by (GameSection, team):".format(
        time.time() - t0))
    print(pd.Series(["{} / {}".format(s, k) for s, k, _ in sets]).value_counts().to_string())
    print("FrameSet attributes: {}".format(sorted({a for _, _, at in sets for a in at})))
    for key, names in frame_attrs.items():
        print("Frame attributes ({}): {}".format(key, sorted(names)))
    for key, counts in sorted(values.items(), key=str):
        print("values {}: {}".format(key, dict(sorted(counts.items()))))
    for key, r in sorted(ranges.items(), key=str):
        if key[0] == "N":
            r = np.array(r)
            print("N {}: {} framesets, first N min {} / max {}, last N min {} / max {}, frames "
                  "{}..{}, framesets with jumps {}".format(
                      key[1:], len(r), r[:, 0].min(), r[:, 0].max(), r[:, 1].min(), r[:, 1].max(),
                      r[:, 2].min(), r[:, 2].max(), int((r[:, 3] > 0).sum())))
        else:
            print("range {}: {:.2f} .. {:.2f}".format(key, *r))


# --- parsing (streaming) ----------------------------------------------------------------------

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
    """Event data -> columns of validation/metrica.load_events (team H/A/None, type, subtype,
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


# --- the interface of validation/metrica.py -------------------------------------------------

def load_tracking(root, game, cache_dir=None):
    """Same dict as validation/metrica.load_tracking (cache_dir unused: the parsed arrays live in
    <root>/parsed)."""
    from metrica import player_speed
    z = _parsed(root, game)
    pitch = tuple(float(v) for v in z["pitch"])
    players = z["players"]
    return {"ball": z["ball"], "players": players, "speed": player_speed(players, pitch=pitch),
            "team": z["team"], "pitch": pitch, "n_frames": len(z["ball"]), "game": game,
            "period": z["period"].astype(int)}


def load_events(root, game):
    """DataFrame like validation/metrica.load_events: team ('H'/'A'/None), type, subtype,
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
    validation/gt_events.absorb_short_spells."""
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

def gt_stats(labels, blocks, fps=FPS):
    """Aggregates of a per-frame H/A/D ground truth over its graded blocks."""
    lengths, direct, stops = [], 0, 0
    for _, a, b in blocks:
        lab = labels[a:b + 1]
        change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        starts, ends = np.r_[0, change], np.r_[change, len(lab)]
        kinds = lab[starts]
        for s, e, k in zip(starts, ends, kinds):
            if k in ("H", "A"):
                lengths.append((e - s) / fps)
            elif k == "D":
                stops += 1
        direct += int(np.sum(np.isin(kinds[:-1], ["H", "A"]) & np.isin(kinds[1:], ["H", "A"])
                             & (kinds[:-1] != kinds[1:])))
    graded = labels != "X"
    lengths = np.array(lengths)
    minutes = graded.sum() / fps / 60
    return {"minutes": minutes, "live frames": int(np.isin(labels, ["H", "A"]).sum()),
            "dead %": 100 * float((labels[graded] == "D").mean()),
            "possessions": len(lengths), "poss/min": len(lengths) / minutes,
            "<2s %": 100 * float((lengths < 2).mean()),
            "2-5s %": 100 * float(((lengths >= 2) & (lengths < 5)).mean()),
            "mean s": float(lengths.mean()), "median s": float(np.median(lengths)),
            "H<->A direct": direct, "stoppages": stops}


def tracking_checks(tr, gt, blocks):
    """Label sanity (aggregates): nearest player within 1 m of the ball on live frames has the
    ground-truth team; home team's mean x per half; ball seen / outside the pitch, alive vs dead."""
    from possession.context import edge_distance_m
    ball, P, team = tr["ball"], tr["players"], tr["team"]
    scale = np.array(tr["pitch"])
    live = np.isin(gt, ["H", "A"])
    dead = gt == "D"
    seen = ~np.isnan(ball[:, 0])
    d = np.hypot(*np.moveaxis((P - ball[:, None, :]) * scale, -1, 0))
    d = np.where(np.isnan(d), np.inf, d)
    j = np.argmin(d, axis=1)
    near = d[np.arange(len(d)), j] < 1.0
    m = live & seen & near
    agree = np.mean(np.where(team[j[m]] == HOME, "H", "A") == gt[m])
    out = {"near<1m agrees %": 100 * float(agree), "near<1m share of live %":
           100 * float(m.sum() / max(1, (live & seen).sum()))}
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for p, a, b in blocks[:2]:
                hx = float(np.nanmean(P[a:b + 1, team == HOME, 0]))
                ax = float(np.nanmean(P[a:b + 1, team == AWAY, 0]))
                out["home x P{}".format(p)] = hx
                out["away x P{}".format(p)] = ax
                # control.attack_sign takes home_x < 0.5 as "home attacks towards x = 1"; the
                # team nearer x = 0 on average (home_x < away_x) is the robust reading
                out["attack_sign ok P{}".format(p)] = int((hx < 0.5) == (hx < ax))
    out["ball seen live %"] = 100 * float(seen[live].mean())
    out["ball seen dead %"] = 100 * float(seen[dead].mean())
    edge = edge_distance_m(ball[:, 0], ball[:, 1], tr["pitch"])
    out["ball out >1m, dead seen %"] = 100 * float(np.mean(edge[dead & seen] < -1.0))
    out["ball out >1m, live seen %"] = 100 * float(np.mean(edge[live & seen] < -1.0))
    return out


def restart_alignment(events, gt):
    """Share of restart events with a dead -> alive change of the ground truth within 1 / 3 s."""
    import gt_events
    r = events[gt_events.is_restart(events)]
    alive = np.isin(gt, ["H", "A"]).astype(np.int8)
    up = np.flatnonzero((alive[1:] == 1) & (gt[:-1] == "D")) + 1
    if not len(r) or not len(up):
        return {"restarts": len(r)}
    signed = np.array([(up - s)[np.argmin(np.abs(up - s))] for s in r["start"]])
    dist = np.abs(signed)
    return {"restarts": len(r), "D->alive within 1 s %": 100 * float(np.mean(dist <= FPS)),
            "within 3 s %": 100 * float(np.mean(dist <= 3 * FPS)),
            "median |frames|": float(np.median(dist)),
            "median GT-event frames": float(np.median(signed))}


def summary(root, games, metrica_games=True):
    import bench as B
    import gt_events
    import metrica
    rows, checks = [], []
    for g in games:
        tr = load_tracking(root, g)
        gt, blocks = ground_truth(root, g)
        coarse, _ = ground_truth(root, g, 2.0)
        ev = load_events(root, g)
        rows.append({"game": g, "GT": "BallStatus/BallPossession", **gt_stats(gt, blocks),
                     "coarse poss": gt_stats(coarse, blocks)["possessions"]})
        checks.append({"game": g, **tracking_checks(tr, gt, blocks), **restart_alignment(ev, gt)})
    if metrica_games:
        bench = B.load_bench(os.path.join(HERE, "bench.yaml"))
        cache = B.load_config(os.path.join(REPO, "config.yaml"))["paths"]["cache_dir"]
        for g in metrica.GAMES:
            tr = metrica.load_tracking(bench["metrica_dir"], g, cache_dir=cache)
            ev = metrica.load_events(bench["metrica_dir"], g)
            gt, blocks = gt_events.ground_truth(ev, tr["n_frames"])
            coarse, _ = gt_events.ground_truth(ev, tr["n_frames"], 2.0)
            rows.append({"game": g, "GT": "events (gt_events.py)", **gt_stats(gt, blocks),
                         "coarse poss": gt_stats(coarse, blocks)["possessions"]})
            checks.append({"game": g, **tracking_checks(tr, gt, blocks)})
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.float_format", "{:.2f}".format):
        df = pd.DataFrame(rows).set_index("game")
        print("ground truth per game (graded frames):")
        print(df.to_string())
        print("\nmean by source:")
        print(df.groupby("GT").mean(numeric_only=True).to_string())
        print("\nlabel and tracking checks (clean tracking):")
        print(pd.DataFrame(checks).set_index("game").to_string())


# --- command line -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["download", "inspect", "parse", "summary"])
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--games", nargs="+", default=None)
    parser.add_argument("--events-only", action="store_true", help="inspect: skip the positions")
    parser.add_argument("--no-metrica", action="store_true",
                        help="summary: without the Metrica games for comparison")
    args = parser.parse_args()
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import bench  # noqa: F401  (sets the paths to src/ and the official repo)
    games = args.games or list(GAMES)
    if args.command == "download":
        download(args.root)
    elif args.command == "inspect":
        for g in games:
            inspect(args.root, g, positions=not args.events_only)
    elif args.command == "parse":
        for g in games:
            parse(args.root, g)
    elif args.command == "summary":
        summary(args.root, games, metrica_games=not args.no_metrica)


if __name__ == "__main__":
    main()
