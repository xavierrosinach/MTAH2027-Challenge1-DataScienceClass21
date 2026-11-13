"""SkillCorner open data: A-League 2024/25 matches with broadcast tracking, as a third source of
OTHER games for the bench and for training (never BAR-ATH).

Source: https://github.com/SkillCorner/opendata (MIT licence, public, no registration), pinned to
commit COMMIT below: 20 matches (the README still says 10: ten were added on 2026-09-08).
Broadcast tracking (computer vision on the TV feed, 10 fps; players off camera are extrapolated
and flagged), plus SkillCorner's own algorithmic "dynamic events" and "phases of play". No human
labels stated.

    python validation/skillcorner.py licence     # licence, README terms, commit, file list (no data)
    python validation/skillcorner.py download    # the match files -> ../external/skillcorner
    python validation/skillcorner.py inspect     # structure of the files: keys, value counts
    python validation/skillcorner.py parse       # -> ../external/skillcorner/parsed (outside the repo)
    python validation/skillcorner.py summary     # aggregate check of the ground truth (no rows)
    python validation/skillcorner.py summary --compare   # ... next to the IDSSE and Metrica games
    python validation/skillcorner.py diagnose    # possession group vs events, offsets, axes

Games are named SKC_<match id> (e.g. SKC_1886347). `load_tracking`, `load_events` and
`ground_truth` return the same as validation/idsse.py and validation/metrica.py: 0-1 coordinates
(x along the length from the left goal line, y across from the top touch line), 25 fps, "period" 0
between the halves (HALF_GAP empty frames), graded blocks = every frame of each half.

Parsing decisions:
- 10 -> 25 fps: output frame k of a half is at 0.4 k of the 10 fps frames; positions are
  interpolated linearly between the two neighbouring samples only when both exist (an exact sample
  needs only itself), so a missing ball is never filled in.
- Ball: only `is_detected` true counts as seen. SkillCorner also gives positions it did not detect
  (is_detected false: estimated); those are dropped, as is the ball without coordinates.
- Players: `is_detected` false means "extrapolated" (off camera, position estimated). They are
  dropped (not seen) unless KEEP_EXTRAPOLATED: our tracking only has players it sees, and the bench
  degradation removes more. The parsed arrays keep both, so the choice costs no re-parse.
- Axes: x along the length, y up (checked: event coordinates are in the attacking team's frame,
  where wide_left has y > 0, and they equal the tracking's when the team attacks left to right);
  normalised as validation/idsse.py: x / length + 0.5, 0.5 - y / width.
Ground truth (SkillCorner's algorithms; no ball status in the tracking):
- D outside the phases of play ("only defined when the ball is in play"); frames of a phase are
  frame_start..frame_end inclusive.
- Alive, source "group" (GT_SOURCE, the default): the team of the last non-null
  `possession.group` of the tracking, held until the other team has it; the frames of a live
  stretch before its first group get that first group (the restarting team); a live stretch
  without any group takes the phase's team_in_possession (counted in the summary).
- Alive, source "events" (bench: `skillcorner_gt: events`): the same with the team of the
  player_possession events instead of the group. On SKC_1886347 (`diagnose`) the nearest detected
  player within 1 m of the detected ball is of the event team on 96.9 % of frames, of the group's
  on 85.5 %; where the two disagree, of the event team on 91.9 %.
Restarts (degraded+tags): kick-off at each half's first frame, and every player_possession with a
`game_interruption_before` (throw-in, free kick, corner, goal kick, penalty, goal -> kick-off), at
the start of its live stretch if that is at most 5 s earlier, else at the possession's start.

Only aggregates are printed.
"""

import argparse
import base64
import datetime as dt
import hashlib
import json
import os
import sys
import time
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_ROOT = os.path.normpath(os.path.join(REPO, "..", "external", "skillcorner"))
GITHUB = "SkillCorner/opendata"
API = "https://api.github.com/repos/" + GITHUB
COMMIT = "4340d274572876239c154c90bc507a9b3250a656"     # master on 2026-11-13 (2026-09-14)
LICENCE = "MIT"
FPS_IN, FPS = 10, 25
HOME, AWAY = 0, 1
HALF_GAP = 10 * FPS          # NaN frames inserted between the halves (as validation/idsse.py)
PARSE_VERSION = 2            # bump when the parsed arrays change
KEEP_EXTRAPOLATED = False    # True: keep the extrapolated (off camera) players as seen
GT_SOURCES = ("group", "events")
GT_SOURCE = "group"          # default team source of the ground truth (bench: skillcorner_gt)
RESTART_LINK_S = 5.0         # a restart is put at the start of its live stretch if within this

# the 10 matches of the 2025-10-06 release (the "10 matches" of the README) ...
RELEASE_2025 = ("1886347", "1899585", "1925299", "1953632", "1996435", "2006229", "2011166",
                "2013725", "2015213", "2017461")
# ... and the 10 added on 2026-09-08 (commit abbc3ff, "Add 10 A-League matches")
ADDED_2026 = ("1874553", "1927964", "1959846", "1986691", "1996436", "2006363", "2007448",
              "2007721", "2010085", "2016236")
MATCHES = tuple(sorted(RELEASE_2025 + ADDED_2026))
KINDS = ("match.json", "tracking_extrapolated.jsonl", "dynamic_events.csv", "phases_of_play.csv")
GAMES = tuple("SKC_" + m for m in MATCHES)


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "mtah2027-bench/1.0 (urllib)"})
    return urllib.request.urlopen(req, timeout=timeout)


def _json(url):
    with _get(url) as r:
        return json.loads(r.read().decode("utf-8"))


def file_name(match, kind):
    return "{}_{}".format(match, kind)


def match_id(game):
    if not game.startswith("SKC_") or game[4:] not in MATCHES:
        raise ValueError("not a SkillCorner game: {}".format(game))
    return game[4:]


def is_skc(game):
    return game.startswith("SKC_")


def path_of(root, game, kind):
    path = os.path.join(root, "matches", match_id(game), file_name(match_id(game), kind))
    if not os.path.isfile(path):
        raise FileNotFoundError("SkillCorner file missing: {}\n  python validation/skillcorner.py "
                                "download".format(path))
    return path


# --- licence and documentation (no data) ------------------------------------------------------

def readme():
    """The README (public documentation, no data) and the keys / ids of data/matches.json."""
    text = base64.b64decode(_json("{}/readme".format(API))["content"]).decode("utf-8")
    print(text)
    with _get("https://raw.githubusercontent.com/{}/master/data/matches.json".format(GITHUB)) as r:
        matches = json.loads(r.read().decode("utf-8"))
    print("----- data/matches.json: {} entries -----".format(len(matches)))
    for m in matches:
        print({k: (v if not isinstance(v, (dict, list)) else type(v).__name__)
               for k, v in m.items()})
    print("----- commits touching data/matches.json -----")
    commits = _json("{}/commits?path=data/matches.json&per_page=50".format(API))
    for c in commits:
        print(c["sha"][:10], c["commit"]["committer"]["date"],
              c["commit"]["message"].splitlines()[0][:100])
    for c in commits:
        try:
            with _get("https://raw.githubusercontent.com/{}/{}/data/matches.json".format(
                    GITHUB, c["sha"])) as r:
                ids = sorted(m["id"] for m in json.loads(r.read().decode("utf-8")))
        except Exception as exc:  # noqa: BLE001
            ids = ["error: {}".format(exc)]
        print(c["sha"][:10], len(ids), ids)


def licence():
    repo = _json(API)
    branch = repo["default_branch"]
    head = _json("{}/commits/{}".format(API, branch))
    print("repo {}: default branch {}, licence (GitHub) {} / {}".format(
        GITHUB, branch, (repo.get("license") or {}).get("spdx_id"),
        (repo.get("license") or {}).get("name")))
    print("head commit {} ({}); pinned commit {}".format(
        head["sha"], head["commit"]["committer"]["date"], COMMIT))
    lic = _json("{}/license".format(API))
    print("licence file: {} ({} bytes)".format(lic["path"], lic["size"]))
    print("----- licence text -----")
    print(base64.b64decode(lic["content"]).decode("utf-8"))
    print("----- end licence text -----")
    text = base64.b64decode(_json("{}/readme".format(API))["content"]).decode("utf-8")
    print("README: {} lines; lines mentioning licence / terms / register / LFS / cite:".format(
        len(text.splitlines())))
    keys = ("licen", "term", "regist", "lfs", "cite", "citation", "permission", "commercial",
            "copyright", "attribution", "sign up", "account", "huggingface", "hugging face")
    for i, line in enumerate(text.splitlines(), 1):
        if any(k in line.lower() for k in keys):
            print("  {:4d}: {}".format(i, line.strip()))
    tree = _json("{}/git/trees/{}?recursive=1".format(API, head["sha"]))
    print("tree: {} entries, truncated {}".format(len(tree["tree"]), tree.get("truncated")))
    for t in tree["tree"]:
        if t["type"] == "blob":
            print("  {:>12}  {}".format(t.get("size", ""), t["path"]))


# --- download ---------------------------------------------------------------------------------

def _hash_file(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_blob_sha1(path):
    h = hashlib.sha1(b"blob %d\0" % os.path.getsize(path))
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url, dest, timeout=300):
    tmp = dest + ".partial"
    with _get(url, timeout) as r, open(tmp, "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    return tmp


def _lfs_pointer(raw):
    """(sha256, size) of a Git LFS pointer file, or None if it is real content."""
    if not raw.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return None
    fields = dict(line.split(" ", 1) for line in raw.decode("ascii").splitlines() if " " in line)
    return fields["oid"].split(":", 1)[1], int(fields["size"])


def download(root, matches, dry_run=False):
    """Every file of the chosen matches at the pinned commit, checked against the git tree (blob
    sha1) or, for the Git LFS tracking files, against the pointer (sha256 and size). Files of
    the right hash are kept; nothing else of the repository is fetched (no body pose)."""
    tree = {t["path"]: t for t in _json("{}/git/trees/{}?recursive=1".format(API, COMMIT))["tree"]
            if t["type"] == "blob"}
    plan = []
    for m in matches:
        for kind in KINDS:
            rel = "data/matches/{}/{}".format(m, file_name(m, kind))
            if rel not in tree:
                raise ValueError("{} not in the tree of {}".format(rel, COMMIT))
            plan.append((m, kind, rel, tree[rel]))
    # LFS pointers first, to know the real sizes before downloading
    lfs = {}
    for m, kind, rel, t in plan:
        if t["size"] < 300:
            url = "https://raw.githubusercontent.com/{}/{}/{}".format(GITHUB, COMMIT, rel)
            with _get(url) as r:
                ptr = _lfs_pointer(r.read())
            if ptr:
                lfs[rel] = ptr
    total = sum(lfs[r][1] if r in lfs else t["size"] for _, _, r, t in plan)
    print("{} files of {} matches at {}: {} in Git LFS, {:.1f} MB in all".format(
        len(plan), len(matches), COMMIT[:10], len(lfs), total / 1e6), flush=True)
    if dry_run:
        for rel, (_, size) in sorted(lfs.items()):
            print("  LFS {:>10.1f} MB  {}".format(size / 1e6, rel))
        return
    rows = []
    for m, kind, rel, t in plan:
        dest = os.path.join(root, "matches", m, file_name(m, kind))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if rel in lfs:
            sha, size = lfs[rel]
            url = "https://media.githubusercontent.com/media/{}/{}/{}".format(GITHUB, COMMIT, rel)
            check = lambda p, sha=sha, size=size: (os.path.getsize(p) == size  # noqa: E731
                                                   and _hash_file(p, "sha256") == sha)
            ref = "sha256:" + sha
        else:
            url = "https://raw.githubusercontent.com/{}/{}/{}".format(GITHUB, COMMIT, rel)
            check = lambda p, sha=t["sha"]: _git_blob_sha1(p) == sha  # noqa: E731
            ref = "git-sha1:" + t["sha"]
        if os.path.isfile(dest) and check(dest):
            print("have  {} ({:.1f} MB)".format(rel, os.path.getsize(dest) / 1e6), flush=True)
        else:
            t0 = time.time()
            tmp = _fetch(url, dest)
            if not check(tmp):
                raise IOError("{}: size or hash mismatch ({} bytes)".format(rel, os.path.getsize(tmp)))
            os.replace(tmp, dest)
            print("got   {} ({:.1f} MB, {:.0f} s)".format(rel, os.path.getsize(dest) / 1e6,
                                                        time.time() - t0), flush=True)
        rows.append((rel, os.path.getsize(dest), ref))
    write_source(root, rows)


def write_source(root, rows):
    lic = base64.b64decode(_json("{}/license".format(API))["content"]).decode("utf-8")
    path = os.path.join(root, "SOURCE.txt")
    old = []
    if os.path.isfile(path):   # keep the rows of matches downloaded by an earlier call
        with open(path, encoding="utf-8") as fh:
            body = fh.read().split("\nfile\tbytes\thash\n", 1)
        if len(body) == 2:
            old = [ln.split("\t") for ln in body[1].splitlines() if ln.strip()]
    have = {r[0] for r in rows}
    rows = sorted([tuple(r) for r in old if r[0] not in have] + [tuple(map(str, r)) for r in rows])
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("SkillCorner open data (broadcast tracking, A-League 2024/25)\n")
        fh.write("Source: https://github.com/{} at commit {}\n".format(GITHUB, COMMIT))
        fh.write("Licence: {} (the repository's LICENSE, copied below). The README asks users to "
                 "credit SkillCorner.\n".format(LICENCE))
        fh.write("Access: public GitHub repository, no registration. Tracking files are in Git LFS "
                 "(https://media.githubusercontent.com/media/...); the rest from "
                 "raw.githubusercontent.com.\n")
        fh.write("Content: per match {}_match.json, _tracking_extrapolated.jsonl (10 fps), "
                 "_dynamic_events.csv, _phases_of_play.csv; all produced by SkillCorner's "
                 "algorithms (computer vision on the broadcast video), no human annotation "
                 "stated. Body pose (Hugging Face) not downloaded.\n".format("<id>"))
        fh.write("Matches: 10 of the 2025-10-06 release ({}) and 10 added on 2026-09-08 ({}).\n"
                 .format(", ".join(RELEASE_2025), ", ".join(ADDED_2026)))
        fh.write("Downloaded: {} by validation/skillcorner.py download (MTAH 2027, "
                 "DataScienceClass21; other matches, used for validation and training only)\n"
                 .format(dt.datetime.now().strftime("%Y-%m-%d %H:%M")))
        fh.write("\n----- LICENSE -----\n{}\n----- end LICENSE -----\n".format(lic.strip()))
        fh.write("\nfile\tbytes\thash\n")
        for row in rows:
            fh.write("\t".join(row) + "\n")
    print("{} files listed in {}".format(len(rows), path))


# --- structure of the files (aggregates only) -------------------------------------------------

def _mask(value):
    """A value's shape without its content: digits -> d, letters -> a."""
    return "".join("d" if c.isdigit() else ("a" if c.isalpha() else c) for c in str(value))


def _shape(obj):
    """Keys and value types of a JSON object, recursively for dicts (lists: type of the first)."""
    if isinstance(obj, dict):
        return {k: _shape(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return ["list[{}] of".format(len(obj)), _shape(obj[0]) if obj else None]
    return type(obj).__name__


def inspect(root, game):
    with open(path_of(root, game, "match.json"), encoding="utf-8") as fh:
        info = json.load(fh)
    print("== {}: match.json top-level keys and types".format(game))
    for k, v in info.items():
        s = _shape(v)
        if isinstance(v, list):
            s = "list[{}] of {}".format(len(v), sorted(v[0].keys()) if v and isinstance(v[0], dict)
                                        else type(v[0]).__name__ if v else "-")
        elif isinstance(v, dict):
            s = "dict {}".format(sorted(v.keys()))
        print("  {}: {}".format(k, s))
    for k in ("pitch_length", "pitch_width", "home_team_side", "status", "match_periods"):
        if k in info:
            print("  value {}: {}".format(k, info[k] if not isinstance(info[k], list)
                                          else [_shape(x) for x in info[k]][:2]))
    home, away = info["home_team"]["id"], info["away_team"]["id"]
    pl = info.get("players") or []
    print("  players: {} (home {}, away {}, other {})".format(
        len(pl), sum(p.get("team_id") == home for p in pl), sum(p.get("team_id") == away for p in pl),
        sum(p.get("team_id") not in (home, away) for p in pl)))
    if pl:
        print("  player keys: {}".format(_shape(pl[0])))

    # tracking: one JSON object per line
    t0 = time.time()
    keys, periods, ts_mask = Counter(), Counter(), Counter()
    ball_keys, ball_det, ball_null = Counter(), Counter(), Counter()
    poss_keys, groups, poss_pid_null = Counter(), Counter(), Counter()
    player_keys, det, n_players = Counter(), Counter(), []
    frames = {}
    zs = []
    n = 0
    with open(path_of(root, game, "tracking_extrapolated.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            fr = json.loads(line)
            n += 1
            keys.update(fr.keys())
            p = fr.get("period")
            periods[p] += 1
            ts_mask[_mask(fr.get("timestamp"))] += 1
            frames.setdefault(p, []).append(fr["frame"])
            b = fr.get("ball_data") or {}
            ball_keys.update(b.keys())
            ball_det[(p is not None, b.get("is_detected"))] += 1
            ball_null[(p is not None, b.get("x") is None)] += 1
            if b.get("z") is not None:
                zs.append(b["z"])
            q = fr.get("possession") or {}
            poss_keys.update(q.keys())
            groups[(p is not None, q.get("group"))] += 1
            poss_pid_null[(q.get("group") is not None, q.get("player_id") is None)] += 1
            pd_ = fr.get("player_data") or []
            n_players.append(len(pd_))
            for e in pd_:
                player_keys.update(e.keys())
                det[(p is not None, e.get("is_detected"))] += 1
    print("\ntracking: {} lines read in {:.0f} s".format(n, time.time() - t0))
    print("frame keys: {}".format(dict(keys)))
    print("period: {}".format(dict(periods)))
    print("timestamp formats: {}".format(dict(ts_mask.most_common(4))))
    for p, f in frames.items():
        f = np.array(f)
        d = np.diff(f)
        print("period {}: {} lines, frame {}..{}, steps {}".format(
            p, len(f), f.min(), f.max(), dict(Counter(d.tolist()).most_common(4))))
    print("ball_data keys: {}".format(dict(ball_keys)))
    print("ball is_detected by (in a period, value): {}".format(dict(ball_det)))
    print("ball x is None by (in a period, None): {}".format(dict(ball_null)))
    if zs:
        z = np.array(zs, dtype=float)
        print("ball z: {} values, quantiles 1/50/99 % {}".format(
            len(z), np.round(np.nanpercentile(z, [1, 50, 99]), 2).tolist()))
    print("possession keys: {}".format(dict(poss_keys)))
    print("possession group by (in a period, group): {}".format(dict(groups)))
    print("possession player_id None by (group set, None): {}".format(dict(poss_pid_null)))
    print("player_data entry keys: {}".format(dict(player_keys)))
    print("player is_detected by (in a period, value): {}".format(dict(det)))
    npl = np.array(n_players)
    print("players per line: quantiles 0/10/50/90/100 % {}".format(
        np.percentile(npl, [0, 10, 50, 90, 100]).tolist()))

    # phases of play
    ph = pd.read_csv(path_of(root, game, "phases_of_play.csv"))
    print("\nphases_of_play: {} rows, columns {}".format(len(ph), list(ph.columns)))
    for c in ph.columns:
        if not pd.api.types.is_numeric_dtype(ph[c]) and ph[c].nunique() <= 20:
            print("  {}: {}".format(c, {str(k): int(v) for k, v in
                                        ph[c].value_counts(dropna=False).items()}))
    print("  team_in_possession_id is home / away / other: {} / {} / {}".format(
        int((ph["team_in_possession_id"] == home).sum()),
        int((ph["team_in_possession_id"] == away).sum()),
        int((~ph["team_in_possession_id"].isin([home, away])).sum())))
    if {"frame_start", "frame_end"} <= set(ph.columns):
        ph = ph.sort_values("frame_start")
        gap = ph["frame_start"].to_numpy()[1:] - ph["frame_end"].to_numpy()[:-1]
        print("  length (frames) quantiles 0/50/100 %: {}".format(
            np.percentile(ph["frame_end"] - ph["frame_start"], [0, 50, 100]).tolist()))
        print("  next start - end: {}".format(dict(Counter(np.clip(gap, -5, 30).tolist())
                                               .most_common(12))))

    # dynamic events
    ev = pd.read_csv(path_of(root, game, "dynamic_events.csv"), low_memory=False)
    print("\ndynamic_events: {} rows, {} columns".format(len(ev), len(ev.columns)))
    for c in ("event_type", "event_subtype", "start_type", "end_type", "game_interruption_before",
              "game_interruption_after", "pass_outcome"):
        if c in ev.columns:
            print("  {}: {}".format(c, {str(k): int(v) for k, v in
                                        ev[c].value_counts(dropna=False).items()}))
    pp = ev[ev["event_type"] == "player_possession"]
    print("  player_possession rows: {}; team home / away / other {} / {} / {}".format(
        len(pp), int((pp["team_id"] == home).sum()), int((pp["team_id"] == away).sum()),
        int((~pp["team_id"].isin([home, away])).sum())))
    for c in ("game_interruption_before", "game_interruption_after", "start_type"):
        if c in pp.columns:
            print("  player_possession {}: {}".format(c, {str(k): int(v) for k, v in
                                                          pp[c].value_counts(dropna=False).items()}))
    print("  frame_end - frame_start of player_possession, quantiles 0/50/100 %: {}".format(
        np.percentile(pp["frame_end"] - pp["frame_start"], [0, 50, 100]).tolist()))


# --- parsing (streaming) ----------------------------------------------------------------------

def match_info(root, game):
    with open(path_of(root, game, "match.json"), encoding="utf-8") as fh:
        info = json.load(fh)
    home, away = info["home_team"]["id"], info["away_team"]["id"]
    players = {p["id"]: (HOME if p["team_id"] == home else AWAY) for p in info["players"]
               if p["team_id"] in (home, away)}
    periods = {int(m["period"]): (int(m["start_frame"]), int(m["end_frame"]))
               for m in info.get("match_periods") or []}
    return {"pitch": (float(info["pitch_length"]), float(info["pitch_width"])), "home": home,
            "away": away, "players": players, "periods": periods,
            "home_side": list(info.get("home_team_side") or [])}


def _count_lines(path):
    with open(path, "rb") as fh:
        return sum(1 for line in fh if line.strip())


def parse_tracking(path, info, max_slots=64):
    """Stream the tracking file (one JSON object per line). Per line: frame, period (0 = none),
    ball (m, NaN without coordinates), ball state (1 detected, 0 not detected, -1 no coordinates),
    possession group (1 home, 2 away, 0 none); per player slot: position (m) and detected flag.
    Returns arrays and the player ids of the slots, in order of first appearance."""
    n = _count_lines(path)
    frame = np.full(n, -1, dtype=np.int64)
    period = np.zeros(n, dtype=np.int8)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    bstate = np.full(n, -1, dtype=np.int8)
    group = np.zeros(n, dtype=np.int8)
    pos = np.full((n, max_slots, 2), np.nan, dtype=np.float32)
    det = np.zeros((n, max_slots), dtype=bool)
    slots = {}
    odd = Counter()
    i = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            fr = json.loads(line)
            frame[i] = fr["frame"]
            period[i] = fr.get("period") or 0
            b = fr.get("ball_data") or {}
            if b.get("x") is not None and b.get("y") is not None:
                ball[i] = (b["x"], b["y"])
                bstate[i] = 1 if b.get("is_detected") else 0
            g = (fr.get("possession") or {}).get("group")
            group[i] = 1 if g == "home team" else 2 if g == "away team" else 0
            if g not in (None, "home team", "away team"):
                odd["group " + str(g)] += 1
            for e in fr.get("player_data") or []:
                pid = e.get("player_id")
                if pid not in info["players"]:
                    odd["player not in match.json"] += 1
                    continue
                if e.get("x") is None or e.get("y") is None:
                    odd["player without x/y"] += 1
                    continue
                j = slots.setdefault(pid, len(slots))
                pos[i, j] = (e["x"], e["y"])
                det[i, j] = bool(e.get("is_detected"))
            i += 1
    S = len(slots)
    return {"frame": frame, "period": period, "ball": ball, "bstate": bstate, "group": group,
            "pos": pos[:, :S], "det": det[:, :S], "ids": list(slots), "odd": dict(odd)}


def _resample(a, ok, n_out):
    """10 -> 25 fps: output k at 0.4 k of the input; linear between the two neighbours only when
    both are ok (an exact sample needs only itself); NaN elsewhere. a: (n, ...) with ok (n, ...)
    of a's shape without its last axis."""
    k = np.arange(n_out)
    i = (2 * k) // 5
    w = ((2 * k) % 5) / 5.0
    j = np.minimum(i + 1, len(a) - 1)
    a0 = np.nan_to_num(a.astype(np.float64))
    wb = w.reshape((-1,) + (1,) * (a.ndim - 1))
    out = a0[i] * (1 - wb) + a0[j] * wb
    good = ok[i] & ((w == 0).reshape((-1,) + (1,) * (ok.ndim - 1)) | ok[j])
    out[~good] = np.nan
    return out, good


def _nearest10(n_out, n10):
    """Index of the nearest 10 fps sample of each 25 fps frame (0.4 k never ends in .5)."""
    return np.minimum((4 * np.arange(n_out) + 5) // 10, n10 - 1)


def _runs(mask):
    """(starts, ends) of the runs of True (end exclusive)."""
    m = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(m == 1), np.flatnonzero(m == -1)


INTERRUPTIONS = {"throw_in": ("SET PIECE", "THROW IN"), "free_kick": ("SET PIECE", "FREE KICK"),
                 "corner": ("SET PIECE", "CORNER KICK"), "penalty": ("SET PIECE", "PENALTY"),
                 "goal_kick": ("PASS", "GOAL KICK"), "goal": ("SET PIECE", "KICK OFF")}


def parsed_path(root, game):
    return os.path.join(root, "parsed", "{}.npz".format(game))


def parse(root, game):
    """Parse one game and save its 25 fps arrays, ground truth, restarts and parse statistics
    under <root>/parsed (outside the repository). Prints aggregates only."""
    t0 = time.time()
    info = match_info(root, game)
    length, width = info["pitch"]
    tr = parse_tracking(path_of(root, game, "tracking_extrapolated.jsonl"), info)
    ph = pd.read_csv(path_of(root, game, "phases_of_play.csv"))
    ev = pd.read_csv(path_of(root, game, "dynamic_events.csv"), low_memory=False,
                     usecols=["event_type", "frame_start", "frame_end", "period", "team_id",
                              "game_interruption_before", "attacking_side", "channel_start",
                              "y_start"])
    team_code = {info["home"]: 1, info["away"]: 2}
    stats = Counter()
    stats.update({"odd " + k: v for k, v in tr["odd"].items()})

    # slots: home first, each team in order of first appearance
    present = ~np.isnan(tr["pos"][..., 0])
    first = np.where(present.any(axis=0), present.argmax(axis=0), len(present))
    order = sorted(range(len(tr["ids"])), key=lambda j: (info["players"][tr["ids"][j]], first[j]))
    team = np.array([info["players"][tr["ids"][j]] for j in order], dtype=np.int8)
    pos_all, det_all = tr["pos"][:, order], tr["det"][:, order]

    periods = sorted(int(p) for p in np.unique(tr["period"]) if p > 0)
    lay, n = [], 0
    for p in periods:
        rows = np.flatnonzero(tr["period"] == p)
        f0, f1 = int(tr["frame"][rows].min()), int(tr["frame"][rows].max())
        n10 = f1 - f0 + 1
        n_out = 5 * (n10 - 1) // 2 + 1
        if lay:
            n += HALF_GAP
        lay.append((p, rows, f0, f1, n10, n, n_out))
        n += n_out
        if p in info["periods"]:
            a, b = info["periods"][p]
            stats["match_periods |start diff| frames"] = max(
                stats["match_periods |start diff| frames"], abs(a - f0))
            stats["match_periods |end diff| frames"] = max(
                stats["match_periods |end diff| frames"], abs(b - f1))

    S = len(order)
    ball = np.full((n, 2), np.nan)
    ball_any = np.zeros(n, dtype=bool)          # a ball position, detected or not (stats only)
    players = np.full((n, S, 2), np.nan, dtype=np.float32)
    pdet = np.zeros((n, S), dtype=bool)
    period = np.zeros(n, dtype=np.int8)
    labels = {s: np.full(n, "X", dtype="<U1") for s in GT_SOURCES}
    blocks, restarts = [], []
    for p, rows, f0, f1, n10, start, n_out in lay:
        idx = tr["frame"][rows] - f0
        if len(np.unique(idx)) != len(idx):
            raise ValueError("{}: repeated frame numbers in period {}".format(game, p))
        stats["missing 10 fps frames"] += n10 - len(idx)
        # inputs on the 10 fps grid of the half (frames the file skips stay empty)
        b10 = np.full((n10, 2), np.nan)
        bs10 = np.full(n10, -1, dtype=np.int8)
        g10 = np.zeros(n10, dtype=np.int8)
        p10 = np.full((n10, S, 2), np.nan, dtype=np.float32)
        d10 = np.zeros((n10, S), dtype=bool)
        b10[idx] = tr["ball"][rows]
        bs10[idx] = tr["bstate"][rows]
        g10[idx] = tr["group"][rows]
        p10[idx] = pos_all[rows]
        d10[idx] = det_all[rows]
        sl = slice(start, start + n_out)

        # normalised coordinates: x from the left goal line, y from the top touch line (y up)
        bn = np.column_stack([b10[:, 0] / length + 0.5, 0.5 - b10[:, 1] / width])
        pn = np.stack([p10[..., 0] / length + 0.5, 0.5 - p10[..., 1] / width], axis=-1)
        ball[sl], _ = _resample(bn, bs10 == 1, n_out)
        _, any_ok = _resample(bn, bs10 >= 0, n_out)
        ball_any[sl] = any_ok
        has = ~np.isnan(pn[..., 0])
        pr, _ = _resample(pn, has, n_out)
        players[sl] = pr
        _, both_det = _resample(pn, has & d10, n_out)
        pdet[sl] = both_det
        period[sl] = p

        # ground truth on the 10 fps grid
        alive = np.zeros(n10, dtype=bool)
        phase_team = np.zeros(n10, dtype=np.int8)
        for r in ph[ph["period"] == p].itertuples():
            a, b = max(int(r.frame_start), f0) - f0, min(int(r.frame_end), f1) - f0
            if a <= b:
                alive[a:b + 1] = True
                phase_team[a:b + 1] = team_code.get(r.team_in_possession_id, 0)
        pp = ev[(ev["event_type"] == "player_possession") & (ev["period"] == p)]
        e10 = np.zeros(n10, dtype=np.int8)        # team of the player_possession events
        for r in pp.itertuples():
            a, b = max(int(r.frame_start), f0) - f0, min(int(r.frame_end), f1) - f0
            if a <= b:
                e10[a:b + 1] = team_code.get(r.team_id, 0)
        seg_a, seg_b = _runs(alive)
        stats["live stretches"] += len(seg_a)
        owners = {}
        for src, x10 in (("group", g10), ("events", e10)):
            owner = np.zeros(n10, dtype=np.int8)
            for a, b in zip(seg_a, seg_b):
                g = x10[a:b]
                nz = np.flatnonzero(g)
                if len(nz):
                    last = np.full(b - a, -1)
                    last[nz] = nz
                    last = np.maximum.accumulate(last)
                    owner[a:b] = np.where(last >= 0, g[np.maximum(last, 0)], g[nz[0]])
                    stats["{}: live frames before the first one (10 fps)".format(src)] += int(nz[0])
                else:
                    owner[a:b] = phase_team[a:b]
                    stats["{}: live stretches without any".format(src)] += 1
                    stats["{}: live frames from the phase team (10 fps)".format(src)] += b - a
            if np.any(alive & (owner == 0)):
                raise ValueError("{}: live frame without a team ({})".format(game, src))
            lab10 = np.where(alive, np.where(owner == 1, "H", "A"), "D")
            labels[src][sl] = lab10[_nearest10(n_out, n10)]
            stats["{}: live frames GT = phase team".format(src)] += int(
                np.sum(alive & (owner == phase_team)))
            owners[src] = owner
        owner = owners["events"]
        stats["live frames GT group = GT events"] += int(
            np.sum(alive & (owners["group"] == owners["events"])))
        blocks.append((p, start, start + n_out - 1))

        # aggregates of the inputs (10 fps, this half)
        stats["frames 10 fps"] += n10
        stats["frames 10 fps live"] += int(alive.sum())
        stats["ball detected live"] += int(np.sum((bs10 == 1) & alive))
        stats["ball detected dead"] += int(np.sum((bs10 == 1) & ~alive))
        stats["ball undetected position live"] += int(np.sum((bs10 == 0) & alive))
        stats["ball no position live"] += int(np.sum((bs10 == -1) & alive))
        stats["group null live"] += int(np.sum((g10 == 0) & alive))
        stats["group set dead"] += int(np.sum((g10 != 0) & ~alive))
        stats["frames without players"] += int(np.sum(~has.any(axis=1)))
        stats["player samples detected"] += int(np.sum(has & d10))
        stats["player samples extrapolated"] += int(np.sum(has & ~d10))

        # restarts: kick-off (team of the first player possession), then the player
        # possessions after an interruption
        def at25(r10):
            return start + min(n_out - 1, (5 * int(r10) + 1) // 2)
        if len(seg_a):
            restarts.append((("H" if owner[seg_a[0]] == 1 else "A"), "SET PIECE", "KICK OFF", p,
                             start))
        taken = {int(seg_a[0])} if len(seg_a) else set()
        for r in pp.itertuples():
            gi = r.game_interruption_before
            if not isinstance(gi, str):
                continue
            kind = gi.rsplit("_", 1)[0]
            if kind not in INTERRUPTIONS:
                stats["interruption not used: " + gi] += 1
                continue
            mine = team_code.get(r.team_id, 0)
            if not mine:
                continue
            # "<kind>_for": the restart is for the possessing player's team; a kick-off after
            # "goal_against" is taken by that team, after "goal_for" by the other one
            ours = gi.endswith("_for") != (kind == "goal")
            code = mine if ours else 3 - mine
            f = int(r.frame_start) - f0
            if not 0 <= f < n10:
                continue
            k = np.searchsorted(seg_a, f, side="right") - 1
            if k >= 0 and f - seg_a[k] <= RESTART_LINK_S * FPS_IN:
                r10 = int(seg_a[k])
                stats["restarts at a live-stretch start"] += 1
            else:
                r10 = f
                stats["restarts without a stoppage before"] += 1
            if r10 in taken:
                stats["restarts repeated"] += 1
                continue
            taken.add(r10)
            t, sub = INTERRUPTIONS[kind]
            restarts.append(("H" if code == 1 else "A", t, sub, p, at25(r10)))
            stats["restart " + sub] += 1

        # the possession events against the group ground truth (mid frame)
        mids = ((pp["frame_start"] + pp["frame_end"]) // 2 - f0).to_numpy()
        teams = pp["team_id"].map(team_code).fillna(0).to_numpy()
        ok = (mids >= 0) & (mids < n10) & (teams > 0)
        stats["player possessions"] += int(ok.sum())
        stats["player possessions alive"] += int(alive[mids[ok]].sum())
        stats["player possessions = GT group team"] += int(np.sum(
            alive[mids[ok]] & (owners["group"][mids[ok]] == teams[ok])))

        # attacking direction: the labeller's relative reading (detected players, home mean x
        # below the away mean x -> home attacks towards x = 1) against match.json
        side = info["home_side"][p - 1] if p - 1 < len(info["home_side"]) else None
        if side in ("left_to_right", "right_to_left"):
            px = np.where(d10 & has, pn[..., 0], np.nan)
            with np.errstate(invalid="ignore"):
                hx = np.nanmean(px[:, team == HOME])
                ax = np.nanmean(px[:, team == AWAY])
            stats["direction checks"] += 1
            stats["direction agrees with match.json"] += int((hx < ax) == (side == "left_to_right"))
    # y axis: event coordinates are in the attacking team's frame (x towards its goal); there a
    # team's left wing is at y > 0 if y points up
    w = ev[ev["channel_start"].isin(["wide_left", "wide_right"]) & (ev["y_start"] != 0)].dropna(
        subset=["y_start"])
    stats["y-up check events"] = len(w)
    stats["y-up check agree"] = int(np.sum(np.sign(w["y_start"]) == np.where(
        w["channel_start"] == "wide_left", 1, -1)))

    events = pd.DataFrame(restarts, columns=["team", "type", "subtype", "period", "start"])
    events["end"] = events["start"]
    events = events.sort_values(["start", "end"], kind="stable").reset_index(drop=True)

    out = parsed_path(root, game)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".{}.tmp.npz".format(os.getpid())
    np.savez(tmp, version=PARSE_VERSION, ball=ball, ball_any=ball_any, players=players,
             pdet=pdet, team=team, period=period, labels_group=labels["group"],
             labels_events=labels["events"], pitch=np.array(info["pitch"]),
             blocks=np.array(blocks, dtype=np.int64),
             ev_team=events["team"].to_numpy(dtype="<U1"),
             ev_type=events["type"].to_numpy(dtype=str),
             ev_subtype=events["subtype"].to_numpy(dtype=str),
             ev_period=events["period"].to_numpy(dtype=np.int64),
             ev_start=events["start"].to_numpy(dtype=np.int64),
             ev_end=events["end"].to_numpy(dtype=np.int64),
             stats=json.dumps({k: int(v) for k, v in stats.items()}))
    os.replace(tmp, out)
    print("parsed {}: {} frames, {} slots ({} home), {} restarts, periods {}  ({:.0f} s)".format(
        game, n, S, int((team == HOME).sum()), len(events), periods, time.time() - t0),
        flush=True)


def diagnose(root, game):
    """Aggregates only: the raw possession group of the tracking against SkillCorner's own
    player_possession events and phases, with time shifts (is there an offset?), the length of
    the group's runs, and the attacking direction of match.json against where the teams stand."""
    info = match_info(root, game)
    with open(path_of(root, game, "match.json"), encoding="utf-8") as fh:
        side = json.load(fh).get("home_team_side")
    tr = parse_tracking(path_of(root, game, "tracking_extrapolated.jsonl"), info)
    ph = pd.read_csv(path_of(root, game, "phases_of_play.csv"))
    ev = pd.read_csv(path_of(root, game, "dynamic_events.csv"), low_memory=False,
                     usecols=["event_type", "frame_start", "frame_end", "period", "team_id",
                              "x_start", "y_start", "attacking_side", "channel_start",
                              "player_id"])
    code = {info["home"]: 1, info["away"]: 2}
    f0 = int(tr["frame"].min())
    nf = int(tr["frame"].max()) - f0 + 1
    grp = np.zeros(nf, dtype=np.int8)
    grp[tr["frame"] - f0] = tr["group"]
    print("== {}: home_team_side {}".format(game, side))
    for p in sorted(int(x) for x in np.unique(tr["period"]) if x > 0):
        rows = tr["period"] == p
        P = tr["pos"][rows]
        hx_all = float(np.nanmean(P[:, [info["players"][i] == HOME for i in tr["ids"]], 0]))
        ax_all = float(np.nanmean(P[:, [info["players"][i] == AWAY for i in tr["ids"]], 0]))
        D = np.where(tr["det"][rows], P[..., 0], np.nan)
        hx_d = float(np.nanmean(D[:, [info["players"][i] == HOME for i in tr["ids"]]]))
        ax_d = float(np.nanmean(D[:, [info["players"][i] == AWAY for i in tr["ids"]]]))
        print("P{}: mean x (m) home / away, all players {:.1f} / {:.1f}, detected only {:.1f} / "
              "{:.1f}; relative reading home attacks +x: all {}, detected {}".format(
                  p, hx_all, ax_all, hx_d, ax_d, hx_all < ax_all, hx_d < ax_d))
    pp = ev[ev["event_type"] == "player_possession"].copy()
    pp["t"] = pp["team_id"].map(code)
    print("player_possession events: {}".format(len(pp)))
    print("raw group at the event frame (+shift, 10 fps frames): same / other / null %")
    for at in ("frame_start", "mid", "frame_end"):
        base = ((pp["frame_start"] + pp["frame_end"]) // 2 if at == "mid" else pp[at]).to_numpy()
        for s in (-20, -10, -5, -2, 0, 2, 5, 10, 20):
            i = base - f0 + s
            ok = (i >= 0) & (i < nf)
            g = grp[i[ok]]
            t = pp["t"].to_numpy()[ok]
            print("  {:11s} {:+3d}: {:5.1f} / {:5.1f} / {:5.1f}".format(
                at, s, 100 * np.mean(g == t), 100 * np.mean((g != t) & (g > 0)),
                100 * np.mean(g == 0)))
    # within the event: share of its frames with the same group
    within = []
    for r in pp.itertuples():
        g = grp[max(0, r.frame_start - f0):max(0, r.frame_end - f0) + 1]
        if len(g):
            within.append((np.mean(g == r.t), np.mean((g != r.t) & (g > 0)), len(g)))
    w = np.array(within)
    print("frames inside the events: same {:.1f} %, other {:.1f} %, (frame-weighted same {:.1f} "
          "%); events with any frame of the other team {:.1f} %".format(
              100 * w[:, 0].mean(), 100 * w[:, 1].mean(),
              100 * np.average(w[:, 0], weights=w[:, 2]), 100 * np.mean(w[:, 1] > 0)))
    # runs of the raw group (non-null) and their lengths
    change = np.flatnonzero(np.diff(grp) != 0) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, nf]
    kinds = grp[starts]
    lens = (ends - starts)[kinds > 0] / FPS_IN
    print("runs of a non-null group: {}, length s quantiles 10/50/90 % {}; < 1 s {:.1f} %".format(
        len(lens), np.round(np.percentile(lens, [10, 50, 90]), 1).tolist(),
        100 * np.mean(lens < 1)))
    seq = kinds[kinds > 0]
    print("consecutive non-null runs with a different team: {:.1f} % of {} changes".format(
        100 * np.mean(seq[1:] != seq[:-1]), len(seq) - 1))
    # phases: team in possession vs raw group
    alive = np.zeros(nf, dtype=bool)
    pteam = np.zeros(nf, dtype=np.int8)
    for r in ph.itertuples():
        a, b = int(r.frame_start) - f0, int(r.frame_end) - f0
        alive[max(a, 0):b + 1] = True
        pteam[max(a, 0):b + 1] = code.get(r.team_in_possession_id, 0)
    m = alive & (grp > 0)
    print("live frames with a group: phase team same {:.1f} %".format(
        100 * np.mean(grp[m] == pteam[m])))
    # event coordinates: are they in the attacking team's frame? (wide_left -> y sign)
    wl = ev[ev["channel_start"] == "wide_left"]
    for s in ("left_to_right", "right_to_left"):
        y = wl.loc[wl["attacking_side"] == s, "y_start"].dropna()
        print("wide_left events attacking {}: {} with y > 0 {:.1f} %".format(
            s, len(y), 100 * np.mean(y > 0)))
    # compare the event x/y with the possessor's tracking x/y at frame_start (sign relation)
    slot = {pid: j for j, pid in enumerate(tr["ids"])}
    fr_row = {int(f): i for i, f in enumerate(tr["frame"])}
    same = []
    for r in pp.itertuples():
        i, j = fr_row.get(int(r.frame_start)), slot.get(r.player_id)
        if i is None or j is None or np.isnan(tr["pos"][i, j, 0]):
            continue
        px, py = tr["pos"][i, j]
        same.append((np.sign(px) == np.sign(r.x_start), np.sign(py) == np.sign(r.y_start),
                     r.attacking_side == "left_to_right"))
    s = np.array(same)
    for flag, name in ((True, "left_to_right"), (False, "right_to_left")):
        k = s[s[:, 2] == flag]
        print("possessions attacking {}: event x sign = tracking x sign {:.1f} %, y {:.1f} % "
              "({})".format(name, 100 * k[:, 0].mean(), 100 * k[:, 1].mean(), len(k)))
    # where do the group and the events disagree? by half and by 10-minute window
    evteam = np.zeros(nf, dtype=np.int8)
    for r in pp.itertuples():
        evteam[max(0, r.frame_start - f0):max(0, r.frame_end - f0) + 1] = r.t
    per = np.zeros(nf, dtype=np.int8)
    per[tr["frame"] - f0] = tr["period"]
    m = (evteam > 0) & (grp > 0)
    print("frames inside possession events with a group: {}; group = event team {:.1f} %".format(
        int(m.sum()), 100 * np.mean(grp[m] == evteam[m])))
    for p in (1, 2):
        mp = m & (per == p)
        win = (np.flatnonzero(mp) - np.flatnonzero(per == p)[0]) // (10 * 60 * FPS_IN)
        agree = (grp[mp] == evteam[mp])
        print("  P{}: {:.1f} %; by 10-min window: {}".format(p, 100 * agree.mean(), [
            round(100 * float(agree[win == k].mean()), 1) for k in np.unique(win)]))
    # who is nearest the detected ball (detected players, < 1 m), against each source
    ball_ok = tr["bstate"] == 1
    d = np.hypot(tr["pos"][..., 0] - tr["ball"][:, None, 0], tr["pos"][..., 1] - tr["ball"][:, None, 1])
    d = np.where(tr["det"] & ball_ok[:, None] & ~np.isnan(d), d, np.inf)
    j = np.argmin(d, axis=1)
    near = d[np.arange(len(d)), j] < 1.0
    nteam = np.array([1 if info["players"][pid] == HOME else 2 for pid in tr["ids"]])[j]
    rows_f = tr["frame"] - f0
    g_rows, e_rows = grp[rows_f], evteam[rows_f]
    for name, src in (("tracking group", g_rows), ("event team", e_rows)):
        k = near & (src > 0)
        print("nearest detected player < 1 m of the detected ball has the {} team: {:.1f} % of "
              "{}".format(name, 100 * np.mean(nteam[k] == src[k]), int(k.sum())))
    both = near & (g_rows > 0) & (e_rows > 0) & (g_rows != e_rows)
    print("  where group and event team disagree ({} frames): nearest = group {:.1f} %, = event "
          "{:.1f} %".format(int(both.sum()), 100 * np.mean(nteam[both] == g_rows[both]),
                            100 * np.mean(nteam[both] == e_rows[both])))
    # the player id of the possession dict against the event's player
    print("possession groups home / away on live frames: {:.1f} / {:.1f} %".format(
        100 * np.mean(grp[alive] == 1), 100 * np.mean(grp[alive] == 2)))


def _load_npz(path):
    """Every array of a parsed file, with the file closed (Windows cannot replace an open file)."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _parsed(root, game):
    path = parsed_path(root, game)
    if os.path.isfile(path):
        z = _load_npz(path)
        if int(z["version"]) == PARSE_VERSION:
            return z
    parse(root, game)
    return _load_npz(path)


# --- the interface of validation/metrica.py and validation/idsse.py ---------------------------

def load_tracking(root, game, cache_dir=None, keep_extrapolated=None):
    """Same dict as validation/metrica.load_tracking (cache_dir unused: the parsed arrays live in
    <root>/parsed). Extrapolated players are not seen unless keep_extrapolated (default
    KEEP_EXTRAPOLATED)."""
    from metrica import player_speed
    z = _parsed(root, game)
    keep = KEEP_EXTRAPOLATED if keep_extrapolated is None else keep_extrapolated
    players = z["players"].astype(np.float64)
    if not keep:
        players[~z["pdet"]] = np.nan
    pitch = tuple(float(v) for v in z["pitch"])
    return {"ball": z["ball"], "players": players, "speed": player_speed(players, pitch=pitch),
            "team": z["team"], "pitch": pitch, "n_frames": len(z["ball"]), "game": game,
            "period": z["period"].astype(int)}


def load_events(root, game):
    """DataFrame like validation/metrica.load_events with the restarts only: team ('H'/'A'),
    type, subtype (SET PIECE + KICK OFF / THROW IN / FREE KICK / CORNER KICK / PENALTY; goal kick
    = PASS + GOAL KICK), period, start, end (0-based frame indices)."""
    z = _parsed(root, game)
    return pd.DataFrame({"team": z["ev_team"].astype(object), "type": z["ev_type"].astype(str),
                         "subtype": z["ev_subtype"].astype(str), "period": z["ev_period"],
                         "start": z["ev_start"], "end": z["ev_end"]})


def ground_truth(root, game, min_spell_s=0.0, source=None):
    """(labels (N,) 'H'/'A'/'D'/'X', blocks [(period, first, last)]): D outside the phases of play,
    else the team of the last possession group of the tracking (source "group") or of the last
    player_possession event (source "events"); see the module docstring. Default GT_SOURCE.
    Graded: every frame of each half. min_spell_s > 0: the coarse variant of
    validation/gt_events.absorb_short_spells."""
    import gt_events
    source = source or GT_SOURCE
    if source not in GT_SOURCES:
        raise ValueError("SkillCorner ground truth source must be one of {}: {}".format(
            GT_SOURCES, source))
    z = _parsed(root, game)
    labels = z["labels_" + source].astype("<U1")
    blocks = [(int(p), int(a), int(b)) for p, a, b in z["blocks"]]
    if min_spell_s > 0:
        labels = gt_events.absorb_short_spells(labels, blocks, min_spell_s)
    return labels, blocks


def parse_stats(root, game):
    return json.loads(str(_parsed(root, game)["stats"]))


# --- aggregate check of the ground truth --------------------------------------------------------

def nearest_distance(tr, live):
    """Median distance (m) from the seen ball to the nearest seen player on live frames, and the
    median number of seen players on live frames."""
    ball, P = tr["ball"], tr["players"]
    m = live & ~np.isnan(ball[:, 0])
    scale = np.array(tr["pitch"])
    d = np.hypot(*np.moveaxis((P[m] - ball[m][:, None, :]) * scale, -1, 0))
    with np.errstate(invalid="ignore"):
        dmin = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
    dmin = dmin[np.isfinite(dmin)]
    seen = (~np.isnan(P[live][..., 0])).sum(axis=1)
    return float(np.median(dmin)) if len(dmin) else float("nan"), float(np.median(seen))


def game_rows(game, tr, gt, blocks, coarse, events, source, extra=None):
    import idsse
    live = np.isin(gt, ["H", "A"])
    dist, n_seen = nearest_distance(tr, live)
    short1 = short3 = 0
    for _, a, b in blocks:       # stoppages between two live frames, by length
        lab = gt[a:b + 1]
        s, e = _runs(lab == "D")
        inner = (s > 0) & (e < len(lab))
        short1 += int(np.sum(inner & (e - s < FPS)))
        short3 += int(np.sum(inner & (e - s < 3 * FPS)))
    row = {"game": game, "source": source, **idsse.gt_stats(gt, blocks),
           "coarse poss": idsse.gt_stats(coarse, blocks)["possessions"],
           "stops<1s": short1, "stops<3s": short3}
    chk = {"game": game, "source": source, **idsse.tracking_checks(tr, gt, blocks),
           "ball-nearest median m": dist, "seen players live (median)": n_seen}
    if events is not None:
        chk.update(idsse.restart_alignment(events, gt))
    chk.update(extra or {})
    return row, chk


def skc_extra(st):
    pct = lambda a, b: 100.0 * st.get(a, 0) / max(1, st.get(b, 0))  # noqa: E731
    live = st.get("frames 10 fps live", 0)
    dead = st.get("frames 10 fps", 0) - live
    samples = st.get("player samples detected", 0) + st.get("player samples extrapolated", 0)
    return {
        "ball undetected-but-placed live %": 100.0 * st.get("ball undetected position live", 0)
        / max(1, live),
        "ball no position live %": 100.0 * st.get("ball no position live", 0) / max(1, live),
        "ball detected dead % (10fps)": 100.0 * st.get("ball detected dead", 0) / max(1, dead),
        "players detected %": 100.0 * st.get("player samples detected", 0) / max(1, samples),
        "frames without players %": pct("frames without players", "frames 10 fps"),
        "group null live %": 100.0 * st.get("group null live", 0) / max(1, live),
        "group set dead %": 100.0 * st.get("group set dead", 0) / max(1, dead),
        "GT group = phase team %": 100.0 * st.get("group: live frames GT = phase team", 0)
        / max(1, live),
        "GT events = phase team %": 100.0 * st.get("events: live frames GT = phase team", 0)
        / max(1, live),
        "GT group = GT events %": 100.0 * st.get("live frames GT group = GT events", 0)
        / max(1, live),
        "possession events alive %": pct("player possessions alive", "player possessions"),
        "possession events = GT group team %": pct("player possessions = GT group team",
                                                   "player possessions alive"),
        "live stretches": st.get("live stretches", 0),
        "stretches w/o group": st.get("group: live stretches without any", 0),
        "stretches w/o events": st.get("events: live stretches without any", 0),
        "direction ok (of halves)": "{}/{}".format(st.get("direction agrees with match.json", 0),
                                                   st.get("direction checks", 0)),
        "y-up agree %": pct("y-up check agree", "y-up check events"),
        "restarts linked %": 100.0 * st.get("restarts at a live-stretch start", 0) / max(1, (
            st.get("restarts at a live-stretch start", 0)
            + st.get("restarts without a stoppage before", 0))),
    }


def summary(root, games, compare=False):
    """One game at a time (one process): ground-truth aggregates and tracking checks; with
    compare, the same for the IDSSE and Metrica games (clean tracking)."""
    import bench as B
    import gt_events
    import idsse
    import metrica
    rows, checks = [], []
    for g in games:
        tr = load_tracking(root, g)
        st = parse_stats(root, g)
        events = load_events(root, g)
        for src in GT_SOURCES:
            gt, blocks = ground_truth(root, g, source=src)
            coarse, _ = ground_truth(root, g, 2.0, source=src)
            r, c = game_rows(g, tr, gt, blocks, coarse, events, "SKC " + src,
                             skc_extra(st) if src == GT_SOURCES[0] else None)
            rows.append(r)
            checks.append(c)
        odd = {k: v for k, v in st.items() if k.startswith(("odd", "interruption not used",
                                                             "restart", "match_periods",
                                                             "missing"))}
        print("{}: {:.1f} min, restarts and parse notes {}".format(g, r["minutes"], odd),
              flush=True)
        del tr, gt, coarse
    if compare:
        bench = B.load_bench(os.path.join(HERE, "bench_all.yaml"))
        for g in idsse.GAMES:
            tr = idsse.load_tracking(bench["idsse_dir"], g)
            gt, blocks = idsse.ground_truth(bench["idsse_dir"], g)
            coarse, _ = idsse.ground_truth(bench["idsse_dir"], g, 2.0)
            r, c = game_rows(g, tr, gt, blocks, coarse, idsse.load_events(bench["idsse_dir"], g),
                             "IDSSE")
            rows.append(r)
            checks.append(c)
            del tr, gt, coarse
        for g in metrica.GAMES:
            tr = metrica.load_tracking(bench["metrica_dir"], g, cache_dir=None)
            ev = metrica.load_events(bench["metrica_dir"], g)
            gt, blocks = gt_events.ground_truth(ev, tr["n_frames"])
            coarse, _ = gt_events.ground_truth(ev, tr["n_frames"], 2.0)
            r, c = game_rows(g, tr, gt, blocks, coarse, None, "Metrica")
            rows.append(r)
            checks.append(c)
            del tr, gt, coarse
    with pd.option_context("display.width", 250, "display.max_columns", 40,
                           "display.max_rows", 200, "display.float_format", "{:.2f}".format):
        df = pd.DataFrame(rows).set_index(["game", "source"])
        print("\nground truth per game (graded frames):")
        print(df.to_string())
        print("\nmean by source:")
        print(df.groupby(level="source").mean(numeric_only=True).to_string())
        ch = pd.DataFrame(checks).set_index(["game", "source"])
        cols = [c for c in ("near<1m agrees %", "near<1m share of live %", "ball seen live %",
                            "ball seen dead %", "ball-nearest median m",
                            "seen players live (median)", "restarts", "D->alive within 1 s %",
                            "within 3 s %") if c in ch.columns]
        print("\nlabel and tracking checks (tracking as loaded, not degraded):")
        print(ch[cols].to_string())
        print("\nchecks, mean by source:")
        print(ch[cols].groupby(level="source").mean(numeric_only=True).to_string())
        extra = [c for c in ch.columns if c not in cols and not c.startswith(
            ("home x", "away x", "attack_sign", "median |frames|", "median GT-event"))]
        skc = ch.xs("SKC " + GT_SOURCES[0], level="source")[extra]
        if len(skc):
            skc.index = [g.replace("SKC_", "") for g in skc.index]
            print("\nSkillCorner inputs per game (10 fps unless noted):")
            print(skc.T.to_string())
            num = skc.apply(pd.to_numeric, errors="coerce")
            print("\nmean over the SkillCorner games:")
            print(num.mean().dropna().to_string())


# --- command line -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["licence", "readme", "download", "inspect", "parse",
                                            "summary", "diagnose"])
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--games", nargs="+", default=None, help="SKC_<id> (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="download: sizes only")
    parser.add_argument("--compare", action="store_true",
                        help="summary: also the IDSSE and Metrica games, one at a time")
    args = parser.parse_args()
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import bench  # noqa: F401  (sets the paths to src/ and the official repo)
    games = args.games or list(GAMES)
    if args.command == "licence":
        licence()
    elif args.command == "readme":
        readme()
    elif args.command == "download":
        download(args.root, [match_id(g) for g in games], args.dry_run)
    elif args.command == "inspect":
        for g in games:
            inspect(args.root, g)
    elif args.command == "parse":
        for g in games:
            parse(args.root, g)
    elif args.command == "summary":
        summary(args.root, games, compare=args.compare)
    elif args.command == "diagnose":
        for g in games:
            diagnose(args.root, g)


if __name__ == "__main__":
    sys.exit(main())
