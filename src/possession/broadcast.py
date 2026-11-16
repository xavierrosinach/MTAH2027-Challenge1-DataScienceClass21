"""What the broadcast recording adds to the tactical one.

The tactical recording is the reference: every player and every ball position it has stays as it
is. The broadcast camera films the same match, and its tracking is used for two things only:

  - the ball, on frames where the tactical tracking does not see it;
  - a second opinion on who has the ball, from what the broadcast camera sees by itself
    (`broadcast_view`), with its tracks' teams corrected by shirt colour on its own video;
  - whether the ball is in the air (`ball_air`): the two cameras then put it in different places;
  - the referee's whistles in its sound track (`whistle`, src/possession/whistle.py);
  - whether the TV is showing the live wide shot (`broadcast_live`). Directors cut to replays and
    close-ups mostly while the ball is dead, so a stoppage spent away from the live shot has a
    reason to be one (context.py, `tv_cut`).

  1. align()   the two recordings are not synchronised. Both tracks are in pitch coordinates, so
               the offset is the one that puts the broadcast ball on the tactical ball. It is
               searched per half on one-minute windows; the offset of a half is the one most
               windows agree on (live TV runs in real time, so it is constant within a half).
  2. live      a broadcast frame is used only if it shows the same moment: its players must sit
               on tactical players (replays, close-ups and other cuts do not).
  3. fuse()    where the tactical ball is missing on a live frame, the broadcast ball is used.

No player is added or moved, nothing is created by hand and no label is used.
"""

import os

import numpy as np

CHUNK = 4096     # candidate offsets per step of the search (memory only: the result does not depend on it)


def load(cfg, game="BAR-ATH"):
    """Broadcast tracking as arrays (its own timeline), or None if the files are not there."""
    import games
    from .data import _parse_raw, team_of_each_slot

    try:
        paths = games.asset_paths(game, cfg["paths"]["data_dir"], source="broadcast")
    except Exception:
        return None
    if not (os.path.isfile(paths["atd_raw"]) and os.path.isfile(paths["atd_meta"])):
        return None
    cache = os.path.join(cfg["paths"]["cache_dir"], "{}_broadcast_atd.npz".format(game))
    if os.path.isfile(cache) and os.path.getmtime(cache) >= os.path.getmtime(paths["atd_raw"]):
        return dict(np.load(cache))
    print("reading {} (1-2 min the first time)...".format(os.path.basename(paths["atd_raw"])))
    team = team_of_each_slot(paths["atd_meta"])
    with open(paths["atd_raw"]) as handle:
        n = sum(1 for _ in handle)
    ball, players, speed = _parse_raw(paths["atd_raw"], n, len(team))
    os.makedirs(cfg["paths"]["cache_dir"], exist_ok=True)
    np.savez(cache, ball=ball, players=players, speed=speed, team=team)
    return {"ball": ball, "players": players, "speed": speed, "team": team}


def align(tracking, bro, blocks, cfg):
    """Offset of each half: tactical frame = broadcast frame + offset.

    Returns {half: (offset, windows that agree, windows tried)}; offset None if no agreement.
    """
    c = cfg["broadcast"]
    pitch = np.array(tracking["pitch"])
    Bt, Bb = tracking["ball"] * pitch, bro["ball"] * pitch
    nt, nb = len(Bt), len(Bb)
    W = int(c["window_s"] * 25)
    offsets = np.arange(-int(c["max_lead_s"] * 25), nt - nb + int(c["max_lead_s"] * 25))
    out = {}
    for half, first, last in blocks:
        found = []
        starts = np.linspace(first, last - W, int(c["windows_per_half"])).astype(int)
        for w0 in starts:
            idx = np.flatnonzero(~np.isnan(Bt[w0:w0 + W, 0])) + w0
            if len(idx) < c["min_ball_frames"]:
                continue
            idx = idx[::max(1, len(idx) // 300)]
            best = (np.inf, None)
            for k in range(0, len(offsets), CHUNK):                # in chunks: memory
                off = offsets[k:k + CHUNK]
                B = idx[:, None] - off[None, :]
                ok = (B >= 0) & (B < nb)
                d = np.hypot(*(Bb[np.clip(B, 0, nb - 1)] - Bt[idx][:, None, :]).transpose(2, 0, 1))
                d[~ok] = np.nan
                seen = (~np.isnan(d)).sum(axis=0)
                d = np.where(np.isnan(d), np.inf, d)
                med = np.median(d, axis=0)                         # inf unless > half are seen
                med[seen < 0.5 * len(idx)] = np.inf
                j = int(np.argmin(med))
                if med[j] < best[0]:
                    best = (float(med[j]), int(off[j]))
            if best[0] < c["match_m"]:
                found.append(best[1])
        if not found:
            out[half] = (None, 0, len(starts))
            continue
        found = np.array(found)
        mode = int(np.median(found))
        agree = int((np.abs(found - mode) <= c["offset_tolerance_frames"]).sum())
        out[half] = (mode if agree >= c["min_windows"] else None, agree, len(starts))
    return out


def agreement_gate(ball, bball, pitch, agree_m, window_s, min_frames=5, fps=25):
    """(N,) bool, True on the frames of each tactical-ball gap whose surroundings (window_s before
    and after it, where both balls are seen) have the two balls within agree_m metres (median).
    A gap with fewer than min_frames such frames around it is not trusted."""
    miss = np.isnan(ball[:, 0])
    both = ~miss & ~np.isnan(bball[:, 0])
    d = np.full(len(ball), np.nan)
    d[both] = np.hypot(*((ball[both] - bball[both]) * np.asarray(pitch)).T)
    m = np.diff(np.concatenate(([0], miss.astype(np.int8), [0])))
    ok = np.zeros(len(ball), dtype=bool)
    w = int(round(window_s * fps))
    for s, e in zip(np.flatnonzero(m == 1), np.flatnonzero(m == -1)):
        around = np.concatenate((d[max(0, s - w):s], d[e:e + w]))
        around = around[~np.isnan(around)]
        if len(around) >= min_frames and np.median(around) < agree_m:
            ok[s:e] = True
    return ok


def ball_in_air(ball, bball, live, pitch, c):
    """(N,) bool: the two cameras put the ball in different places because it is off the ground.

    Each camera projects the ball onto the pitch along its own line of sight. On the ground both
    land on the same point; in the air they land apart, along the line that goes from the cameras
    (both in the main stand, at `camera_xy_m`, at different heights) through the ball. So: both
    see it, they differ by more than `air_m`, the difference lies along that line (within
    `air_angle_deg`), for at least `air_min_frames` in a row."""
    dif = (bball - ball) * pitch
    rad = ball * pitch - np.array(c["camera_xy_m"], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.hypot(dif[:, 0], dif[:, 1])
        cos = np.abs((dif * rad).sum(axis=1)) / (d * np.hypot(rad[:, 0], rad[:, 1]))
        air = live & (d > c["air_m"]) & (cos > np.cos(np.radians(c["air_angle_deg"])))
    air = np.nan_to_num(air).astype(bool)
    edge = np.diff(np.r_[0, air.astype(int), 0])
    for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)):
        if b - a < c["air_min_frames"]:
            air[a:b] = False
    return air


def view_team_by_shirt(bro, offs, blocks, n, cfg, game, pitch, verbose=True):
    """(N, Sb) team of each broadcast track on the tactical timeline, from the shirt colour in
    the broadcast video (jersey.py on that video), or None without it."""
    from . import jersey

    nb = len(bro["ball"])
    tr = {"players": bro["players"], "team": bro["team"].astype(np.int8), "pitch": pitch,
          "game": game}
    blocks_b = []
    for half, first, last in blocks:
        off = offs[half][0]
        if off is not None:
            blocks_b.append((half, max(0, first - off), min(nb - 1, last - off)))
    if not blocks_b:
        return None
    T_b, changed = jersey.team_by_shirt(tr, blocks_b, cfg, verbose, source="broadcast")
    if T_b is None:
        return None
    if verbose:
        print("broadcast tracks given to the other team by shirt colour: {} ({:.0f} s)".format(
            len(changed), sum(b - a + 1 for _, a, b in changed) / 25))
    T = np.repeat(bro["team"].astype(np.int8)[None, :], n, axis=0)
    for half, first, last in blocks:
        off = offs[half][0]
        if off is None:
            continue
        f = np.arange(first, last + 1)
        k = f - off
        ok = (k >= 0) & (k < nb)
        T[f[ok]] = T_b[k[ok]]
    return T


def on_tactical_timeline(x, offs, blocks, n):
    """(N,) bool: a per-broadcast-frame flag moved to the tactical frames of the graded halves."""
    out = np.zeros(n, dtype=bool)
    for half, first, last in blocks:
        off = offs[half][0]
        if off is None:
            continue
        f = np.arange(first, last + 1)
        k = f - off
        ok = (k >= 0) & (k < len(x))
        out[f[ok]] = x[k[ok]]
    return out


def fuse(tracking, cfg, verbose=True):
    """Tracking with the broadcast ball where the tactical one is missing, plus `broadcast_live`
    (N,): the TV shows the live wide shot. Only called with `broadcast.enabled`: without the
    broadcast files it stops, because carrying on would silently give other labels (another CSV)."""
    import games

    c = cfg["broadcast"]
    game = tracking.get("game", "BAR-ATH")
    bro = load(cfg, game)
    if bro is None:
        raise FileNotFoundError(
            "broadcast.enabled is true but the broadcast tracking of {} (ATD .txt and .xml in "
            "the data folder's broadcast/) was not found. Without it the labels, and the CSV, "
            "would differ. Download it or set broadcast.enabled: false.".format(game))
    blocks = games.half_video_blocks(game)
    offs = align(tracking, bro, blocks, cfg)
    pitch = np.array(tracking["pitch"])
    P, ball = tracking["players"], tracking["ball"]
    n, nb = len(P), len(bro["ball"])
    ball2 = ball.copy()
    bball = np.full_like(ball, np.nan)              # broadcast ball on live-shot frames
    live = np.zeros(n, dtype=bool)
    filled = 0
    Sb = bro["players"].shape[1]
    # the broadcast tracking on the tactical timeline, live-shot frames only (its own players,
    # teams and ball, untouched): a second opinion on who has the ball (context.py,
    # control.second_view_weight)
    view = {"players": np.full((n, Sb, 2), np.nan), "ball": bball,
            "speed": np.full((n, Sb), np.nan, dtype=np.float32), "team": bro["team"].astype(np.int8),
            "pitch": tracking["pitch"], "game": game, "other_camera": True}
    for half, first, last in blocks:
        off = offs[half][0]
        if verbose:
            print("broadcast, half {}: offset {} frames ({} of {} windows agree)".format(half, *offs[half]))
        if off is None:
            print("WARNING broadcast, half {}: no offset found ({} of {} windows agree); the "
                  "broadcast is not used in this half".format(half, offs[half][1], offs[half][2]))
            continue
        for t in range(max(first, off), min(last, nb - 1 + off) + 1):
            q = bro["players"][t - off]
            qs = np.flatnonzero(~np.isnan(q[:, 0]))
            ps = np.flatnonzero(~np.isnan(P[t, :, 0]))
            if len(qs) < c["min_players"] or len(ps) < c["min_players"]:
                continue
            D = np.hypot((q[qs][:, None, 0] - P[t, ps][None, :, 0]) * pitch[0],
                         (q[qs][:, None, 1] - P[t, ps][None, :, 1]) * pitch[1])
            if np.median(D.min(axis=1)) > c["live_m"]:
                continue                                           # replay, cut or bad homography
            live[t] = True
            bball[t] = bro["ball"][t - off]
            view["players"][t], view["speed"][t] = q, bro["speed"][t - off]
            if c["fill_ball"] and np.isnan(ball[t, 0]) and not np.isnan(bro["ball"][t - off, 0]):
                ball2[t] = bro["ball"][t - off]
                filled += 1
    if c.get("fill_agree_m") is not None:
        # fill a tactical gap only where the two balls agree around it (EXPERIMENTS.md #34)
        drop = ~agreement_gate(ball, bball, pitch, c["fill_agree_m"], c["fill_agree_s"]) \
            & np.isnan(ball[:, 0]) & ~np.isnan(ball2[:, 0])
        ball2[drop] = np.nan
        filled -= int(drop.sum())
    if verbose:
        graded = sum(last - first + 1 for _, first, last in blocks)
        print("broadcast: live on {:.1f} % of the graded frames; ball filled on {} frames "
              "({:.1f} %)".format(100 * live.sum() / graded, filled, 100 * filled / graded))
    out = dict(tracking)
    ctx = cfg["context"]
    if (ctx.get("jersey") or {}).get("enabled") and (ctx.get("jersey") or {}).get("broadcast"):
        # the broadcast tracker gives tracks to the wrong team too: its shirts, from its video
        shirt = view_team_by_shirt(bro, offs, blocks, n, cfg, game, tracking["pitch"], verbose)
        if shirt is not None:
            view["team_shirt"] = shirt
    if c.get("air_m"):
        out["ball_air"] = ball_in_air(ball, bball, live, pitch, c)
        if verbose:
            print("ball in the air (the two cameras differ along their line of sight): "
                  "{:.1f} % of the frames with both balls".format(
                      100 * out["ball_air"].sum() / max(1, (~np.isnan(ball[:, 0]) & ~np.isnan(bball[:, 0])).sum())))
    if (ctx.get("whistle") or {}).get("enabled"):
        from . import whistle
        heard = whistle.whistles(cfg, game, verbose)
        if heard is not None:
            out["whistle"] = on_tactical_timeline(heard, offs, blocks, n)
    out.update(ball=ball2, broadcast_live=live, ball_tactical=ball, broadcast_view=view)   # tactical kept: fill_team_only
    return out
