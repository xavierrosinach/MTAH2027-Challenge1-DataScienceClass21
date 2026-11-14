"""What the broadcast recording adds to the tactical one.

The tactical recording is the reference: every player and every ball position it has stays as it
is. The broadcast camera films the same match, and its tracking is used for two things only:

  - the ball, on frames where the tactical tracking does not see it;
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
            for k in range(0, len(offsets), 5000):                 # in chunks: memory
                off = offsets[k:k + 5000]
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


def fuse(tracking, cfg, verbose=True):
    """Tracking with the broadcast ball where the tactical one is missing, plus `broadcast_live`
    (N,): the TV shows the live wide shot. Returns the same tracking if there is no broadcast."""
    import games

    c = cfg["broadcast"]
    game = tracking.get("game", "BAR-ATH")
    bro = load(cfg, game)
    if bro is None:
        return tracking
    blocks = games.half_video_blocks(game)
    offs = align(tracking, bro, blocks, cfg)
    pitch = np.array(tracking["pitch"])
    P, ball = tracking["players"], tracking["ball"]
    n, nb = len(P), len(bro["ball"])
    ball2 = ball.copy()
    live = np.zeros(n, dtype=bool)
    filled = 0
    for half, first, last in blocks:
        off = offs[half][0]
        if verbose:
            print("broadcast, half {}: offset {} frames ({} of {} windows agree)".format(half, *offs[half]))
        if off is None:
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
            if c["fill_ball"] and np.isnan(ball[t, 0]) and not np.isnan(bro["ball"][t - off, 0]):
                ball2[t] = bro["ball"][t - off]
                filled += 1
    if verbose:
        graded = sum(last - first + 1 for _, first, last in blocks)
        print("broadcast: live on {:.1f} % of the graded frames; ball filled on {} frames "
              "({:.1f} %)".format(100 * live.sum() / graded, filled, 100 * filled / graded))
    out = dict(tracking)
    out.update(ball=ball2, broadcast_live=live)
    return out
