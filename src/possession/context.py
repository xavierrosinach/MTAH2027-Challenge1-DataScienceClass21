"""Possession labeller: per-frame evidence for H / A / D, decoded with Viterbi.

The ball is missing on ~41 % of the graded frames, often for many seconds, so a frame-by-frame "nearest team" rule has nothing to say on much of the match. Instead:

1. per-frame features that exist with or without a ball (player speeds, time since the ball was seen, where it was last / next seen, Smart Tagging);
2. P(dead ball) from a logistic regression trained on *algorithmic pseudo-labels* of this match (ball clearly out, start of a SET PIECES tag, 2nd-half pre-kick-off = dead; ball moving at a player's feet in the middle of the pitch = live), plus rules on where the ball is;
3. team evidence (H vs A) from a model of who controls the ball, learnt on other games (control.py), completed passes, kicks and Smart Tagging;
4. Viterbi over the whole half, with a cost for every change of state: the change lands where the evidence changes, and a possession survives a ball gap unless something says otherwise;
5. three corrective passes (team shape where the ball is missing, stoppages without a reason, restarts without play) and a final trim of the stoppage edges.

No human label of this match is used anywhere. All constants are in config.yaml (`context`).
"""

import os
import warnings
import numpy as np

H, A, D = 0, 1, 2
FPS = 25
ATTACK_CODES = ("PROGRESSION", "BUILD UP", "BALL IN FINAL THIRD", "ATTACKING TRANSITION", "CREATING CHANCES", "BALL IN THE BOX", "LONG BALL")

# HELPERS
def interpolate_ball(ball, max_gap):
    """Linear interpolation of ball gaps up to max_gap frames."""

    if max_gap <= 0:
        return ball
    ball = ball.copy()
    seen = ~np.isnan(ball[:, 0])
    idx = np.flatnonzero(seen)
    if len(idx) < 2:
        return ball
    gaps = np.diff(idx) - 1
    for start, gap in zip(idx[:-1][(gaps > 0) & (gaps <= max_gap)],
                          gaps[(gaps > 0) & (gaps <= max_gap)]):
        end = start + gap + 1
        t = (np.arange(1, gap + 1) / (gap + 1))[:, None]
        ball[start + 1:end] = ball[start] + t * (ball[end] - ball[start])
    return ball

def rolling_nanmean(x, half_window):
    """Centred mean over +/- half_window frames, ignoring NaN."""

    ok = ~np.isnan(x)
    c_val = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    c_n = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - half_window), np.minimum(len(x), i + half_window + 1)
    n = c_n[hi] - c_n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (c_val[hi] - c_val[lo]) / n, np.nan)

def ffill_index(mask):
    """Index of the last True at or before each frame (-1 if none)."""

    idx = np.where(mask, np.arange(len(mask)), -1)
    return np.maximum.accumulate(idx)

def bfill_index(mask):
    """Index of the next True at or after each frame (len if none)."""

    n = len(mask)
    idx = np.where(mask, np.arange(n), n)
    return np.minimum.accumulate(idx[::-1])[::-1]

def edge_distance_m(x, y, pitch):
    """Distance (m) from a 0-1 position to the nearest pitch line; negative outside."""

    length, width = pitch
    return np.minimum.reduce([x * length, (1 - x) * length, y * width, (1 - y) * width])

def spells(labels, first, last):
    """[(start, end)] runs of equal labels inside one half (end exclusive, absolute frames)."""

    seg = labels[first:last + 1]
    change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    return list(zip(np.r_[0, change] + first, np.r_[change, len(seg)] + first))

# TEAMS
def keeper_teams(tracking, blocks, cfg):
    """(N, S) team of each slot per frame, with goalkeeper tracks fixed by the goal they guard.

    The tracker sometimes swaps the identities of the two goalkeepers. A goalkeeper slot is one that spends more than `share` of a half within `depth_m` of one goal line; on every frame it is given to the team that defends the goal it is nearer to. Other slots keep their metadata team.
    """

    c = cfg["context"]["keepers"]
    P, team = tracking["players"], tracking["team"]
    length = tracking["pitch"][0]
    n, S = P.shape[:2]
    T = np.repeat(team[None, :], n, axis=0).astype(np.int8)
    fixed = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, first, last in blocks:
            X = P[first:last + 1, :, 0]
            seen = ~np.isnan(X)
            near = (np.minimum(X, 1 - X) * length < c["depth_m"]) & seen
            share = near.sum(axis=0) / np.maximum(1, seen.sum(axis=0))
            keepers = np.flatnonzero((share > c["share"]) & (seen.mean(axis=0) > 0.2))
            outfield = np.setdiff1d(np.arange(S), keepers)
            hx = np.nanmean(X[:, outfield[team[outfield] == 0]])
            ax = np.nanmean(X[:, outfield[team[outfield] == 1]])
            home_goal_left = hx < ax                   # home defends x = 0
            for s in keepers:
                left = X[:, s] < 0.5
                new = np.where(left == home_goal_left, 0, 1).astype(np.int8)
                new[~seen[:, s]] = team[s]
                fixed += int((new != team[s]).sum())
                T[first:last + 1, s] = new
    return T, fixed

def frame_teams(tracking, blocks, cfg, shirt=None):
    """Team of each slot per frame: goalkeepers by their goal, everybody else by shirt colour (jersey.py) when available, else the metadata."""
    c = cfg["context"]
    team_frame = None
    if c["keepers"]["enabled"]:
        team_frame, _ = keeper_teams(tracking, blocks, cfg)
    if shirt is None:
        return team_frame
    if team_frame is None:
        return shirt
    return np.where(team_frame != tracking["team"][None, :], team_frame, shirt)

# FEATURES
def tag_arrays(tags, n):
    """Per frame: attacking-phase votes for H and A, and frames since a SET PIECES tag started."""

    vote_h, vote_a = np.zeros(n), np.zeros(n)
    for row in tags.itertuples():
        if row.code in ATTACK_CODES and row.team in ("H", "A"):
            target = vote_h if row.team == "H" else vote_a
            target[max(0, row.start_frame):min(n, row.end_frame)] += 1
    sp_start = np.zeros(n, dtype=bool)
    for s in tags.loc[tags["code"] == "SET PIECES", "start_frame"]:
        if 0 <= s < n:
            sp_start[s] = True
    last = ffill_index(sp_start)
    since_sp = np.where(last >= 0, np.arange(n) - last, 10 ** 6)
    return vote_h, vote_a, since_sp

def false_kicks(seen, ball_speed, ball, pitch, near_slot, near_d, cfg):
    """Frames of kicks that do not put the ball in play: the ball is moved, not played.

    During a stoppage the ball is thrown back, rolled to the spot or placed by one player. That looks like a kick, but nobody else plays it and it comes to rest again. A kick is false when, within `max_s`, the ball is at rest (below `rest_speed` for `rest_s`) and from the kick to that moment the only player within touch distance was the kicker.
    """

    c = cfg["context"]["restart_check"]
    out = np.zeros(len(seen), dtype=bool)
    if not c["enabled"]:
        return out
    n = len(seen)
    cb, cp = cfg["context"]["ball_dead"], cfg["context"]["passes"]
    edge = edge_distance_m(ball[:, 0], ball[:, 1], pitch)
    inside = seen & (edge > cb["inside_m"])
    if cb["kick_from_line"]:
        # a ball on a line that moves into the pitch (corner, throw-in, goal kick)
        later = np.r_[edge[10:], np.full(10, np.nan)]
        with np.errstate(invalid="ignore"):
            inside = inside | (seen & (edge > 0) & (later > edge + 0.5))
    kick = inside & (np.nan_to_num(ball_speed) > cb["kick_speed"])
    still = seen & (np.nan_to_num(ball_speed, nan=99.0) < c["rest_speed"])
    rest_f, max_f = int(c["rest_s"] * FPS), int(c["max_s"] * FPS)
    run = np.concatenate(([0], np.cumsum(still)))
    rest_start = np.zeros(n, dtype=bool)                  # a rest of rest_f frames begins here
    rest_start[:n - rest_f] = (run[rest_f:n] - run[:n - rest_f]) == rest_f
    touch = seen & (near_d < cp["touch_m"])
    k = np.flatnonzero(kick)
    if not len(k):
        return out
    for g in np.split(k, np.flatnonzero(np.diff(k) > 5) + 1):    # one kick = nearby fast frames
        k0, k1 = int(g[0]), int(g[-1])
        before = np.flatnonzero(touch[max(0, k0 - 12):k0 + 3])
        if not len(before):
            continue                                              # nobody seen kicking it
        kicker = near_slot[max(0, k0 - 12) + before[-1]]
        rests = np.flatnonzero(rest_start[k1:min(n, k1 + max_f)])
        if not len(rests):
            continue                                              # still rolling: in play
        r = k1 + int(rests[0])
        if not seen[k0:r].all():
            continue                                              # lost from view: cannot tell
        if (touch[k0:r + rest_f] & (near_slot[k0:r + rest_f] != kicker)).any():
            continue                                              # somebody else played it
        out[k0:k1 + 1] = True
    return out

def restart_gaps(seen, ball, pitch, cfg):
    """Ball gaps that are a stoppage: the ball comes back where it went missing.

    In live play a ball unseen for seconds reappears far away. After a foul or an offside it is put down at the spot, so a gap of at least `min_s` whose ball reappears within `max_move_m` of where it was lost is dead time.
    """

    c = cfg["context"]["gap_restart"]
    out = np.zeros(len(seen), dtype=bool)
    if not c["enabled"]:
        return out
    m = np.diff(np.concatenate(([0], (~seen).astype(np.int8), [0])))
    scale = np.array(pitch)
    for s, e in zip(np.flatnonzero(m == 1), np.flatnonzero(m == -1)):
        if s == 0 or e >= len(seen) or (e - s) < c["min_s"] * FPS:
            continue
        if np.hypot(*((ball[e] - ball[s - 1]) * scale)) < c["max_move_m"]:
            out[s:e] = True
    return out

def frame_features(tracking, ball, tags, cfg):
    """Per-frame features (dict of arrays) shared by the dead-ball model and the team evidence."""
    n = len(ball)
    pitch = tracking["pitch"]
    length, width = pitch
    players, speed = tracking["players"], tracking["speed"]
    team = tracking.get("team_frame", tracking["team"][None, :])   # (N, S) or (1, S)

    seen = ~np.isnan(ball[:, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spd = np.nanmean(speed, axis=1).astype(np.float64)
    spd = np.where(np.isnan(spd), np.nanmedian(spd), spd)

    # ball geometry, carried across gaps from both sides
    last_i, next_i = ffill_index(seen), bfill_index(seen)
    has_last, has_next = last_i >= 0, next_i < n
    bl = ball[np.clip(last_i, 0, n - 1)]
    bn = ball[np.clip(next_i, 0, n - 1)]
    edge_last = np.where(has_last, edge_distance_m(bl[:, 0], bl[:, 1], pitch), 0.0)
    edge_next = np.where(has_next, edge_distance_m(bn[:, 0], bn[:, 1], pitch), 0.0)
    t_since = np.where(has_last, np.arange(n) - last_i, 10 ** 6) / FPS
    t_until = np.where(has_next, next_i - np.arange(n), 10 ** 6) / FPS

    # ball speed (m/s) from positions 4 frames apart, NaN when not seen at both ends
    k = 4
    disp = np.full(n, np.nan)
    d = (ball[k:] - ball[:-k]) * np.array([length, width])
    disp[k:] = np.hypot(d[:, 0], d[:, 1]) * FPS / k

    # nearest player of each team (m)
    diff = (players - ball[:, None, :]) * np.array([length, width])
    dist = np.hypot(diff[..., 0], diff[..., 1])
    dist = np.where(np.isnan(dist), np.inf, dist)
    d_h = np.where(team == 0, dist, np.inf).min(axis=1)
    d_a = np.where(team == 1, dist, np.inf).min(axis=1)

    # who is on the ball: nearest slot within touch_m of a seen ball, no rival within contest_m
    near_slot = np.argmin(dist, axis=1)
    near_d = dist[np.arange(n), near_slot]
    cp = cfg["context"]["passes"]
    near_team = np.broadcast_to(team, dist.shape)[np.arange(n), near_slot]
    d_rival = np.where(near_team == 0, d_a, d_h)
    clean = seen & (near_d < cp["touch_m"]) & (d_rival > cp["contest_m"])
    touch_slot = np.where(clean, near_slot, -1)
    touch_team = np.where(clean, near_team, -1)

    vote_h, vote_a, since_sp = tag_arrays(tags, n)

    # where the ball was heading when last seen: extrapolated `exit_pred_s` ahead
    c = cfg["context"]["ball_dead"]
    vel = np.full((n, 2), np.nan)
    vel[k:] = (ball[k:] - ball[:-k]) / k                       # per frame, 0-1 units
    vl = vel[np.clip(last_i, 0, n - 1)]
    pred = bl + np.nan_to_num(vl) * c["exit_pred_s"] * FPS
    edge_pred = np.where(has_last, edge_distance_m(pred[:, 0], pred[:, 1], pitch), 0.0)

    # how long the (seen) ball has had nobody near it while rolling or still
    alone = (seen & (np.minimum(d_h, d_a) > c["isolated_m"])
             & (np.nan_to_num(disp, nan=99) < c["isolated_max_speed"]))
    start = ffill_index(~alone)
    alone_s = np.where(alone, (np.arange(n) - start) / FPS, 0.0)

    return {
        "seen": seen, "spd": spd,
        "spd_2s": rolling_nanmean(spd, 2 * FPS), "spd_6s": rolling_nanmean(spd, 6 * FPS),
        "edge_last": edge_last, "edge_next": edge_next,
        "t_since": t_since, "t_until": t_until,
        "ball_speed": disp, "d_h": d_h, "d_a": d_a,
        "vote_h": vote_h, "vote_a": vote_a, "since_sp": since_sp,
        "edge_pred": edge_pred, "alone_s": alone_s, "abandoned": alone_s >= c["isolated_s"],
        "touch_slot": touch_slot, "touch_team": touch_team, "ball": ball,
        "near_slot": near_slot, "near_d": near_d,
        "false_kick": false_kicks(seen, disp, ball, pitch, near_slot, near_d, cfg),
        "gap_restart": restart_gaps(seen, ball, pitch, cfg),
    }

# DEAD BALL EVIDENCE
def design_matrix(f):
    """Inputs of the dead-ball regression: how the players move.

    Ball visibility and position are left out on purpose: the pseudo-labels are defined from them, so a model given them would only learn "no ball => dead". They enter in `ball_dead_llr`.
    """
    return np.column_stack([np.clip(f["spd_2s"], 0, 5), np.clip(f["spd_6s"], 0, 5)])

def ball_dead_llr(f, cfg):
    """Rule-based dead-ball log-odds from where the ball is, or was last seen."""

    c = cfg["context"]["ball_dead"]
    llr = np.zeros(len(f["seen"]))
    out_now = f["seen"] & (f["edge_last"] < -c["out_m"])
    inside_now = f["seen"] & (f["edge_last"] > c["inside_m"])
    llr[out_now] += c["out_llr"]
    llr[inside_now] -= c["inside_llr"]
    # a gap that started with the ball out, or heading out fast enough to cross a line
    gap_after_out = ~f["seen"] & ((f["edge_last"] < 0) | (f["edge_pred"] < -c["out_m"]))
    llr[gap_after_out] += c["gap_after_out_llr"]
    # ball left alone (nobody near, not flying) for a while: nobody is playing it
    llr[f["abandoned"]] += c["isolated_llr"]
    # ball kicked inside the pitch: play is (re)starting
    kicked = inside_now & (np.nan_to_num(f["ball_speed"]) > c["kick_speed"]) & ~f["false_kick"]
    if c["kick_from_line"]:
        # corners, throw-ins and goal kicks start ON a line: inside and moving away from it
        edge = np.where(f["seen"], f["edge_last"], np.nan)
        later = np.r_[edge[10:], np.full(10, np.nan)]
        with np.errstate(invalid="ignore"):
            inward = (edge > 0) & (later > edge + 0.5)
        kicked |= (inward & (np.nan_to_num(f["ball_speed"]) > c["kick_speed"]) & ~f["false_kick"])
    llr[kicked] -= c["kick_llr"]
    if f["gap_restart"].any():
        llr[f["gap_restart"]] += cfg["context"]["gap_restart"]["llr"]
    # once the ball has gone out, play stays stopped until it is kicked again (the restart),
    # even if the ball is seen back inside (carried, placed, a new ball thrown in)
    went_out = out_now | gap_after_out
    last_out, last_kick = ffill_index(went_out), ffill_index(kicked)
    since = np.arange(len(llr)) - last_out
    latched = ((last_out >= 0) & (last_out > last_kick) & ~went_out
               & (since <= c["latch_max_s"] * FPS))
    llr[latched & inside_now] += c["inside_llr"]        # being inside says nothing here
    llr[latched] += c["latch_llr"]
    return llr

def goal_dead_mask(f, ball, tags, pitch, cfg):
    """True from a goal to the kick-off that follows it.

    A "Kick Off (After Goal)" SET PIECES tag means a goal was scored before it. The stoppage starts at the last sighting of the ball near a goal before the tag (at most `goal_lookback_s` earlier).
    """

    c = cfg["context"]["goal"]
    length, width = pitch
    n = len(ball)
    mask = np.zeros(n, dtype=bool)
    with np.errstate(invalid="ignore"):
        near_goal = (f["seen"]
                     & (np.minimum(ball[:, 0], 1 - ball[:, 0]) * length < c["zone_depth_m"])
                     & (np.abs(ball[:, 1] - 0.5) * width < c["zone_half_width_m"]))
    kicks = tags[(tags["code"] == "SET PIECES") & (tags["type"] == "Kick Off (After Goal)")]
    for k in kicks["start_frame"]:
        lo = max(0, k - int(c["goal_lookback_s"] * FPS))
        hits = np.flatnonzero(near_goal[lo:k])
        if len(hits):
            mask[lo + hits[-1] + 1:min(n, k)] = True
    return mask

def pseudo_labels(f, frame_period, cfg):
    """1 = dead, 0 = live, -1 = unknown. Rules only, no human decision about any frame."""

    c = cfg["context"]["pseudo"]
    y = np.full(len(f["seen"]), -1, dtype=np.int8)
    out_now = f["seen"] & (f["edge_last"] < -c["out_m"])
    live = (f["seen"] & (f["edge_last"] > c["live_edge_m"])
            & (np.minimum(f["d_h"], f["d_a"]) < c["live_near_m"])
            & (np.nan_to_num(f["ball_speed"]) > c["live_ball_speed"])
            & (f["since_sp"] > c["set_piece_s"] * FPS * 2))
    dead = out_now | (f["since_sp"] < c["set_piece_s"] * FPS)
    if frame_period is not None:
        dead |= frame_period == "2nd_half_pre_kickoff"
    y[live] = 0
    y[dead] = 1
    return y

def fit_logistic(X, y, l2=1.0, iters=50):
    """Class-balanced L2 logistic regression by Newton's method. Returns (mean, std, w)."""

    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    sw = np.where(y == 1, 0.5 / max(1, (y == 1).sum()), 0.5 / max(1, (y == 0).sum())) * len(y)
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (sw * (p - y)) + l2 * np.r_[0, w[1:]]
        Hs = (Z * (sw * p * (1 - p))[:, None]).T @ Z + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        step = np.linalg.solve(Hs, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return mu, sd, w

def predict_logistic(model, X):
    mu, sd, w = model
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    return 1 / (1 + np.exp(-Z @ w))

# TEAM EVIDENCE
def add_team_llr(e, llr):
    """Add H-vs-A log-odds (positive = H) that say nothing about dead vs alive.

    Only the disfavoured team is penalised, so the favoured one keeps its score against D: a player standing on the ball to take a free kick says who restarts, not that play is live.
    """
    
    e[:, H] -= np.maximum(0.0, -llr)
    e[:, A] -= np.maximum(0.0, llr)

def emissions(f, p_dead, cfg, proximity=True):
    """(N, 3) log-likelihood scores for H, A, D (up to a per-frame constant)."""

    c = cfg["context"]
    e = np.zeros((len(p_dead), 3))

    # dead vs alive: learnt from player movement + rules on the ball
    p = np.clip(p_dead, 1e-3, 1 - 1e-3)
    llr_dead = np.clip(np.log(p / (1 - p)), -c["dead_llr_clip"], c["dead_llr_clip"])
    e[:, D] += c["dead_weight"] * llr_dead + ball_dead_llr(f, cfg)

    # nearest team to the ball (off when the learnt control model replaces it)
    s = c["ball"]["scale_m"]
    w_h, w_a = np.exp(-f["d_h"] / s), np.exp(-f["d_a"] / s)
    tot = w_h + w_a
    with np.errstate(invalid="ignore", divide="ignore"):
        share_h = np.where(tot > 0, w_h / tot, 0.5)
    strength = c["ball"]["weight"] * np.maximum(w_h, w_a) * proximity
    strength[np.minimum(f["d_h"], f["d_a"]) > c["ball"]["contact_m"]] = 0.0   # loose ball
    add_team_llr(e, strength * np.log(np.clip(share_h, 0.02, 0.98)
                                      / np.clip(1 - share_h, 0.02, 0.98)))

    # Smart Tagging attacking phases (weak, coarse in time): only while the ball is not seen
    add_team_llr(e, c["tag_weight"] * np.sign(f["vote_h"] - f["vote_a"]) * ~f["seen"])
    return e

def home_attack_sign(tracking, blocks):
    """+1 on the frames of a block where the home team attacks towards x = 1, -1 towards x = 0.

    The team whose players stand nearer x = 0 on average over the block defends x = 0. Frames outside the blocks get +1.
    """

    P, team = tracking["players"], tracking["team"]
    sign = np.ones(len(P))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, first, last in blocks:
            home_x = np.nanmean(P[first:last + 1, team == 0, 0])
            away_x = np.nanmean(P[first:last + 1, team == 1, 0])
            sign[first:last + 1] = 1 if home_x < away_x else -1
    return sign

def team_shape_features(tracking, blocks, smooth_frames):
    """Per frame, how each team is laid out and moving, with x towards its attacking goal.

    The team with the ball stretches and pushes up; the other compacts and drops. None of these use the ball, so they still speak while it is missing.
    """

    P, speed = tracking["players"], tracking["speed"]
    team = tracking.get("team_frame", tracking["team"][None, :])
    length, width = tracking["pitch"]
    n = len(P)
    sign = home_attack_sign(tracking, blocks)

    cols = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in (0, 1):
            mine = team == t
            px = np.where(mine, P[:, :, 0], np.nan) * length
            py = np.where(mine, P[:, :, 1], np.nan) * width
            fwd = (px - length / 2) * (sign if t == 0 else -sign)[:, None]
            centre = np.nanmean(fwd, axis=1)
            step = np.full(n, np.nan)
            step[25:] = centre[25:] - centre[:-25]     # centroid advance over 1 s
            cols += [centre, np.nanstd(px, axis=1), np.nanstd(py, axis=1),
                     np.nanmean(np.where(mine, speed, np.nan), axis=1),
                     np.nanmax(fwd, axis=1), np.nanmin(fwd, axis=1), step]
    F = np.column_stack(cols)
    k = len(cols) // 2
    F = np.column_stack([F, F[:, 0] + F[:, k],          # both centroids: where the play is
                         F[:, 1] - F[:, k + 1], F[:, 2] - F[:, k + 2]])   # who is stretched
    F = np.nan_to_num(F, nan=0.0)
    return np.column_stack([rolling_nanmean(F[:, j], smooth_frames) for j in range(F.shape[1])])

def shape_evidence(tracking, f, labels, graded, blocks, cfg):
    """H-vs-A log-odds from team shape, learnt on this match from frames where the first pass is sure: ball seen, the labelled team within own_m of it and the other team beyond opp_m."""
    
    c = cfg["context"]["shape"]
    F = team_shape_features(tracking, blocks, int(c["smooth_s"] * FPS))
    d_own = np.where(labels == "H", f["d_h"], f["d_a"])
    d_opp = np.where(labels == "H", f["d_a"], f["d_h"])
    sure = (graded & f["seen"] & np.isin(labels, ["H", "A"])
            & (d_own < c["own_m"]) & (d_opp > c["opp_m"]))
    y = (labels == "H").astype(float)
    idx = np.flatnonzero(sure)
    model = fit_logistic(F[idx], y[idx])
    p = np.clip(predict_logistic(model, F), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))

def touch_events(f, min_frames):
    """[(start, end, slot, team)] runs of the same player on the ball for >= min_frames."""

    ts, tt = f["touch_slot"], f["touch_team"]
    change = np.flatnonzero(ts[1:] != ts[:-1]) + 1
    starts, ends = np.r_[0, change], np.r_[change, len(ts)]
    return [(s, e, int(ts[s]), int(tt[s])) for s, e in zip(starts, ends)
            if ts[s] >= 0 and e - s >= min_frames]

def completed_passes(f, cfg):
    """[(start, end, team)] ball going from one player to a different teammate.

    The receiver must be on the ball for >= min_receive_frames; the passer may touch it for as little as min_touch_frames (a first-time pass). The ball travels at least min_pass_m within max_gap_s, and no opponent has it for >= min_receive_frames in between (a shorter touch is a deflection, which does not change possession).
    """

    c = cfg["context"]["passes"]
    ev = touch_events(f, c["min_touch_frames"])

    def is_pass(a, b):
        (s0, e0, p0, t0), (s1, e1, p1, t1) = a, b
        if t0 != t1 or p0 == p1 or s1 - e0 > c["max_gap_s"] * FPS:
            return False
        b0, b1 = f["ball"][e0 - 1], f["ball"][s1]
        return np.hypot((b1[0] - b0[0]) * 105.0, (b1[1] - b0[1]) * 68.0) >= c["min_pass_m"]

    out, prev, last = [], None, None
    for cur in ev:
        s1, e1, p1, t1 = cur
        long_touch = e1 - s1 >= c["min_receive_frames"]
        if long_touch:
            # from the last touch of any kind, else from the last one that counts
            for src in (last, prev):
                if src is not None and is_pass(src, cur):
                    out.append((src[0], e1, t1))
                    break
        # a short touch by the other team is a deflection: it does not take over as `prev`
        if prev is None or long_touch or t1 == prev[3]:
            prev = cur
        last = cur
    return out

def kicker_evidence(e, f, cfg):
    """Whoever kicks the ball gets kick_team_llr for their team over the kick.

    The kicker is the player nearest to the ball (within kicker_m) just before it speeds up past kick_speed. In open play one kick (a clearance) is less than a change of possession; after a stoppage it says who restarts.
    """

    c = cfg["context"]["passes"]
    fast = f["seen"] & (np.nan_to_num(f["ball_speed"]) > cfg["context"]["ball_dead"]["kick_speed"])
    llr = np.zeros(len(e))
    for k in np.flatnonzero(fast[1:] & ~fast[:-1]) + 1:
        lo = max(0, k - 6)
        d = np.minimum(f["d_h"][lo:k], f["d_a"][lo:k])
        if not len(d) or not np.isfinite(d).any() or d.min() > c["kicker_m"]:
            continue
        j = lo + int(np.argmin(d))
        team = 0 if f["d_h"][j] <= f["d_a"][j] else 1
        llr[j:k + 1] += (c["kick_team_llr"] / (k + 1 - j)) * (1 if team == 0 else -1)
    add_team_llr(e, llr)

def pass_evidence(e, f, cfg):
    """Spread pass_llr over each completed pass, against the other team."""

    c = cfg["context"]["passes"]
    llr = np.zeros(len(e))
    for s, end, t in completed_passes(f, cfg):
        llr[s:end] += (c["pass_llr"] / (end - s)) * (1 if t == 0 else -1)
    add_team_llr(e, llr)

def restart_evidence(e, tags, cfg):
    """SET PIECES with a team: that team restarts. The tag starts a few seconds before the restart, so the window is [delay, delay + length] seconds after its start."""
    
    c = cfg["context"]["restart"]
    n = len(e)
    for row in tags[tags["code"] == "SET PIECES"].itertuples():
        if row.team not in ("H", "A"):
            continue
        lo = max(0, row.start_frame + int(c["delay_s"] * FPS))
        hi = min(n, lo + int(c["length_s"] * FPS))
        e[lo:hi, A if row.team == "H" else H] -= c["weight"]

def control_evidence(tracking, ball, blocks, cfg, verbose):
    """(H-vs-A log-odds, ball-seen mask) of the learnt control model (control.py).

    With the broadcast tracking, the same model is also applied to what the broadcast camera sees by itself (its ball, players and teams); where both cameras see the ball the two log-odds are averaged (`second_view_weight` for the broadcast one). A player one camera loses in a duel, or gives to the other team, is usually right in the other.
    """
    
    from . import control
    from .config import REPO_ROOT

    c = cfg["context"]
    ctl = c["control"]
    paths = [m if os.path.isabs(m) else os.path.join(REPO_ROOT, m)
             for m in [ctl["model"]] + list(ctl.get("ensemble") or [])]
    groups = tuple(ctl["features"])
    llr, seen = control.control_llr(tracking, ball, blocks, paths, groups)
    llr = np.clip(llr, -ctl["clip"], ctl["clip"])

    view, w2 = tracking.get("broadcast_view"), float(ctl.get("second_view_weight") or 0.0)
    if view is not None and w2 > 0:
        v = dict(view)
        team_frame = frame_teams(v, blocks, cfg)
        if team_frame is not None:
            v["team_frame"] = team_frame
        vball = interpolate_ball(view["ball"], cfg["ball_gaps"]["interpolate_max_frames"])
        llr2, seen2 = control.control_llr(v, vball, blocks, paths, groups)
        llr2 = np.clip(llr2, -ctl["clip"], ctl["clip"])
        both = seen & seen2 & ~np.isnan(tracking["ball_tactical"][:, 0])
        if verbose:
            print("second view: both cameras see the ball on {:.0%} of the frames with ball; "
                  "they disagree on the team on {:.1%} of those".format(
                      both.sum() / max(1, seen.sum()),
                      np.mean(np.sign(llr[both]) != np.sign(llr2[both]))))
        llr = np.where(both, (1 - w2) * llr + w2 * llr2, llr)
    return llr, seen

# DECODING (VITERBI)
def viterbi(e, team_switch, to_dead, from_dead, min_dead):
    """Max-score H/A/D path where every dead-ball stretch lasts at least min_dead frames.

    States: 0 H, 1 A, then a chain D_1 ... D_k (k = min_dead): entering D lands on D_1, the chain must be walked to D_k, and only D_k may stay or restart play. Without the minimum, a hop H -> D -> A of one frame would be cheaper than a direct hand-over and every change of possession would grow a fake stoppage. from_dead may be a per-frame array.
    """

    n, k = len(e), max(1, int(min_dead))
    eh, ea, ed = e[:, H].tolist(), e[:, A].tolist(), e[:, D].tolist()
    fd = (np.broadcast_to(np.asarray(from_dead, dtype=float), (n,))).tolist()
    ts = (np.broadcast_to(np.asarray(team_switch, dtype=float), (n,))).tolist()
    back_h = np.zeros(n, dtype=np.int8)     # 0 from H, 1 from A, 2 from D_k
    back_a = np.zeros(n, dtype=np.int8)
    back_d1 = np.zeros(n, dtype=np.int8)    # 0 from H, 1 from A
    back_dk = np.zeros(n, dtype=np.int8)    # 0 from D_{k-1}, 1 from D_k
    NEG = -1e18
    sh, sa = eh[0], ea[0]
    chain = [NEG] * k
    chain[0] = ed[0]
    for t in range(1, n):
        dk = chain[-1]
        # ties go to the change: with no evidence either way, the change lands as late as possible, i.e. when the new team first touches the ball
        c0, c1, c2 = sh, sa - ts[t], dk - fd[t]
        if c0 > c1 and c0 > c2:
            nh, bh = c0, 0
        elif c1 >= c2:
            nh, bh = c1, 1
        else:
            nh, bh = c2, 2
        c0, c1, c2 = sa, sh - ts[t], dk - fd[t]
        if c0 > c1 and c0 > c2:
            na, ba = c0, 1
        elif c1 >= c2:
            na, ba = c1, 0
        else:
            na, ba = c2, 2
        enter_h, enter_a = sh - to_dead, sa - to_dead
        d1, b1 = (enter_h, 0) if enter_h >= enter_a else (enter_a, 1)
        if k == 1:
            if dk >= d1:
                new_chain, bk = [dk + ed[t]], 1
            else:
                new_chain, bk = [d1 + ed[t]], 0
        else:
            stay, step = chain[-1], chain[-2]
            last, bk = (stay, 1) if stay > step else (step, 0)
            new_chain = [v + ed[t] for v in [d1] + chain[:-2] + [last]]
        back_d1[t] = b1
        back_h[t], back_a[t], back_dk[t] = bh, ba, bk
        sh, sa, chain = nh + eh[t], na + ea[t], new_chain

    # backtrack; state code: 0 H, 1 A, 2+i = D_{i+1}
    path = np.empty(n, dtype=np.int8)
    s = int(np.argmax([sh, sa] + chain))
    for t in range(n - 1, -1, -1):
        path[t] = H if s == 0 else A if s == 1 else D
        if t == 0:
            break
        if s == 0:
            s = [0, 1, 1 + k][back_h[t]]
        elif s == 1:
            s = [0, 1, 1 + k][back_a[t]]
        elif s == 2 and k > 1:            # D_1 came from H or A
            s = int(back_d1[t])
        elif s == 1 + k:                  # D_k
            if k == 1:
                s = 1 + k if back_dk[t] == 1 else int(back_d1[t])
            else:
                s = 1 + k if back_dk[t] == 1 else k
        else:                             # D_i came from D_{i-1}
            s -= 1
    return path

def decode(e, cost, blocks):
    """Viterbi independently on each graded half; ungraded frames stay D. Returns labels."""

    path = np.full(len(e), D, dtype=np.int8)
    for _, first, last in blocks:
        c = dict(cost)
        for key in ("from_dead", "team_switch"):
            if np.ndim(c[key]):
                c[key] = c[key][first:last + 1]
        path[first:last + 1] = viterbi(e[first:last + 1], **c)
    return np.array(["H", "A", "D"])[path]

def restart_cost(f, cfg):
    """Per-frame cost of restarting play: dearer while the ball is seen and nobody kicks it within +/- kick_window_s (the ball is being placed or carried, play has not restarted)."""

    t = cfg["context"]["transition_cost"]
    c = cfg["context"]["ball_dead"]
    kick = f["seen"] & (np.nan_to_num(f["ball_speed"]) > c["kick_speed"]) & ~f["false_kick"]
    near_kick = rolling_nanmean(kick.astype(float), int(t["kick_window_s"] * FPS)) > 0
    cost = np.full(len(kick), float(t["from_dead"]))
    cost[f["seen"] & ~near_kick] += t["restart_without_kick"]
    return cost

# CORRECTIVE PASSES
def unsupported_dead_spells(labels, f, ball, tags, pitch, blocks, frame_period, tv_cut, cfg):
    """D spells with no sign of a stoppage other than slow players.

    A spell is supported if any of these holds: the ball is seen out (or lost on a line, heading out, left alone, or back where it was lost); a SET PIECES tag starts around it; play restarts on a line or on the centre spot; it lasts at least `long_s`; it is the 2nd-half pre-kick-off; or the TV spent most of it away from the live wide shot (replays, close-ups).
    """

    c = cfg["context"]["dead_support"]
    length, width = pitch
    n = len(labels)
    sp_starts = tags.loc[tags["code"] == "SET PIECES", "start_frame"].to_numpy()
    look = int(c["restart_window_s"] * FPS)
    before = int(c["out_before_s"] * FPS)
    out_m = cfg["context"]["ball_dead"]["out_m"]
    bad = []
    for _, first, last in blocks:
        for s, end in spells(labels, first, last):
            if labels[s] != "D":
                continue
            if frame_period is not None and (frame_period[s:end] == "2nd_half_pre_kickoff").any():
                continue
            if (end - s) >= c["long_s"] * FPS:
                continue
            lo = max(0, s - before)
            if (f["seen"][lo:end] & (f["edge_last"][lo:end] < 0)).any():
                continue
            if not f["seen"][s] and f["edge_last"][s] < c["line_m"]:
                continue
            if (~f["seen"][lo:end] & (f["edge_pred"][lo:end] < -out_m)).any():
                continue
            if f["abandoned"][lo:end].any() or f["gap_restart"][lo:end].any():
                continue
            if ((sp_starts >= s - c["sp_before_s"] * FPS)
                    & (sp_starts < end + c["sp_after_s"] * FPS)).any():
                continue
            nxt = np.flatnonzero(f["seen"][end:min(n, end + look)])
            if len(nxt):
                bx, by = ball[end + nxt[0]]
                if edge_distance_m(bx, by, pitch) < c["line_m"]:
                    continue
                if np.hypot((bx - 0.5) * length, (by - 0.5) * width) < c["centre_m"]:
                    continue
            if tv_cut is not None and tv_cut[s:end].mean() >= cfg["context"]["tv_cut"]["support_share"]:
                continue
            bad.append((s, end))
    return bad

def false_restarts(labels, f, blocks, cfg):
    """Live spells between two stoppages in which one player kicks the ball to nobody.

    D -> team -> D within `max_s`, only one player touches the ball in between, and the second stoppage starts with the ball alone: he was sending the ball back to the spot, play never restarted. Returns [(start, end)] to be made dead.
    """

    c = cfg["context"]["restart_check"]
    touch = f["seen"] & (f["near_d"] < cfg["context"]["passes"]["touch_m"])
    bad = []
    for _, first, last in blocks:
        runs = spells(labels, first, last)
        for i in range(1, len(runs) - 1):
            s, end = runs[i]
            if labels[s] == "D" or labels[s - 1] != "D" or labels[end] != "D":
                continue
            if (end - s) > c["max_s"] * FPS or not f["alone_s"][end] > 0:
                continue
            if len(np.unique(f["near_slot"][s:end][touch[s:end]])) <= 1:
                bad.append((s, end))
    return bad

def trim_dead(labels, e, blocks, cfg):
    """Shorten each decoded stoppage: its start start_s later, its restart end_s earlier.

    On the validation games (other matches) the decoded stoppages start ~0.35-0.4 s early and end ~0.15-0.2 s late. The trimmed frames take the label of the neighbouring possession. Forced dead frames are never trimmed, nor a stoppage that would get shorter than min_s.
    """

    c = cfg["context"]["dead_trim"]
    if not c["enabled"]:
        return labels
    a_n, b_n, m_n = (int(round(c[k] * FPS)) for k in ("start_s", "end_s", "min_s"))
    forced = e[:, D] >= 1e5
    out = labels.copy()
    for _, first, last in blocks:
        seg = labels[first:last + 1]
        cut = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        bounds = np.concatenate(([0], cut, [len(seg)]))
        for i in range(len(bounds) - 1):
            s, t = bounds[i], bounds[i + 1]
            if seg[s] != "D" or (t - s) - a_n - b_n < m_n:
                continue
            lo, hi = first + s, first + t
            if s > 0 and not forced[lo:lo + a_n].any():
                out[lo:lo + a_n] = seg[s - 1]
            if t < len(seg) and not forced[hi - b_n:hi].any():
                out[hi - b_n:hi] = seg[t]
    return out

# MAIN ENTRY POINT
def label(tracking, ball, tags, cfg, blocks, frame_period=None, verbose=False):
    """Labels (N,) in {'H', 'A', 'D'} for every frame (ungraded frames stay D)."""
    c = cfg["context"]

    # team of each track per frame: goalkeepers by their goal, the rest by shirt colour
    shirt = None
    if c["jersey"]["enabled"]:
        from . import jersey
        shirt, changed = jersey.team_by_shirt(tracking, blocks, cfg, verbose)
        if verbose and shirt is not None:
            print("tracks given to the other team by shirt colour: {} ({:.0f} s)".format(
                len(changed), sum(b - a + 1 for _, a, b in changed) / FPS))
    team_frame = frame_teams(tracking, blocks, cfg, shirt)
    if team_frame is not None:
        tracking = dict(tracking, team_frame=team_frame)

    # dead-ball model, fitted on this match's pseudo-labels
    f = frame_features(tracking, ball, tags, cfg)
    y = pseudo_labels(f, frame_period, cfg)
    graded = np.zeros(len(y), dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    train = graded & (y >= 0)
    p_dead = predict_logistic(fit_logistic(design_matrix(f)[train], y[train].astype(float)), design_matrix(f))

    ctl = c["control"]
    e = emissions(f, p_dead, cfg, proximity=not (ctl["enabled"] and ctl["replace_proximity"]))
    if ctl["enabled"]:
        llr, seen = control_evidence(tracking, ball, blocks, cfg, verbose)
        add_team_llr(e, np.where(seen, ctl["weight"], ctl["gap_weight"]) * llr)

    # forced stoppages: before the 2nd-half kick-off (FAQ) and after a goal
    if frame_period is not None:
        e[frame_period == "2nd_half_pre_kickoff", D] += 1e6
    e[goal_dead_mask(f, ball, tags, tracking["pitch"], cfg), D] += 1e6

    restart_evidence(e, tags, cfg)
    if c["passes"]["kick_team_llr"] > 0:
        kicker_evidence(e, f, cfg)
    if c["passes"]["pass_llr"] > 0:
        pass_evidence(e, f, cfg)

    t = c["transition_cost"]
    cost = {"team_switch": t["team_switch"], "to_dead": t["to_dead"],
            "from_dead": restart_cost(f, cfg), "min_dead": int(round(t["min_dead_s"] * FPS))}
    labels = decode(e, cost, blocks)

    # second pass: team shape speaks where the ball is missing
    if c["shape"]["weight"] > 0:
        shape = shape_evidence(tracking, f, labels, graded, blocks, cfg)
        llr = c["shape"]["weight"] * np.clip(shape, -c["shape"]["clip"], c["shape"]["clip"])
        add_team_llr(e, llr * ~f["seen"])
        labels = decode(e, cost, blocks)

    # third pass: a stoppage needs a reason; slow players alone are not one
    if c["dead_support"]["enabled"]:
        tv_cut = None
        if c["tv_cut"]["enabled"]:
            if tracking.get("broadcast_live") is None:
                raise ValueError("context.tv_cut needs broadcast.enabled (the broadcast tracking)")
            tv_cut = ~tracking["broadcast_live"]
        bad = unsupported_dead_spells(labels, f, ball, tags, tracking["pitch"], blocks,
                                      frame_period, tv_cut, cfg)
        for s, end in bad:
            e[s:end, D] -= 1e6
        if bad:
            labels = decode(e, cost, blocks)
        if verbose:
            print("dead spells without support, made live: {}".format(len(bad)))

    # fourth pass: a restart needs play; one player moving a dead ball is not one
    if c["restart_check"]["enabled"]:
        moved = false_restarts(labels, f, blocks, cfg)
        for s, end in moved:
            e[s:end, D] += 1e6
        if moved:
            labels = decode(e, cost, blocks)
        if verbose:
            print("false restarts made dead: {}".format(len(moved)))

    # last: stoppage edges corrected for the bias measured on the validation games
    return trim_dead(labels, e, blocks, cfg)
