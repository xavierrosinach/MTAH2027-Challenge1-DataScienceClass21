"""Context labeller: dead-ball model + team evidence + Viterbi over {H, A, D}.

Why: the ball is missing on 41% of graded frames, in long stretches (median gap ~11 s). A
frame-by-frame "nearest team" rule has nothing to say there, and a causal hysteresis switches
~switch_frames late at every hand-over. This module instead

1. builds per-frame features that exist with or without a ball (player speeds, time since the
   ball was seen, where it was last / next seen, Smart Tagging),
2. learns P(dead ball) with a logistic regression trained on *algorithmic pseudo-labels*
   (ball clearly out of the pitch, start of a SET PIECES tag, 2nd-half pre-kickoff = dead;
   ball moving at a player's feet in the middle of the pitch = live). No human labels, no GT.
3. turns the ball-player geometry and the Smart Tagging phases into team log-likelihoods (the
   geometry through a model of who controls the ball learnt on other games: control.py),
4. decodes the whole match at once with Viterbi, with a cost for every change of state. Being
   non-causal, the change lands where the evidence changes instead of N frames later, and a
   possession survives a ball gap unless something says otherwise ("possession stays with the
   last team that had it").

All constants live in config.yaml (section `context`) and are set from physical reasoning and
looking at the data, never from the public ground truth.
"""

import os
import warnings

import numpy as np

H, A, D = 0, 1, 2
ATTACK_CODES = ("PROGRESSION", "BUILD UP", "BALL IN FINAL THIRD", "ATTACKING TRANSITION",
                "CREATING CHANCES", "BALL IN THE BOX", "LONG BALL")


# --- helpers

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


# --- features

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


def keeper_teams(tracking, blocks, cfg):
    """(N, S) team of each slot per frame, with goalkeeper tracks fixed by the goal they guard.

    The tracker sometimes swaps the identities of the two goalkeepers, so the home keeper's
    slot turns up in the away goal. A goalkeeper slot is one that spends more than
    `share` of a half within `depth_m` of one goal line. On every frame it is given to the team
    that defends the goal it is nearer to (from where each team's outfield players stand in
    that half). Other slots keep the team in the metadata.
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


def frame_features(tracking, ball, tags, cfg):
    """Per-frame features (dict of arrays) shared by the dead-ball model and the team evidence."""
    n = len(ball)
    pitch = tracking["pitch"]
    length, width = pitch
    fps = 25
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
    t_since = np.where(has_last, np.arange(n) - last_i, 10 ** 6) / fps
    t_until = np.where(has_next, next_i - np.arange(n), 10 ** 6) / fps

    # ball speed (m/s) from positions 4 frames apart, NaN when not seen at both ends
    k = 4
    disp = np.full(n, np.nan)
    d = (ball[k:] - ball[:-k]) * np.array([length, width])
    disp[k:] = np.hypot(d[:, 0], d[:, 1]) * fps / k

    # nearest player of each team (m)
    diff = (players - ball[:, None, :]) * np.array([length, width])
    dist = np.hypot(diff[..., 0], diff[..., 1])
    dist = np.where(np.isnan(dist), np.inf, dist)
    d_h = np.where(team == 0, dist, np.inf).min(axis=1)
    d_a = np.where(team == 1, dist, np.inf).min(axis=1)

    # who is on the ball: nearest slot within touch_m of a seen ball (-1 = nobody)
    near_slot = np.argmin(dist, axis=1)
    near_d = dist[np.arange(n), near_slot]
    cp = cfg["context"]["passes"]
    team_full = np.broadcast_to(team, dist.shape)
    near_team = team_full[np.arange(n), near_slot]
    d_rival = np.where(near_team == 0, d_a, d_h)          # nearest opponent of that player
    clean = seen & (near_d < cp["touch_m"]) & (d_rival > cp["contest_m"])
    touch_slot = np.where(clean, near_slot, -1)
    touch_team = np.where(clean, near_team, -1)

    vote_h, vote_a, since_sp = tag_arrays(tags, n)

    # where the ball was heading when last seen: position extrapolated `exit_pred_s` ahead with
    # the velocity of its last 4 frames (NaN velocity -> no extrapolation)
    c = cfg["context"]["ball_dead"]
    vel = np.full((n, 2), np.nan)
    vel[k:] = (ball[k:] - ball[:-k]) / k                       # per frame, 0-1 units
    vl = vel[np.clip(last_i, 0, n - 1)]
    pred = bl + np.nan_to_num(vl) * c["exit_pred_s"] * fps
    edge_pred = np.where(has_last, edge_distance_m(pred[:, 0], pred[:, 1], pitch), 0.0)

    # how long the (seen) ball has had nobody near it while rolling or still
    alone = (seen & (np.minimum(d_h, d_a) > c["isolated_m"])
             & (np.nan_to_num(disp, nan=99) < c["isolated_max_speed"]))
    start = ffill_index(~alone)
    alone_s = np.where(alone, (np.arange(n) - start) / fps, 0.0)

    return {
        "seen": seen, "spd": spd,
        "spd_2s": rolling_nanmean(spd, 2 * fps), "spd_6s": rolling_nanmean(spd, 6 * fps),
        "seen_4s": rolling_nanmean(seen.astype(float), 4 * fps),
        "edge_last": edge_last, "edge_next": edge_next,
        "t_since": t_since, "t_until": t_until,
        "ball_speed": disp, "d_h": d_h, "d_a": d_a,
        "vote_h": vote_h, "vote_a": vote_a, "since_sp": since_sp,
        "edge_pred": edge_pred, "alone_s": alone_s,
        "touch_slot": touch_slot, "touch_team": touch_team, "ball": ball,
    }


# --- dead-ball model

def design_matrix(f):
    """Features for the dead-ball logistic regression: how the players move.

    Deliberately excludes ball visibility and ball position: the pseudo-labels are defined from
    those (live needs a seen ball, dead is mostly ball out / unseen), so a model given them
    would only learn "no ball => dead". Ball position enters separately in `ball_dead_llr`.
    """
    clip = np.clip
    return np.column_stack([
        clip(f["spd_2s"], 0, 5),
        clip(f["spd_6s"], 0, 5),
    ])


def ball_dead_llr(f, cfg):
    """Rule-based dead-ball log-odds from where the ball is, or was last seen."""
    c = cfg["context"]["ball_dead"]
    llr = np.zeros(len(f["seen"]))
    out_now = f["seen"] & (f["edge_last"] < -c["out_m"])
    inside_now = f["seen"] & (f["edge_last"] > c["inside_m"])
    llr[out_now] += c["out_llr"]
    llr[inside_now] -= c["inside_llr"]
    # in a gap that started with the ball out, or with the ball last seen heading out fast
    # enough to cross a line within exit_pred_s: play stays stopped until the ball is seen again
    gap_after_out = ~f["seen"] & ((f["edge_last"] < 0) | (f["edge_pred"] < -c["out_m"]))
    llr[gap_after_out] += c["gap_after_out_llr"]
    # ball left alone (nobody within isolated_m, not flying) for isolated_s: nobody is playing it
    llr[f["alone_s"] >= c["isolated_s"]] += c["isolated_llr"]
    # ball kicked inside the pitch: play is (re)starting
    kicked = inside_now & (np.nan_to_num(f["ball_speed"]) > c["kick_speed"])
    llr[kicked] -= c["kick_llr"]
    # once the ball has gone out, play stays stopped until it is kicked again (the restart),
    # even if the ball is seen back inside: the keeper carrying it to the six-yard box, a player
    # placing it, a ball boy throwing a new one in
    went_out = out_now | gap_after_out
    last_out, last_kick = ffill_index(went_out), ffill_index(kicked)
    since = np.arange(len(llr)) - last_out
    latched = ((last_out >= 0) & (last_out > last_kick) & ~went_out
               & (since <= c["latch_max_s"] * 25))
    llr[latched & inside_now] += c["inside_llr"]        # being inside says nothing here
    llr[latched] += c["latch_llr"]
    return llr


def goal_dead_mask(f, ball, tags, pitch, cfg):
    """True from a goal to the kick-off that follows it.

    A "Kick Off (After Goal)" SET PIECES tag means a goal was scored before it. Players run to
    celebrate, so speed says nothing; the stoppage starts at the last sighting of the ball near
    a goal (inside a penalty-box-sized zone) before the tag, at most `goal_lookback_s` earlier.
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
        lo = max(0, k - int(c["goal_lookback_s"] * 25))
        hits = np.flatnonzero(near_goal[lo:k])
        if len(hits):
            mask[lo + hits[-1] + 1:min(n, k)] = True
    return mask


def pseudo_labels(f, frame_period, cfg):
    """1 = dead, 0 = live, -1 = unknown. Rules only, no human decision about any frame."""
    c = cfg["context"]["pseudo"]
    n = len(f["seen"])
    y = np.full(n, -1, dtype=np.int8)
    out_now = f["seen"] & (f["edge_last"] < -c["out_m"])
    live = (f["seen"] & (f["edge_last"] > c["live_edge_m"])
            & (np.minimum(f["d_h"], f["d_a"]) < c["live_near_m"])
            & (np.nan_to_num(f["ball_speed"]) > c["live_ball_speed"])
            & (f["since_sp"] > c["set_piece_s"] * 25 * 2))
    dead = out_now | (f["since_sp"] < c["set_piece_s"] * 25)
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


# --- emissions and decoding

def add_team_llr(e, llr):
    """H-vs-A log-odds (positive = H) that say nothing about dead vs alive.

    Only the disfavoured team is penalised, so the favoured one keeps its score against D: a
    player standing on the ball to take a free kick says who restarts, not that play is live.
    """
    e[:, H] -= np.maximum(0.0, -llr)
    e[:, A] -= np.maximum(0.0, llr)


def emissions(f, p_dead, cfg, proximity=True):
    """(N, 3) log-likelihood scores for H, A, D (up to a per-frame constant)."""
    c = cfg["context"]
    n = len(p_dead)
    e = np.zeros((n, 3))

    # dead vs alive
    p = np.clip(p_dead, 1e-3, 1 - 1e-3)
    llr_dead = np.clip(np.log(p / (1 - p)), -c["dead_llr_clip"], c["dead_llr_clip"])
    e[:, D] += c["dead_weight"] * llr_dead + ball_dead_llr(f, cfg)

    # ball at a player's feet: exp(-d/scale) per team, team share weighted by how close anyone is
    # (off when the learnt control model replaces it)
    s = c["ball"]["scale_m"]
    w_h, w_a = np.exp(-f["d_h"] / s), np.exp(-f["d_a"] / s)    # 0 when inf (no ball / no players)
    tot = w_h + w_a
    with np.errstate(invalid="ignore", divide="ignore"):
        share_h = np.where(tot > 0, w_h / tot, 0.5)
    strength = c["ball"]["weight"] * np.maximum(w_h, w_a) * proximity
    # loose ball (nobody in contact): exactly no evidence, possession stays with the last team
    strength[np.minimum(f["d_h"], f["d_a"]) > c["ball"]["contact_m"]] = 0.0
    team_llr = strength * np.log(np.clip(share_h, 0.02, 0.98) / np.clip(1 - share_h, 0.02, 0.98))
    add_team_llr(e, team_llr)

    # Smart Tagging attacking phases (weak, coarse in time): only while the ball is not seen,
    # where there is nothing better; with a visible ball the geometry decides
    tag = c["tag_weight"] * np.sign(f["vote_h"] - f["vote_a"]) * ~f["seen"]
    add_team_llr(e, tag)
    return e


def team_shape_features(tracking, blocks, smooth_frames):
    """Per frame, how each team is laid out and moving, with x towards its attacking goal.

    The team with the ball stretches and pushes up; the other compacts and drops. None of these
    use the ball, so they still speak while it is missing.
    """
    P, speed = tracking["players"], tracking["speed"]
    team = tracking.get("team_frame", tracking["team"][None, :])
    length, width = tracking["pitch"]
    n = len(P)
    sign = np.ones(n)                                  # +1: home attacks towards x = 1
    for _, first, last in blocks:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            home_x = np.nanmean(P[first:last + 1, tracking["team"] == 0, 0])
        sign[first:last + 1] = 1 if home_x < 0.5 else -1

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


def shape_evidence(tracking, f, labels, graded, blocks, cfg, verbose):
    """H-vs-A log-odds from team shape, learnt from frames where the first pass is sure.

    Pseudo-labels: frames with the ball seen, the labelled team within own_m of it and the
    other team farther than opp_m. Checked with a blocked (by time) cross-validation.
    """
    c = cfg["context"]["shape"]
    F = team_shape_features(tracking, blocks, int(c["smooth_s"] * 25))
    d_own = np.where(labels == "H", f["d_h"], f["d_a"])
    d_opp = np.where(labels == "H", f["d_a"], f["d_h"])
    sure = (graded & f["seen"] & np.isin(labels, ["H", "A"])
            & (d_own < c["own_m"]) & (d_opp > c["opp_m"]))
    y = (labels == "H").astype(float)
    idx = np.flatnonzero(sure)

    if verbose:
        folds = np.array_split(idx, 10)
        acc = []
        for k in range(10):
            fit = np.concatenate([folds[j] for j in range(10) if j != k])
            m = fit_logistic(F[fit], y[fit])
            acc.append(np.mean((predict_logistic(m, F[folds[k]]) > 0.5) == (y[folds[k]] == 1)))
        print("shape model: {} pseudo-labelled frames, blocked-CV accuracy {:.3f} +/- {:.3f}"
              .format(len(idx), np.mean(acc), np.std(acc)))

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

    The receiver must be on the ball for >= min_receive_frames (control); the passer may touch
    it for as little as min_touch_frames (a first-time pass). Between the two, the ball travels
    at least min_pass_m within max_gap_s, and no opponent has it for >= min_receive_frames;
    a shorter opponent touch is a deflection that did not intercept, which per CHALLENGE.md
    does not change possession. A touch needs no opponent within contest_m: in a crowded box
    the nearest player flickers between defenders as the ball flies over them.
    CHALLENGE.md names consecutive passes as a proxy for controlled possession; a short
    recovery (win it, pass, lose it) shows up as one or two of these.
    """
    c = cfg["context"]["passes"]
    ev = touch_events(f, c["min_touch_frames"])
    def is_pass(a, b):
        (s0, e0, p0, t0), (s1, e1, p1, t1) = a, b
        if t0 != t1 or p0 == p1 or s1 - e0 > c["max_gap_s"] * 25:
            return False
        b0, b1 = f["ball"][e0 - 1], f["ball"][s1]
        return np.hypot((b1[0] - b0[0]) * 105.0, (b1[1] - b0[1]) * 68.0) >= c["min_pass_m"]

    out, prev, last = [], None, None
    for cur in ev:
        s1, e1, p1, t1 = cur
        long_touch = e1 - s1 >= c["min_receive_frames"]
        if long_touch:
            # from the last touch of any kind (a first-time interception that found a teammate),
            # else from the last one that counts (skipping short opposing deflections)
            for src in (last, prev):
                if src is not None and is_pass(src, cur):
                    out.append((src[0], e1, t1))
                    break
        # a short touch by the other team is a deflection unless a teammate then controls the
        # ball (handled above through `last`): it does not take over as `prev`
        if prev is None or long_touch or t1 == prev[3]:
            prev = cur
        last = cur
    return out


def kicker_evidence(e, f, cfg):
    """Whoever kicks the ball gets kick_team_llr for their team over the kick.

    A kick is the ball speeding up past kick_speed; the kicker is the player nearest to the ball
    (within kicker_m) just before it. In open play one kick (a clearance) is less than a change
    of possession; after a stoppage it says who restarts (throw-ins, whose tags carry no team).
    """
    c = cfg["context"]["passes"]
    ks = cfg["context"]["ball_dead"]["kick_speed"]
    bs = np.nan_to_num(f["ball_speed"])
    fast = f["seen"] & (bs > ks)
    starts = np.flatnonzero(fast[1:] & ~fast[:-1]) + 1
    llr = np.zeros(len(e))
    n_kicks = 0
    for k in starts:
        lo = max(0, k - 6)
        d = np.minimum(f["d_h"][lo:k], f["d_a"][lo:k])
        if not len(d) or not np.isfinite(d).any() or d.min() > c["kicker_m"]:
            continue
        j = lo + int(np.argmin(d))
        team = 0 if f["d_h"][j] <= f["d_a"][j] else 1
        llr[j:k + 1] += (c["kick_team_llr"] / (k + 1 - j)) * (1 if team == 0 else -1)
        n_kicks += 1
    add_team_llr(e, llr)
    return n_kicks


def pass_evidence(e, f, cfg):
    """Spread pass_llr over each completed pass, against the other team."""
    c = cfg["context"]["passes"]
    llr = np.zeros(len(e))
    passes = completed_passes(f, cfg)
    for s, end, t in passes:
        llr[s:end] += (c["pass_llr"] / (end - s)) * (1 if t == 0 else -1)
    add_team_llr(e, llr)
    return passes


def unsupported_dead_spells(labels, f, ball, tags, pitch, blocks, frame_period, cfg):
    """D spells with no sign of a stoppage other than slow players.

    A spell is supported if any of these holds:
      - the ball is seen outside the pitch during it (or just before), or it was last seen
        within `line_m` of a line when it went missing;
      - a SET PIECES tag starts during it, up to `sp_before_s` before or `sp_after_s` after
        (the tag starts a few seconds before the restart, and the ball being placed can end
        the spell early);
      - play restarts on a line or on the centre spot: the first ball seen within
        `restart_window_s` after the spell is within `line_m` of a line or `centre_m` of the spot;
      - it lasts at least `long_s` (goals, injuries, substitutions);
      - it is the 2nd-half pre-kickoff stretch.
    """
    c = cfg["context"]["dead_support"]
    length, width = pitch
    n = len(labels)
    sp_starts = tags.loc[tags["code"] == "SET PIECES", "start_frame"].to_numpy()
    look = int(c["restart_window_s"] * 25)
    before = int(c["out_before_s"] * 25)
    bad = []
    for _, first, last in blocks:
        lab = labels[first:last + 1]
        change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        starts, ends = np.r_[0, change] + first, np.r_[change, len(lab)] + first
        for s, end in zip(starts, ends):
            if labels[s] != "D":
                continue
            if frame_period is not None and (frame_period[s:end] == "2nd_half_pre_kickoff").any():
                continue
            if (end - s) >= c["long_s"] * 25:
                continue
            lo = max(0, s - before)
            seen = f["seen"][lo:end]
            if (seen & (f["edge_last"][lo:end] < 0)).any():
                continue
            if not f["seen"][s] and f["edge_last"][s] < c["line_m"]:
                continue
            # ball lost heading out, or left alone with nobody playing it
            out_m = cfg["context"]["ball_dead"]["out_m"]
            if (~f["seen"][lo:end] & (f["edge_pred"][lo:end] < -out_m)).any():
                continue
            if (f["alone_s"][lo:end] >= cfg["context"]["ball_dead"]["isolated_s"]).any():
                continue
            if ((sp_starts >= s - c["sp_before_s"] * 25)
                    & (sp_starts < end + c["sp_after_s"] * 25)).any():
                continue
            nxt = np.flatnonzero(f["seen"][end:min(n, end + look)])
            if len(nxt):
                bx, by = ball[end + nxt[0]]
                if edge_distance_m(bx, by, pitch) < c["line_m"]:
                    continue
                if np.hypot((bx - 0.5) * length, (by - 0.5) * width) < c["centre_m"]:
                    continue
            bad.append((s, end))
    return bad


def restart_evidence(e, tags, cfg):
    """SET PIECES with a team: that team restarts, so favour it just after the tag starts.

    The tag starts a few seconds before the restart (players slowest ~2-4 s in, back to speed
    by ~8 s), so the window is [delay, delay + length] seconds after the start.
    """
    c = cfg["context"]["restart"]
    n = len(e)
    for row in tags[tags["code"] == "SET PIECES"].itertuples():
        if row.team not in ("H", "A"):
            continue
        lo = max(0, row.start_frame + int(c["delay_s"] * 25))
        hi = min(n, lo + int(c["length_s"] * 25))
        own, other = (H, A) if row.team == "H" else (A, H)
        e[lo:hi, other] -= c["weight"]


def viterbi(e, team_switch, to_dead, from_dead, min_dead):
    """Max-score H/A/D path where every dead-ball stretch lasts at least min_dead frames.

    States: 0 H, 1 A, then a chain D_1 ... D_k (k = min_dead): entering D lands on D_1, the chain
    must be walked to D_k, and only D_k may stay or restart play. Without the minimum, a hop
    H -> D -> A of one frame is cheaper than a direct hand-over whenever
    to_dead + from_dead < team_switch, and every change of possession grows a fake stoppage.

    from_dead may be a scalar or a per-frame array (the cost of restarting play at frame t).
    """
    n, k = len(e), max(1, int(min_dead))
    eh, ea, ed = e[:, H].tolist(), e[:, A].tolist(), e[:, D].tolist()
    fd = (np.broadcast_to(np.asarray(from_dead, dtype=float), (n,))).tolist()
    back_h = np.zeros(n, dtype=np.int8)     # 0 from H, 1 from A, 2 from D_k
    back_a = np.zeros(n, dtype=np.int8)
    back_d1 = np.zeros(n, dtype=np.int8)    # 0 from H, 1 from A
    back_dk = np.zeros(n, dtype=np.int8)    # 0 from D_{k-1}, 1 from D_k (only if k > 1)
    NEG = -1e18
    sh, sa = eh[0], ea[0]
    chain = [NEG] * k
    chain[0] = ed[0]
    for t in range(1, n):
        dk = chain[-1]
        # H
        # ties go to the change: with no evidence either way, the change lands as late as
        # possible, i.e. when the new team first touches the ball, not when the old one lost it
        c0, c1, c2 = sh, sa - team_switch, dk - fd[t]
        if c0 > c1 and c0 > c2:
            nh, bh = c0, 0
        elif c1 >= c2:
            nh, bh = c1, 1
        else:
            nh, bh = c2, 2
        # A
        c0, c1, c2 = sa, sh - team_switch, dk - fd[t]
        if c0 > c1 and c0 > c2:
            na, ba = c0, 1
        elif c1 >= c2:
            na, ba = c1, 0
        else:
            na, ba = c2, 2
        # D chain
        enter_h, enter_a = sh - to_dead, sa - to_dead
        d1, b1 = (enter_h, 0) if enter_h >= enter_a else (enter_a, 1)
        if k == 1:
            if dk >= d1:
                new_chain, bk = [dk + ed[t]], 1
            else:
                new_chain, bk = [d1 + ed[t]], 0
            back_d1[t] = b1
        else:
            stay, step = chain[-1], chain[-2]
            last, bk = (stay, 1) if stay > step else (step, 0)
            new_chain = [d1] + chain[:-2] + [last]
            new_chain = [v + ed[t] for v in new_chain]
            back_d1[t] = b1
        back_h[t], back_a[t], back_dk[t] = bh, ba, bk
        sh, sa, chain = nh + eh[t], na + ea[t], new_chain

    # backtrack; state code: 0 H, 1 A, 2+i = D_{i+1}
    path = np.empty(n, dtype=np.int8)
    finals = [sh, sa] + chain
    s = int(np.argmax(finals))
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
        else:                             # D_i, 1 < i < k, came from D_{i-1}
            s -= 1
    return path


def decode_by_half(e, cost, blocks):
    """Viterbi independently on each graded half; ungraded frames stay D."""
    path = np.full(len(e), D, dtype=np.int8)
    for _, first, last in blocks:
        c = dict(cost)
        if np.ndim(c["from_dead"]):
            c["from_dead"] = c["from_dead"][first:last + 1]
        path[first:last + 1] = viterbi(e[first:last + 1], **c)
    return path


def restart_cost(f, cfg):
    """Per-frame cost of restarting play: dearer while the ball is seen and not being kicked.

    A restart is a kick (or throw). If the ball is visible and nobody kicks it within
    +/- kick_window_s, play has not restarted yet: the ball is being placed or carried.
    With the ball unseen there is no way to tell, so the plain cost applies.
    """
    t = cfg["context"]["transition_cost"]
    c = cfg["context"]["ball_dead"]
    w = int(t["kick_window_s"] * 25)
    kick = f["seen"] & (np.nan_to_num(f["ball_speed"]) > c["kick_speed"])
    near_kick = rolling_nanmean(kick.astype(float), w) > 0
    cost = np.full(len(kick), float(t["from_dead"]))
    cost[f["seen"] & ~near_kick] += t["restart_without_kick"]
    return cost


def label(tracking, ball, tags, cfg, blocks, frame_period=None, verbose=True):
    """Full context labeller. Returns labels (N,) in {'H','A','D'} and diagnostics."""
    c = cfg["context"]
    if c["keepers"]["enabled"]:
        tracking = dict(tracking)
        tracking["team_frame"], fixed = keeper_teams(tracking, blocks, cfg)
        if verbose:
            print("goalkeeper slots re-assigned by goal on {} slot-frames".format(fixed))
    f = frame_features(tracking, ball, tags, cfg)
    X = design_matrix(f)
    y = pseudo_labels(f, frame_period, cfg)
    graded = np.zeros(len(y), dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    train = graded & (y >= 0)
    model = fit_logistic(X[train], y[train].astype(float))
    p_dead = predict_logistic(model, X)

    ctl = c.get("control") or {"enabled": False}
    e = emissions(f, p_dead, cfg, proximity=not (ctl["enabled"] and ctl["replace_proximity"]))
    if ctl["enabled"]:
        # learnt "who controls the ball" (src/possession/control.py), trained on other games
        from . import control
        from .config import REPO_ROOT
        path = ctl["model"] if os.path.isabs(ctl["model"]) else os.path.join(REPO_ROOT,
                                                                              ctl["model"])
        llr, seen = control.control_llr(tracking, ball, blocks, path, control.feature_groups(cfg))
        llr = np.clip(llr, -ctl["clip"], ctl["clip"])
        add_team_llr(e, np.where(seen, ctl["weight"], ctl["gap_weight"]) * llr)
        if verbose:
            print("control model {}: |llr| > 2 on {:.0%} of frames with ball".format(
                os.path.basename(path), np.mean(np.abs(llr[seen]) > 2)))
    if frame_period is not None:                       # FAQ: always D before the 2nd kickoff
        e[frame_period == "2nd_half_pre_kickoff", D] += 1e6
    goal = goal_dead_mask(f, ball, tags, tracking["pitch"], cfg)
    e[goal, D] += 1e6
    if verbose:
        print("goal stoppages: {:.0f} s forced dead".format(goal.sum() / 25))

    restart_evidence(e, tags, cfg)
    if c["passes"]["kick_team_llr"] > 0:
        n_kicks = kicker_evidence(e, f, cfg)
        if verbose:
            print("kicks with a kicker: {}".format(n_kicks))
    if c["passes"]["pass_llr"] > 0:
        passes = pass_evidence(e, f, cfg)
        if verbose:
            print("completed passes found: {}".format(len(passes)))

    t = c["transition_cost"]
    cost = {"team_switch": t["team_switch"], "to_dead": t["to_dead"],
            "from_dead": restart_cost(f, cfg), "min_dead": int(round(t["min_dead_s"] * 25))}
    path = decode_by_half(e, cost, blocks)
    labels = np.array(["H", "A", "D"])[path]

    # second pass: team shape speaks where the ball is missing
    if c["shape"]["weight"] > 0:
        shape = shape_evidence(tracking, f, labels, graded, blocks, cfg, verbose)
        llr = c["shape"]["weight"] * np.clip(shape, -c["shape"]["clip"], c["shape"]["clip"])
        llr = llr * ~f["seen"]
        add_team_llr(e, llr)
        path = decode_by_half(e, cost, blocks)
        labels = np.array(["H", "A", "D"])[path]

    # third pass: a stoppage needs a reason; slow players alone are not one
    if c["dead_support"]["enabled"]:
        bad = unsupported_dead_spells(labels, f, ball, tags, tracking["pitch"], blocks,
                                      frame_period, cfg)
        for s, end in bad:
            e[s:end, D] -= 1e6
        if bad:
            path = decode_by_half(e, cost, blocks)
            labels = np.array(["H", "A", "D"])[path]
        if verbose:
            print("dead spells without support, made live: {} ({:.0f} s)".format(
                len(bad), sum(end - s for s, end in bad) / 25))

    if verbose:
        print("dead model: {} pseudo-dead, {} pseudo-live frames; weights {}".format(
            int((y[train] == 1).sum()), int((y[train] == 0).sum()),
            np.round(model[2], 2).tolist()))
    return labels, {"p_dead": p_dead, "features": f, "pseudo": y, "emissions": e}
