"""Make a clean tracking look more like the BAR-ATH ATD, which is automatic and uncorrected.

Calibrated only on aggregate statistics of the BAR-ATH tracking (no labels): 46 slots for ~22
players at a time, a median of 11 visible players per team, track fragments of a median ~35 s,
the ball missing on ~41 % of graded frames in long gaps.

- players: each player's track is cut into fragments (log-normal length) separated by short
  unseen gaps, and consecutive fragments alternate between two slots, as an automatic tracker that
  loses and re-acquires a player under a new identity.
- ball_live_gaps: extra ball gaps while the ball is alive (the Metrica ball is already missing
  during most stoppages, like ours); placed at random, at a rate per minute of live play.
- noise_m: Gaussian noise (m) on every position.

Every option is seeded; the same seed gives the same degraded game.
"""

import numpy as np


def _lognormal_frames(rng, median_s, sigma, size, fps=25):
    return np.maximum(1, np.round(rng.lognormal(np.log(median_s * fps), sigma, size))).astype(int)


def fragment_players(players, team, rng, fragment_median_s, gap_median_s, sigma=1.0):
    """(players (N, 2S, 2), team (2S,)): fragments alternate between slot s and slot S + s."""
    n, S = players.shape[:2]
    out = np.full((n, 2 * S, 2), np.nan)
    for s in range(S):
        t, k = 0, int(rng.integers(2))
        while t < n:
            length = int(_lognormal_frames(rng, fragment_median_s, sigma, 1)[0])
            gap = int(_lognormal_frames(rng, gap_median_s, sigma, 1)[0])
            out[t:t + length, s + k * S] = players[t:t + length, s]
            t += length + gap
            k = 1 - k
    return out, np.r_[team, team].astype(np.int8)


def add_ball_gaps(ball, live, rng, per_min, median_s, sigma=0.8, fps=25):
    """Remove the ball on random stretches starting in live play."""
    ball = ball.copy()
    idx = np.flatnonzero(live)
    n_gaps = rng.poisson(per_min * len(idx) / fps / 60)
    for start in rng.choice(idx, size=n_gaps, replace=False) if n_gaps else []:
        ball[start:start + int(_lognormal_frames(rng, median_s, sigma, 1)[0])] = np.nan
    return ball


def lose_ball_early(ball, rng, median_s, min_gap_s=5.0, sigma=0.8, fps=25):
    """Start every long ball gap earlier: an automatic tracker loses the ball in the air or among
    players before it goes out, not exactly when play stops."""
    ball = ball.copy()
    seen = ~np.isnan(ball[:, 0])
    m = np.diff(np.concatenate(([0], (~seen).astype(np.int8), [0])))
    starts, ends = np.flatnonzero(m == 1), np.flatnonzero(m == -1)
    for s in starts[(ends - starts) >= min_gap_s * fps]:
        lead = int(_lognormal_frames(rng, median_s, sigma, 1)[0])
        ball[max(0, s - lead):s] = np.nan
    return ball


def degrade(tracking, gt_labels, cfg, seed):
    """Degraded copy of a tracking dict (the speeds are recomputed from the new positions)."""
    from metrica import player_speed

    rng = np.random.default_rng(seed)
    out = dict(tracking)
    players, team, ball = tracking["players"], tracking["team"], tracking["ball"]
    scale = np.array(tracking["pitch"])
    if cfg.get("noise_m", 0) > 0:
        players = players + rng.normal(0, cfg["noise_m"], players.shape) / scale
        ball = ball + rng.normal(0, cfg["noise_m"], ball.shape) / scale
    p = cfg.get("players") or {}
    if p.get("enabled", False):
        players, team = fragment_players(players, team, rng, p["fragment_median_s"],
                                         p["gap_median_s"])
    lead = cfg.get("ball_lost_early_s", 0)
    if lead > 0:
        ball = lose_ball_early(ball, rng, lead)
    b = cfg.get("ball_live_gaps") or {}
    if b.get("per_min", 0) > 0:
        ball = add_ball_gaps(ball, np.isin(gt_labels, ["H", "A"]), rng, b["per_min"],
                             b["median_s"])
    out.update(players=players, team=team, ball=ball, speed=player_speed(players))
    return out
