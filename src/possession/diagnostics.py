"""Checks of a labelling that need no ground truth: is it consistent with the match itself?

Used to compare variants instead of the public 5 minutes (which may not be used for tuning).
Every check states what a sensible labelling should do; none of them reads any answer key.
"""

import numpy as np

FPS = 25


def runs(labels):
    """(start, end_exclusive, label) of each maximal run."""
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(labels)]
    return [(s, e, labels[s]) for s, e in zip(starts, ends)]


def rival_control_episodes(labels, f, graded, near_m=1.0, min_s=0.5):
    """(start, end) runs where the team NOT labelled in possession is on the ball."""
    d_own = np.where(labels == "H", f["d_h"], f["d_a"])
    d_opp = np.where(labels == "H", f["d_a"], f["d_h"])
    rival = (graded & f["seen"] & np.isin(labels, ["H", "A"])
             & (d_opp < near_m) & (d_opp < d_own))
    out = []
    for s, e, v in runs(rival):
        if v and e - s >= min_s * FPS:
            out.append((s, e))
    return out


def report(labels, f, tags, blocks, p_dead=None, passes=None):
    """Dict of diagnostics over the graded frames."""
    out = {}
    graded = np.zeros(len(labels), dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True

    spells, dead_spells, switches = [], [], []
    for _, first, last in blocks:
        lab = labels[first:last + 1]
        for s, e, v in runs(lab):
            s, e = s + first, e + first
            (spells if v in "HA" else dead_spells).append((s, e, v))
        # direct hand-overs H<->A
        ch = np.flatnonzero((lab[1:] != lab[:-1]) & np.isin(lab[1:], ["H", "A"])
                            & np.isin(lab[:-1], ["H", "A"])) + 1 + first
        switches.extend(ch.tolist())

    dur = np.array([(e - s) / FPS for s, e, _ in spells])
    ddur = np.array([(e - s) / FPS for s, e, _ in dead_spells])
    out["share_D"] = float(np.mean(labels[graded] == "D"))
    out["share_H_of_alive"] = float(np.mean(labels[graded & (labels != "D")] == "H"))
    out["n_possessions"] = len(spells)
    out["poss_median_s"] = float(np.median(dur))
    out["poss_lt_2s"] = int((dur < 2).sum())
    out["poss_lt_5s"] = int((dur < 5).sum())
    out["n_dead_spells"] = len(dead_spells)
    out["dead_lt_1s"] = int((ddur < 1).sum())
    out["n_direct_switches"] = len(switches)

    # a hand-over should come with the new team near the ball (within +/- 1 s)
    if switches:
        near, in_gap = [], []
        for t in switches:
            new = labels[t]
            lo, hi = max(0, t - FPS), min(len(labels), t + FPS)
            d_new = f["d_h"][lo:hi] if new == "H" else f["d_a"][lo:hi]
            near.append(np.min(d_new) < 1.5)
            in_gap.append(not f["seen"][t])
        out["switch_new_team_near_ball"] = float(np.mean(near))
        out["switch_inside_ball_gap"] = float(np.mean(in_gap))

    # each SET PIECES tag is a restart: D just before it is taken, then the tag's team on the ball
    sp = tags[(tags["code"] == "SET PIECES")]
    sp = sp[[graded[min(s, len(labels) - 1)] for s in sp["start_frame"]]]
    dead_at, team_ok, team_n = [], 0, 0
    for row in sp.itertuples():
        s = row.start_frame
        win = labels[s:s + 6 * FPS]
        dead_at.append(bool((win == "D").any()))
        after = labels[s:s + 20 * FPS]
        alive = after[after != "D"]
        if row.team in ("H", "A") and (win == "D").any() and len(alive):
            team_n += 1
            team_ok += alive[0] == row.team
    out["set_piece_has_D_within_6s"] = float(np.mean(dead_at)) if dead_at else float("nan")
    out["set_piece_restart_team_matches"] = team_ok / team_n if team_n else float("nan")
    out["set_piece_n_with_team"] = team_n

    # time per possession with the team that has it actually near the ball (when the ball is seen)
    seen_alive = graded & f["seen"] & np.isin(labels, ["H", "A"])
    d_own = np.where(labels == "H", f["d_h"], f["d_a"])
    d_opp = np.where(labels == "H", f["d_a"], f["d_h"])
    close = seen_alive & (np.minimum(d_own, d_opp) < 1.5)
    out["own_team_closer_when_contact"] = float(np.mean(d_own[close] < d_opp[close]))

    # opponent clearly on the ball (within 1 m, nearer than anyone of the labelled team) for
    # >= 0.5 s, with no change of possession: either a touch without control or a missed change
    eps = rival_control_episodes(labels, f, graded)
    lens = np.array([e - s for s, e in eps]) / FPS
    out["rival_control_ge_0.5s"] = len(eps)
    out["rival_control_ge_1s"] = int((lens >= 1).sum())
    out["rival_control_ge_2s"] = int((lens >= 2).sum())

    # every restart (D -> H/A) should be a kick: ball seen moving > 6 m/s within 1 s of it.
    # A restart with no kick around it ended the stoppage while the ball was still being placed
    restarts = [e for s, e, v in dead_spells if e < len(labels) and labels[e] in "HA"]
    if restarts:
        bs = np.nan_to_num(f["ball_speed"])
        kick = [(bs[max(0, t - FPS // 2):t + FPS] > 6).any() for t in restarts]
        out["restart_with_kick"] = float(np.mean(kick))
        out["n_restarts"] = len(restarts)

    # completed passes by the team NOT labelled in possession (a short possession left out)
    if passes is not None:
        rival, chains, last_end = 0, 0, -10 ** 9
        for s, e, t in passes:
            if not graded[s]:
                continue
            lab = labels[s:e]
            other = "A" if t == 0 else "H"
            if np.mean(lab == other) > 0.5:
                rival += 1
                chains += s - last_end <= 4 * FPS     # a second rival pass right after
                last_end = e
        out["passes_total"] = sum(graded[s] for s, _, _ in passes)
        out["rival_passes_inside_possession"] = rival
        out["rival_pass_pairs_inside_possession"] = int(chains)

    if p_dead is not None:
        out["mean_p_dead_on_D"] = float(np.mean(p_dead[graded & (labels == "D")]))
        out["mean_p_dead_on_alive"] = float(np.mean(p_dead[graded & (labels != "D")]))
    return out


def print_report(rep, title="diagnostics"):
    print("\n{} (no ground truth used)".format(title))
    for k, v in rep.items():
        print("  {:34s} {}".format(k, round(v, 3) if isinstance(v, float) else v))
