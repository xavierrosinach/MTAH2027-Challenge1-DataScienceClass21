"""A per-frame H/A/D ground truth for the Metrica sample games, derived mechanically from their
(hand-collected) event data. Other games only: RULES.md allows human labels on other matches.

The rules follow CHALLENGE.md ("Possession rules"):

- Graded frames: each period, from its kick-off to the end of its last event. Everything else X.
- Dead ball (D): from the end of the last action before a restart to the restart itself. A
  restart is a SET PIECE event (kick-off, throw-in, free kick, corner, penalty) or a goal kick
  (coded by Metrica as a PASS with subtype GOAL KICK). Cards are given during the stoppage and are
  not actions.
- Ball alive: the team in possession is the team of the last *controlled* action -- a pass, a
  carry, a recovery, a shot, a restart, or a ball lost (the team losing it had it until then).
  Challenges (duels, aerial or ground), fouls received and balls out are not control, so a
  contested or loose ball stays with the last team that had it, and a touch, deflection or aerial
  duel without control does not change possession.

The change of possession lands on the first frame of the new team's controlled action.
This is an approximation of the challenge's ground truth (which comes from corrected tracking),
documented in the technical report; every version is scored against the same one.
"""

import numpy as np

RESTART_TYPES = ("SET PIECE",)
CONTROL_TYPES = ("PASS", "CARRY", "RECOVERY", "SHOT", "SET PIECE", "BALL LOST")
NOT_ACTIONS = ("CARD",)


def is_restart(ev):
    return ev["type"].isin(RESTART_TYPES) | ((ev["type"] == "PASS")
                                             & ev["subtype"].str.contains("GOAL KICK"))


def absorb_short_spells(labels, blocks, min_s, fps=25):
    """Stress-test variant of the ground truth: a possession shorter than min_s between two
    possessions of the other team (A-H-A) is taken as a touch without control and given to them.

    The challenge's own ground truth does have very short possessions, so this is not the
    definition; it checks that a decision does not hinge on how short possessions are counted.
    """
    labels = labels.copy()
    for _, a, b in blocks:
        lab = labels[a:b + 1]
        change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        starts, ends = np.r_[0, change], np.r_[change, len(lab)]
        for i in range(1, len(starts) - 1):
            mid, prev, nxt = lab[starts[i]], lab[starts[i - 1]], lab[starts[i + 1]]
            if (mid in ("H", "A") and prev == nxt and prev in ("H", "A") and prev != mid
                    and ends[i] - starts[i] < min_s * fps):
                lab[starts[i]:ends[i]] = prev
        labels[a:b + 1] = lab
    return labels


def ground_truth(events, n_frames, min_spell_s=0.0):
    """(labels (N,) of 'H'/'A'/'D'/'X', blocks [(period, first, last)] graded, inclusive)."""
    labels = np.full(n_frames, "X", dtype="<U1")
    restart = is_restart(events)
    blocks = []
    for period, ev in events.groupby("period", sort=True):
        ev = ev.sort_values(["start", "end"], kind="stable")
        r = restart.loc[ev.index]
        kickoffs = ev[r & (ev["subtype"].str.contains("KICK OFF"))]
        if not len(kickoffs):
            continue
        first = int(kickoffs["start"].iloc[0])
        last = int(min(n_frames - 1, ev["end"].max()))
        blocks.append((int(period), first, last))

        # team of the last controlled action, forward-filled
        owner = np.full(last - first + 1, "", dtype="<U1")
        ctrl = ev[ev["type"].isin(CONTROL_TYPES) & ev["team"].isin(["H", "A"])
                  & (ev["start"] >= first) & (ev["start"] <= last)]
        idx = np.full(last - first + 1, -1)
        idx[(ctrl["start"] - first).to_numpy()] = np.arange(len(ctrl))
        idx = np.maximum.accumulate(idx)
        teams = ctrl["team"].to_numpy()
        owner[idx >= 0] = teams[idx[idx >= 0]]
        lab = np.where(owner == "", "D", owner)

        # stoppages: from the end of the last action to each restart
        actions = ev[~ev["type"].isin(NOT_ACTIONS)]
        for s in ev.loc[r & (ev["start"] > first), "start"].to_numpy():
            before = actions[actions["start"] < s]
            if not len(before):
                continue
            stop = int(min(s - 1, before["end"].max()))
            if stop + 1 < s:
                lab[stop + 1 - first:s - first] = "D"
        labels[first:last + 1] = lab
    if min_spell_s > 0:
        labels = absorb_short_spells(labels, blocks, min_spell_s)
    return labels, blocks


def summary(labels, blocks, fps=25):
    """Counts a coach would read: graded minutes, share dead, possessions and their length."""
    n_spells, lengths = 0, []
    for _, a, b in blocks:
        lab = labels[a:b + 1]
        change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        starts, ends = np.r_[0, change], np.r_[change, len(lab)]
        for s, e in zip(starts, ends):
            if lab[s] in ("H", "A"):
                n_spells += 1
                lengths.append(e - s)
    graded = labels != "X"
    return {"graded_min": graded.sum() / fps / 60,
            "dead_share": float((labels[graded] == "D").mean()),
            "home_share_alive": float((labels == "H").sum() / max(1, np.isin(labels, ["H", "A"]).sum())),
            "possessions": n_spells,
            "per_min": n_spells / (graded.sum() / fps / 60),
            "mean_s": float(np.mean(lengths)) / fps if lengths else 0.0,
            "median_s": float(np.median(lengths)) / fps if lengths else 0.0}
