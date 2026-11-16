"""Team of each track from the shirt colour in the video.

The provided tracking gives every slot a fixed team (the metadata). When two rivals overlap in
the image the tracker loses one of them, and the track that is born afterwards is sometimes given
to the wrong team, or a live track slides onto the rival. From then on a home player is drawn
(and counted) as away. The video still shows the shirt.

Three steps, none of them uses a label:
  1. sample()    every `step` frames, the colour of a box on the torso of every seen slot (pitch
                 position -> pixel with the homography), described by a few numbers.
  2. fit()       a logistic model shirt colour -> team, trained with the metadata team of each
                 slot as the target, only on players alone in the image. Most tracks are right,
                 so the model learns the two kits; samples it contradicts are dropped and it is
                 fitted again.
  3. decode()    per slot, the most likely team over time (two states, a cost to change) from the
                 model's evidence on the frames where the player is alone in the image.

    python src/possession/jersey.py [--config config.yaml]     # sample + report, fills the cache
"""

import os
import sys

import numpy as np

FEATURES = ("b", "g", "r", "sb", "sg", "sr", "light", "dark", "red", "blue", "rel_v", "cover")


def _project(H, x, y):
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    return (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w, (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w


def keeper_slots(tracking, blocks, cfg):
    """Slots that live by one goal line in some half (same test as context.keeper_teams)."""
    c = cfg["context"]["keepers"]
    X = tracking["players"][:, :, 0]
    length = tracking["pitch"][0]
    keep = np.zeros(X.shape[1], dtype=bool)
    for _, first, last in blocks:
        x = X[first:last + 1]
        seen = ~np.isnan(x)
        near = (np.minimum(x, 1 - x) * length < c["depth_m"]) & seen
        share = near.sum(axis=0) / np.maximum(1, seen.sum(axis=0))
        keep |= (share > c["share"]) & (seen.mean(axis=0) > 0.2)
    return keep


def sample(tracking, blocks, video_path, homography, cfg, verbose=True, valid=None):
    """Torso colour of every seen slot every `step` frames.

    Returns dict of arrays, one row per (frame, slot) sample: frame, slot, X (features),
    ok (enough shirt pixels in the box), alone (no other tracked player over the box).
    """
    import cv2

    c = cfg["context"]["jersey"]
    step = int(c["step"])
    P = tracking["players"]
    L = tracking["pitch"][0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("cannot open " + video_path)
    lo, hi, half_w = c["torso_from_m"], c["torso_to_m"], c["torso_half_width_m"]
    rows_f, rows_s, rows_x, rows_ok, rows_alone = [], [], [], [], []
    for _, first, last in blocks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        for f in range(first, last + 1):
            if (f - first) % step:
                cap.grab()
                continue
            ok, img = cap.read()
            if not ok:
                break
            if valid is not None and not valid[f]:
                continue                                  # no homography on this frame (close-up)
            if verbose and (f - first) % (step * 4000) == 0:
                print("  jersey: frame {} / {}".format(f, last), flush=True)
            slots = np.flatnonzero(~np.isnan(P[f, :, 0]))
            if not len(slots):
                continue
            H = homography[f]
            x, y = P[f, slots, 0], P[f, slots, 1]
            u, v = _project(H, x, y)
            u1, _ = _project(H, x + 0.5 / L, y)
            u0, _ = _project(H, x - 0.5 / L, y)
            s = np.abs(u1 - u0)                                   # pixels per metre at the player
            du = u[None, :] - u[:, None]                          # other - me
            dv = v[None, :] - v[:, None]
            over = (np.abs(du) < 0.9 * s[:, None]) & (dv > -1.7 * s[:, None]) & (dv < 1.1 * s[:, None])
            np.fill_diagonal(over, False)
            alone = ~over.any(axis=1)
            hh, ww = img.shape[:2]
            for k, slot in enumerate(slots):
                if not (np.isfinite(u[k]) and np.isfinite(v[k]) and s[k] > 4):
                    continue
                x0, x1 = int(round(u[k] - half_w * s[k])), int(round(u[k] + half_w * s[k])) + 1
                y0, y1 = int(round(v[k] - hi * s[k])), int(round(v[k] - lo * s[k])) + 1
                if x0 < 2 or y0 < 2 or x1 > ww - 2 or y1 > hh - 2:
                    continue
                m = int(round(1.2 * s[k]))                        # surroundings: the grass around him
                Y0, Y1, X0, X1 = max(0, y0 - m), min(hh, y1 + m), max(0, x0 - m), min(ww, x1 + m)
                big = cv2.cvtColor(img[Y0:Y1, X0:X1], cv2.COLOR_BGR2HSV)
                grass_big = (big[:, :, 0] > 32) & (big[:, :, 0] < 92) & (big[:, :, 1] > 60)
                grass_v = float(big[:, :, 2][grass_big].mean()) if grass_big.any() else 128.0
                hsv = big[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0].reshape(-1, 3).astype(np.int32)
                bgr = img[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
                shirt = ~grass_big[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0].reshape(-1)
                cover = float(shirt.mean())
                good = shirt.sum() >= 4 and cover > 0.25
                if good:
                    px, hs = bgr[shirt], hsv[shirt]
                    hue, sat, val = hs[:, 0], hs[:, 1], hs[:, 2]
                    feat = (*(px.mean(axis=0) / 255), *(px.std(axis=0) / 255),
                            float(((val > 150) & (sat < 90)).mean()), float((val < 90).mean()),
                            float((((hue < 12) | (hue > 165)) & (sat > 90)).mean()),
                            float(((hue > 100) & (hue < 135) & (sat > 90)).mean()),
                            float(val.mean() / max(grass_v, 1.0)), cover)
                else:
                    feat = (0.0,) * len(FEATURES)
                rows_f.append(f)
                rows_s.append(slot)
                rows_x.append(feat)
                rows_ok.append(good)
                rows_alone.append(bool(alone[k]))
    cap.release()
    return {"frame": np.array(rows_f, dtype=np.int32), "slot": np.array(rows_s, dtype=np.int16),
            "X": np.array(rows_x, dtype=np.float32), "ok": np.array(rows_ok, dtype=bool),
            "alone": np.array(rows_alone, dtype=bool)}


def _fit_logistic(X, y, l2=1e-3, iters=30):
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-6
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    R = l2 * len(Z) * np.eye(Z.shape[1])
    R[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - y) + R @ w
        Hs = (Z * (p * (1 - p))[:, None]).T @ Z + R
        w -= np.linalg.solve(Hs, g)
    return {"mu": mu, "sd": sd, "w": w}


def _score(model, X):
    return np.c_[np.ones(len(X)), (X - model["mu"]) / model["sd"]] @ model["w"]


def _features(X):
    return np.c_[X, X[:, :3] * X[:, 10:11], X[:, 6:10] ** 2]


def fit(samples, team, keepers, cfg, verbose=True):
    """Evidence for "away" (log odds) of every sample; NaN where the box had no shirt.

    The target is the metadata team. Checked on time the model was not fitted on (odd / even
    10 s chunks). Returns (llr, report dict).
    """
    c = cfg["context"]["jersey"]
    X = _features(samples["X"].astype(np.float64))
    y = team[samples["slot"]].astype(float)
    use = samples["ok"] & samples["alone"] & ~keepers[samples["slot"]]
    chunk = (samples["frame"] // 250) % 2
    report = {}
    keep = use.copy()
    for round_ in range(2):                    # second round: without the samples the model contradicts
        model = _fit_logistic(X[keep], y[keep], l2=c["l2"])
        z = _score(model, X)
        if round_ == 0:
            keep = use & ((z > 0) == (y > 0.5))
    half = _fit_logistic(X[use & (chunk == 0)], y[use & (chunk == 0)], l2=c["l2"])
    zt = _score(half, X)
    test = use & (chunk == 1)
    hit = (zt[test] > 0) == (y[test] > 0.5)
    sure = np.abs(zt[test]) > 2
    report["n_samples"] = int(len(y))
    report["n_alone"] = int(use.sum())
    report["agree_heldout"] = float(hit.mean())
    report["agree_heldout_confident"] = float(hit[sure].mean())
    report["share_confident"] = float(sure.mean())
    # a colour that is neither kit (referee, goalkeeper, staff) is no evidence: distance to the
    # nearer kit, in units of that kit's own spread, beyond what `kit_quantile` of its samples have
    C = samples["X"][:, [0, 1, 2, 10]].astype(np.float64)
    d2 = np.full(len(C), np.inf)
    for g in (0, 1):
        own = keep & (y == g)
        mu, cov = C[own].mean(axis=0), np.cov(C[own].T) + 1e-6 * np.eye(C.shape[1])
        d = np.einsum("ni,ij,nj->n", C - mu, np.linalg.inv(cov), C - mu)
        d2 = np.minimum(d2, d / np.quantile(d[own], c["kit_quantile"]))
    report["share_not_a_kit"] = float((d2[use] > 1).mean())
    z = np.where(samples["ok"] & (d2 <= 1), z, np.nan)
    if verbose:
        print("  jersey model: {} samples alone, agreement with the metadata on held-out time {:.3f} "
              "({:.3f} on the {:.0f} % with |log odds| > 2)".format(
                  report["n_alone"], report["agree_heldout"], report["agree_heldout_confident"],
                  100 * report["share_confident"]))
    return z, report


def decode(samples, llr, team, keepers, n_frames, cfg, conflict=None, vote=None):
    """(N, S) team per frame: per slot, Viterbi over its samples with two states (home, away).

    Evidence clipped to +-`clip`, times `overlap_weight` where another tracked player is over the box.
    Leaving the metadata team on a continuous track costs `switch_cost`, coming back
    `return_cost`. After a gap longer than `gap_s` it is a new track: it starts afresh, and
    starting against the metadata costs `prior_cost`. Goalkeeper slots are left alone.
    Returns (T, list of (slot, first frame, last frame) segments given to the other team).
    """
    c = cfg["context"]["jersey"]
    S = len(team)
    T = np.repeat(team[None, :], n_frames, axis=0).astype(np.int8)
    step, gap = int(c["step"]), int(c["gap_s"] * 25)
    changed = []
    order = np.lexsort((samples["frame"], samples["slot"]))
    fr, sl, al = samples["frame"][order], samples["slot"][order], samples["alone"][order]
    z = llr[order]
    # a sample under another tracked player counts less (the box may show the other shirt)
    ev = np.clip(np.nan_to_num(z), -c["clip"], c["clip"]) * np.where(al, 1.0, c["overlap_weight"])
    bounds = np.searchsorted(sl, np.arange(S + 1))
    for s in range(S):
        a, b = bounds[s], bounds[s + 1]
        if b - a < 2 or keepers[s]:
            continue
        f, e = fr[a:b], ev[a:b]
        m = len(f)
        new_track = np.diff(f) > gap
        # where another camera gives this player to the other team the metadata is no prior
        doubt = np.zeros(m, dtype=bool)
        if conflict is not None:
            for k in range(-2, 3):
                doubt |= conflict[np.clip(f + k, 0, n_frames - 1), s]
        if vote is not None:                                # what the other camera says, as evidence
            v = np.stack([vote[np.clip(f + k, 0, n_frames - 1), s] for k in range(-2, 3)])
            e = e + v[np.abs(v).argmax(axis=0), np.arange(m)]
        home, other = int(team[s]), 1 - int(team[s])
        gain = np.stack([-e / 2, e / 2], axis=1)            # state 0 home, state 1 away
        best = np.zeros((m, 2))
        back = np.zeros((m, 2), dtype=np.int8)
        best[0] = gain[0]
        best[0, other] -= 0.0 if doubt[0] else c["prior_cost"]
        for i in range(1, m):
            if doubt[i]:                                    # free to follow the shirt
                top = int(np.argmax(best[i - 1]))
                for st in (0, 1):
                    best[i, st], back[i, st] = best[i - 1, top] + gain[i, st], top
                continue
            if new_track[i - 1]:                            # the tracker decided again: start afresh
                top = int(np.argmax(best[i - 1]))
                best[i, home], back[i, home] = best[i - 1, top] + gain[i, home], top
                best[i, other], back[i, other] = best[i - 1, top] - c["prior_cost"] + gain[i, other], top
                continue
            for st, cost in ((home, c["return_cost"]), (other, c["switch_cost"])):
                stay, move = best[i - 1, st], best[i - 1, 1 - st] - cost
                if move > stay:
                    best[i, st], back[i, st] = move + gain[i, st], 1 - st
                else:
                    best[i, st], back[i, st] = stay + gain[i, st], st
        path = np.zeros(m, dtype=np.int8)
        path[-1] = int(np.argmax(best[-1]))
        for i in range(m - 1, 0, -1):
            path[i - 1] = back[i, path[i]]
        # samples -> frames: each sample owns the frames around it up to the next sample of the
        # same track (a change across a gap takes effect when the track comes back)
        off = path != team[s]
        for i in np.flatnonzero(off):
            new_track = i == 0 or f[i] - f[i - 1] > gap
            lo = f[i] - (step - 1 if new_track or not off[i - 1] else 0)
            last_of_track = i + 1 == m or f[i + 1] - f[i] > gap
            hi = f[i] + step if last_of_track else f[i + 1]
            T[max(lo, 0):min(hi, n_frames), s] = path[i]
        d = np.diff(np.r_[0, off.astype(int), 0])
        for i0, i1 in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            changed.append((s, int(f[i0]), int(f[i1 - 1])))
    return T, changed


def load_samples(tracking, blocks, cfg, verbose=True, source="tactical"):
    """Samples from the cache, or from the video (~10 min). None if the game has no video.
    `source` "broadcast" reads the broadcast video with its own homography (frames without one,
    close-ups and the like, are skipped); `tracking` and `blocks` are then on its timeline."""
    import games
    import pandas as pd

    game = tracking.get("game", "BAR-ATH")
    try:
        paths = (games.asset_paths(game, cfg["paths"]["data_dir"]) if source == "tactical" else
                 games.asset_paths(game, cfg["paths"]["data_dir"], source=source))
        video, hom = paths["video"], paths["homography"]
    except Exception:
        return None
    if not (os.path.isfile(video) and os.path.isfile(hom)):
        return None
    c = cfg["context"]["jersey"]
    cache = os.path.join(cfg["paths"]["cache_dir"], "{}_jersey{}_step{}_{:g}_{:g}_{:g}.npz".format(
        game, "" if source == "tactical" else "_" + source, c["step"], c["torso_from_m"],
        c["torso_to_m"], c["torso_half_width_m"]))
    if os.path.isfile(cache) and os.path.getmtime(cache) >= os.path.getmtime(video):
        return dict(np.load(cache))
    if verbose:
        print("reading shirt colours from {} (~10 min the first time)...".format(os.path.basename(video)))
    table = pd.read_csv(hom)
    H = table[["h{}{}".format(r, k) for r in range(3) for k in range(3)]].to_numpy(
        np.float64).reshape(-1, 3, 3)
    valid = (table["valid"].to_numpy() == 1) if "valid" in table.columns else None
    samples = sample(tracking, blocks, video, H, cfg, verbose, valid)
    os.makedirs(cfg["paths"]["cache_dir"], exist_ok=True)
    np.savez(cache, **samples)
    return samples


def team_by_shirt(tracking, blocks, cfg, verbose=True, source="tactical"):
    """((N, S) team of each slot per frame, changed segments), or (None, []) without video."""
    samples = load_samples(tracking, blocks, cfg, verbose, source)
    if samples is None:
        return None, []
    keepers = keeper_slots(tracking, blocks, cfg)
    llr, _ = fit(samples, tracking["team"], keepers, cfg, verbose)
    return decode(samples, llr, tracking["team"], keepers, len(tracking["players"]), cfg,
                  tracking.get("team_conflict"), tracking.get("team_vote"))


if __name__ == "__main__":
    import argparse

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from possession.config import REPO_ROOT, load_config
    from possession.data import load_tracking

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "config.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    import games
    tr = load_tracking(cfg)
    T, changed = team_by_shirt(tr, games.half_video_blocks(tr["game"]), cfg)
    if T is None:
        sys.exit("no video")
    tot = sum(b - a + 1 for _, a, b in changed)
    print("segments given to the other team: {} ({:.0f} s of track)".format(len(changed), tot / 25))
