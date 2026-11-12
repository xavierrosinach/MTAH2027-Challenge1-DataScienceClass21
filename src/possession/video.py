"""Videos of the match with the tracking and our labels on top, drawn with OpenCV.

Called from run.py when `video.enabled` is true in config.yaml (see render_blocks).

modes:
    overlay  real tactical video + tracking on top (+ minimap)
    pitch    2D pitch only (no video decoding, faster)
"""

import os
import time

import numpy as np
import pandas as pd

H_COLS = [f"h{r}{c}" for r in range(3) for c in range(3)]
# BGR
HOME = (156, 78, 31)
AWAY = (43, 57, 192)
BALL = (15, 196, 241)
LOST = (200, 200, 200)
WHITE = (255, 255, 255)
GRASS = (50, 125, 46)


# --- helpers

def _project(H, x, y):
    p = np.einsum("nij,nj->ni", H, np.stack([x, y, np.ones_like(x)], axis=1))
    return p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]


def _active_tags(sel):
    """'CODE(team), ...' per frame from the st_ columns."""
    cols = [c for c in sel.columns if c.startswith("st_") and c not in ("st_set_piece_type", "st_n_active")]
    out = np.full(len(sel), "", dtype=object)
    for c in cols:
        v = sel[c].astype(str).to_numpy()
        name = c[3:].replace("_", " ").upper()
        if c == "st_set_pieces":
            sp = sel["st_set_piece_type"].astype(str).to_numpy()
            txt = np.char.add(np.char.add(name + " ", sp.astype(str)), np.char.add("(", np.char.add(v.astype(str), ")")))
        else:
            txt = np.char.add(name + "(", np.char.add(v.astype(str), ")"))
        on = v != ""
        out[on] = [a + ("  " if a else "") + b for a, b in zip(out[on], txt[on])]
    return out


class Pitch:
    """2D pitch canvas in pixels."""

    def __init__(self, width_px, pitch=(105.3, 68.0), margin_m=4):
        import cv2

        self.cv2 = cv2
        L, W = pitch
        self.L, self.W, self.m = L, W, margin_m
        self.s = width_px / (L + 2 * margin_m)
        self.size = (int(width_px), int((W + 2 * margin_m) * self.s))
        self.base = self._draw()

    def xy(self, x_m, y_m):
        return ((x_m + self.m) * self.s).astype(int), ((y_m + self.m) * self.s).astype(int)

    def _p(self, x, y):
        return int((x + self.m) * self.s), int((y + self.m) * self.s)

    def _draw(self):
        cv2, L, W = self.cv2, self.L, self.W
        img = np.full((self.size[1], self.size[0], 3), GRASS, dtype=np.uint8)
        t = max(1, int(self.s * 0.25))
        cv2.rectangle(img, self._p(0, 0), self._p(L, W), WHITE, t)
        cv2.line(img, self._p(L / 2, 0), self._p(L / 2, W), WHITE, t)
        cv2.circle(img, self._p(L / 2, W / 2), int(9.15 * self.s), WHITE, t)
        for x0, sg in ((0, 1), (L, -1)):
            cv2.rectangle(img, self._p(x0, W / 2 - 20.16), self._p(x0 + sg * 16.5, W / 2 + 20.16), WHITE, t)
            cv2.rectangle(img, self._p(x0, W / 2 - 9.16), self._p(x0 + sg * 5.5, W / 2 + 9.16), WHITE, t)
        return img


# --- main

def make_video(sel, players, cfg, out_path, mode="overlay", scale=0.5, minimap=True,
               label_col=None, step=1, fps=25, progress_every=2500):
    """Write an mp4 with every frame in `sel` (rows of the frames table).

    sel        frames DataFrame (filtered), needs frame, ball_*, h00..h22, clock, period, st_*
    players    players DataFrame from data_proc
    mode       "overlay" (video + tracking) or "pitch" (2D only)
    scale      output size vs 1920x1080 (overlay) -- 0.5 = 960x540
    label_col  optional column (or list of columns) in sel to show as big letters, top right,
               left to right in the given order; when two differ they get a yellow frame
    step       keep 1 frame every `step`
    """
    import cv2
    import games

    sel = sel.sort_values("frame").iloc[::step].reset_index(drop=True)
    fr = sel["frame"].to_numpy()
    n = len(sel)

    # players indexed by frame (players is sorted by frame)
    pl = players.sort_values("frame", kind="stable") if not players["frame"].is_monotonic_increasing else players
    pf = pl["frame"].to_numpy()
    lo, hi = np.searchsorted(pf, fr, "left"), np.searchsorted(pf, fr, "right")
    p_px, p_py = pl["px"].to_numpy(), pl["py"].to_numpy()
    p_xm, p_ym = pl["x_m"].to_numpy(), pl["y_m"].to_numpy()
    p_home = (pl["team"].astype(str) == "H").to_numpy()
    p_slot = pl["slot"].to_numpy()

    # ball: current and last seen (in pitch coords, projected with the current homography)
    seen = sel["ball_seen"].to_numpy()
    last_x = sel["ball_x"].ffill().to_numpy(np.float64)
    last_y = sel["ball_y"].ffill().to_numpy(np.float64)
    last_f = pd.Series(np.where(seen, fr, np.nan)).ffill().to_numpy()
    Hm = sel[H_COLS].to_numpy(np.float64).reshape(-1, 3, 3)
    bx, by = _project(Hm, last_x, last_y)                  # last seen (= current when seen)
    b_xm, b_ym = last_x * games.GAMES["BAR-ATH"]["pitch"][0], last_y * games.GAMES["BAR-ATH"]["pitch"][1]

    tags_txt = _active_tags(sel)
    cols = [label_col] if isinstance(label_col, str) else list(label_col or [])
    labels = {c: sel[c].astype(str).to_numpy() for c in cols}

    # canvas
    if mode == "overlay":
        out_w, out_h = int(1920 * scale), int(1080 * scale)
        cap = cv2.VideoCapture(games.asset_paths("BAR-ATH", cfg["paths"]["data_dir"])["video"])
        if not cap.isOpened():
            raise IOError("cannot open video")
        mini = Pitch(int(out_w * 0.28)) if minimap else None
    elif mode == "pitch":
        mini = Pitch(int(2560 * scale))   # 0.5 -> 1280 px wide
        out_w, out_h = mini.size[0], mini.size[1] + 60
        cap = None
    else:
        raise ValueError(mode)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.4, 0.55 * scale / 0.5) if mode == "overlay" else 0.6
    t0, next_read = time.time(), None

    def draw_mini(canvas, i, x0, y0):
        img = mini.base.copy()
        a, b = lo[i], hi[i]
        xs, ys = mini.xy(p_xm[a:b], p_ym[a:b])
        r = max(3, int(mini.s * 0.9))
        for x, y, h in zip(xs, ys, p_home[a:b]):
            cv2.circle(img, (x, y), r, HOME if h else AWAY, -1)
        if not np.isnan(b_xm[i]):
            x, y = mini.xy(np.array([b_xm[i]]), np.array([b_ym[i]]))
            if seen[i]:
                cv2.circle(img, (x[0], y[0]), max(2, r // 2 + 1), BALL, -1)
            else:
                cv2.drawMarker(img, (x[0], y[0]), LOST, cv2.MARKER_TILTED_CROSS, r * 2, 2)
        h_, w_ = img.shape[:2]
        canvas[y0:y0 + h_, x0:x0 + w_] = img

    for i in range(n):
        f = fr[i]
        if mode == "overlay":
            if next_read != f:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, img = cap.read()
            next_read = f + 1
            if step > 1:   # skip frames by grabbing
                for _ in range(step - 1):
                    cap.grab()
                next_read = f + step
            if not ok:
                img = np.zeros((1080, 1920, 3), np.uint8)
            if scale != 1:
                img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
            a, b = lo[i], hi[i]
            r = max(6, int(14 * scale))
            for x, y, h, s in zip(p_px[a:b] * scale, p_py[a:b] * scale, p_home[a:b], p_slot[a:b]):
                if np.isfinite(x) and np.isfinite(y):
                    c = HOME if h else AWAY
                    cv2.circle(img, (int(x), int(y)), r, c, 2)
                    cv2.putText(img, str(s), (int(x) - r, int(y) - r - 2), font, fs * 0.7, c, 1, cv2.LINE_AA)
            if np.isfinite(bx[i]) and np.isfinite(by[i]):
                x, y = int(bx[i] * scale), int(by[i] * scale)
                if seen[i]:
                    cv2.rectangle(img, (x - r // 2 - 2, y - r // 2 - 2), (x + r // 2 + 2, y + r // 2 + 2), BALL, 2)
                else:
                    cv2.drawMarker(img, (x, y), LOST, cv2.MARKER_TILTED_CROSS, r * 2, 2)
            canvas = img
            if mini is not None:
                draw_mini(canvas, i, out_w - mini.size[0] - 8, out_h - mini.size[1] - 8)
        else:
            canvas = np.zeros((out_h, out_w, 3), np.uint8)
            draw_mini(canvas, i, 0, 60)

        # header
        status = "ball seen" if seen[i] else (
            "BALL LOST {:.1f}s".format((f - last_f[i]) / 25) if not np.isnan(last_f[i]) else "BALL LOST")
        line1 = "{} {}  |  frame {}  |  {}".format(sel["period"].iat[i], sel["clock"].iat[i], f, status)
        for c in cols:
            line1 += "  |  {}: {}".format(c, labels[c][i])
        bar_h = int(52 * fs / 0.55)
        cv2.rectangle(canvas, (0, 0), (out_w, bar_h), (0, 0, 0), -1)
        cv2.putText(canvas, line1, (8, int(bar_h * 0.42)), font, fs, WHITE, 1, cv2.LINE_AA)
        cv2.putText(canvas, tags_txt[i][:160], (8, int(bar_h * 0.88)), font, fs * 0.8, BALL, 1, cv2.LINE_AA)
        # big label letters, top right, in the team colour (grey = dead ball, black = X)
        box = int(out_h * 0.16)
        title_h = int(box * 0.28) if len(cols) > 1 else 0
        differ = len(cols) > 1 and len({labels[c][i] for c in cols}) > 1
        for j, c in enumerate(cols):
            lab = labels[c][i]
            colour = {"H": HOME, "A": AWAY, "D": (110, 110, 110)}.get(lab, (0, 0, 0))
            x1 = out_w - 8 - (len(cols) - 1 - j) * (box + 8)
            y0 = bar_h + 8
            y1 = y0 + title_h + box
            cv2.rectangle(canvas, (x1 - box, y0), (x1, y1), colour, -1)
            cv2.rectangle(canvas, (x1 - box, y0), (x1, y1), BALL if differ else WHITE,
                          5 if differ else 2)
            if title_h:
                (tw, th), _ = cv2.getTextSize(c, font, fs, 1)
                cv2.putText(canvas, c, (x1 - box + (box - tw) // 2, y0 + title_h - 4), font, fs,
                            WHITE, 1, cv2.LINE_AA)
            scale_txt = box / 40
            (tw, th), _ = cv2.getTextSize(lab, font, scale_txt, 3)
            cv2.putText(canvas, lab, (x1 - box + (box - tw) // 2, y1 - (box - th) // 2), font,
                        scale_txt, WHITE, 3, cv2.LINE_AA)
        writer.write(canvas)

        if progress_every and (i + 1) % progress_every == 0:
            el = time.time() - t0
            print("{}/{} frames  {:.0f}s elapsed  ~{:.0f}s left".format(i + 1, n, el, el / (i + 1) * (n - i - 1)),
                  flush=True)

    writer.release()
    if cap is not None:
        cap.release()
    print("video {} ({:.0f} s)".format(out_path, time.time() - t0))
    return out_path


GT_FILE = os.path.join("evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def video_tables(cfg, labels, tracking, public_gt=False):
    """Frames and players tables for make_video, with our labels (and the public GT) as columns.

    Goalkeeper tracks are coloured by the goal they guard, the same fix the labeller applies.
    Returns (frames, players, label columns to show).
    """
    import games
    from possession.context import keeper_teams
    from possession.data_proc import load_all

    d = load_all(cfg)
    frames, players = d["frames"].copy(), d["players"].copy()
    frames["modelo"] = labels
    shown = ["modelo"]
    if public_gt:
        # official public ground truth (5 min): only to understand how situations are labelled,
        # never to tune parameters (RULES.md)
        gt = pd.read_csv(os.path.join(cfg["paths"]["official_repo"], GT_FILE))["label"].to_numpy()
        frames["GT"] = gt
        shown = ["GT", "modelo"]
    if cfg["context"]["keepers"]["enabled"]:
        T, _ = keeper_teams(tracking, games.half_video_blocks("BAR-ATH"), cfg)
        t = T[players["frame"].to_numpy(), players["slot"].to_numpy() - 1]
        players["team"] = pd.Categorical(np.where(t == 0, "H", "A"))
    return frames, players, shown


def render_blocks(cfg, labels, tracking):
    """Videos of the graded match in blocks of `video.block_minutes`, with our labels on top.

    Each block is outputs/videos/<half>T_<nn>_<from>-<to>.mp4 (match clock). `video.blocks` picks
    which ones by name prefix ("1T_01", "2T", ...); an empty list renders them all.
    """
    import games

    vc = cfg["video"]
    frames, players, shown = video_tables(cfg, labels, tracking, vc.get("public_gt"))
    out_dir = os.path.join(cfg["paths"]["output_dir"], "videos")
    size = int(vc["block_minutes"] * 60 * 25)
    wanted = vc.get("blocks") or []
    written = []
    for half, first, last in games.half_video_blocks("BAR-ATH"):
        for k, start in enumerate(range(first, last + 1, size)):
            sel = frames[(frames.frame >= start) & (frames.frame < min(start + size, last + 1))]
            c0 = sel.clock.iloc[0] or "pre"
            c1 = sel.clock.iloc[-1] or "pre"
            name = "{}T_{:02d}_{}-{}".format(half, k + 1, c0.replace(":", "m"), c1.replace(":", "m"))
            if wanted and not any(name.startswith(w) for w in wanted):
                continue
            path = os.path.join(out_dir, name + ("_GT" if len(shown) > 1 else "") + ".mp4")
            make_video(sel, players, cfg, path, mode=vc["mode"], scale=vc["scale"],
                       minimap=vc["minimap"], label_col=shown, progress_every=0)
            written.append(path)
    return written
