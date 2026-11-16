"""Referee whistles in the audio of the broadcast video.

A foul or an offside stops play with a whistle, and the Smart Tagging tags few of them. The
broadcast video carries the stadium sound (the tactical one has no audio), and a referee whistle
is a loud, narrow tone around 4 kHz, unlike the crowd, which is broadband.

  1. audio()    the sound track as mono 16 kHz samples, decoded with ffmpeg (imageio-ffmpeg).
  2. features() one spectrum per video frame (25 per second): the share of the energy between
                1 and 8 kHz that falls in the whistle band, and how peaked that band is.
  3. detect()   a whistle is a run of frames whose band share stands out from the rest of the
                match (median + k robust deviations) and whose band is one narrow peak.

No label is used. Returns frames of the broadcast video; broadcast.py puts them on the tactical
timeline.

    python src/possession/whistle.py [--config config.yaml]     # report, fills the cache
"""

import os
import subprocess
import sys

import numpy as np

RATE = 16000
FPS = 25


def audio(video_path):
    """Mono samples at 16 kHz (float32), or None if ffmpeg or the sound track is missing."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
    cmd = [exe, "-v", "error", "-i", video_path, "-vn", "-ac", "1", "-ar", str(RATE),
           "-f", "s16le", "-"]
    run = subprocess.run(cmd, capture_output=True)
    if run.returncode != 0 or len(run.stdout) < RATE * 2:
        return None
    return np.frombuffer(run.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def features(x, c, chunk=4000):
    """(share, peak) per video frame: energy of the whistle band over the 1-8 kHz energy, and the
    strongest bin of the band over the mean bin of 1-8 kHz."""
    hop, win = RATE // FPS, 1024
    n = max(0, (len(x) - win) // hop + 1)
    freqs = np.fft.rfftfreq(win, 1.0 / RATE)
    band = (freqs >= c["band_hz"][0]) & (freqs <= c["band_hz"][1])
    ref = (freqs >= 1000) & (freqs <= 8000)
    window = np.hanning(win).astype(np.float32)
    share, peak = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        idx = (np.arange(a, b) * hop)[:, None] + np.arange(win)[None, :]
        S = np.abs(np.fft.rfft(x[idx] * window, axis=1)) ** 2
        tot = S[:, ref].sum(axis=1) + 1e-12
        share[a:b] = S[:, band].sum(axis=1) / tot
        peak[a:b] = S[:, band].max(axis=1) / (tot / ref.sum())
    return share, peak


def detect(share, peak, c):
    """(n,) bool per video frame: inside a whistle."""
    med = float(np.median(share))
    mad = float(np.median(np.abs(share - med))) + 1e-9
    hit = (share > med + c["k_mad"] * 1.4826 * mad) & (peak > c["min_peak"])
    # close short holes, then keep runs long enough to be a blown whistle
    gap = int(c["join_frames"])
    edge = np.diff(np.r_[0, (~hit).astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)):
        if 0 < a and b < len(hit) and b - a <= gap:
            hit[a:b] = True
    edge = np.diff(np.r_[0, hit.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)):
        if b - a < c["min_frames"]:
            hit[a:b] = False
    return hit


def whistles(cfg, game="BAR-ATH", verbose=True):
    """(n_broadcast_frames,) bool, or None without the broadcast video, its sound or ffmpeg."""
    import games

    c = cfg["context"]["whistle"]
    try:
        video = games.asset_paths(game, cfg["paths"]["data_dir"], source="broadcast")["video"]
    except Exception:
        return None
    if not os.path.isfile(video):
        return None
    cache = os.path.join(cfg["paths"]["cache_dir"], "{}_whistle_{}_{}.npz".format(
        game, int(c["band_hz"][0]), int(c["band_hz"][1])))
    if os.path.isfile(cache) and os.path.getmtime(cache) >= os.path.getmtime(video):
        z = np.load(cache)
        share, peak = z["share"], z["peak"]
    else:
        if verbose:
            print("reading the sound of {} (1-2 min the first time)...".format(os.path.basename(video)))
        x = audio(video)
        if x is None:
            if verbose:
                print("whistle: no sound track or no ffmpeg (pip install imageio-ffmpeg); not used")
            return None
        share, peak = features(x, c)
        os.makedirs(cfg["paths"]["cache_dir"], exist_ok=True)
        np.savez(cache, share=share, peak=peak)
    hit = detect(share, peak, c)
    if verbose:
        starts = int((np.diff(np.r_[0, hit.astype(np.int8)]) == 1).sum())
        print("whistles heard in the broadcast sound: {} ({:.1f} per minute)".format(
            starts, starts / max(1e-9, len(hit) / FPS / 60)))
    return hit


if __name__ == "__main__":
    import argparse

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from possession.config import REPO_ROOT, load_config

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "config.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    w = whistles(cfg)
    if w is None:
        sys.exit("no broadcast video, no sound track or no ffmpeg")
    edge = np.diff(np.r_[0, w.astype(np.int8), 0])
    a, b = np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)
    print("duration (s): median {:.2f}, p90 {:.2f}".format(np.median(b - a) / FPS, np.percentile(b - a, 90) / FPS))
    print("first ones (broadcast frame): {}".format(a[:20].tolist()))
