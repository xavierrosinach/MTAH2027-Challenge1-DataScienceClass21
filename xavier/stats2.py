"""Banco: cuando la posesión cambia dentro de un hueco de balón, ¿en qué punto del hueco cambia?
Y: ¿cuánto tarda el GT en dar por reanudado el juego respecto al primer balón visto tras una parada?
python xavier/stats2.py"""
import os, sys
import numpy as np, pandas as pd
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raíz del repo
sys.path.insert(0, os.path.join(repo, "validation")); sys.path.insert(0, os.path.join(repo, "src"))
import bench as B
from possession import context
from possession.config import load_config

bench = B.load_bench(os.path.join(repo, "validation", "bench.yaml"))
cfg = load_config(os.path.join(repo, "config.yaml"))
frac, durs, n_same, n_diff, n_dead = [], [], 0, 0, 0
for game in bench["games"]:
    for seed in (0, 1, 2, 3):
        tr, events, gt, blocks, tags = B.load_case(game, seed, bench, cfg["paths"]["cache_dir"])
        ball = context.interpolate_ball(tr["ball"], 12)
        seen = ~np.isnan(ball[:, 0])
        m = np.diff(np.r_[0, (~seen).astype(int), 0])
        for s, e in zip(np.flatnonzero(m == 1), np.flatnonzero(m == -1)):
            if s == 0 or e >= len(seen) or (e - s) < 25:
                continue
            g = gt[s - 1:e + 1]
            if (g == "X").any():
                continue
            if (g == "D").any():
                n_dead += 1
                continue
            if g[0] == g[-1]:
                n_same += 1
                continue
            ch = np.flatnonzero(g[1:] != g[:-1])
            if len(ch) != 1:
                continue
            n_diff += 1
            frac.append(ch[0] / (e - s))
            durs.append((e - s) / 25)
frac, durs = np.array(frac), np.array(durs)
print("huecos >= 1 s en juego vivo: mismo equipo antes y despues {}, cambia de equipo (1 cambio) {}, con parada {}".format(n_same, n_diff, n_dead))
print("posicion del cambio dentro del hueco (0 = inicio, 1 = final): percentiles 10/25/50/75/90 =",
      np.percentile(frac, [10, 25, 50, 75, 90]).round(2).tolist(), " media %.2f" % frac.mean())
for lo, hi in ((1, 3), (3, 6), (6, 100)):
    k = (durs >= lo) & (durs < hi)
    print("  huecos de {}-{} s: n={}  mediana {:.2f}  media {:.2f}".format(lo, hi, k.sum(), np.median(frac[k]), frac[k].mean()))
