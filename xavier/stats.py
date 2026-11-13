"""Estadísticas en los partidos del banco (nunca BAR-ATH): premisas de las reglas.
python xavier/stats.py"""
import os, sys
import numpy as np, pandas as pd
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raíz del repo
sys.path.insert(0, os.path.join(repo, "validation")); sys.path.insert(0, os.path.join(repo, "src"))
import bench as B
from possession import context
from possession.config import load_config

bench = B.load_bench(os.path.join(repo, "validation", "bench.yaml"))
cfg = load_config(os.path.join(repo, "config.yaml"))
rows, durs, short = [], [], []
for game in bench["games"]:
    for seed in (0, 1):
        tr, events, gt, blocks, tags = B.load_case(game, seed, bench, cfg["paths"]["cache_dir"])
        ball = context.interpolate_ball(tr["ball"], 12)
        seen = ~np.isnan(ball[:, 0])
        scale = np.array(tr["pitch"])
        graded = gt != "X"
        m = np.diff(np.r_[0, (~seen).astype(int), 0])
        for s, e in zip(np.flatnonzero(m == 1), np.flatnonzero(m == -1)):
            if s == 0 or e >= len(seen) or not graded[s:e].all():
                continue
            mv = np.hypot(*((ball[e] - ball[s - 1]) * scale))
            rows.append((game, (e - s) / 25, mv, (gt[s:e] == "D").mean()))
        if seed == 0:
            # duración de las paradas reales y de las posesiones
            for _, first, last in blocks:
                g = gt[first:last + 1]
                ch = np.flatnonzero(g[1:] != g[:-1]) + 1
                b = np.r_[0, ch, len(g)]
                for i in range(len(b) - 1):
                    if g[b[i]] == "D" and 0 < i < len(b) - 2:
                        durs.append((b[i + 1] - b[i]) / 25)
d = pd.DataFrame(rows, columns=["game", "dur", "move", "dead"])
d["is_dead"] = d.dead > 0.5
print("HUECOS DE BALON (tracking degradado, 3 partidos x 2 semillas): % que son parada segun el GT")
d["dbin"] = pd.cut(d.dur, [0.5, 1, 2, 3, 5, 10, 1000])
d["mbin"] = pd.cut(d.move, [0, 3, 6, 10, 20, 200], include_lowest=True)
print(d.pivot_table(index="dbin", columns="mbin", values="is_dead", aggfunc="mean", observed=False).round(2).to_string())
print("n por celda")
print(d.pivot_table(index="dbin", columns="mbin", values="is_dead", aggfunc="size", observed=False).to_string())
durs = np.array(durs)
print("\nPARADAS REALES (GT, entre dos tramos vivos): n={}  percentiles 1/5/10/25/50 (s): {}".format(
    len(durs), np.percentile(durs, [1, 5, 10, 25, 50]).round(1).tolist()))
for th in (1, 2, 3, 4, 5):
    print("  paradas reales de menos de {} s: {:.1f} %".format(th, 100 * (durs < th).mean()))
