"""Una imagen por cada frame en que el modelo no coincide con el ground truth público.

    python xavier/diff_frames.py
    python xavier/diff_frames.py --pred outputs/submissions/main_xavier.csv
    python xavier/diff_frames.py --zoom 3 --step 5

Para cada frame distinto (solo en los 5 min públicos, frames 11185-18684) guarda
outputs/diff_frames/<nombre del csv>/<frame>.png con:
  - izquierda: el vídeo táctico ampliado alrededor del balón, con puntos pequeños
    (azul = H, rojo = A, amarillo = balón; cruz gris = último balón visto si ahora no se ve)
  - derecha: el mismo zoom en planta (campo 2D, ±20 m alrededor del balón)
  - arriba: frame, reloj, GT y modelo
Además escribe index.csv con todos los frames distintos.

Es para entender la definición de posesión mirando casos, no para ajustar parámetros con los
5 min públicos (RULES.md).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raíz del repo
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, HERE)

from possession.config import load_config  # noqa: E402

# BGR
HOME = (235, 120, 30)
AWAY = (40, 40, 220)
BALL = (0, 230, 255)
LOST = (200, 200, 200)
WHITE = (255, 255, 255)
GRASS = (50, 125, 46)
LABEL_COLOUR = {"H": HOME, "A": AWAY, "D": (110, 110, 110), "X": (0, 0, 0)}


def video_crop(cv2, img, cx, cy, zoom, out_w, out_h):
    """Recorte de img centrado en (cx, cy), 1/zoom del tamaño, ampliado a out_w x out_h.
    Devuelve el recorte y la transformación (x0, y0, factor) para dibujar encima."""
    H, W = img.shape[:2]
    cw, ch = int(W / zoom), int(H / zoom)
    if not (np.isfinite(cx) and np.isfinite(cy)):
        cx, cy = W / 2, H / 2
    x0 = int(np.clip(cx - cw / 2, 0, W - cw))
    y0 = int(np.clip(cy - ch / 2, 0, H - ch))
    crop = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    return crop, (x0, y0, out_w / cw)


def pitch_zoom(cv2, pitch_lw, cx, cy, half_m, size, pl_x, pl_y, pl_home, ball_xy, ball_seen):
    """Planta del campo centrada en el balón (cx, cy en metros), ±half_m, cuadrada de `size` px."""
    L, W = pitch_lw
    s = size / (2 * half_m)

    def p(x, y):
        return int((x - cx + half_m) * s), int((y - cy + half_m) * s)

    img = np.full((size, size, 3), GRASS, np.uint8)
    t = 2
    cv2.rectangle(img, p(0, 0), p(L, W), WHITE, t)
    cv2.line(img, p(L / 2, 0), p(L / 2, W), WHITE, t)
    cv2.circle(img, p(L / 2, W / 2), int(9.15 * s), WHITE, t)
    for x0, sg in ((0, 1), (L, -1)):
        cv2.rectangle(img, p(x0, W / 2 - 20.16), p(x0 + sg * 16.5, W / 2 + 20.16), WHITE, t)
        cv2.rectangle(img, p(x0, W / 2 - 9.16), p(x0 + sg * 5.5, W / 2 + 9.16), WHITE, t)
    for r in (5, 10):                                   # anillos de 5 y 10 m alrededor del balón
        cv2.circle(img, p(cx, cy), int(r * s), (70, 150, 66), 1, cv2.LINE_AA)
    for x, y, h in zip(pl_x, pl_y, pl_home):
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(img, p(x, y), 6, HOME if h else AWAY, -1, cv2.LINE_AA)
    if ball_xy is not None:
        if ball_seen:
            cv2.circle(img, p(*ball_xy), 5, BALL, -1, cv2.LINE_AA)
        else:
            cv2.drawMarker(img, p(*ball_xy), LOST, cv2.MARKER_TILTED_CROSS, 14, 2)
    cv2.putText(img, "planta +/-{} m".format(int(half_m)), (8, size - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    return img


def label_box(cv2, canvas, x, y, title, lab, size=70):
    cv2.rectangle(canvas, (x, y), (x + size, y + size + 22), LABEL_COLOUR.get(lab, (0, 0, 0)), -1)
    cv2.rectangle(canvas, (x, y), (x + size, y + size + 22), BALL, 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(title, font, 0.55, 1)
    cv2.putText(canvas, title, (x + (size - tw) // 2, y + 18), font, 0.55, WHITE, 1, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(lab, font, 1.8, 3)
    cv2.putText(canvas, lab, (x + (size - tw) // 2, y + 22 + (size + th) // 2), font, 1.8, WHITE, 3,
                cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    parser.add_argument("--pred", default=os.path.join(HERE, "outputs", "submissions",
                                                       "DataScienceClass21.csv"),
                        help="CSV a comparar con el GT (por defecto el de run.py)")
    parser.add_argument("--name", help="nombre del modelo en las imágenes (por defecto, el del CSV)")
    parser.add_argument("--out", help="carpeta de salida (por defecto outputs/diff_frames/<csv>)")
    parser.add_argument("--zoom", type=float, default=2.5, help="ampliación del vídeo (2.5 = 1/2.5 de la imagen)")
    parser.add_argument("--half-m", type=float, default=20.0, help="medio lado de la planta, en metros")
    parser.add_argument("--step", type=int, default=1,
                        help="guardar 1 de cada N frames distintos consecutivos (1 = todos)")
    parser.add_argument("--frames", type=int, nargs=2, metavar=("DESDE", "HASTA"),
                        help="solo este rango de frames, y TODOS (coincidan o no con el GT)")
    args = parser.parse_args()

    import cv2

    cfg = load_config(args.config)
    # después de load_config: necesitan games.py (repo oficial) en el path
    import games
    from possession.data import load_tracking
    from possession.video import H_COLS, _project, video_tables
    stem = os.path.splitext(os.path.basename(args.pred))[0]
    name = args.name or ("modelo" if stem == cfg["team_name"] else stem)
    out_dir = args.out or os.path.join(cfg["paths"]["output_dir"], "diff_frames", stem)
    os.makedirs(out_dir, exist_ok=True)

    pred = pd.read_csv(args.pred).sort_values("frame")
    tracking = load_tracking(cfg)
    n = len(tracking["ball"])
    labels = np.full(n, "X", dtype=object)
    labels[pred["frame"].to_numpy()] = pred["label"].astype(str).to_numpy()
    frames, players, _ = video_tables(cfg, labels, tracking, public_gt=True)

    sel = frames[(frames["GT"] != "X")]
    diff = sel[sel["GT"] != sel["modelo"]].sort_values("frame")
    print("ventana pública: {} frames, distintos: {} ({:.1f} %)".format(
        len(sel), len(diff), 100 * len(diff) / max(1, len(sel))))
    if args.frames:
        diff = frames[frames["frame"].between(*args.frames)].sort_values("frame")
    if args.step > 1:
        run_id = (diff["frame"].diff() != 1).cumsum()
        diff = diff[diff.groupby(run_id).cumcount() % args.step == 0]
    diff.assign(modelo_nombre=name)[["frame", "clock", "GT", "modelo", "ball_seen"]].to_csv(
        os.path.join(out_dir, "index.csv"), index=False)

    # jugadores por frame
    pl = players.sort_values("frame", kind="stable")
    pf = pl["frame"].to_numpy()
    p_px, p_py = pl["px"].to_numpy(), pl["py"].to_numpy()
    p_xm, p_ym = pl["x_m"].to_numpy(), pl["y_m"].to_numpy()
    p_home = (pl["team"].astype(str) == "H").to_numpy()

    # balón: el actual o el último visto (posición en píxeles con la homografía de este frame)
    L, W = games.GAMES["BAR-ATH"]["pitch"]
    seen_all = frames["ball_seen"].to_numpy()
    last_x = frames["ball_x"].ffill().to_numpy(np.float64)
    last_y = frames["ball_y"].ffill().to_numpy(np.float64)
    last_f = pd.Series(np.where(seen_all, frames["frame"], np.nan)).ffill().to_numpy()

    cap = cv2.VideoCapture(games.asset_paths("BAR-ATH", cfg["paths"]["data_dir"])["video"])
    if not cap.isOpened():
        sys.exit("no se puede abrir el vídeo")
    vw, vh, side, bar = 960, 540, 540, 46
    font = cv2.FONT_HERSHEY_SIMPLEX
    next_read = None

    for k, row in enumerate(diff.itertuples(index=False)):
        f = int(row.frame)
        if next_read != f:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        next_read = f + 1
        if not ok:
            img = np.zeros((1080, 1920, 3), np.uint8)

        Hm = frames.loc[f, H_COLS].to_numpy(np.float64).reshape(1, 3, 3)
        bx, by = _project(Hm, np.array([last_x[f]]), np.array([last_y[f]]))
        bx, by = float(bx[0]), float(by[0])
        seen = bool(seen_all[f])

        # vídeo ampliado alrededor del balón, con puntos pequeños
        crop, (x0, y0, fac) = video_crop(cv2, img, bx, by, args.zoom, vw, vh)
        a, b = np.searchsorted(pf, f, "left"), np.searchsorted(pf, f, "right")
        for x, y, h in zip(p_px[a:b], p_py[a:b], p_home[a:b]):
            if np.isfinite(x) and np.isfinite(y):
                pt = (int((x - x0) * fac), int((y - y0) * fac))
                cv2.circle(crop, pt, 6, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(crop, pt, 4, HOME if h else AWAY, -1, cv2.LINE_AA)
        if np.isfinite(bx) and np.isfinite(by):
            pt = (int((bx - x0) * fac), int((by - y0) * fac))
            if seen:
                cv2.circle(crop, pt, 5, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(crop, pt, 3, BALL, -1, cv2.LINE_AA)
            else:
                cv2.drawMarker(crop, pt, LOST, cv2.MARKER_TILTED_CROSS, 16, 2)

        # planta
        bxm, bym = last_x[f] * L, last_y[f] * W
        if np.isfinite(bxm):
            plan = pitch_zoom(cv2, (L, W), bxm, bym, args.half_m, side, p_xm[a:b], p_ym[a:b],
                              p_home[a:b], (bxm, bym), seen)
        else:
            plan = np.full((side, side, 3), GRASS, np.uint8)

        canvas = np.zeros((bar + vh, vw + side, 3), np.uint8)
        canvas[bar:, :vw] = crop
        canvas[bar:, vw:] = plan
        status = "balon visto" if seen else (
            "balon perdido {:.1f} s".format((f - last_f[f]) / 25) if np.isfinite(last_f[f]) else "balon perdido")
        txt = "frame {}   |   {} {}   |   {}   |   GT: {}   {}: {}".format(
            f, row.period if hasattr(row, "period") else "", row.clock, status, row.GT, name, row.modelo)
        cv2.putText(canvas, txt, (10, 30), font, 0.65, WHITE, 1, cv2.LINE_AA)
        label_box(cv2, canvas, vw - 2 * 80 - 10, bar + 10, "GT", row.GT)
        label_box(cv2, canvas, vw - 80 - 10, bar + 10, name[:9], row.modelo)

        cv2.imwrite(os.path.join(out_dir, "{}.png".format(f)), canvas)
        if (k + 1) % 200 == 0:
            print("{}/{} imágenes".format(k + 1, len(diff)), flush=True)

    cap.release()
    print("{} imágenes en {}".format(len(diff), out_dir))


if __name__ == "__main__":
    main()
