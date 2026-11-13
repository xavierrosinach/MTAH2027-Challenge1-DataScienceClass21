"""Variantes de config: python xavier/mkcfg.py <out_dir> <bench|bar>"""
import copy, os, sys, yaml
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raíz del repo
out, mode = sys.argv[1:3]
base = yaml.safe_load(open(os.path.join(repo, "config.yaml"), encoding="utf-8"))
base["output"]["verbose"] = False
if mode == "bench":      # modelo de control entrenado solo con IDSSE: sin fuga en el banco de Metrica
    base["context"]["control"]["model"] = "models/control_v5dfl_mlp.json"
    base["context"]["control"]["ensemble"] = ["models/control_v5dfl_lin.json"]
FLAGS = {
    "falso": lambda c: c["ball_dead"].update(isolated_max_player_speed=1.6),
    "oclus": lambda c: c["occlusion"].update(enabled=True),
    "back": lambda c: c["ball_dead"].update(isolated_backdate=True),
    "rcheck": lambda c: c["restart_check"].update(enabled=True),
    "gap": lambda c: c["gap_restart"].update(enabled=True),
    "md2": lambda c: c["transition_cost"].update(min_dead_s=2.0),
    "md3": lambda c: c["transition_cost"].update(min_dead_s=3.0),
    "kline": lambda c: c["ball_dead"].update(kick_from_line=True),
    "gap10": lambda c: c["gap_restart"].update(max_move_m=10.0),
    "gapv2": lambda c: c["gap_restart"].update(max_speed=2.0),
    "gapv3": lambda c: c["gap_restart"].update(max_speed=3.0),
    "gs2": lambda c: c["gap_restart"].update(min_s=2),
    "td": lambda c: c["transition_cost"].update(to_dead=12),
    "v1": lambda c: c["gap_restart"].update(max_speed=1.0),
    "v15": lambda c: c["gap_restart"].update(max_speed=1.5),
    "md4": lambda c: c["transition_cost"].update(min_dead_s=4.0),
    "gap8": lambda c: c["gap_restart"].update(max_move_m=8.0),
    "gap13": lambda c: c["gap_restart"].update(max_move_m=13.0),
    "rk": lambda c: c["restart_check"].update(enabled=True, kicks=True, spells=False),
    "rs": lambda c: c["restart_check"].update(enabled=True, kicks=False, spells=True),
}
variants = {"base": [], **{k: [k] for k in FLAGS}}
for extra in sys.argv[3:]:                      # p.ej. combo=falso+back+rcheck
    name, _, keys = extra.partition("=")
    variants[name] = keys.split("+")
os.makedirs(out, exist_ok=True)
for name, keys in variants.items():
    cfg = copy.deepcopy(base)
    cfg["team_name"] = "x_" + name
    for k in keys:
        FLAGS[k](cfg["context"])
    yaml.safe_dump(cfg, open(os.path.join(out, name + ".yaml"), "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
print(list(variants))
