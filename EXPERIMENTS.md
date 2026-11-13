# Experiments

How every change to the labeller is decided. RULES.md forbids tuning or choosing versions with the
public 5 minutes of BAR-ATH (and 17 possessions could not rank anything anyway), so the public
window is only a smoke test (`python run.py`) and decisions are made on **other games**.

## The validation bench (`validation/`)

```
git clone https://github.com/metrica-sports/sample-data ../external/metrica-sample-data
python validation/bench.py --suite                                   # config.yaml vs official baselines
python validation/bench.py --suite --configs ref=a.yaml new=b.yaml   # two versions side by side
python validation/sweep.py context.transition_cost.team_switch 12 16 # one parameter vs the config
python validation/calibrate.py [--grid]                              # how the degradation was chosen
python validation/train_control.py                                   # learnt control model (models/control.json)
python validation/check_control.py                                   # its features on BAR-ATH vs the bench (aggregates)
python validation/errors.py --configs ref=a.yaml new=b.yaml          # where spellF1 is lost (oracles, durations, causes)
python validation/compare.py x.csv --ref ref --new new              # paired difference of two versions in a bench CSV
python validation/check_bar.py --configs ref=a.yaml new=b.yaml       # label-free effect on BAR-ATH (stoppages, Smart Tagging)
python validation/train_dead.py / check_dead.py                      # learnt dead-ball model (#7, rejected, off)
```

Each control model file gets its own leave-one-game-out copies in
`outputs/validation/control/<model file name>/`, so versions with different features
(`context.control.features`) can be benched side by side. Experiments live on `exp/...` branches;
only what passes the decision rule is merged into `josep/mejoras`.

- **Games**: the three public Metrica Sports sample matches (anonymised, 25 fps, 105 x 68 m,
  tracking + event data). Not BAR-ATH.
- **Ground truth** (`gt_events.py`): derived mechanically from their event data with the rules of
  CHALLENGE.md: dead ball from the end of the last action to each restart (set piece or goal kick);
  while alive, the team of the last controlled action (pass, carry, recovery, shot, restart, ball
  lost); duels, fouls received and balls out do not change possession. ~384 possessions per game,
  ~4 per minute, ~35 % dead ball. Human event data of other games is allowed (RULES.md).
- **Degradation** (`degrade.py`): Metrica's sample tracking is much cleaner than our automatic ATD.
  Calibrated on label-free aggregate statistics of BAR-ATH only (`calibrate.py`), never on labels:
  - smooth error on the ball position (sigma 1.2 m, ~1 s): Metrica's ball sits on the possessor's
    feet (median distance to the nearest player 0.3 m), ours does not (1.9 m);
  - long ball gaps start ~0.6 s earlier (only 32 % of our gaps start at a line, 68 % in Metrica);
  - extra ball gaps in live play (0.5 per live minute, ~8 s) -> ~42 % of frames without ball;
  - player tracks cut into fragments (~35 s) alternating between two slots.
- **Check of the bench**: with this degradation the four official baselines come out in the same
  order as their published full-match BAR-ATH scores (smooth50 > smooth125 > smooth25 > raw), with
  similar ratios, and a bit higher (smooth50 0.30 vs 0.24). Without the ball noise they did not.
- **Smart Tagging stand-in**: none, or SET PIECES tags built from the restarts of the events
  (4 s early, 80 % recall).
- **Scorer**: the official `evaluation/possession_metrics.py`, unchanged.

**Decision rule.** The suite has four conditions: `light` (ball noise 0.6 m only), `degraded`,
`degraded+tags`, `degraded+coarseGT` (possessions < 2 s between two of the rival absorbed: checks
that a change does not hinge on how short possessions are counted). 3 games x 4 degradation seeds.
A change is kept only if its mean spellF1 improves under **every** condition and in most
(condition, game) cells, without losing macroF1. Among passing values, a round one in a flat region.
Then two transfer checks on BAR-ATH, label-free (learnt from #5 and #7):
`check_control.py` / `check_dead.py` (the model's inputs and outputs look like the bench's), and
`check_bar.py` (possessions, stoppages, share of Smart Tagging set pieces with a predicted stoppage:
a change must not make these implausible). Game 3, the most different bench game, gets a look of
its own. Exploratory runs on BAR-ATH go through `check_bar.py`, never `run.py` with the public
table on.

## Every path explored: branches while open, tags once closed

`josep/mejoras` is the best version so far.

- **While open**, an experiment lives on a branch `exp/<idea>` cut from `josep/mejoras`.
- **Once decided**, it becomes an annotated tag and the branch is deleted, so the branch list only
  shows work in progress. Tags are named `<decision>/<group>/<idea>`, with groups `viterbi`,
  `cercania`, `control`, `muerto` (more as needed).
  - An `aceptado/...` tag points at the commit that was merged.
  - A `descartado/...` tag points at one commit on top of the commit where the experiment was
    tested. That commit switches `config.yaml` to the variant and adds its models, so
    `git checkout <tag>` followed by `python run.py --no-public` reproduces it. The commit message
    gives the bench result.

On GitHub the tags are under "Tags". Locally: `git tag -l "descartado/*" -n1`.

| tag | entry | change |
|---|---|---|
| `aceptado/viterbi/team-switch-16` | 2 | `team_switch` 20 -> 16 |
| `aceptado/control/v1-control-aprendido` | 3 | learnt control model (linear) |
| `aceptado/control/v2-rasgos-movimiento` | 4 | movement features |
| `aceptado/control/v3-red-promediada` | 5 | average with the antisymmetric network |
| `aceptado/control/v4-peso-posesion` | 6 | possession-length weighting 0.25 |
| `descartado/viterbi/team-switch-13`, `-20` | 6 | hand-over cost on v4 (coarseGT / worse) |
| `descartado/cercania/contacto-1m` | rejected table | `contact_m` 1.0 |
| `descartado/cercania/cercania-ademas` | 3 | keep nearest-player evidence as well |
| `descartado/control/gap-weight` | 3 | control model in ball gaps too (noise) |
| `descartado/control/rasgos-long`, `-touch`, `-todos` | 4 | other feature groups |
| `descartado/control/red-sola` | 5 | the network alone (game 3) |
| `descartado/control/peso-0.5`, `peso-1` | 6 | stronger weighting (coarseGT) |
| `descartado/control/rasgos-short` | 6 | half-second features on v4 |
| `descartado/muerto/modelo-aprendido` | 7 | learnt dead-ball model, on (does not transfer) |
| `descartado/control/semillas` | 8 | mean of 5 networks with different seeds (noise) |
| `aceptado/muerto/dead-weight-03` | 9 | `dead_weight` 0.5 -> 0.3 |
| `descartado/muerto/to-dead-20` | 9 | `to_dead` 10 -> 20 (fewer stoppages on BAR-ATH) |
| `aceptado/datos/v6-idsse` | 10 | control model trained on 10 games, IDSSE added (current) |
| `descartado/control/train-mix-light` | 11 | also trained on `light` copies (cuts finer, loses coarseGT) |
| `descartado/huecos/interp-50` | 12 | interpolate ball gaps up to 2 s (noise) |
| `descartado/muerto/modelo-10-partidos` | 13 | learnt dead-ball model on 10 games, ball hidden in dead time (fewer stoppages on BAR-ATH) |
| `descartado/huecos/gap-weight-03` | 14 | team evidence of the gap model, weight 0.3 (BAR-ATH stoppages 105 < 106) |
| `descartado/datos/v9-skillcorner` | 15 | control model trained on 20 games, SkillCorner added (coarser cuts) |

Not tagged: `ball.weight`, `passes.pass_llr` and "ignore proximity when the ball flies".
These were first-bench trials whose code or settings were not kept; the table below gives their
results.

## Results (current bench: commit b259b60)

Mean over 3 games x 4 seeds. spellF1 per condition; mean macroF1; predicted possessions (GT 384).

| # | version | light | degraded | +tags | +coarseGT | mean spell | mean macro | poss | decision |
|---|---------|------:|---------:|------:|----------:|-----------:|-----------:|-----:|----------|
| 0 | official baseline_smooth50 | 0.360 | 0.303 | 0.303 | 0.310 | 0.319 | 0.769 | ~590 | floor |
| 1 | main 6791db5 (Xavier, Viterbi) | 0.507 | 0.404 | 0.435 | 0.473 | 0.455 | 0.812 | ~215 | reference |
| 2 | #1 + `team_switch` 20 -> 16 | 0.520 | 0.415 | 0.443 | 0.476 | 0.463 | 0.812 | ~237 | kept |
| 3 | #2 + learnt control model (`context.control`, weight 1) | 0.609 | 0.540 | 0.565 | 0.617 | 0.583 | 0.847 | ~244 | kept |
| 4 | #3 + movement features (`features: [base, motion]`) | 0.615 | 0.555 | 0.579 | 0.629 | 0.594 | 0.849 | ~250 | kept |
| 5 | #4 averaged with a non-linear model (`model_type: mlp`, `ensemble`) | 0.627 | 0.564 | 0.591 | 0.638 | 0.605 | 0.851 | ~251 | kept |
| 6 | #5 trained with frames weighted by possession length^-0.25 (`weight_power`) | 0.640 | 0.573 | 0.599 | 0.640 | 0.613 | 0.852 | ~260 | kept |
| 9 | #6 + `dead_weight` 0.5 -> 0.3 | 0.647 | 0.578 | 0.604 | 0.647 | 0.619 | 0.855 | ~260 | kept |
| 10 | #9 with the control model trained on 10 games (3 Metrica + 7 IDSSE) | 0.656 | 0.590 | 0.617 | 0.653 | **0.629** | **0.856** | ~260 | **kept** |

Row 10 is on seeds 4-7, the only seeds measured on #9's base; #9 itself scores 0.616 / 0.854 on
those seeds, so the paired gain is +0.013. Rows 0-9 are seeds 0-3.

Entry 2: +0.009 spellF1, better under all 4 conditions (worst +0.003), 10/12 cells, games 1 and 3 up,
game 2 flat (-0.003). The reference under-segments (215 predicted vs 384 true possessions); a
cheaper hand-over lets it keep more of the short ones. 12 scores a little more on average but
loses under coarseGT, so 16.

Entry 3: **+0.119 spellF1, +0.035 macroF1, better under all 4 conditions (worst +0.089), 12/12
cells.** `src/possession/control.py` replaces the nearest-player evidence with a model of who
controls the ball, learnt on the other games:

- *Model*: conditional logit, P(H) = sigmoid(w . (g(H) - g(A))), one model for frames with the
  ball, one for gaps. 42 features per team, all relative to the ball or to the team itself, on the
  frame, the second before, the second after and +/- 2 s: nearest and second-nearest player
  (exp(-d/1, 2.5, 6 m)), players within 5 / 10 m, share of the team goal-side of the ball, ball
  position and velocity towards the team's attacking goal, team shape (context.team_shape_features).
  Symmetric by construction: no home bias, nothing specific to a team.
- *Training* (`python validation/train_control.py`): live frames of the 3 Metrica games, tracking
  degraded as in `degraded` with seeds 100 and 101 (not the bench's 0-3), event-derived possession.
  Deterministic (retraining gives a byte-identical `models/control.json`, which `run.py` reads; it
  needs neither Metrica data nor training).
- *No leakage into the bench*: the bench labels each game with `logo_<game>.json`, trained on the
  other two games only.
- *Held-out per-frame accuracy* (live frames, ball seen): 92.4-92.6 % vs 72.9-74.9 % for "team of
  the nearest player" (81-83 % on the half of the frames where someone is within 2 m). Stable
  across training seeds (200/201, a single seed) and L2 (1 vs 100): +-0.2 pt.
- *Same signal on BAR-ATH* (`python validation/check_control.py`, label-free aggregates): the model
  is as confident on BAR-ATH as on the degraded games (|llr| > 2 on 79 % vs 79 % of frames with
  ball, median |llr| 4.26 vs 4.23, sign flips 5.4 vs 6.2 per minute). Every feature has the same
  scale (median |H - A| within 0.7-1.5x, 90th percentile within 0.97-1.4x) except the deepest
  player and the length of the team (1.4-1.7x at the 90th percentile: tactics or keepers); speeds
  are in the same unit.
- *Harsher tracking than it was trained on* (ball noise 2 m, player fragments 12 s, twice the ball
  gaps): 0.508 vs 0.361 spellF1 for #2. The gain grows when the tracking gets worse.
- *Error analysis* (degraded, #2 -> #3): possessions found at IoU 0.5: 2-5 s 18 -> 35 %, 5-10 s
  44 -> 63 %, 10-20 s 69 -> 75 %, > 20 s 70 -> 77 %; < 2 s still 2 -> 8 %. Hand-overs with a
  predicted one within 5 s 53 -> 64 %, mean timing error 1.66 -> 1.21 s.
- *Weight* (per-frame llr, clip 4): 0.05 / 0.1 / 0.2 / 0.4 / 0.7 / 1 / 1.5 / 2.5 -> +0.005 / +0.040
  / +0.068 / +0.098 / +0.112 / +0.119 / +0.124 / +0.123: flat from 1 to 2.5, so 1.
- Tried on top and not kept (all within +-0.006, i.e. bench noise): `team_switch` 10 / 13 / 20 / 24
  (+0.006 but loses coarseGT / +0.004 / -0.006 / -0.010); gap model weight 0.1 / 0.3 / 0.6
  (+0.001 / +0.003 / +0.003); keeping the nearest-player evidence as well (-0.000, 5/12).

Entry 4 (branch `exp/rasgos`): **+0.0115 spellF1 (paired over the 48 runs: sd 0.009, standard
error 0.0014, 88 % of runs better), +0.002 macroF1, better under all 4 conditions (light +0.006,
degraded +0.015, +tags +0.014, coarseGT +0.012), 11/12 cells** (the one below: game 3 under
`light`, -0.005). New group `motion` in `control.py`, per team, over +/- 0.5 s, the second before and
the second after: closing speed of its nearest player towards the ball, whether the ball moves with
him (exp(-|v_player - v_ball| / 3 m/s): a carry), his speed, and how many of its players within
12 m run at the ball faster than 2 m/s (pressing). Held-out per-frame accuracy 92.6-93.1 %
(base 92.4-92.6 %). Same signal on BAR-ATH (`check_control.py`, label-free): every motion feature
has the same scale (90th percentile ratio 0.99-1.09, medians within 0.92-1.04x), the model as
confident (|llr| > 2 on 79 % vs 80 % of frames with ball). Model file `models/control_v2.json`.

Tried in the same run and not kept (paired mean vs #3, cells better):
`long` (base features over 3 s before / after and +/- 5 s): -0.002, 7/12; `touch` (share of
frames in control before / after, time since / until the team's last / next control): -0.003,
3/12; all three groups together: +0.009, 11/12 but below `motion` alone and +0.001 under `light`.
The linear model gains little from more of the same evidence; what it lacked was how players move.

Entry 5 (branch `exp/nolineal`): **+0.0104 spellF1 (paired over 48 runs: standard error 0.0013,
90 % of runs better), +0.0023 macroF1, better in all 12 (condition, game) cells for both metrics.**
The log-odds of the linear model #4 are averaged with those of an antisymmetric one-hidden-layer
network on the same features (`control.fit_mlp`): z = h([D, S]) - h([-D, S]), D = G_H - G_A,
S = G_H + G_A, h = w2 . tanh(W1 x + b1). Swapping the teams flips z, so no home bias, and it can
use interactions a linear score cannot (nearest distance matters only while the ball is slow...).
Plain numpy, Adam, fixed seed; retraining gives the same weights. Size chosen on held-out-game
log-loss in `train_control.py`, not on the bench: 32 hidden / 12 epochs overfit (log-loss 0.22-0.28
vs 0.17-0.19 linear); 16 / 4 / l2 1e-3 gives 0.155-0.178 and 92.9-93.9 % accuracy (linear 92.6-93.1 %);
16 / 6 and 32 / 4 the same, so the smaller.

- *The network alone was not kept* although it passed the rule (+0.0139, se 0.0023, all 4
  conditions, 9/12 cells): it lost on game 3 under 3 of 4 conditions (-0.006). Its gain came from
  games 1 and 2; game 3, the most different of the three, is the best stand-in for an unseen match.
  On BAR-ATH (`check_control.py`) it was also less like the bench than the linear model: confident
  (|llr| > 2) on 77 % vs 82 % of frames with ball, 7.6 vs 7.0 sign flips per minute (linear: 79 vs
  80 %, 6.4 vs 7.0).
- *The average* keeps most of the gain and loses nowhere (game 3: +0.002 to +0.008); on BAR-ATH it
  sits between the two (79 vs 81 %, 7.0 vs 6.8 flips per minute).
- Files (replaced by #6's): `models/control_mlp.json` (the network) and `models/control_v2.json`
  (the linear #4).

Entry 6 (branch `exp/peso-posesion`): **+0.0081 spellF1 (paired, standard error 0.0015, 79 % of
runs better), macroF1 +0.0005, better under all 4 conditions (coarseGT +0.002), 11/12 cells**
(the one below: game 3 `degraded`, -0.002). Both models of the average are trained with each frame
weighted by (length of its true possession)^-p, normalised to mean 1: spellF1 counts every
possession once, per-frame training counts long ones many times. The error analysis of #5 showed
the evidence missing short possessions. One command trains both (`train_control.py` now trains
the `ensemble` members too): `models/control_v4_mlp.json`, `models/control_v4_lin.json`.
On BAR-ATH the output is as like the bench as #5's (confident on 76 vs 79 % of frames with ball,
7.7 vs 7.6 sign flips per minute).

| p | mean spell | light | degraded | +tags | +coarseGT | cells | macro | poss | |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| 0 (#5) | 0.605 | 0.627 | 0.564 | 0.591 | 0.638 | | 0.851 | 248 | |
| 0.25 | 0.613 | 0.640 | 0.573 | 0.599 | 0.640 | 11/12 | 0.852 | 256 | kept |
| 0.5 | 0.617 | 0.645 | 0.581 | 0.604 | **0.636** | 10/12 | 0.851 | 266 | loses under coarseGT |
| 1 | 0.602 | 0.646 | 0.568 | 0.593 | **0.601** | 6/12 | 0.846 | 295 | over-segments |

The stronger the weighting, the more short possessions it finds, which pays with the fine ground
truth and costs with the coarse one. BAR-ATH's public window has fewer short possessions than the
bench's fine ground truth (2 of 17 under 2 s vs 27 %), so only the value that wins under both is
kept.

Also tried (same run as 0.25): group `short` (the proximity features over +/- 0.5 s and the half
second before / after) on #5: +0.0029 (se 0.0009), all 4 conditions but 9/12 cells.

Tried on top of #6 and not kept (paired vs #6, by condition light / degraded / +tags / +coarseGT):
- `short`: -0.0003 overall (-0.001 / +0.002 / -0.001 / -0.002), 6/12 cells: what it added to #5 is
  covered by the possession weighting.
- `team_switch` 13: +0.0035 (+0.007 / +0.007 / +0.004 / **-0.003**), 8/12; 20: -0.0029 (-0.006 /
  -0.006 / -0.005 / **+0.005**), 3/12. Cheaper hand-overs win with the fine ground truth and lose
  with the coarse one, like stronger weighting: how finely to cut is a knob whose best value
  depends on how fine the real ground truth is, which we cannot measure. 16 with p = 0.25 is the
  setting that wins under both.

Entry 7, **rejected** (branch `exp/muerto`; code kept, off: `context.dead_model`): a dead-ball
model learnt on the other games (`src/possession/dead.py`, `validation/train_dead.py`), a plain
logistic regression on the signals the default dead-ball evidence uses (players' speed over several
windows, where the ball was last / next seen, time since / until seen, ball speed, ball alone,
nearest player) plus the default rules' output, replacing the default evidence. Held-out AUC
0.957-0.979 vs 0.928-0.962 for the default.
- Trained on `degraded` copies only: +0.0067 spellF1 but **-0.016 under `light`**, macroF1 -0.007.
- Trained on `degraded` and `light` copies (`train_mix: [light]`): **+0.0171 spellF1 (se 0.0028),
  all 4 conditions, 10/12 cells, macroF1 +0.001, alive/dead F1 +0.005; game 3 +0.03 to +0.05.**
- **But it does not transfer to BAR-ATH** (`check_bar.py`, label-free): stoppages over the match
  104 -> 93, dead share 40.6 -> 35.5 %, Smart Tagging set pieces with a predicted stoppage 96.6 ->
  91.4 % (two throw-ins and a goal kick lost). A match has more stoppages, not fewer; the learnt
  model is too ready to call play alive on our tracking, whose stoppages the degradation of the
  bench imitates only coarsely. (`check_dead.py`: same feature scales, but its output flips sign
  3.6 times per minute on BAR-ATH vs 2.9 on the bench.)
- Process note: this candidate was first run on BAR-ATH with `run.py`, which prints the public
  table; the decision rests on the label-free checks above, and exploratory runs now use
  `check_bar.py`. The accepted #2-#6 pass `check_bar.py` unchanged: 103-104 stoppages, 96.6 % of
  set pieces with a stoppage; possessions 246 (#3) -> 254 -> 267 -> 270 (#6).

Entry 8, **rejected** (tag `descartado/control/semillas`; code kept, off: `context.control.mlp_seeds`):
the network of #6 replaced by the mean of K networks trained identically with different seeds
(the mean then averaged 50/50 with the linear model, as before). Held-out log-loss improves a little
(game 3 0.179 -> 0.175 with K=5) and not further with K=10. Bench, paired over 48 runs vs #6 (0.6128):
K=5 0.6138, +0.0010 (se 0.0010, 48 % of runs better; light -0.0013, degraded +0.0027, +tags -0.0008,
coarseGT +0.0034; 9/12 cells; game 1 loses under 3 of 4 conditions); K=10 +0.0008 (se 0.0012, light
-0.0019, +tags -0.0006, 7/12). Noise: the network's seed is not where the error is. With `[0]` the
code retrains byte-identical v4 files.

*What we learn (#8):* the network's variance between seeds is small next to the bench noise.
Lower held-out log-loss (−0.004) does not turn into spellF1. The remaining error is bias (what
the features cannot see), not variance, so more capacity or averaging on the same features will
not pay; new information will (features, data, labels).

Entry 9 (branch `exp/reglas`, tag `aceptado/muerto/dead-weight-03`): **`dead_weight` 0.5 -> 0.3.**
The rule-made constants were set when the team evidence was nearest-player proximity. With the
learnt model in charge, they were swept one by one on #6 (`sweep.py`, seeds 0-3, mean spellF1 vs
0.6128; cells better of 12):

| parameter (now) | values: change in mean spellF1 (cells) | |
|---|---|---|
| `passes.pass_llr` (15) | 0: -0.0015 (4) · 8: -0.0001 (5) · 25: +0.0000 (5) | flat: no effect |
| `passes.kick_team_llr` (10) | 0: -0.0010 (4) · 5: -0.0009 (4) · 15: +0.0014 (8, coarseGT -0.002) | noise |
| `transition_cost.to_dead` (10) | 6: -0.0023 (0) · 14: +0.0022 (11) · 20: +0.0030 (11) · 30: +0.0042 (11) | see below |
| `transition_cost.from_dead` (5) | 3: -0.0015 (1) · 8: +0.0019 (11) | small |
| `dead_weight` (0.5) | 0: -0.0374 (0) · 0.1: +0.0042 (8, macro -0.0014) · 0.3: **+0.0060 (12, macro +0.0034)** · 0.8: -0.0082 (0) | kept 0.3 |

- `dead_weight` 0.3 (the speed model of dead ball, fitted on each match with rule-made
  pseudo-labels, gets less weight per frame): seeds 0-3 +0.0060, all 4 conditions (+0.004 to
  +0.007), 12/12 cells for spellF1 and macroF1; **new seeds 4-7: +0.0088 (se 0.0008, 98 % of runs
  better), all 4 conditions (+0.007 to +0.010), 12/12 cells, macroF1 +0.0038, game 3 +0.011.**
  On BAR-ATH (`check_bar.py`, label-free): 268 possessions (270), 106 stoppages (104), dead share
  40.3 % (40.6 %), 96.6 % of Smart Tagging set pieces with a stoppage (unchanged), 1.7 % of frames
  changed. 0.1 to 0.3 is a plateau; 0.3 is better on macroF1.
- `to_dead` 20 (and 20 together with `dead_weight` 0.3: +0.0072) passed the bench rule but **not
  the BAR-ATH check**: stoppages 104 -> 94, set pieces with a stoppage 96.6 -> 94.8 % (a goal
  kick lost). Same pattern as #7: a dearer stoppage costs real ones on our tracking. Tag
  `descartado/muerto/to-dead-20`.
- Second round on #9 (branch `exp/muerto2`, no candidate, not tagged): the dead-ball rule
  constants, change in mean spellF1 (cells better of 12):
  - `dead_llr_clip` (3): 2 -0.0001 (1), 5 -0.0003 (0);
  - `ball_dead.inside_llr` (0.5): 0 **-0.0203** (0), 1 +0.0003 (7, macroF1 -0.005);
  - `out_llr` (2): 1 +0.0004 (7), 4 -0.0003 (0);
  - `isolated_llr` (1.5): 0.5 +0.0010 (9), 3 -0.0016 (0);
  - `latch_llr` (1.5): 0.75 +0.0004 (7), 3 -0.0004 (0);
  - `kick_llr` (2): 1 +0.0003 (6), 4 -0.0005 (3).

  All flat within noise: the rules are at a sensible point; only the "ball seen inside the pitch"
  evidence matters (removing it costs 0.02).
- *What we learn (#9):*
  - The only rule constant that mattered was the weight of the self-fitted speed model. It was
    too strong once the learnt team model sharpened the hand-overs.
  - Changes that make stoppages dearer win on the bench and lose real stoppages on BAR-ATH
    (twice now: #7, `to_dead` 20). The bench imitates our stoppages badly, so any dead-ball change
    needs `check_bar` before anything else.
  - The pass and kick rules are now redundant with the learnt model (flat sweeps). They are
    harmless, so they stay.
- Error analysis of #6 (`errors.py`, `degraded`): spellF1 0.573; perfect stoppages 0.632, perfect
  team 0.853. True possessions found at IoU 0.5: < 2 s 11 %, 2-5 s 42 %, 5-10 s 67 %, 10-20 s 78 %,
  > 20 s 79 %. Missed: 20 % to the rival, 21 % merged or wrong edges, 8 % called dead.

### Error analysis of #3 (`validation/errors.py`, condition `degraded`)

- **Oracles**: 0.540 as is; 0.602 with perfect stoppages (our teams); **0.853 with the perfect team**
  (our stoppages). The work that pays is *who has the ball*, not dead-ball detection.
- Wrong team on 7.0 % of the frames both call alive with the ball seen, 21.7 % in ball gaps; gaps
  are 9.6 % of those frames, so 75 % of team errors happen with the ball visible.
- 241 predicted vs 384 true possessions: still under-segmented. True possessions found at IoU 0.5:
  < 2 s 8 %, 2-5 s 35 %, 5-10 s 62 %, 10-20 s 76 %, > 20 s 77 % (< 2 s are 27 % of all true
  possessions, 2-5 s 25 %). Missed: 22 % of true possessions mostly given to the rival, 23 % right
  team but merged with neighbours or wrong edges, 8 % called dead.
- Hand-overs with a predicted one of the same kind within 1 / 2 / 5 s: 34 / 44 / 54 % (main: 16 /
  25 / 37 %).
- For scale, the public BAR-ATH window (aggregates of the public file, to understand the definition,
  not to tune): 17 possessions, 2 under 2 s, 4 of 2-5 s; 12 of its 20 transitions are direct H<->A.
  Short possessions exist in the real ground truth too, a bit fewer than in the bench's fine one.

### Error analysis of #5 (`validation/errors.py --switch-oracles`, condition `degraded`)

- 0.564 as is; perfect stoppages 0.626; perfect team 0.854. **A perfect hand-over detector adds
  little**: switches nearly free within +/- 0.5 s of every true H<->A hand-over gives 0.600, also
  2.5x dearer elsewhere 0.611. Timing the switches is not the bottleneck; the evidence is.
- Team evidence (emissions H - A, towards the true team) inside true possessions, per run:
  missed < 2 s (95 per game, a quarter of all true possessions): mean -0.15, right on only 41 % of
  their frames, so the evidence does not see them; missed 2-5 s (58): +0.66, 59 %; found ones of any
  length: +2.0 to +3.3, 88-96 %. Missed 10-20 s ones have strong right evidence (+2.9, 94 %): they
  are merged across a short rival possession that was missed, or have wrong edges.
- Why: the model's strongest features are +/- 2 s averages, which drown a 1 s possession, and it is
  trained with one weight per frame, so long possessions dominate; spellF1 counts each possession
  once whatever its length.

### Tried and rejected

| change | result on the bench | why rejected |
|--------|---------------------|--------------|
| `ball.contact_m` 2.0 -> 1.0 | +0.013, 12/12 cells on the first bench; **-0.025, 0/12** after adding ball noise | artefact of Metrica's ball glued to the feet; on our ATD only 28 % of ball frames have a player < 1 m |
| `ball.contact_m` 3.0 | +0.001, 6/12 | no effect |
| ignore proximity when the ball flies (> 5/8/12 m/s) | -0.027 / -0.005 / -0.002 | receiving a long pass of the rival is real evidence |
| `ball.weight` 2, 3 (first bench) | gains only with the fine GT, -0.016 / -0.037 under coarseGT | not robust |
| `passes.pass_llr` 8, 25 (first bench) | +-0.001 | no effect |
| `team_switch` 28 | -0.021, 0/12 | |

### Where the labeller still fails (degraded bench, error analysis of #2; see entry 3 for #3)

- Possessions shorter than 5 s are almost never found (2 % of < 2 s, 16 % of 2-5 s matched at
  IoU 0.5), and they are half of all possessions. Missing a short one in A-H-A merges the two A
  possessions into one, so even 20 s+ possessions are matched only 66 % of the time.
- Only 47 % of true hand-overs have a predicted one of the same kind within 5 s.
- Stoppages are fine: restarts land ~0.2 s late (median), stoppage starts ~0.4 s early.
- The ball is visible on 85 % of possession time, so the information is there: what is missing
  is better evidence of *who controls* the ball (touch sequences, a learned model trained on these
  other games), not more tuning of the costs.

## Open experiments: pre-registered before measuring

Method since 13-nov (the coordinator decides): before measuring, the hypothesis, the single change
and the acceptance rule are written here and committed. After measuring there are three outcomes:
- everything passes: accepted;
- something fails clearly: rejected;
- doubtful (gain below one standard error, signs differ between conditions, or a secondary check
  fails narrowly): one more round with seeds 8-11; if still doubtful, rejected.

Thresholds are never changed after seeing a result. Each closed experiment gets a "What we learn"
paragraph, and each accepted version gets `errors.py` to reorder the queue.

### #10 v6: control model trained on 10 games (branch `exp/idsse`)

- **Change:** `context.control.model` / `ensemble` -> `models/control_v6_mlp.json` /
  `control_v6_lin.json`. Same features, network + linear, `weight_power` 0.25, trained on the
  3 Metrica + 7 IDSSE games (`validation/bench_all.yaml`) with the relative attack direction.
  Nothing else changes.
- **Hypothesis:** more and more varied training data (a second provider) gives better log-odds on
  an unseen provider. The IDSSE ground truth is built like the official one (ball status and
  possession from the tracking).
- **Rule:** first written 12-nov in PROGRESS.md ("Regla para la línea IDSSE"). The seeds 0-3
  rounds were measured on v4 with `dead_weight` 0.5: Metrica +0.0094 (se 0.0022), 10 games
  +0.0177 (se 0.0010).
- **Rule restated 13-nov 11:50, before any result on the current base:**
  - Reference: `josep/mejoras` 7c9e6c4 (v4 + `dead_weight` 0.3). Candidate: the same config with
    the v6 models.
  - The only measurement on this base is seeds 4-7, so the full conditions apply to it:
    - (a) 10-game bench (Metrica + DFL, seeds 4-7): mean spellF1 up under each of the 4 conditions,
      more than 20 of the 40 (condition, game) cells better, mean macroF1 not down;
    - (b) Metrica bench alone (seeds 4-7): mean spellF1 up under each of the 4 conditions, at least
      7 of the 12 cells better, mean macroF1 not down, game 3 (mean over its conditions) not down;
    - (d) `check_control`: the model's behaviour on BAR-ATH relative to the bench as with v4.
      Measured 13-nov: |llr| > 2 on 72.1 % of BAR-ATH frames with ball vs 75.2 % on the bench,
      7.13 vs 7.16 sign flips per minute. It does not depend on `dead_weight`.
    - (e) `check_bar` on the current base: 240-300 possessions; stoppages and the share of set
      pieces with a stoppage not below the current version (106; 96.6 %).
  - Doubtful (see above): seeds 8-11 on both benches.
  - Before merging: Codex critiques the decision, and each objection is answered here.
- **Threshold change on (d), recorded as required.** The 12-nov rule said "|llr| > 2 on 76-82 %
  of BAR-ATH frames with ball", an absolute range taken from models trained on Metrica only. v6
  gave 72.1 % on BAR-ATH and 75.2 % on the bench: it is less confident everywhere, and BAR-ATH
  relative to the bench is the same as v4's (76 vs 79 %). Josep ruled on 13-nov, AFTER the
  measurement (in his decisions file), that (d) is read relatively. This is a change made after
  measuring. It applies to v6 by his decision, and to later models from now on.

**Result (13-nov), ACCEPTED: tag `aceptado/datos/v6-idsse`, `josep/mejoras` e652fd2.**

| check | seeds 4-7 on #9's base | passes |
|---|---|---|
| (a) 10 games | +0.0190 (se 0.0010, 92 % of runs better); light +0.022, degraded +0.020, +tags +0.019, coarseGT +0.015; 39/40 cells; macroF1 +0.0037 | yes |
| (b) Metrica | 0.6160 -> 0.6292, +0.0132 (se 0.0020, 81 % better); light +0.011, degraded +0.014, +tags +0.014, coarseGT +0.014; 11/12 cells (game 1 under light -0.0045); macroF1 +0.0024; game 3 +0.026, games 1 and 2 +0.007 | yes |
| DFL alone | +0.0215 (se 0.0011), 28/28 cells, every game +0.015 to +0.028 | (part of a) |
| (d) `check_control` | 72.1 vs 75.2 %, 7.13 vs 7.16 sign flips / min (relative reading, see above) | yes, by Josep's ruling |
| (e) `check_bar` | 268 possessions (268), 106 stoppages (106), 96.6 % of set pieces with a stoppage (96.6 %), 1.4 % of frames changed | yes |

CSV: SHA256 `c53b0b1baece1e93cf017ac49ebb51b37c93f4d9c406e44fe99b1a668e8b1a02`, the same twice;
H 54,891 / A 39,856 / D 64,110 / X 40,462. `check_rules` PASS.

**Codex's critique (consulta-03, `C:/consultas-ia/consulta-03-*.md`) and our answers:**
1. *(d) was not met as written; moving it afterwards is post hoc.* True, and recorded above. The
   exception was made by the team lead, not by the coordinator. What (d) is meant to catch
   (behaviour on BAR-ATH unlike the bench) is not present: v6 is less confident on the bench too
   (79 -> 75 %), by the same amount as on BAR-ATH.
2. *Many correlated tests; fix a primary metric.* The primary metric is the mean paired difference
   in spellF1 with its standard error. Conditions and cells are robustness filters, not counted
   as independent evidence. +0.019 is 19 standard errors on 10 games, and +0.013 is 6.6 on Metrica
   alone, on seeds not used in round 1.
3. *The bench does not measure transfer to a new provider: on B, v6 saw the provider.* That is
   why (b) requires the Metrica bench alone, where the comparison favours the reference: the
   reference was trained on 2 games of the evaluated provider, v6 on the same 2 plus 7 of
   another. v6 still wins there under all 4 conditions. The closest leave-provider-out we have
   (12-nov, exploration): a model trained on the 7 IDSSE games only, never having seen Metrica,
   scored +0.0027 over v4 on the Metrica bench. Another provider's data transfers at least as well
   as two games of the same provider.
4. *The gain could come from the GT or from the direction fix, not from data variety.*
   - The direction fix leaves v4's bench runs and our CSV identical. It only fixes the IDSSE
     training data, and the reference got it too in round 2.
   - Data and its GT are one change by design: adding IDSSE means adding its labels. The decision
     does not depend on splitting them.
5. *Separate the effects; per-game persistence.* Per game, the gain holds in all 10:
   - Metrica: +0.007 / +0.007 / +0.026;
   - IDSSE: +0.015 to +0.028.

   Labels of a third provider do not exist for us. The target's public window is forbidden for
   decisions.
6. *Errors by context and calibration before trusting it.* `errors.py` runs on every accepted
   version, and runs next on this one. Shared systematic errors between IDSSE's GT and its own
   evaluation can only inflate the DFL bench, which is why (b) is required.
7. *Less confident means ambiguous frames near 0 may move edges.* On BAR-ATH, 1.4 % of frames
   change and the number of possessions and stoppages is identical. On the bench, spellF1 rises
   under the coarse GT too (+0.014 Metrica, +0.015 overall), so edges are not getting worse.
8. *Accept as a candidate, not as validated for the target.* Agreed in spirit. `josep/mejoras`
   is "the best version by our evidence". The submission is decided by the team on 16-nov, and
   every earlier version is one tag away.

*What we learn (#10):*
- **Data was the bottleneck, as #8 suggested.** More games and a second provider gave more than
  any feature or capacity change since v2. The largest gain was on the most different Metrica
  game (+0.026).
- A model trained on another provider's labels built from tracking (the official construction) did
  not hurt on the event-labelled games.
- Next in this line:
  - re-run `errors.py` on v6;
  - re-measure `train_mix` on the 10 games (#11);
  - look for more open tracking data with possession labels (SkillCorner, MIT).

*Error analysis of #10* (`errors.py`, Metrica, `degraded`, seeds 0-3; #6 in brackets):
- spellF1 0.590 [0.573]. Perfect stoppages 0.647 [0.632]; perfect team 0.859 [0.853].
- Wrong team on 5.8 % [6.1 %] of live frames with the ball, and on 21.1 % [21.6 %] of the gaps.
- True possessions found at IoU 0.5:

  | < 2 s | 2-5 s | 5-10 s | 10-20 s | > 20 s |
  |--:|--:|--:|--:|--:|
  | 12.0 % [11.1] | 45.2 % [41.5] | 67.8 % [67.0] | 78.1 % [77.6] | 79.6 % [78.7] |

- Missed < 2 s possessions still have no team evidence (−0.10 towards the true team, 42 % of
  frames right). Missed 2-5 s ones have some (+0.71, 62 %).
- 256 possessions predicted vs 384 true.
- Missed: 19.5 % to the rival, 20.4 % merged or wrong edges, 8.3 % called dead.
- Hand-overs with a predicted one within 1 / 2 / 5 s: 41 / 50 / 60 %.
- **For the queue:**
  - the team is still the lever (+0.27 to the team oracle), then stoppages (+0.057);
  - short possessions are invisible to the features, so more data and robustness pay more than
    decoder tricks;
  - for stoppages, the blocker is validation: neither bench imitates our stoppages, because
    IDSSE's ball never disappears in dead time.

### #11 control model also trained on `light` copies (`context.control.train_mix`, branch `exp/mezcla`)

- **Change:** the control models are trained on the usual degraded copies plus copies degraded as
  the bench's `light` condition (seeds 100-101). Nothing else changes.
- **Hypothesis:** the real tracking is not exactly our imitation of it. A model that has seen
  two degradation strengths should transfer better (Codex, consulta-01, idea 4).
- **Plan, depending on #10:**
  - If #10 is accepted: the v7 models (trained on Metrica only) would be two changes against v6.
    `train_mix [light]` is therefore retrained on the 10 games of v6 and measured against v6 with
    the same (a), (b), (d), (e) as #10. First seeds 0-3; if that passes, seeds 4-7 on both benches.
  - If #10 is rejected: v7 (trained on Metrica) against the current version with the usual rule:
    Metrica bench seeds 0-3, all 4 conditions up, at least 7/12 cells, macroF1 not down, game 3
    looked at; then seeds 4-7 with the same rule, `check_control` (relative) and `check_bar` as
    in (e).

**Result (13-nov), REJECTED: tag `descartado/control/train-mix-light`.**
- Retrained on the 10 games: models `control_v8_*`, trained with `--workers 1` because with 2 the
  run was killed for lack of memory.
- Held-out log-loss of the network: Metrica 0.167 / 0.162 / 0.174 (v6 0.162 / 0.159 / 0.172);
  IDSSE about the same.
- Metrica bench, seeds 0-3, vs v6 (0.6292): 0.6318, +0.0027 (se 0.0012, 62 % of runs better).
  - By condition: light +0.0067, degraded +0.0046, +tags +0.0047, **coarseGT −0.0053** (se
    0.0019, 25 % better; all 3 coarseGT cells negative).
  - 9/12 cells, macroF1 +0.0006.
  - +6 predicted possessions per game (265 vs 259).
- Rule (b) fails clearly (coarseGT down), so the 10-game bench was not run.

*Codex's critique (consulta-04) and our answers:*
1. *No flaw in rejecting it.* Agreed.
2. *"Cuts finer" is plausible but not proven: the boundaries' calibration could change, or it
   could fit bench A's style.* True: we did not separate these, and for the decision we do not
   need to. Whatever the mechanism, the effect on the scores is the known pattern: more
   possessions, a gain only with the fine ground truth.
3. *A smaller dose of light copies only with a rule fixed beforehand that coarseGT does not drop;
   otherwise close the line.* Closed. A dose is one more "how finely to cut" knob, like
   `weight_power` 0.5 and `team_switch` 13: the bench can only tell us that it trades the fine
   ground truth against the coarse one.

*What we learn (#11):*
- This is the fourth change whose gain comes from more possessions and is paid for under
  coarseGT: stronger weighting, cheaper hand-overs, and now cleaner training copies.
- The current setting sits where the fine and coarse ground truths balance. Any further gain has
  to come from better evidence at the same cut, not from cutting finer.
- Training on cleaner copies makes the model more confident in short touches. That is exactly
  what moves the cut.

### #12 interpolate longer ball gaps (`ball_gaps.interpolate_max_frames`, branch `exp/interp`)

Pre-registered 13-nov, before measuring.
- **Change:** `ball_gaps.interpolate_max_frames` 12 (0.5 s) -> 25 or 50. Ball gaps up to that
  length are filled by linear interpolation, at inference only; the models are not retrained.
  Nothing else changes.
- **Hypothesis:** about 41 % of our frames miss the ball, many in short gaps inside live play.
  With the ball interpolated, the learnt team model (used only where the ball is "seen") gives
  evidence on more frames. Interpolated positions between two touches lie on the pass line, so
  they should not mislead.
- **Rule (the usual one):**
  - A sweep on the Metrica bench, seeds 0-3, with values 25 and 50. A value is a candidate if its
    mean spellF1 is up under each of the 4 conditions, at least 7 of the 12 cells are better,
    macroF1 is not down and game 3 is not down. If both pass, the higher mean is the candidate.
  - The candidate needs the same on seeds 4-7.
  - `check_control` (relative: BAR-ATH vs bench as with v6).
  - `check_bar` (240-300 possessions; stoppages and set pieces with a stoppage not below the
    current 106 and 96.6 %).
  - Doubtful: seeds 8-11.

### #13 learnt dead-ball model, retried on 10 games with our kind of stoppages (branch `exp/muerto-idsse`)

Pre-registered 13-nov, before measuring.
- **Change:** `context.dead_model` on, with `models/dead_v2.json`. Same model, features, weight
  (0.5), clip and `train_mix [light]` as the rejected #7. Only the training data changes: the
  10 games of `validation/bench_all_dead.yaml` (Metrica + IDSSE, ball hidden while dead) instead
  of the 3 Metrica games. `train_dead.py` now loads any bench game through `bench.load_game`.
- **New reason** to retry a rejected idea:
  - #7 failed on BAR-ATH by calling play alive too readily (stoppages 104 -> 93), having learnt
    stoppages from 3 games of one provider.
  - IDSSE adds 7 games whose dead ball comes from the tracking's own ball status (the official
    construction).
  - The training copies now lose the ball in dead time, as ours does.
- **Hypothesis:** a dead-ball model learnt on more, and more realistic, stoppages closes part of
  the stoppage oracle gap (+0.057 on v6) and transfers to BAR-ATH.
- **Rule**, checked in this order; the first clear failure rejects:
  1. `check_bar` against the current version: 240-300 possessions, stoppages >= 106, share of
     set pieces with a stoppage >= 96.6 %;
  2. Metrica bench, seeds 0-3: mean spellF1 up under each of the 4 conditions, >= 7/12 cells,
     macroF1 not down, game 3 not down;
  3. IDSSE bench with the ball hidden in dead time (`bench_dfl_dead.yaml`), seeds 0-3: mean
     spellF1 up under each of the 4 conditions, more than 14 of 28 cells, macroF1 not down;
  4. Metrica bench, seeds 4-7: as in 2;
  5. `check_dead`: the learnt model's sign flips per minute on BAR-ATH at most 1.2x the bench's
     (#7: 3.6 vs 2.9, 1.24x).

  Doubtful: seeds 8-11.

**Result of #12 (13-nov), REJECTED: tag `descartado/huecos/interp-50`.**
- Sweep, seeds 0-3, vs v6:
  - 25 frames: +0.0005 (se 0.0004), but macroF1 −0.0002 (2 se) and coarseGT −0.0006: rejected;
  - 50 frames: +0.0005 (se 0.0005), +tags 0.0000, coarseGT −0.0002: doubtful.
- Extra round for 50, seeds 8-11: −0.0004 (se 0.0005), macroF1 −0.0007, 4/12 cells; only light
  goes up (+0.0018).

*Codex (consulta-05, part 1):* the rejection is justified. Caveat: it rejects this inference-time
intervention, not the idea that more reliable ball observations help. Interpolated positions can
be wrong, and the model was not retrained on them. Answer: agreed. We do not reopen it, because
our tracking's gaps cannot be filled with real observations.

*What we learn (#12):* the team model gets little from interpolated ball positions in longer
gaps. Where the ball is missing for more than half a second, the line between two touches is not
evidence of who controls it. The only condition that liked it, `light`, has the fewest and
shortest gaps.

### #14 team evidence in ball gaps from v6's gap model (`context.control.gap_weight`, branch `exp/gap`)

Pre-registered 13-nov, before measuring.
- **Change:** `context.control.gap_weight` 0 -> 0.1 or 0.3. The learnt control model has a
  conditional logit for frames without the ball, trained with the rest of v6; today its output
  is not used. Nothing is retrained.
- **New reason** to retry: tried on v3 (+0.001 to +0.003, noise), when the gap model was trained
  on 3 games of one provider. v6's is trained on 10 games of 2 providers, and Codex (consulta-05,
  path 1) points at the gaps as the largest team error: 21 % wrong vs 5.8 % with the ball seen.
- **Rule (the usual one):**
  - A sweep, Metrica bench, seeds 0-3, values 0.1 and 0.3. A candidate has mean spellF1 up under
    each of the 4 conditions, >= 7/12 cells, macroF1 not down and game 3 not down; if both pass,
    the higher mean.
  - The candidate needs the same on seeds 4-7, plus `check_bar` (240-300 possessions; stoppages
    and set pieces with a stoppage not below 106 / 96.6 %).
  - Doubtful: seeds 8-11.

**Result of #13 (13-nov), REJECTED at the first check: tag `descartado/muerto/modelo-10-partidos`.**
- Held-out games: AUC 0.963-0.998 vs 0.916-0.982 for the default evidence. Largest weight: "ball
  seen" (−3.8, i.e. seen means alive).
- `check_bar` on BAR-ATH:
  - stoppages 106 -> **82**;
  - set pieces with a stoppage 96.6 -> **84.5 %** (throw-ins 26/26 -> 20/26);
  - dead share 40.4 -> 33.6 %;
  - possessions 268 -> 274.
- Clear failure of check 1: the benches were not run.

*Codex's critique (consulta-06) and our answers:*
1. *The rejection follows the rule.* Agreed.
2. *Our reading (BAR-ATH sees the ball in part of its stoppages, so hiding it in all dead time
   taught "seen = alive") is plausible but not proven by the aggregates; the throw-ins could also
   be the rules' fault.* True. It stays a hypothesis.
3. *The model may have learnt something specific to the providers or their labels.* Possible,
   and this is the second learnt dead-ball model to fail the same way (#7). The pattern is
   clear: what predicts stoppages on other providers' tracking does not carry over to ours.
4. *Do not tune a hidden-ball fraction against BAR-ATH's aggregates: that calibrates against our
   own gate.* Agreed. Not done, and the line is closed.
5. *If anything, work by type of stoppage (throw-ins).* Noted in the queue. It is not tried here:
   there is no labelled throw-in timing on our match, and the bench does not imitate our
   stoppages.

*What we learn (#13):*
- Learnt dead-ball models win on the benches and lose stoppages on BAR-ATH, twice now.
- The default speed model, fitted on our own match with rule-made pseudo-labels, transfers by
  construction. Only its weight was worth tuning (#9).
- The dead-ball line is closed for this deadline.
- The tool `degrade.hide_dead_ball` stays: it makes the IDSSE bench's ball visibility closer to
  ours. But hiding ALL dead time is too strong a model of our tracking.

### #15 v9: control model trained on 20 games, SkillCorner added (branch `exp/skillcorner`)

Pre-registered 13-nov, before measuring.
- **Change:** `context.control.model` / `ensemble` -> `models/control_v9_*`.
  - Same features, network + linear, `weight_power` 0.25.
  - Trained on `validation/bench_all20_events.yaml`: v6's 10 games plus 10 SkillCorner games
    (A-League 2024/25, broadcast tracking at 10 fps resampled to 25, MIT licence).
  - SkillCorner ground truth: dead outside its "phases of play"; while alive, the team of the
    last `player_possession` event. Its "possession group" field disagrees with those events on
    17 % of live frames and with the nearest-player check on 15 %.
  - One SkillCorner game is left out: its phases break during play (41 stoppages under 3 s).
  - Training memory: `train_control.py --no-heldout --logo-games <the 10 bench games>`, a
    memory-only option: same model files, no held-out copies.
- **Hypothesis:** going from 3 to 10 games was the largest gain (+0.013 / +0.019). A third
  provider, with tracking even worse than ours (broadcast: ~14.5 players seen, ball 1.5 m from
  the nearest player), adds variety for an unseen provider.
- **Risk:** its ground truth is coarser (2.6 possessions per minute, 12 % < 2 s), which could teach
  coarser cuts; the 4-condition rule catches that.
- **Rule** (as #10, on v6):
  - (a) 10-game bench (Metrica + IDSSE, `bench.yaml` + `bench_dfl.yaml`), seeds 0-3: mean spellF1
    up under each of the 4 conditions, more than 20/40 cells, macroF1 not down;
  - (b) Metrica alone, seeds 0-3: up under each of the 4 conditions, >= 7/12 cells, macroF1 not
    down, game 3 not down;
  - (c) the same (a) and (b) on seeds 4-7;
  - (d) `check_control`, relative: |llr| > 2 on BAR-ATH within 5 points of the bench's, and sign
    flips per minute on BAR-ATH at most 1.1x the bench's (v6: 72.1 vs 75.2 %, 7.13 vs 7.16);
  - (e) `check_bar`: 240-300 possessions; stoppages >= 106; set pieces with a stoppage >= 96.6 %.
  - Doubtful: seeds 8-11.
- Not measured on a SkillCorner bench: its ground truth is not the official construction and its
  tracking is harder than ours. It is training data only.

**Result of #14 (13-nov), REJECTED by its written rule: tag `descartado/huecos/gap-weight-03`.**
- Sweep, seeds 0-3, vs v6:
  - 0.3: +0.0050 (se 0.0008); light +0.0017, degraded +0.0055, +tags +0.0069, coarseGT +0.0059;
    12/12 cells; macroF1 +0.0014; game 3 +0.0041;
  - 0.1: +0.0031 (se 0.0007), all 4 up, 11/12 cells.

  Candidate 0.3.
- Seeds 4-7 (0.3): +0.0044 (se 0.0008, 85 % of runs better); light +0.0023, degraded +0.0052,
  +tags +0.0055, coarseGT +0.0044; 11/12 cells; macroF1 +0.0015; game 3 +0.0044.
- `check_bar`:
  - possessions 268 -> 269;
  - **stoppages 106 -> 105** (the rule asked >= 106);
  - set pieces with a stoppage 96.6 % -> 96.6 %, every type identical;
  - dead share 40.4 -> 40.5 %;
  - 0.7 % of frames changed.
- A secondary check failing narrowly makes it doubtful. The only extra round allowed (seeds 8-11)
  cannot change a deterministic check on BAR-ATH, so it stays doubtful and is rejected as the rule
  says. Thresholds are not changed after the result.

*Codex's critique (consulta-07) and our answers:*
1. *Rejecting 0.3 is right; re-testing it under a new threshold would be changing the rule
   afterwards.* Agreed. 0.3 is not reopened.
2. *Testing 0.1 (seen in the sweep) under a corrected check would select it a posteriori; a new
   rule does not erase what was seen.* Agreed. Only a confirmation on data never used for that
   setting, with the earlier result explicitly not counted, would be legitimate (#16, if time
   allows).
3. *Apply the correction only to future changes.* Done:

**Threshold correction, written 13-nov after #14, for experiments registered from now on (#16 on).**
In `check_bar`, the
rule "stoppages >= current" was written for dead-ball changes, which lost real stoppages
(104-106 -> 82-94). For changes that do not touch the dead-ball evidence it becomes: stoppages
within ±2 of the current version, and set pieces with a stoppage not below it. Dead-ball changes
keep ">= current". This affects #15: its pre-registered (e) said ">= 106", and since the
correction comes after #15 was registered, #15 keeps ">= 106".

*What we learn (#14):*
- The gap model trained on 10 games does carry team evidence: +0.004 to +0.005 on two seed
  sets, 11-12/12 cells. On v3 (3 games) it was noise.
- It moves one stoppage edge on BAR-ATH.
- The gate was written for another kind of change. The lesson is about rules: a check that is
  right for one family of changes can be too tight for another, so write it per family.

### #16 `gap_weight` 0.1, confirmatory only (branch `exp/gap01`)

Pre-registered 13-nov, before any new measurement, as Codex advised in consulta-07.
- **Change:** `context.control.gap_weight` 0 -> 0.1 (the team evidence of the learnt gap model
  in frames without the ball). Measured against `josep/mejoras` as it is when #15 closes; if
  #15 is accepted, the gap model is v9's.
- **The earlier result does not count.** 0.1 was seen in #14's sweep (seeds 0-3: +0.0031) and
  came second; 0.3 failed `check_bar` by one stoppage. To avoid selecting 0.1 after the fact, the
  decision rests only on new measurements:
  - Metrica bench on seeds **8-11** (never run for this setting), with the usual rule: mean
    spellF1 up under each of the 4 conditions, >= 7/12 cells, macroF1 not down, game 3 not down;
  - and seeds **12-15** with the same rule, since doubtful has no further round here;
  - `check_bar` with the corrected threshold for non-dead-ball changes: 240-300 possessions,
    stoppages within ±2 of the current version, set pieces with a stoppage not below it.
- **Outcomes:** all pass -> accepted; anything fails -> rejected (no doubtful round).

**Result of #15 (13-nov), REJECTED: tag `descartado/datos/v9-skillcorner`.** The tag keeps the
SkillCorner integration: `validation/skillcorner.py`, its bench files, and the memory-only
`train_control.py --no-heldout / --logo-games`.

- Metrica bench, seeds 0-3, vs v6 (0.6292): 0.6229, **−0.0063** (se 0.0015, 27 % of runs better).
  - By condition: light −0.0114, degraded −0.0073, +tags −0.0052, coarseGT −0.0011.
  - 1/12 cells; macroF1 −0.0001; game 3 −0.0034.
  - **−6.7 predicted possessions per game** (252 vs 259).
- (b) fails clearly, so the 10-game bench was not run.

*Codex's critique (consulta-08) and our answers:*
1. *No flaw in rejecting it.* Agreed.
2. *Besides the coarser ground truth, the input distribution differs:* 10 fps resampled, fewer
   players seen, the ball farther from the players, camera and selection effects, and an
   algorithmic ground truth that may share biases with its own signals. We cannot split these
   with one bench. The fewer predicted possessions point at the ground truth's granularity first.
3. *Keep it out of training; at most use it as a robustness check, without its possession
   labels.* Agreed. It is not used: its ground truth is not the official construction, so it
   would not validate anything either.

*What we learn (#15):*
- More data is not automatically better. The labels must be cut like the target's.
- IDSSE helped because its ground truth is built like the official one, at a similar or finer
  granularity. SkillCorner's coarser labels taught coarser cuts (−6.7 possessions), and that
  lost under every fine ground truth.
- This is the "how finely to cut" lesson again, from the training labels' side.
