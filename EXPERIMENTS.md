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

## Results (current bench: commit b259b60)

Mean over 3 games x 4 seeds. spellF1 per condition; mean macroF1; predicted possessions (GT 384).

| # | version | light | degraded | +tags | +coarseGT | mean spell | mean macro | poss | decision |
|---|---------|------:|---------:|------:|----------:|-----------:|-----------:|-----:|----------|
| 0 | official baseline_smooth50 | 0.360 | 0.303 | 0.303 | 0.310 | 0.319 | 0.769 | ~590 | floor |
| 1 | main 6791db5 (Xavier, Viterbi) | 0.507 | 0.404 | 0.435 | 0.473 | 0.455 | 0.812 | ~215 | reference |
| 2 | #1 + `team_switch` 20 -> 16 | 0.520 | 0.415 | 0.443 | 0.476 | 0.463 | 0.812 | ~237 | kept |
| 3 | #2 + learnt control model (`context.control`, weight 1) | 0.609 | 0.540 | 0.565 | 0.617 | 0.583 | 0.847 | ~244 | kept |
| 4 | #3 + movement features (`features: [base, motion]`) | 0.615 | 0.555 | 0.579 | 0.629 | **0.594** | **0.849** | ~250 | **kept** |

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
