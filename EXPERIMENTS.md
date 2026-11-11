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
```

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
| 2 | #1 + `team_switch` 20 -> 16 | 0.520 | 0.415 | 0.443 | 0.476 | 0.463 | 0.812 | ~237 | **kept** |

Entry 2: +0.009 spellF1, better under all 4 conditions (worst +0.003), 10/12 cells, games 1 and 3 up,
game 2 flat (-0.003). The reference under-segments (215 predicted vs 384 true possessions); a
cheaper hand-over lets it keep more of the short ones. 12 scores a little more on average but
loses under coarseGT, so 16.

### Tried and rejected

| change | result on the bench | why rejected |
|--------|---------------------|--------------|
| `ball.contact_m` 2.0 -> 1.0 | +0.013, 12/12 cells on the first bench; **-0.025, 0/12** after adding ball noise | artefact of Metrica's ball glued to the feet; on our ATD only 28 % of ball frames have a player < 1 m |
| `ball.contact_m` 3.0 | +0.001, 6/12 | no effect |
| ignore proximity when the ball flies (> 5/8/12 m/s) | -0.027 / -0.005 / -0.002 | receiving a long pass of the rival is real evidence |
| `ball.weight` 2, 3 (first bench) | gains only with the fine GT, -0.016 / -0.037 under coarseGT | not robust |
| `passes.pass_llr` 8, 25 (first bench) | +-0.001 | no effect |
| `team_switch` 28 | -0.021, 0/12 | |

### Where the labeller still fails (degraded bench, error analysis)

- Possessions shorter than 5 s are almost never found (2 % of < 2 s, 16 % of 2-5 s matched at
  IoU 0.5), and they are half of all possessions. Missing a short one in A-H-A merges the two A
  possessions into one, so even 20 s+ possessions are matched only 66 % of the time.
- Only 47 % of true hand-overs have a predicted one of the same kind within 5 s.
- Stoppages are fine: restarts land ~0.2 s late (median), stoppage starts ~0.4 s early.
- The ball is visible on 85 % of possession time, so the information is there: what is missing
  is better evidence of *who controls* the ball (touch sequences, a learned model trained on these
  other games), not more tuning of the costs.
