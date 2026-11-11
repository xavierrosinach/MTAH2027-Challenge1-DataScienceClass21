# Experiments

How every change to the labeller is decided. RULES.md forbids tuning or choosing versions with the
public 5 minutes of BAR-ATH (and 17 possessions could not rank anything anyway), so the public
window is only a smoke test (`python run.py`) and decisions are made on **other games**.

## The validation bench (`validation/`)

```
git clone https://github.com/metrica-sports/sample-data ../external/metrica-sample-data
python validation/bench.py --suite                                  # config.yaml vs official baselines
python validation/bench.py --suite --configs base=config.yaml new=experiment.yaml
python validation/calibrate.py [--grid]                             # how the degradation was chosen
```

- **Games**: the three public Metrica Sports sample matches (anonymised, 25 fps, 105 x 68 m,
  tracking + event data). Not BAR-ATH.
- **Ground truth** (`gt_events.py`): derived mechanically from their event data with the rules of
  CHALLENGE.md: dead ball from the end of the last action to each restart (set piece or goal kick);
  while alive, the team of the last controlled action (pass, carry, recovery, shot, restart, ball
  lost); duels, fouls received and balls out do not change possession. ~384 possessions per game,
  ~4 per minute, ~35 % dead ball. Human event data of other games is allowed (RULES.md).
- **Degradation** (`degrade.py`): our ATD is automatic and uncorrected, Metrica's sample tracking
  is cleaner. Calibrated on label-free aggregate statistics only (`calibrate.py`): player tracks
  cut into fragments that alternate between two slots, long ball gaps starting earlier (the tracker
  loses the ball before play stops), extra ball gaps in live play.
- **Smart Tagging stand-in**: none, or SET PIECES tags built from the restarts of the events
  (4 s early, 80 % recall). Optimistic, so it is only one of the conditions.
- **Scorer**: the official `evaluation/possession_metrics.py`, unchanged.

Known limits: the bench is easier than BAR-ATH in absolute terms (the official baselines score
~0.38 spellF1 on it vs 0.24 on BAR-ATH) and no degradation setting closed that gap, so absolute
numbers do not transfer. What we use is the **ranking of versions**, and a change is only kept if
it wins under every condition of the suite (clean, degraded, degraded + tags) and on most games.

## Results

Suite = mean over 3 games (x 2 degradation seeds where degraded). spellF1 / macroF1.

| # | date | version | clean | degraded | degraded+tags | mean spell | mean macro | poss (GT 384) | decision |
|---|------|---------|-------|----------|---------------|-----------:|-----------:|---------------|----------|
| 0 | 2026-11-11 | official baseline_smooth50 | 0.413 | 0.377 | 0.377 | 0.389 | 0.797 | ~510 | floor |
| 1 | 2026-11-11 | main 6791db5 (Xavier, Viterbi) | 0.549 | 0.496 | 0.534 | 0.526 | 0.840 | ~225 | reference |
