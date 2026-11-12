# Ball possession per frame — Data Science Class 21

More than a Hack 2027, Phase 1. Technical document of the submission `DataScienceClass21.csv`.

> Living document: it describes the version on branch `josep/mejoras` (control model v4) and is
> updated with every accepted change. The full experiment log is [EXPERIMENTS.md](../EXPERIMENTS.md).

## 1. Summary

Every frame gets one of three states, H, A or D. For each frame we add up evidence for each state:

- dead-ball rules and a model of the players' speed;
- **a learnt model of which team controls the ball, trained on three public matches of other
  teams** (Metrica Sports sample data, whose event data were collected by people);
- completed passes, kicks and restarts.

A Viterbi decoder then picks the most likely sequence over each half, with a cost for every change
of state. No label of the provided match was made or used by a person.

Every design decision was taken on a **validation bench built from those other matches**, never on
the public 5 minutes. On that bench (3 matches × 4 tracking-degradation seeds × 4 conditions) the
mean spellF1 went from 0.319 (official `baseline_smooth50`), and from 0.455 for our first
rule-based version, to **0.613**, with macroF1 0.852.

## 2. Approach

### 2.1 Why this design

- **The ball is often missing.** It is absent on ~41 % of graded frames, in long gaps, and when
  visible it sits a median 1.9 m from the nearest player (0.3 m in clean tracking). "Who is
  nearest" flickers, so the evidence has to look at a few seconds of context. The decision has to
  look at the whole half, not frame by frame.
- **spellF1 punishes split possessions and late hand-overs.** A non-causal decoder (Viterbi) with a
  cost per change puts each change where the evidence changes, with no filter lag. It does not
  split a possession on a single noisy frame.
- **The lever is the team, not the stoppages.** We measured this on the bench with oracles. Perfect
  team labels with our stoppages give 0.85 spellF1; perfect stoppages with our teams give 0.63; a
  perfect hand-over timing adds only +0.04. So most of the effort went into learning *who controls
  the ball*.

### 2.2 Pipeline (`src/possession/context.py`)

1. **Inputs.** We read the ATD tracking and the Smart Tagging and interpolate ball gaps of up to
   0.5 s. Goalkeeper slots get their team from the goal they defend, because the tracking
   sometimes swaps them.
2. **Dead-ball evidence.**
   - A logistic regression on the players' speed. It is fitted on the provided match itself with
     **rule-made pseudo-labels** (ball seen > 1 m out of the pitch or the first 3 s of a SET PIECES
     tag = dead; ball moving at a player's feet well inside the pitch = alive). No person decided
     any of these labels.
   - Ball rules: the ball seen out; lost while crossing, or about to cross, a line; or abandoned
     (nobody within 6 m for 2 s).
   - After a ball out, dead until the next kick.
   - Goals: dead from the last ball near the goal to the `Kick Off (After Goal)` tag.
   - The second half is always dead before its kick-off at frame 113,742 (FAQ).
3. **Team evidence, ball seen.**
   - The learnt control model (2.3).
   - +15 log-odds for each completed pass (a touch by one player, controlled by a team-mate at
     least 5 m away within 4 s).
   - +10 to the team of the player who kicks the ball. This decides who takes a restart, but a
     lone clearance does not change possession.
   - The team of a SET PIECES tag restarts.
4. **Team evidence, ball unseen.**
   - A team-shape model (width, depth, advance). It is fitted on frames where the first decode's
     team is within 1 m of the visible ball and the rival is more than 2 m away, so it is
     self-supervised as well.
   - The attacking phases of the Smart Tagging, as weak evidence.
5. **Viterbi decoding per half** over {H, A, D}.
   - Switching H↔A costs 16, going dead costs 10 and restarting costs 5.
   - Restarting costs 10 more if the ball is seen and nobody kicks it within ±0.5 s.
   - A stoppage lasts at least 1 s.
   - The decode runs three times: after the first decode the shape evidence is added; after the
     second, stoppages with no reason are made live.

### 2.3 The learnt control model (`src/possession/control.py`)

- **Features: 54 per team, all relative to the ball or to the team itself.**
  - How close its nearest and second-nearest players are to the ball (exponential kernels at 1,
    2.5 and 6 m).
  - How many of its players are within 5 and 10 m of the ball, and how many are goal-side of it.
  - The ball's position and velocity relative to the team's attacking direction, and the team's
    shape.
  - These are taken over the frame, the second before, the second after and ±2 s.
  - Movement features, over ±0.5 s and the second before and after: the nearest player's closing
    speed towards the ball, whether the ball moves with him (a carry), his speed, and how many
    team-mates run at the ball (pressing).
- **Symmetric by construction.** Each team's features go into the same function, and P(H) depends
  on the difference between the two teams. Swapping the teams flips the answer, so there is no home
  bias and nothing specific to a team.
- **Two models, averaged in log-odds.**
  - A conditional logit with L2 regularisation.
  - An antisymmetric one-hidden-layer network, z = h([D, S]) − h([−D, S]), where D and S are the
    teams' feature difference and sum, with 16 tanh units, Adam, 4 epochs and a fixed seed. It can
    use interactions a linear score cannot. Its size was chosen on held-out-game log-loss.
- **Training on another match's tracking.** The model learns from every live frame of the Metrica
  games. Their tracking is first degraded to look like ours (2.4), with seeds the bench does not
  use. Each frame is weighted by (length of its true possession)^-0.25, because spellF1 counts
  possessions, not frames.
- **How it is used.** It runs on frames with the ball seen (weight 1, clipped to ±4 log-odds per
  frame) and replaces the nearest-player evidence.
- **Accuracy.** On a held-out game it gets the team right on ~93 % of live frames with the ball
  seen, against 73–75 % for "team of the nearest player".
- **Implementation.** Plain numpy; the weights are JSON files in `models/`. Retraining
  (`python validation/train_control.py`) gives byte-identical files. `run.py` only reads them, so it
  needs no training and no Metrica data.

### 2.4 How we decided without the public window (`validation/`)

- **Bench.** The three Metrica matches, with ground truth derived from their event data (§4).
  - The tracking is degraded to resemble ours: smooth error on the ball, extra ball gaps in live
    play, ball lost earlier before stoppages, and player tracks cut into fragments with identity
    switches.
  - The degradation was calibrated only on **aggregate, label-free statistics** of our tracking.
  - Sanity check: with it, the four official baselines come out in the same order as their
    published full-match scores on our match.
- **Conditions.** Four of them: `light`, `degraded`, `degraded` with synthetic SET PIECES tags,
  and `degraded` with a coarser ground truth (possessions < 2 s between two of the rival
  absorbed). The last one exists because we cannot know how finely the real ground truth cuts.
- **No leakage.** Each bench game is labelled with a model trained on the other two games only.
- **Decision rule.** A change is kept only if its mean spellF1 improves under every condition and
  in most of the 12 (condition, game) cells, without losing macroF1. Then it must pass two
  label-free checks on our match:
  - `check_control.py`: the model's inputs and outputs have the same scale as on the bench;
  - `check_bar.py`: number of possessions, number of stoppages, and the share of Smart Tagging set
    pieces that get a predicted stoppage must stay plausible.
- **Public 5 minutes.** Only a smoke test after a decision is committed (`run.py` prints it).
  Once, an exploratory run printed it for a candidate. That candidate was rejected on the
  label-free checks (§3, learnt dead-ball model). Since then, exploratory runs on our match use
  `check_bar.py`, and regenerating the file uses `run.py --no-public`. Neither loads the public
  ground truth.
- **Experiment record.** Each experiment is developed on its own branch, `exp/<idea>`.
  - Once decided, it is kept as a tag, `aceptado/<group>/<idea>` or `descartado/<group>/<idea>`.
  - Each rejected tag reproduces its variant.
  - Accepted experiments are merged into `josep/mejoras`.
  - The map is in EXPERIMENTS.md.
- **Use of an AI assistant.** Part of the code was written with an AI coding assistant. In line
  with the competition terms, it never received match data, video or extracts. It worked on code
  and on aggregate metrics printed by scripts run locally. This is enforced, not only instructed:
  a local hook runs before every action of the assistant and blocks reading the match data, its
  caches, the public ground truth and the submission file.
- **Rules gate.** Before any merge, `validation/check_rules.py` checks the change: no data-like
  files in the repository, no new dependencies, no integer constant in the code that could be a
  frame of the match, and models trained only on approved games.

## 3. Assets: used and dropped

| asset | use |
|---|---|
| ATD tracking (players, ball) | main input |
| Smart Tagging | SET PIECES tags: restarting team, reason for a stoppage, goals (`Kick Off (After Goal)`). Attacking phases: weak team evidence where the ball is missing. Also used in the label-free checks |
| Homography, tactical video | only for review videos (`video.py`, `create_video_gt.py`), never for labels |
| Broadcast video | not used (not synchronised) |
| Public 5-minute ground truth | smoke test only |
| Official validator, scorer, baselines | validation of the file; scorer and baselines on the bench |
| Metrica Sports sample data (external) | training and validation of every learnt component (§4) |

Tried and dropped, with the bench numbers in EXPERIMENTS.md:

- **Nearest-player contact evidence.** Replaced by the learnt model. Keeping both did not help.
- **A shorter contact distance (1 m).** It won only on clean tracking, where the ball is glued to
  the feet, and lost on degraded tracking.
- **Ignoring proximity while the ball flies.**
- **Other feature groups** (longer windows, touch sequences, half-second windows): no gain over
  the movement features.
- **The network alone.** It lost on the most different bench game.
- **Stronger possession weighting and cheaper hand-overs.** They win only with the fine ground
  truth.
- **A dead-ball model learnt on the Metrica games.** It gave +0.017 on the bench but did not
  transfer to our match: fewer stoppages (104 → 93) and fewer Smart Tagging set pieces with a
  stoppage (96.6 → 91.4 %). The code is kept, switched off (`context.dead_model`).

## 4. Training data

Only data of **other matches** is used to train or validate. The provided match is used only
self-supervised, with rule-made pseudo-labels (§2.2).

- **Source.** Metrica Sports sample data, <https://github.com/metrica-sports/sample-data> (commit
  `e706dd5`, 2021). It contains three anonymised professional matches with synchronised tracking
  (25 fps, 105 × 68 m) and event data. The repository has no licence file. Its README asks for
  responsible use and acknowledgement of the source, which we give here. The data is cloned beside
  our repository and is not in it.
- **Who labelled it.** The event data was produced by Metrica Sports. We treat it as human labels
  of other games, which RULES.md allows. **We labelled nothing ourselves.**
- **How much.**

  | match | minutes | events | possessions | live frames |
  |---|--:|--:|--:|--:|
  | Sample_Game_1 | 94.7 | 1,745 | 385 | 88,122 |
  | Sample_Game_2 | 93.4 | 1,935 | 361 | 87,082 |
  | Sample_Game_3 | 95.1 | 3,620 | 407 | 101,717 |

- **From events to per-frame labels** (`validation/gt_events.py`, mechanical rules following
  CHALLENGE.md):
  - Dead from the end of the last action before a restart (set piece or goal kick) to the restart.
  - While alive, the team of the last *controlled* action: pass, carry, recovery, shot, restart,
    or ball lost.
  - Duels, fouls received and balls out do not change possession.
  - The change lands on the first frame of the new team's action.

  This approximates the official ground truth, which comes from corrected tracking. Its
  granularity is why the bench also has the coarse condition.

## 5. Possession rules we implemented

- **Loose ball.** Possession stays with the last team until another team controls the ball. When
  nobody is close, there is no evidence, and the decoder keeps the state.
- **Challenges and deflections.**
  - A touch with a rival within 2 m is contested, not a controlled touch.
  - A rival touch shorter than 4 frames between two team-mates is a deflection and does not break
    their pass.
  - The learnt model takes its notion of control from event data, where duels do not change
    possession.
- **Kicks and clearances.** The kicker's team gets 10 log-odds, less than the cost of a hand-over
  (16). A lone clearance therefore does not flip possession, while a restart goes to the team that
  takes it.
- **Restarts.** Dead from the ball going out (seen out, lost at a line, or about to cross it) until
  the next kick, for at most 60 s. Restarting is cheap only when someone kicks the ball. A SET
  PIECES tag with a team favours that team.
- **Stoppages.** Every stoppage needs a reason besides slow players: ball out, ball lost at a line,
  restart at the centre spot, a SET PIECES tag, or ≥ 30 s (injuries, goals). It lasts at least
  1 s. An abandoned slow ball (nobody within 6 m for 2 s) is evidence of a stoppage.
- **Goals.** Dead from the last ball near the goal to the `Kick Off (After Goal)` tag.
- **Abstention.** We never answer X on graded frames: X costs macroF1, and spellF1 would only
  repeat the last label. Ungraded frames are X.

## 6. Results

| version | bench mean spellF1 | bench macroF1 |
|---|--:|--:|
| official `baseline_smooth50` | 0.319 | 0.769 |
| rule-based context labeller + Viterbi (first team version) | 0.455 | 0.812 |
| + cheaper hand-over (`team_switch` 20 → 16) | 0.463 | 0.812 |
| + learnt control model (linear) | 0.583 | 0.847 |
| + movement features | 0.594 | 0.849 |
| + averaged with the antisymmetric network | 0.605 | 0.851 |
| + possession-length weighting (**current**) | **0.613** | **0.852** |

Each accepted step improved all 4 conditions and 10 to 12 of the 12 cells. The last three gained
0.008–0.012 each, with paired standard errors of 0.0013–0.0015. On a single full match, spellF1
noise is about ±0.04, which is why decisions use paired comparisons over 48 bench runs.

On our match, label-free aggregates of the current version:

- 270 possessions with a mean of 14.0 s;
- 40.6 % of graded frames dead, in 104 stoppages;
- 96.6 % of Smart Tagging set pieces have a predicted stoppage.

On the public 5 minutes (smoke test only, 17 possessions): spellF1 0.62, macroF1 0.85.

Submission file: H 55,204 / A 39,221 / D 64,432 / X 40,462 frames. SHA256
`00bf726cd05e4387a4b1a118d51ada585db5c432ef8cb908fd4201c6e19636c7`.

## 7. Limitations and what we would do next

- **Short possessions are still missed.** On the bench, possessions under 2 s are about a quarter
  of the true ones and the evidence barely sees them. The strongest features are averages over a
  few seconds. Errors in the team, not in the stoppages, are where most of the score is lost.
- **The training and validation data are small and come from one provider.** Three anonymised
  matches, a ground truth approximated from events, and a degradation that imitates our tracking
  only in aggregate. Stoppages in particular are not imitated well, which is why the learnt
  dead-ball model did not transfer.
- **Some rule constants were not validated individually.** Many dead-ball and pass constants were
  set by reasoning about football and inspecting data, without hand labels. Only some were swept
  on the bench.
- **The granularity of the real ground truth is unknown.** Settings that cut possessions finer win
  with a fine ground truth and lose with a coarse one, so we only keep what wins under both.
- **Next steps:**
  - more open matches with tracking and events, to train on and to widen the bench;
  - a sequence model over touch sequences;
  - a dead-ball model trained on degradations that reproduce our stoppages;
  - Smart Tagging phases as a prior on the team.

## 8. Reproduction

```bash
# beside this repository: the official repo ../more-than-a-hack-2027 and the match data in ../data
# (or MTAH_DATA_DIR)
python -m venv .venv
.venv\Scripts\activate            # Windows; source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt   # Python 3.12, pinned versions
python run.py                     # -> outputs/submissions/DataScienceClass21.csv, validated
```

`run.py` is deterministic and needs no training: the models are in `models/`. The first run takes
1–2 minutes and caches the tracking. To retrain the models or run the bench, clone the Metrica data
to `../external/metrica-sample-data`, then:

- `python validation/train_control.py` retrains the models (byte-identical files);
- `python validation/bench.py --suite` runs the bench.
