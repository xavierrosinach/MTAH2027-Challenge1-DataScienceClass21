# Ball possession per frame — Data Science Class 21

More than a Hack 2027, Phase 1. Technical document of the submission `DataScienceClass21.csv`.

> Living document: it describes the version on branch `josep/mejoras` (control model v6 trained on
> 10 matches, dead-ball weight 0.3, gap evidence 0.1, kick from the line, gap stoppages) and is updated with every accepted change. The full experiment
> log is [EXPERIMENTS.md](../EXPERIMENTS.md).

## 1. Summary

Every frame gets one of three states, H, A or D. For each frame we add up evidence for each state:

- dead-ball rules and a model of the players' speed;
- **a learnt model of which team controls the ball, trained on ten public matches of other
  teams** from two open datasets:
  - Metrica Sports sample data (3 matches), whose event data were collected by people;
  - IDSSE (7 Bundesliga matches, CC BY 4.0), whose ball status and possession come from the
    tracking, as in the official ground truth;
- completed passes, kicks and restarts.

A Viterbi decoder then picks the most likely sequence over each half, with a cost for every change
of state. No label of the provided match was made or used by a person.

Every design decision was taken on **validation benches built from those other matches**, never
on the public 5 minutes. On the Metrica bench (3 matches × 4 tracking-degradation seeds × 4
conditions), mean spellF1 went from 0.319 (official `baseline_smooth50`), and from 0.455 for our
first rule-based version, to 0.619. The 10-match model then added +0.013 on new seeds (0.629,
macroF1 0.856), and +0.019 over all 10 matches. Its gap model, as weak evidence where the ball is
missing, added +0.001 to +0.002 more (**0.636** on seeds 12-15, macroF1 0.857).

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
   - A logistic regression on the players' speed, weighted 0.3 per frame. It is fitted on the provided match itself with
     **rule-made pseudo-labels** (ball seen > 1 m out of the pitch or the first 3 s of a SET PIECES
     tag = dead; ball moving at a player's feet well inside the pitch = alive). No person decided
     any of these labels.
   - Ball rules: the ball seen out; lost while crossing, or about to cross, a line; or abandoned
     (nobody within 6 m for 2 s).
   - After a ball out, dead until the next kick. A kick from a ball on the line that comes into
     the pitch (corners, throw-ins) also counts as the restart.
   - A ball gap of at least 3 s after which the ball reappears within 10 m of where it went
     missing is a stoppage (a free kick or offside is taken from the spot).
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
   - The learnt control model's part for frames without the ball (2.3), weighted 0.1 per frame.
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
- **Training on other matches' tracking.** The model learns from every live frame of the 3 Metrica
  and 7 IDSSE matches (§4). Their tracking is first degraded to look like ours (2.4), with seeds
  the bench does not use. Each frame is weighted by (length of its true possession)^-0.25, because
  spellF1 counts possessions, not frames.
- **Attack direction.** Some features are measured towards each team's attacking goal. Which goal
  a team attacks in each half is read relatively: the team whose players stand nearer x = 0 on
  average over the half defends x = 0. The first version used "home mean x < half-way", which
  failed in one IDSSE half where the home team spent the half in the rival's half. On our match
  and on Metrica both readings agree in every half (`validation/check_direction.py`), so the
  submission did not change.
- **How it is used.** On frames with the ball seen it replaces the nearest-player evidence
  (weight 1, clipped to ±4 log-odds per frame). A second conditional logit, trained the same
  way on frames without the ball, adds weak evidence in ball gaps (weight 0.1).
- **Accuracy.** On a held-out game it gets the team right on ~93 % of live frames with the ball
  seen, against 73–75 % for "team of the nearest player".
- **Implementation.** Plain numpy; the weights are JSON files in `models/`. Retraining
  (`python validation/train_control.py --bench validation/bench_all.yaml`) gives byte-identical
  files. `run.py` only reads them, so it needs no training and no external data.

### 2.4 How we decided without the public window (`validation/`)

- **Benches.**
  - The three Metrica matches, with ground truth derived from their event data (§4).
    - The tracking is degraded to resemble ours: smooth error on the ball, extra ball gaps in
      live play, ball lost earlier before stoppages, and player tracks cut into fragments with
      identity switches.
    - The degradation was calibrated only on **aggregate, label-free statistics** of our tracking.
    - Sanity check: with it, the four official baselines come out in the same order as their
      published full-match scores on our match.
  - Since 12-nov, also the seven IDSSE matches, with the same degradation and the ground truth of
    their own tracking (§4). Their ball is almost never missing, even degraded (5.5 % of frames
    vs 41 % on ours), so this bench says little about stoppages. A change of training data must
    win on the 10 matches together **and** on Metrica alone. On Metrica alone the comparison
    favours models trained only on Metrica, and on IDSSE it favours models that saw IDSSE.
- **Conditions.** Four of them: `light`, `degraded`, `degraded` with synthetic SET PIECES tags,
  and `degraded` with a coarser ground truth (possessions < 2 s between two of the rival
  absorbed). The last one exists because we cannot know how finely the real ground truth cuts.
- **No leakage.** Each bench game is labelled with a model trained without that game
  (leave-one-game-out over the training games).
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
  - On 13-nov a team member proposed six dead-ball and tracking rules. He had found them by
    inspecting his version's failures in the public 5 minutes, to understand their causes;
    CHALLENGE.md presents that window as the reference for what is expected.
  - We treated those rules as unproven ideas. Each was tested alone, with values chosen on the
    bench, and decided only by the bench, new seeds and the label-free checks. Nothing seen in the
    public window was used to choose a value or a version.
- **Pre-registration (from 13-nov).** Before measuring, each experiment's hypothesis, its single
  change and its acceptance rule are written and committed in EXPERIMENTS.md.
  - There are three outcomes: accepted, rejected, or doubtful. A doubtful result gets one more
    round on new seeds; if it is still doubtful, it is rejected.
  - A threshold is never changed after seeing a result. When one proved mis-specified, it was
    applied as written to that experiment, and corrected in writing only for later ones. Twice:
    the stoppage check for changes that do not touch the dead ball, and the log-loss screen for
    model changes.
  - An outside model (OpenAI Codex, given only the method and aggregate numbers, never match
    data) critiqued each decision before it was closed. Its objections are answered in
    EXPERIMENTS.md.
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
| IDSSE dataset (external, CC BY 4.0) | training of the control model and a second bench (§4) |

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
- **Averaging several networks trained with different seeds.** Within bench noise.
- **Re-tuning the pass and kick evidence for the learnt model.** No effect.
- **A dearer switch to dead ball (`to_dead` 10 → 20).** It won on the bench, but on our match it
  gave fewer stoppages (104 → 94) and fewer set pieces with a stoppage (96.6 → 94.8 %).
- **Training also on lightly degraded copies.** It cut possessions finer and lost with the coarse
  ground truth.
- **Interpolating longer ball gaps (1-2 s).** Noise.
- **The learnt dead-ball model retrained on 10 matches, with the ball hidden in dead time.** On
  our match it lost stoppages again (106 → 82).
- **Training on 10 more matches from a third provider** (SkillCorner open data, broadcast
  tracking, MIT licence). Their possession labels are coarser (2.6 per minute). The model learnt
  to cut coarser and lost 0.006 on the bench. Its data and code are kept in a tag, unused.
- **More weight on the learnt team evidence (1.5, 2).** It cut finer and lost with the coarse
  ground truth.
- **A larger network (32 hidden units) on 10 matches.** Its held-out log-loss was worse.
- **Re-weighting the team-shape evidence in ball gaps, and a network for the frames without the
  ball.** Both were noise on the bench.
- **A dead-ball model learnt on the Metrica games.** It gave +0.017 on the bench but did not
  transfer to our match: fewer stoppages (104 → 93) and fewer Smart Tagging set pieces with a
  stoppage (96.6 → 91.4 %). The code is kept, switched off (`context.dead_model`).

## 4. Training data

Only data of **other matches** is used to train or validate. The provided match is used only
self-supervised, with rule-made pseudo-labels (§2.2). There are two open datasets, both
downloaded beside our repository and never put in it. **We labelled nothing ourselves.**

### 4.1 Metrica Sports sample data

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

### 4.2 IDSSE: integrated dataset of spatiotemporal and event data in elite soccer

- **Source.** Bassek, Rein, Weber and Memmert (2025), *An integrated dataset of spatiotemporal
  and event data in elite soccer*, Scientific Data. DOI
  [10.6084/m9.figshare.28196177](https://doi.org/10.6084/m9.figshare.28196177). Licence
  **CC BY 4.0**, public download from figshare without registration. Downloaded with
  `python validation/idsse.py download` to `../external/idsse`, with a `SOURCE.txt` recording
  the source, licence and date.
- **Content.**
  - 7 complete matches of the 2022/23 season: 2 of the Bundesliga and 5 of the 2. Bundesliga.
  - Optical tracking at 25 Hz (TRACAB, by Sportec Solutions for the DFL), 105 × 68 m.
  - Official DFL event data, collected by people.
- **Who labelled what.**
  - The tracking provider's position files carry, on every ball frame, `BallStatus` (alive /
    dead) and `BallPossession` (home / away). We did not verify how the provider produces them.
  - We use them as the ground truth: D where the ball is dead, otherwise the team in
    possession. This is the construction the challenge describes for its own ground truth.
  - The event data is used only for the synthetic SET PIECES tags of the `degraded+tags`
    condition.
- **How much** (`python validation/idsse.py summary`, graded frames):

  | match | minutes | live frames | dead % | possessions | < 2 s % |
  |---|--:|--:|--:|--:|--:|
  | J03WMX | 97.3 | 85,369 | 41.5 | 511 | 37.4 |
  | J03WN1 | 94.4 | 68,377 | 51.7 | 500 | 34.2 |
  | J03WOH | 91.5 | 71,945 | 47.6 | 402 | 27.4 |
  | J03WOY | 95.0 | 76,286 | 46.5 | 456 | 28.1 |
  | J03WPY | 97.5 | 81,460 | 44.3 | 421 | 29.5 |
  | J03WQQ | 94.9 | 84,352 | 40.7 | 401 | 25.2 |
  | J03WR9 | 97.9 | 85,316 | 41.9 | 370 | 25.4 |

  In total 668 minutes, 553,105 live frames and 3,061 possessions: 4.6 per minute, mean 7.3 s.
  The Metrica event-derived ground truth has 4.1 per minute, mean 9.6 s, and 35 % dead (IDSSE 45 %).
  The IDSSE ground truth is somewhat finer and has more dead ball.

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
| + possession-length weighting | 0.613 | 0.852 |
| + less weight on the speed model of dead ball (0.5 → 0.3) | 0.619 | 0.855 |
| + control model trained on 10 matches, Metrica + IDSSE¹ | 0.629 | 0.856 |
| + its gap model as weak evidence where the ball is missing² | 0.636 | 0.857 |
| + a kick from the line restarts play³ | +0.002 / +0.003 | +0.001 |
| + a long ball gap that ends near where the ball was lost is a stoppage (**current**)⁴ | **+0.027 / +0.029** | **+0.021 / +0.022** |

¹ Measured on new degradation seeds (4-7), where the previous version scores 0.616 / 0.854: a
paired gain of +0.013 (standard error 0.002). Over the 10 matches (Metrica + IDSSE), +0.019
(standard error 0.001), 39 of 40 (condition, match) cells better.

² Confirmed on seeds 8-11 (+0.0011, standard error 0.0005) and 12-15 (+0.0019, standard error
0.0004; the row shows these seeds, where the previous version scores 0.634). A weight of 0.3 had
gained more (+0.005) but moved one stoppage on our match, and was rejected by the rule written
before measuring.

³ Paired gain over the previous version: +0.0023 on seeds 0-3 (0.6322 -> 0.6345, standard error
0.0002) and +0.0032 on seeds 4-7 (standard error 0.0004), 12 of 12 cells in both rounds.

⁴ A ball gap of at least 3 s after which the ball reappears within 10 m of where it was lost is
dead-ball evidence. A free kick or offside is taken from the spot; in live play the ball
reappears far away. Confirmed on seeds 8-11 (+0.0272, standard error 0.0021) and 12-15 (+0.0285,
standard error 0.0020), 12 of 12 cells in both. On the IDSSE bench with the ball hidden in dead
time it gave +0.024 in both spellF1 and macroF1, in all 7 matches. On our match it adds stoppages
(106 → 130).

Each accepted step improved all 4 conditions, and 7 to 12 of the 12 cells in every round it was
measured on. The last ones gained 0.002–0.013 each, with paired standard errors of
0.0004–0.0022. On a single full match, spellF1 noise is about ±0.04, which is why decisions use
paired comparisons over 48 bench runs.

On our match, label-free aggregates of the current version:

- 281 possessions with a mean of 12.4 s;
- 45.4 % of graded frames dead, in 130 stoppages (the real matches we have show 92-113);
- 96.6 % of Smart Tagging set pieces have a predicted stoppage.

The public 5 minutes are no longer looked at, not even as a smoke test (`run.py --no-public`). The last smoke test, of control model v4, gave spellF1 0.62 and macroF1 0.85 (17 possessions).

Submission file: H 52,362 / A 34,398 / D 72,097 / X 40,462 frames. SHA256
`567f0d710f4d605f00489e474b068390edce119120b7bba29f8098b6f16f0aae`.

## 7. Limitations and what we would do next

- **Short possessions are still missed.** On the bench, possessions under 2 s are about a quarter
  of the true ones and the evidence barely sees them. The strongest features are averages over a
  few seconds. Errors in the team, not in the stoppages, are where most of the score is lost.
- **The training and validation data are small and come from two providers.**
  - Ten matches in total.
  - On Metrica, the ground truth is approximated from events.
  - The degradation imitates our tracking only in aggregate. Stoppages in particular are not
    imitated well, which is why the learnt dead-ball model did not transfer. IDSSE's ball is
    almost never missing, so its bench cannot test stoppages either.
  - Neither bench is a third, unseen provider like ours.
- **Some rule constants were not validated individually.** Many dead-ball and pass constants were
  set by reasoning about football and inspecting data, without hand labels. Only some were swept
  on the bench.
- **The granularity of the real ground truth is unknown.**
  - Settings that cut possessions finer win with a fine ground truth and lose with a coarse one,
    so we only keep what wins under both.
  - Five changes failed exactly this way: stronger possession weighting, cheaper hand-overs,
    cleaner training copies, more weight on the team evidence, and, the other way round, coarser
    training labels.
  - CHALLENGE.md asks for *controlled* possession (a few seconds, or more than two passes), which
    argues against cutting finer.
- **Learnt dead-ball models do not transfer.** Twice (3 and 10 matches) they won on the bench and
  lost real stoppages on our match. Only the speed model fitted on our own match, with rule-made
  pseudo-labels, is used.
- **Next steps:**
  - more open matches whose ground truth is built like the official one (ball status and
    possession from the tracking), at a similar granularity;
  - a degradation that reproduces our stoppages (when the ball is seen or lost in dead time), so
    that dead-ball changes can be validated off our match;
  - evidence for short possessions: on the bench, the features do not see the < 2 s ones at all.

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
1–2 minutes and caches the tracking. To retrain the models or run the benches:

- clone the Metrica data to `../external/metrica-sample-data`;
- download IDSSE with `python validation/idsse.py download`, then run
  `python validation/idsse.py parse`;
- `python validation/train_control.py --bench validation/bench_all.yaml` retrains the models
  (byte-identical files);
- `python validation/bench.py --suite` runs the Metrica bench, and `--bench
  validation/bench_dfl.yaml` the IDSSE one.
