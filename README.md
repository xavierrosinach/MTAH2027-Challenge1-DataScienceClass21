# MTAH2027-Challenge1-DataScienceClass21

This repository contains the **Data Science Class 21**'s More Than a Hack 2027, challenge 1: for every video frame of the match, which team has possession of the ball (`H` home, `A` away, `D` dead ball, `X` not graded).

`python run.py` reads the match data and writes the submission CSV.

## Repository layout

```
run.py                  entry point: match data + config.yaml -> outputs/submissions/<team_name>.csv
config.yaml             every parameter of the configuration
models/                 weights of the learnt model of ball control
src/possession/
  config.py             creates the configuration of the code using config.yaml
  data.py               reads the tactical tracking and the Smart Tagging
  broadcast.py          reads the broadcast tracking and aligns it with the tactical tracking
  jersey.py             team of each track from the shirt colour in the tactical video
  control.py            learnt model of which team controls the ball
  context.py            the labeller
  submission.py         writes the CSV and runs the official validator
```

## Requirements

- Python 3.12 and `pip install -r requirements.txt`
- The official challenge repository cloned next to this one (`../more-than-a-hack-2027`) - or change the path in `config.yaml`: its `games.py` describes the match and its validator checks the CSV.
- The match data in `../data`, or in the folder named by the `MTAH_DATA_DIR` environment variable, with the layout of the official bundle:

```
data/
├── tactical/    BAR-ATH_tactical.mp4, _atd.txt, _atd.xml, _homography.csv, _smart_tagging.xml
└── broadcast/   BAR-ATH_broadcast_atd.txt, _atd.xml
```

The team has used this folder configuration:

```
parent folder/
├── data/                                    match data
├── more-than-a-hack-2027/                   official challenge repository
└── MTAH2027-Challenge1-DataScienceClass21/  this repository
```

## Run

```bash
python -m venv .venv
.venv\Scripts\activate              # Windows  (source .venv/bin/activate on Linux / macOS)
pip install -r requirements.txt
python run.py
```

Output:

```
written outputs/submissions/DataScienceClass21.csv   H 53643  A 34286  D 70928  X 40462
SHA256 ed3105549ef72269618c63bda1741e94e01ed98bd87aef01b89c1fbbe09744f8
```

The first run takes ~15 minutes: it parses both tracking files and reads the shirt colours from the tactical video, and stores them in `cache/`. Later runs take about one minute. The CSV is then checked with the official validator - it only prints something if the check fails.

## How it works

The ball is missing on ~41 % of the graded frames, often for many seconds, so the hypothesis "which team is nearest the ball" is not enough. For each frame the labeller adds up evidence for three states (H, A, D), and Viterbi picks the most likely sequence over each half, with a cost for every change of state. Being non-causal, a change lands where the evidence changes, and a possession survives a ball gap unless something says otherwise.

**Dead ball or live?**
- A logistic model on player speeds, fitted on this match with automatic pseudo-labels (ball clearly out, start of a `SET PIECES` tag, before the 2nd-half kick-off is dead; ball moving at a player's feet mid-pitch means live).
- Ball rules: seen out of the pitch; lost while going out; abandoned; after going out it stays dead until the next kick (the restart).
- A ball gap of 3 s or more whose ball reappears within 10 m of where it was lost is supposed to be a free kick or an offside.
- Goals: dead from the last ball near the goal to the `Kick Off (After Goal)` tag.
- A restart needs play: a kick after which the ball comes to rest again with no second player touching it (a keeper placing the ball) does not restart play.
- A stoppage needs a reason (ball out, a set-piece tag, restart on a line or the centre spot, also the TV cutting to replays); slow players alone are not one.
- Stoppage edges are trimmed (start 0.3 s later, restart 0.15 s earlier): on the validation games the decoded stoppages are systematically ~0.5 s too wide.

**Which team?**
- A learnt ball-control model (`control.py`): a small antisymmetric network averaged with a linear logit, from distances to the ball, team shapes and how players move against the ball. It replaces "nearest team".
- Completed passes (a touch, then control by a team-mate at least 5 m away) and the kicker of each kick give evidence to their team; a `SET PIECES` tag with a team says who restarts.
- Goalkeeper identity swaps are fixed by the goal each keeper guards.
- Shirt colour (`jersey.py`): after two rivals overlap the tracker sometimes gives a track to the wrong team. The torso colour of every track is read from the tactical video every 0.2 s, a colour team model is fitted with the metadata team as the target, and each track gets its most likely team over time.
- Where the ball is not seen: team shape (width, depth, advance; fitted on frames where the first pass is sure) and, weakly, the Smart Tagging attacking phases.

**Broadcast tracking** (`broadcast.py`). The tactical tracking is always the reference; no player is added or moved. The broadcast recording is aligned automatically (the offset that puts its ball
on the tactical ball, constant within each half). On frames where it shows the live wide shot:
- Its ball fills the tactical ball gaps.
- The control model is also applied to what the broadcast camera sees by itself, and the two opinions are averaged where both cameras see the ball (weight 0.3 for the broadcast).
- Frames away from the live shot (replays, close-ups) give a stoppage its reason.

## Ball-control model

`models/control_mlp.json` and `models/control_lin.json` were trained offline on 10 public games of other competitions — the 3 Metrica Sports sample games and 7 games of the IDSSE dataset (Bassek et al. 2025, CC BY 4.0) — whose tracking was degraded to look like ours (fragmented player tracks, ball noise and gaps). BAR-ATH was never used for training. 
The training code is in `training/` (`python training/train_control.py`, settings in `training/training.yaml`); it is not needed to produce the submission, since `run.py` only reads the trained weights.

## Validation and result

No constant was chosen with the public 5-minute ground truth (RULES.md). Every change was decided on a validation bench of the 10 other games (degraded tracking, simulated broadcast), with paired differences per game and seed, and kept only if it did not lose in any condition.

The public 5 minutes (17 possessions) were used only to check that the pipeline is correct (frames aligned, file well-formed, labels meaning what we think), never to choose between versions or parameters.

## Data

The match data is not, and must not be, in this repository; `.gitignore` excludes it, together with `cache/` and `outputs/` (the generated CSV).