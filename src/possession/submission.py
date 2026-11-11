"""Write the CSV, run the official validator and compare against the official baselines."""

import glob
import os
import shutil
import subprocess
import sys

import numpy as np


def mark_ungraded(labels, game):
    """X on frames that are not graded."""
    import games

    graded = np.zeros(len(labels), dtype=bool)
    for _, first, last in games.half_video_blocks(game):
        graded[first:last + 1] = True
    labels = labels.copy()
    labels[~graded] = "X"
    return labels


def write_csv(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    return path


def validate(path, official_repo):
    """Official format check. Silent when it passes; prints the validator's output if not."""
    script = os.path.join(official_repo, "evaluation", "validate_submission.py")
    run = subprocess.run([sys.executable, script, path], cwd=official_repo,
                         capture_output=True, text=True)
    if run.returncode != 0:
        print(run.stdout, run.stderr)
    return run.returncode == 0


def compare(submissions_dir, scores_dir, official_repo, bootstrap=0):
    """Score our submission(s) and the official baselines on the public 5 minutes.

    Prints only the ranked table. The public window is 17 possessions: it checks that the
    pipeline works and how far apart models are, it must not be used to pick parameters
    (RULES.md, EVALUATION.md).
    """
    work = os.path.join(scores_dir, "compare")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    baselines = glob.glob(os.path.join(official_repo, "evaluation", "results", "baseline_*.csv"))
    for path in baselines + glob.glob(os.path.join(submissions_dir, "*.csv")):
        shutil.copy(path, work)

    script = os.path.join(official_repo, "evaluation", "run_evaluation.py")
    cmd = [sys.executable, script, "--results", work, "--out", scores_dir, "--no-detail"]
    if bootstrap:
        cmd += ["--bootstrap", str(bootstrap)]
    run = subprocess.run(cmd, cwd=official_repo, capture_output=True, text=True)
    if run.returncode != 0:
        print(run.stdout, run.stderr)
        return
    lines = run.stdout.splitlines()
    table = [l for l in lines if l.startswith("rank") or l.startswith("----")
             or l[:4].strip().isdigit()]
    print("\npublic 5 min (only a check, not for tuning):")
    print("\n".join(table))
