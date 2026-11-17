"""Write the submission CSV and check it with the official validator."""

import os
import subprocess
import sys
import numpy as np

def mark_ungraded(labels, game):
    """X on the frames that are not graded (outside the two halves)."""

    import games

    graded = np.zeros(len(labels), dtype=bool)
    for _, first, last in games.half_video_blocks(game):
        graded[first:last + 1] = True
    labels = labels.copy()
    labels[~graded] = "X"
    return labels

def write_csv(path, labels):
    """frame,label with one row per video frame."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    return path

def validate(path, official_repo):
    """Official format check (evaluation/validate_submission.py). Prints its output if it fails."""

    script = os.path.join(official_repo, "evaluation", "validate_submission.py")
    run = subprocess.run([sys.executable, script, path], cwd=official_repo,
                         capture_output=True, text=True)
    if run.returncode != 0:
        print(run.stdout, run.stderr)
    return run.returncode == 0
