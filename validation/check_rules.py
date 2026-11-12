"""Rules gate before merging a change into josep/mejoras (PROTOCOLO_AGENTES.md, section 4.7).

    python validation/check_rules.py [--base josep/mejoras]

Compares the current checkout (committed and uncommitted changes) with --base and fails (exit 1)
if any of these holds:
  1. a changed or tracked file looks like data: csv, npz, pkl, mp4, xml, txt (other than
     requirements.txt), parquet, feather, h5, zip, gz, or anything under data/, cache/, outputs/;
     or any tracked file over 5 MB;
  2. requirements.txt differs from --base (no new dependencies);
  3. an added line in src/ or config.yaml contains an integer >= 5000 that is not one of the
     official frame constants (it could be a frame of the match);
  4. a model file (models/*.json) records training games that are not approved: the Metrica
     sample games, or those listed in validation/approved_games.txt (added only with Josep's OK).
Prints only file names and line numbers, never data.
"""

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA_EXT = (".csv", ".npz", ".pkl", ".mp4", ".xml", ".txt", ".parquet", ".feather", ".h5",
            ".zip", ".gz")
DATA_DIRS = ("data/", "cache/", "outputs/")
ALLOWED_FILES = ("requirements.txt", "requirements-dev.txt", "validation/approved_games.txt")
OFFICIAL_INTS = {11185, 90651, 110808, 113742, 190197, 199318, 199319, 158857, 94214, 113488}
MAX_BYTES = 5 * 1024 * 1024


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout


def approved_games():
    pats = ["Sample_Game_*"]
    path = os.path.join(HERE, "approved_games.txt")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            pats += [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    return pats


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="josep/mejoras")
    args = parser.parse_args()
    problems = []

    changed = [f for f in git("diff", "--name-only", args.base).splitlines() if f]
    changed += [f for f in git("ls-files", "--others", "--exclude-standard").splitlines() if f]
    tracked = [f for f in git("ls-files").splitlines() if f]

    # 1. data-like files
    for f in sorted(set(changed) | set(tracked)):
        low = f.lower()
        if low in ALLOWED_FILES:
            continue
        if low.endswith(DATA_EXT) or low.startswith(DATA_DIRS):
            problems.append("data-like file in the repo: {}".format(f))
        full = os.path.join(REPO, f)
        if os.path.isfile(full) and os.path.getsize(full) > MAX_BYTES:
            problems.append("file over 5 MB: {}".format(f))

    # 2. dependencies
    if git("diff", args.base, "--", "requirements.txt").strip():
        problems.append("requirements.txt changed (R5: no new dependencies)")

    # 3. possible frame constants in added lines
    diff = git("diff", "-U0", args.base, "--", "src", "config.yaml")
    current = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            current = line[6:]
        elif line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            lineno = int(m.group(1)) if m else 0
        elif line.startswith("+") and not line.startswith("+++"):
            for num in re.findall(r"(?<![\w.])\d{4,}(?![\w.])", line):
                if int(num) >= 5000 and int(num) not in OFFICIAL_INTS:
                    problems.append("{}:{}: new integer {} (a frame of the match?) - justify or "
                                    "remove".format(current, lineno, num))
            lineno += 1

    # 4. training games of the models
    pats = approved_games()
    for f in tracked:
        if not (f.startswith("models/") and f.endswith(".json")):
            continue
        try:
            with open(os.path.join(REPO, f), encoding="utf-8") as fh:
                games = (json.load(fh).get("info") or {}).get("games") or []
        except (OSError, ValueError):
            problems.append("model file not readable: {}".format(f))
            continue
        if not games:
            problems.append("model without recorded training games: {}".format(f))
        for g in games:
            if not any(fnmatch.fnmatch(g, p) for p in pats):
                problems.append("{}: trained on a non-approved game {}".format(f, g))

    print("check_rules vs {}: {} changed files, {} tracked".format(args.base, len(set(changed)),
                                                                   len(tracked)))
    for p in problems:
        print("  FAIL " + p)
    print("RESULT: {}".format("FAIL" if problems else "PASS"))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
