"""Reproducibility entry point for the Portuguese wildfire catastrophe loss model.

Runs each phase notebook top to bottom on a fresh kernel via nbconvert, in
order, and reports what happened. Every notebook is always attempted:
whether a phase is "done" is a call for docs/worklog.md and the README's
Status section, not something main.py can safely infer from source text.
(An earlier version pre-scanned each notebook for the literal string
"NotImplementedError" and skipped the whole notebook if it found one
anywhere - including inside a stub function that a later cell never calls.
That silently skipped Phase 2's completed frequency-model fit, since the
still-unimplemented severity functions sit lower in the same notebook. Do
not reintroduce that check.)

A notebook whose "Run" section is still commented out (a not-yet-started
phase) executes cleanly and produces no new output - that is a correct,
non-failing result, not a bug. A notebook that actually calls a function
still raising NotImplementedError fails with nbconvert's traceback
pointing at exactly what's missing.

Per the PRD's reproducibility criterion: `pip install -r requirements.txt`
then `python main.py` on a fresh clone should produce the same results,
because notebook cell outputs are pinned in git and every stochastic step
(the Monte Carlo simulation, once implemented) uses a fixed random seed.

Usage
-----
    python main.py            # run every phase notebook, in order
    python main.py --phase 1  # run one phase only (1-4; Phase 5 is the technical
                               # write-up, not automated - see PHASES below)
"""

import argparse
import subprocess
import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent / "notebooks"

PHASES = [
    (1, "01_eda.ipynb", "Data & EDA"),
    (2, "02_distribution_fitting.ipynb", "Distribution Fitting"),
    (3, "03_monte_carlo.ipynb", "Monte Carlo Simulation"),
    (4, "04_validation.ipynb", "Validation & Sensitivity"),
]


def run_notebook(notebook_path: Path) -> bool:
    """Execute a notebook in place via nbconvert. Returns True on success."""
    result = subprocess.run(
        [
            sys.executable, "-m", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            "--ExecutePreprocessor.timeout=600",
            str(notebook_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stderr[-4000:], file=sys.stderr)
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", type=int, choices=[p for p, _, _ in PHASES], help="run a single phase (1-4)")
    args = parser.parse_args()

    phases = [p for p in PHASES if p[0] == args.phase] if args.phase else PHASES

    print(f"Portuguese wildfire catastrophe loss model - running {len(phases)} phase(s)\n")
    ran, missing, failed = [], [], []
    for number, filename, title in phases:
        path = NOTEBOOKS_DIR / filename
        if not path.exists():
            print(f"Phase {number} ({title}): MISSING - {filename} not found")
            missing.append(number)
            continue
        print(f"Phase {number} ({title}): running {filename} ...")
        if run_notebook(path):
            print(f"Phase {number} ({title}): done")
            ran.append(number)
        else:
            print(f"Phase {number} ({title}): FAILED - see traceback above")
            failed.append(number)

    print(f"\nSummary: ran {ran}, missing {missing}, failed {failed}")
    print("A notebook that ran without error may still be a not-yet-started phase "
          "(its 'Run' section commented out) - see the README's Status section.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
