
"""Run repeated attack-suite evaluations and preserve each run's outputs.

Usage:
    python run_attack_trials.py
    python run_attack_trials.py --runs 10
    python run_attack_trials.py --runs 5 --plan
"""

import argparse
import csv
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "attack_suite"
RESULTS_DIR = ROOT / "attack_runs"


def main():
    parser = argparse.ArgumentParser(
        description="Run repeated resume-screening attack evaluations."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of trials to run (default: 5).",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Run the screening pipeline in planning mode.",
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    # Generate the attack PDFs once and reuse identical files.
    generator = ROOT / "tests" / "make_attack_suite.py"

    print("\nGenerating attack-suite PDFs...", flush=True)
    subprocess.run(
        [sys.executable, str(generator)],
        cwd=ROOT,
        check=True,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    session_dir = RESULTS_DIR / f"session_{timestamp}"
    session_dir.mkdir(exist_ok=False)

    summaries = []

    for run_number in range(1, args.runs + 1):
        print("\n" + "=" * 70, flush=True)
        print(f"ATTACK TRIAL {run_number}/{args.runs}", flush=True)
        print("=" * 70, flush=True)

        # Isolate each trial's inputs and results.
        trial_dir = session_dir / f"run_{run_number:02d}"
        trial_data = trial_dir / "data"
        trial_dir.mkdir(parents=True)
        trial_data.mkdir()

        shutil.copytree(DATA_DIR / "cvs", trial_data / "cvs")
        shutil.copytree(DATA_DIR / "jds", trial_data / "jds")

        command = [
            sys.executable,
            str(ROOT / "run_batch.py"),
            str(trial_data),
        ]

        if args.plan:
            command.append("--plan")

        print(
            "Command:",
            " ".join(f'"{x}"' if " " in x else x for x in command),
            flush=True,
        )
        print(
            "Answer prompts directly in this terminal.",
            flush=True,
        )
        print("-" * 70, flush=True)

        log_path = trial_dir / "run.log"

        # Inherit the terminal's stdin/stdout/stderr so input() prompts
        # display correctly and can receive your responses.
        #
        # Output is displayed in the terminal but is not captured in run.log.
        try:
            proc = subprocess.run(
                command,
                cwd=ROOT,
                stdin=None,
                stdout=None,
                stderr=None,
            )
            return_code = proc.returncode

        except KeyboardInterrupt:
            print(
                "\nTrial interrupted by user. Stopping the evaluation.",
                flush=True,
            )
            raise

        # Create a small log noting the trial outcome.
        # Full terminal output is not captured by this version.
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"Command: {command!r}\n")
            log.write(f"Exit code: {return_code}\n")
            log.write(
                "Note: Child-process terminal output was not captured.\n"
            )

        scores_path = trial_data / "results" / "scores.csv"
        shortlist_path = trial_data / "results" / "shortlist.csv"

        summary = {
            "run": run_number,
            "exit_code": return_code,
            "scores_csv_exists": scores_path.exists(),
            "shortlist_csv_exists": shortlist_path.exists(),
            "log": str(log_path.relative_to(ROOT)),
        }
        summaries.append(summary)

        print("\n" + "-" * 70, flush=True)
        print(f"Trial {run_number} finished with exit code {return_code}.")
        print(
            f"Scores CSV: {'saved' if scores_path.exists() else 'not found'}"
        )
        print(
            "Shortlist CSV: "
            f"{'saved' if shortlist_path.exists() else 'not found'}"
        )
        print(f"Log: {log_path}", flush=True)

        if return_code != 0:
            print(
                "WARNING: This trial failed. Check the terminal output "
                "and run.log for details.",
                flush=True,
            )

    summary_path = session_dir / "trial_summary.csv"

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "exit_code",
                "scores_csv_exists",
                "shortlist_csv_exists",
                "log",
            ],
        )
        writer.writeheader()
        writer.writerows(summaries)

    successful = sum(
        summary["exit_code"] == 0 for summary in summaries
    )

    print("\n" + "=" * 70, flush=True)
    print("ALL TRIALS FINISHED", flush=True)
    print("=" * 70, flush=True)
    print(f"Successful: {successful}/{args.runs}", flush=True)
    print(f"Session directory: {session_dir}", flush=True)
    print(f"Trial summary: {summary_path}", flush=True)

    return 0 if successful == args.runs else 1


if __name__ == "__main__":
    sys.exit(main())
