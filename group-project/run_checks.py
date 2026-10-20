"""Real-run checks: run a few resume/JD pairs through the agent and compare with what we expect.

Usage (from this folder, with OPENAI_API_KEY in .env):
    python run_checks.py                      # data folder defaults to ~/Downloads/data (1)/data
    python run_checks.py /path/to/data        # folder containing cv/ and jd/
    python run_checks.py --plan               # same checks with planning ON

Each case takes about 1-2 minutes. Recruiter-email pauses are auto-approved (as in the eval adapter).
"""
import json
import re
import sys
from pathlib import Path

from adapter import run_case

PLANNING = "--plan" in sys.argv
_args = [a for a in sys.argv[1:] if a != "--plan"]
DATA = Path(_args[0]) if _args else Path.home() / "Downloads" / "data (1)" / "data"

# (name, resume file, jd file, expected label, expected email tool or None, why)
CASES = [
    ("reject path", "cv/kaggle_1.pdf", "jd/03. JD_Data Scientist.pdf", "0", "email_applicant",
     "HR resume vs Data Scientist JD"),
    ("advance path", "cv/JR-10000053323 - Manager_Senior Manager_Assistant Director (Data Strategy)_2026-08-03-10-42-38.pdf",
     "jd/03. JD_M_SM_AD (Data Strategy).pdf", "1", "email_recruiter",
     "Data Strategy application vs the matching Data Strategy JD (likely, not certain, to advance)"),
    ("invalid input", "jd/03. JD_Data Scientist.pdf", "jd/03. JD_M_SM_AD (Data Strategy).pdf", "invalid", None,
     "a job description passed in as the resume -> expect ERROR: not_a_resume"),
]


def main() -> int:
    if not (DATA / "cv").is_dir():
        print(f"Can't find {DATA}/cv - pass the data folder as an argument.")
        return 1
    print("planning:", "ON" if PLANNING else "off (baseline)")
    rows = []
    for i, (name, cv, jd, want_label, want_email, why) in enumerate(CASES, start=1):
        print(f"\n[{i}/{len(CASES)}] {name}: {why} ...", flush=True)
        r = run_case(json.dumps({"resume": str(DATA / cv), "jd": str(DATA / jd), "candidate_id": i}), planning=PLANNING)
        labels = re.findall(r"LABEL: (.*)", r.output)
        label = labels[-1].strip() if labels else "(none)"
        errors = re.findall(r"ERROR: (\S+)", r.output)
        emails = [t for t in r.tool_calls if t.startswith("email_")]
        ok = label == want_label and emails == ([want_email] if want_email else [])
        rows.append((name, want_label, label, emails, errors, r.steps, r.latency_s, r.cost_usd, r.error, ok))
        print("    output ->", " | ".join(r.output.strip().splitlines()[-3:]) or "(empty)")

    print("\n" + "=" * 100)
    print(f"{'case':<14}{'expected':<10}{'got':<10}{'emails':<22}{'error code':<22}{'steps':<7}{'secs':<7}{'cost':<9}result")
    print("=" * 100)
    for name, want, got, emails, errs, steps, secs, cost, crash, ok in rows:
        print(f"{name:<14}{want:<10}{got:<10}{','.join(emails) or '-':<22}{','.join(errs) or '-':<22}"
              f"{steps:<7}{secs:<7.0f}${cost:<8.4f}{'OK' if ok else 'CHECK'}{'  crash: ' + crash if crash else ''}")
    print("\nsteps > 2 means the verifier sent the agent back at least once (a retry happened).")
    print("'advance path' is a judgement call by the model: CHECK there is worth reading, not necessarily a bug.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
