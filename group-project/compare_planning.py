"""Baseline comparison: run the same cases WITHOUT and WITH planning, and compare tokens, cost,
model turns, time and decisions side by side.

Usage (from this folder, with OPENAI_API_KEY in .env):
    python compare_planning.py                 # data folder defaults to ~/Downloads/data (1)/data
    python compare_planning.py /path/to/data --runs 3

Each case takes about 1-2 minutes per mode. Results also go to planning_comparison.csv.
"""
import csv
import json
import re
import statistics
import sys
from pathlib import Path

from adapter import make_agent_fn

args = [a for a in sys.argv[1:]]
RUNS = int(args[args.index("--runs") + 1]) if "--runs" in args else 1
if "--runs" in args:
    i = args.index("--runs"); del args[i:i + 2]
DATA = Path(args[0]) if args else Path.home() / "Downloads" / "data (1)" / "data"

# (name, resume, jd) - edit freely
CASES = [
    ("HR cv vs Data Scientist JD", "cv/kaggle_1.pdf", "jd/03. JD_Data Scientist.pdf"),
    ("JD given as resume (invalid)", "jd/03. JD_Data Scientist.pdf", "jd/03. JD_M_SM_AD (Data Strategy).pdf"),
]


def label_of(output: str) -> str:
    found = re.findall(r"LABEL: (.*)", output)
    return found[-1].strip() if found else "(none)"


def main() -> int:
    if not (DATA / "cv").is_dir():
        print(f"Can't find {DATA}/cv - pass the data folder as an argument.")
        return 1
    fns = {"baseline": make_agent_fn(planning=False), "planning": make_agent_fn(planning=True)}
    rows = []
    for name, cv, jd in CASES:
        task = json.dumps({"resume": str(DATA / cv), "jd": str(DATA / jd), "candidate_id": 1})
        for mode, fn in fns.items():
            for run in range(1, RUNS + 1):
                print(f"{name} | {mode} | run {run}/{RUNS} ...", flush=True)
                r = fn(task)
                rows.append({"case": name, "mode": mode, "run": run, "label": label_of(r.output),
                             "tokens_in": r.input_tokens, "tokens_out": r.output_tokens,
                             "tokens_total": r.input_tokens + r.output_tokens, "model_turns": r.steps,
                             "cost_usd": round(r.cost_usd, 5), "seconds": round(r.latency_s, 1),
                             "tools": " ".join(r.tool_calls), "error": r.error or ""})

    with open("planning_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 104)
    print(f"{'case':<30}{'mode':<10}{'labels':<14}{'tokens (mean)':<15}{'turns':<7}{'cost (mean)':<13}{'secs (mean)':<12}")
    print("=" * 104)
    summary = {}
    for name, _, _ in CASES:
        for mode in fns:
            rs = [r for r in rows if r["case"] == name and r["mode"] == mode]
            tok = statistics.fmean(r["tokens_total"] for r in rs)
            cost = statistics.fmean(r["cost_usd"] for r in rs)
            summary[(name, mode)] = tok
            print(f"{name:<30}{mode:<10}{','.join(r['label'] for r in rs):<14}{tok:<15.0f}"
                  f"{statistics.fmean(r['model_turns'] for r in rs):<7.1f}${cost:<12.4f}"
                  f"{statistics.fmean(r['seconds'] for r in rs):<12.1f}")
        b, p = summary[(name, "baseline")], summary[(name, "planning")]
        if b:
            print(f"{'':<30}{'':<10}token change with planning: {100 * (p - b) / b:+.0f}%")
    print("\nPer-run details: planning_comparison.csv. Same labels in both modes = planning saved tokens "
          "without changing decisions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
