"""Run the screening agent on ONE resume and ONE job description, and print the result.

Usage (from this folder, with OPENAI_API_KEY in .env):
    python run_one.py <resume.pdf> <jd.pdf> [candidate_id]

The recruiter-email approval pause is auto-approved here (same as the eval adapter).
"""
import json
import sys

from adapter import run_case


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 1
    task = json.dumps({"resume": argv[1], "jd": argv[2],
                       "candidate_id": int(argv[3]) if len(argv) > 3 else 1})
    r = run_case(task)

    print("=" * 70)
    print("OUTPUT (agent's final answer)")
    print("=" * 70)
    print(r.output or "(empty)")
    print("=" * 70)
    print("tool_calls :", r.tool_calls)
    print("steps      :", r.steps)
    print("tokens     :", r.input_tokens, "in /", r.output_tokens, "out")
    print(f"cost       : ${r.cost_usd:.4f}  (check the price settings in adapter.py)")
    print(f"latency    : {r.latency_s:.1f}s")
    print("error      :", r.error)
    return 0 if not r.error else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
