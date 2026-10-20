"""Batch job: screen and score every CV against every JD, then invite the top 20% per JD.

Usage (from this folder, with OPENAI_API_KEY in .env):
    python run_batch.py                     # data folder defaults to ./data (needs data/cvs and data/jds)
    python run_batch.py /path/to/data       # a folder containing cvs/ and jds/
    python run_batch.py --plan              # planning mode for each screening run
    python run_batch.py --yes               # approve the shortlist without asking
    python run_batch.py --workers 4         # pairs screened at the same time (default 8)
    python run_batch.py --limit 3           # only the first 3 CVs (C01-C03)
    python run_batch.py --cvs C07,C12 --jds J1,J4   # only these CVs x these JDs
    Before spending anything it shows the number of pairs and an estimated cost, and asks to go
    ahead (--yes skips the question). Planning mode (--plan) uses fewer tokens.

Writes (inside the data folder, which is gitignored - the files hold applicant file names):
    <data>/results/scores.csv      one row per CV x JD: label, score, six criteria, why, tokens
    <data>/results/shortlist.csv   who is invited to which JD
"""
from __future__ import annotations

import csv
import sys
import time
import uuid
from pathlib import Path

from langgraph.types import Command

import adapter
import batch
from scoring import CRITERIA

# From the full run on 20 Oct (75 pairs, $1.68 -> ~$0.022/pair); planning used ~18% fewer tokens.
EST_COST_PER_PAIR = 0.023
EST_COST_PER_PAIR_PLAN = 0.019


def _arg(flag: str, default):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        return type(default)(sys.argv[i + 1])
    return default


def main() -> int:
    planning, auto_yes = "--plan" in sys.argv, "--yes" in sys.argv
    workers, limit = _arg("--workers", 8), _arg("--limit", 0)
    cv_ids = [x.strip().upper() for x in _arg("--cvs", "").split(",") if x.strip()]
    jd_ids = [x.strip().upper() for x in _arg("--jds", "").split(",") if x.strip()]
    skip = {"--workers", "--limit", "--cvs", "--jds"}
    pos = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] not in skip]
    data = Path(pos[0]) if pos else Path(__file__).parent / "data"
    if not (data / "cvs").is_dir() or not (data / "jds").is_dir():
        print(f"Can't find {data}/cvs and {data}/jds")
        return 1
    if limit and not cv_ids:
        cv_ids = [f"C{i:02d}" for i in range(1, limit + 1)]

    # ---- budget check before any model call
    n_cv = len(cv_ids) if cv_ids else len(list((data / "cvs").glob("*.pdf")))
    n_jd = len(jd_ids) if jd_ids else len(list((data / "jds").glob("*.pdf")))
    per_pair = EST_COST_PER_PAIR_PLAN if planning else EST_COST_PER_PAIR
    print(f"{n_cv} CVs x {n_jd} JDs = {n_cv * n_jd} pairs; estimated cost ~${n_cv * n_jd * per_pair:.2f} "
          f"(+ parsing {n_cv + n_jd} files)")
    if not auto_yes and input("Go ahead? [y/N] ").strip().lower() != "y":
        print("Stopped - nothing spent.")
        return 0
    run_dir = data

    print("planning:", "ON" if planning else "off (baseline)", "| workers:", workers)
    graph = batch.build_batch_graph()
    cfg = {"configurable": {"thread_id": f"batch-{uuid.uuid4().hex[:8]}"},
           "max_concurrency": workers, "recursion_limit": 50}
    t0 = time.perf_counter()
    out = graph.invoke({"data_dir": str(run_dir), "planning": planning, "results": [],
                        "cv_ids": cv_ids, "jd_ids": jd_ids}, cfg)

    # ---- human approval of the shortlist (interrupt)
    while "__interrupt__" in out:
        files = {c["id"]: Path(c["file"]).name for c in out["cvs"]}
        roles = {j["id"]: Path(j["file"]).name for j in out["jds"]}
        print(f"\nSHORTLIST (top {batch.INVITE_SHARE:.0%} of valid applicants per JD, by score):")
        for jd_id, ids in out["shortlist"].items():
            print(f"  {jd_id} {roles[jd_id]}")
            for cv_id in ids:
                row = next(r for r in out["results"] if r["jd"] == jd_id and r["cv"] == cv_id)
                print(f"      {cv_id} score {row['score']:>5}  {files[cv_id]}")
            if not ids:
                print("      (nobody: no valid scored candidates)")
        answer = "y" if auto_yes else input("\nSend invitations and rejection emails? [y/N] ").strip().lower()
        out = graph.invoke(Command(resume="approve" if answer == "y" else "reject"), cfg)
    secs = time.perf_counter() - t0

    # ---- write results
    results_dir = data / "results"
    results_dir.mkdir(exist_ok=True)
    rank_of = {(jd, cv): i for jd, ids in out["ranking"].items() for i, cv in enumerate(ids, start=1)}
    invited = {(jd, cv) for jd, ids in out["shortlist"].items() for cv in ids}
    cols = (["jd", "jd_file", "cv", "cv_file", "final_label", "score", "rank", "agent_label"] + [k for k, _, _ in CRITERIA]
            + ["why", "error", "verifier_notes", "answers", "steps", "tokens_in", "tokens_out", "seconds"])
    rows = sorted(out["results"], key=lambda r: (r["jd"], rank_of.get((r["jd"], r["cv"]), 999), r["cv"]))
    with open(results_dir / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            final = "invalid" if r["label"] not in ("0", "1") else ("1" if (r["jd"], r["cv"]) in invited else "0")
            w.writerow({**r, "rank": rank_of.get((r["jd"], r["cv"]), ""), "final_label": final, "agent_label": r["label"]})
    with open(results_dir / "shortlist.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["jd", "jd_file", "rank", "cv", "cv_file", "score"])
        for r in rows:
            if (r["jd"], r["cv"]) in invited:
                w.writerow([r["jd"], r["jd_file"], rank_of[(r["jd"], r["cv"])], r["cv"], r["cv_file"], r["score"]])

    # ---- summary
    tin = sum(r["tokens_in"] for r in out["results"]); tout = sum(r["tokens_out"] for r in out["results"])
    cost = (tin * adapter.PRICE_IN_PER_M + tout * adapter.PRICE_OUT_PER_M) / 1e6
    print("\n" + "=" * 78)
    print(f"{'JD':<5}{'valid CVs':>10}{'invited (1)':>13}{'not (0)':>9}{'invalid':>9}   invited (score)")
    for jd in out["jds"]:
        rs = [r for r in out["results"] if r["jd"] == jd["id"]]
        valid = sum(r["label"] in ("0", "1") for r in rs)
        inv = out["shortlist"][jd["id"]]
        top = ", ".join(f"{cv}:{next(r['score'] for r in rs if r['cv'] == cv)}" for cv in inv)
        print(f"{jd['id']:<5}{valid:>10}{len(inv):>13}{valid - len(inv):>9}{len(rs) - valid:>9}   {top}")
    skipped = [c["id"] for c in out["cvs"] if c["status"] != "ok"]
    print(f"\nCV files skipped: {len(skipped)} ({', '.join(skipped) or '-'})")
    print(f"emails: {'sent (mock): ' + str(len(out.get('emails', []))) if out.get('approved') else 'NOT sent (not approved)'}")
    print(f"pairs: {len(out['results'])} | tokens {tin} in / {tout} out | cost ~${cost:.2f} "
          f"(screening + scoring, parser not included) | {secs / 60:.1f} min")
    print(f"written: {results_dir / 'scores.csv'}, {results_dir / 'shortlist.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
