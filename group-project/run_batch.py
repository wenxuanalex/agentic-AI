"""Batch job: screen and score every CV against every JD, then invite the top 10% per JD.

Usage (from this folder, with OPENAI_API_KEY in .env):
    python run_batch.py                     # data folder defaults to ./data (needs data/cvs and data/jds)
    python run_batch.py /path/to/data       # a folder containing cvs/ and jds/
    python run_batch.py --plan              # planning mode for each screening run
    python run_batch.py --yes               # approve the shortlist without asking
    python run_batch.py --workers 4         # pairs screened at the same time (default 8)
    python run_batch.py --limit 3           # only the first 3 CVs (a cheap test run)

Writes (inside the data folder, which is gitignored - the files hold applicant file names):
    <data>/results/scores.csv      one row per CV x JD: label, score, six criteria, why, tokens
    <data>/results/shortlist.csv   who is invited to which JD
"""
from __future__ import annotations

import csv
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

from langgraph.types import Command

import adapter
import batch
from scoring import CRITERIA


def _arg(flag: str, default):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        return type(default)(sys.argv[i + 1])
    return default


def main() -> int:
    planning, auto_yes = "--plan" in sys.argv, "--yes" in sys.argv
    workers, limit = _arg("--workers", 8), _arg("--limit", 0)
    skip = {"--workers", "--limit"}
    pos = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] not in skip]
    data = Path(pos[0]) if pos else Path(__file__).parent / "data"
    if not (data / "cvs").is_dir() or not (data / "jds").is_dir():
        print(f"Can't find {data}/cvs and {data}/jds")
        return 1

    run_dir = data
    if limit:  # test run on the first few CVs: link them into a temporary data folder
        run_dir = Path(tempfile.mkdtemp(prefix="batch_"))
        (run_dir / "cvs").mkdir(); (run_dir / "jds").mkdir()
        for f in sorted((data / "cvs").glob("*.pdf"))[:limit]:
            (run_dir / "cvs" / f.name).symlink_to(f.resolve())
        for f in sorted((data / "jds").glob("*.pdf")):
            (run_dir / "jds" / f.name).symlink_to(f.resolve())

    print("planning:", "ON" if planning else "off (baseline)", "| workers:", workers)
    graph = batch.build_batch_graph()
    cfg = {"configurable": {"thread_id": f"batch-{uuid.uuid4().hex[:8]}"},
           "max_concurrency": workers, "recursion_limit": 50}
    t0 = time.perf_counter()
    out = graph.invoke({"data_dir": str(run_dir), "planning": planning, "results": []}, cfg)

    # ---- human approval of the shortlist (interrupt)
    while "__interrupt__" in out:
        files = {c["id"]: Path(c["file"]).name for c in out["cvs"]}
        roles = {j["id"]: Path(j["file"]).name for j in out["jds"]}
        print("\nSHORTLIST (top 10% per JD, LABEL 1 only):")
        for jd_id, ids in out["shortlist"].items():
            print(f"  {jd_id} {roles[jd_id]}")
            for cv_id in ids:
                row = next(r for r in out["results"] if r["jd"] == jd_id and r["cv"] == cv_id)
                print(f"      {cv_id} score {row['score']:>5}  {files[cv_id]}")
            if not ids:
                print("      (nobody: no LABEL 1 candidates)")
        answer = "y" if auto_yes else input("\nSend invitations and rejection emails? [y/N] ").strip().lower()
        out = graph.invoke(Command(resume="approve" if answer == "y" else "reject"), cfg)
    secs = time.perf_counter() - t0

    # ---- write results
    results_dir = data / "results"
    results_dir.mkdir(exist_ok=True)
    rank_of = {(jd, cv): i for jd, ids in out["ranking"].items() for i, cv in enumerate(ids, start=1)}
    invited = {(jd, cv) for jd, ids in out["shortlist"].items() for cv in ids}
    cols = (["jd", "jd_file", "cv", "cv_file", "label", "score", "rank", "invited"] + [k for k, _, _ in CRITERIA]
            + ["why", "error", "verifier_notes", "steps", "tokens_in", "tokens_out", "seconds"])
    rows = sorted(out["results"], key=lambda r: (r["jd"], rank_of.get((r["jd"], r["cv"]), 999), r["cv"]))
    with open(results_dir / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "rank": rank_of.get((r["jd"], r["cv"]), ""), "invited": (r["jd"], r["cv"]) in invited})
    with open(results_dir / "shortlist.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["jd", "jd_file", "rank", "cv", "cv_file", "score"])
        for r in rows:
            if (r["jd"], r["cv"]) in invited:
                w.writerow([r["jd"], r["jd_file"], rank_of[(r["jd"], r["cv"])], r["cv"], r["cv_file"], r["score"]])
    if limit:
        shutil.rmtree(run_dir, ignore_errors=True)

    # ---- summary
    tin = sum(r["tokens_in"] for r in out["results"]); tout = sum(r["tokens_out"] for r in out["results"])
    cost = (tin * adapter.PRICE_IN_PER_M + tout * adapter.PRICE_OUT_PER_M) / 1e6
    print("\n" + "=" * 78)
    print(f"{'JD':<5}{'valid CVs':>10}{'label 1':>9}{'label 0':>9}{'invited':>9}   top scores")
    for jd in out["jds"]:
        rs = [r for r in out["results"] if r["jd"] == jd["id"]]
        ones = [r for r in rs if r["label"] == "1"]
        top = ", ".join(f"{cv}:{next(r['score'] for r in ones if r['cv'] == cv)}" for cv in out["ranking"][jd["id"]][:3])
        print(f"{jd['id']:<5}{sum(r['label'] in ('0', '1') for r in rs):>10}{len(ones):>9}"
              f"{sum(r['label'] == '0' for r in rs):>9}{len(out['shortlist'][jd['id']]):>9}   {top}")
    skipped = [c["id"] for c in out["cvs"] if c["status"] != "ok"]
    print(f"\nCV files skipped: {len(skipped)} ({', '.join(skipped) or '-'})")
    print(f"emails: {'sent (mock): ' + str(len(out.get('emails', []))) if out.get('approved') else 'NOT sent (not approved)'}")
    print(f"pairs: {len(out['results'])} | tokens {tin} in / {tout} out | cost ~${cost:.2f} "
          f"(screening + scoring, parser not included) | {secs / 60:.1f} min")
    print(f"written: {results_dir / 'scores.csv'}, {results_dir / 'shortlist.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
