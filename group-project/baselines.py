"""Two non-agent baselines to compare the screening agent against.

    python baselines.py keyword               # keyword ATS, no LLM in the decision (deterministic, run once)
    python baselines.py estimate              # one-call baseline: prompt sizes and a cost estimate, no model call
    python baselines.py onecall --runs 3      # one gpt-5-mini call per CV x JD pair, no tools / verifier / loop

keyword   For each JD, its must-have requirements come from tools.parsed_jd, parsed ONCE and reused, so every
          CV is checked against the same list (saved to jd_requirements.json). The CV text is everything
          pdfplumber extracts, hidden text included, as a naive ATS would read it. LABEL 1 if every must-have's
          key terms appear in the text (case-insensitive), else 0. Key terms: the requirement's words minus
          stop words and generic words ("experience", "years", "strong", ...) and numbers; "A/B" means A or B;
          a must-have with no key terms left (e.g. "3+ years of experience") counts as met. If the JD can't
          be parsed the label is "invalid". It never outputs "invalid" otherwise.
          --min-share 0.5 is a lenient variant: a must-have counts as met when at least half of its key-term
          groups appear (long sentence-style requirements almost never match word for word).
onecall   The CV's full text (pdfplumber, hidden text included) and the JD's text (pdfplumber) in ONE prompt,
          asking for LABEL 0 or 1 on the must-haves. The answer's last "LABEL: 0|1" line is the label
          ("none" if missing). Same model as the agent (gpt-5-mini, Chat Completions).

Inputs (all with known expectations): the single-CV cases in eval_cases.py that expect a label, the mini folder,
the 5 policy-invariance pairs, the attack suite and the 15 real CVs x Data Scientist JD.
Output: data/baselines/<timestamp>_<system>/labels.csv, one row per input x run (data/ is gitignored).
Only labels and aggregates are printed - never CV text.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE / "data"

PRICE_IN_PER_M, PRICE_OUT_PER_M = 0.25, 2.00   # gpt-5-mini, same as adapter.py
REAL_EXPECTED = {"C03": "1", "C13": "0"}         # CV_MATCH / CV_MISS in eval_cases.py
MINI_EXPECTED = {"01_all_must.pdf": "1", "02_half_must.pdf": "0", "03_none_must.pdf": "0"}


# ---------------------------------------------------------------- inputs
def build_inputs() -> list[dict]:
    """One dict per screening: group, item, cv, jd, expected ("" = none), pair/role for invariance."""
    from eval_cases import CASES
    items = []
    for c in CASES:
        if c.kind == "single" and "label" in c.expect:
            items.append({"group": "single_cases", "item": c.id, "cv": DATA / c.inputs["resume"],
                          "jd": DATA / c.inputs["jd"], "expected": c.expect["label"]})
    mini = DATA / "eval_inputs" / "mini"
    for f in sorted((mini / "cvs").glob("*.pdf")):
        items.append({"group": "mini", "item": f.stem, "cv": f, "jd": mini / "jds" / "jd.pdf",
                      "expected": MINI_EXPECTED.get(f.name, "")})
    for c in CASES:
        if c.kind == "invariance":   # the base CV is screened once per pair, like the agent's eval
            for role in ("base", "variant"):
                items.append({"group": "invariance", "item": f"{c.id}:{role}", "cv": DATA / c.inputs[role],
                              "jd": DATA / c.inputs["jd"], "expected": "", "pair": c.id, "role": role})
    attack = DATA / "attack_suite"
    jd = next((attack / "jds").glob("*.pdf"))
    for f in sorted((attack / "cvs").glob("*.pdf")):
        items.append({"group": "attack", "item": f.stem, "cv": f, "jd": jd,
                      "expected": "1" if f.stem.startswith("A0") else "0"})
    for i, f in enumerate(sorted((DATA / "cvs").glob("*.pdf")), start=1):   # same IDs as the batch
        cid = f"C{i:02d}"
        items.append({"group": "real", "item": cid, "cv": f, "jd": DATA / "jds" / "03. JD_Data Scientist.pdf",
                      "expected": REAL_EXPECTED.get(cid, "")})
    return items


_TEXT: dict = {}


def full_text(path: Path) -> str:
    """Everything pdfplumber extracts, hidden text included (no visibility filter)."""
    key = str(path)
    if key not in _TEXT:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            _TEXT[key] = "\n".join(p.extract_text() or "" for p in pdf.pages)
    return _TEXT[key]


# ---------------------------------------------------------------- keyword baseline
STOP = set("""a an the and or of in on at to for with by from as is are be been being this that these those
such e g eg i.e ie etc including include includes within across into over per via who which what will would
should can could must may might have has having do does
experience experienced experiences year years yrs yr month months minimum min least preferred preferably required
requirement requirements plus strong solid good excellent proven demonstrated hands-on hands ability able
skills skill knowledge proficiency proficient familiarity familiar understanding working work related relevant
similar equivalent field fields area areas using use used apply applying related background level professional
candidate candidates role job other others any some more than
masked [masked] must-have nice-to-have""".split())   # parser artifacts: its mask and tag text, not JD terms


def key_term_groups(requirement: str) -> list[list[str]]:
    """'Experience with Python/R and SQL' -> [['python', 'r'], ['sql']]: every group needs one hit."""
    groups = []
    for chunk in re.split(r"[\s,;:()]+", requirement.lower()):
        alts = [a.strip(".'\"") for a in chunk.split("/")]
        alts = [a for a in alts if a and a not in STOP and not re.fullmatch(r"[\d.+\-%]+", a)
                and (len(a) > 1 or a in ("r", "c"))]
        if alts:
            groups.append(alts)
    return groups


def term_in(term: str, text: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?:s|es)?(?![a-z0-9])", text) is not None


def jd_requirements(jd_path: Path, cache: dict) -> dict:
    """Parse the JD once (tools.parsed_jd) -> {"must": [...], "nice": [...]} or {"error": ...}."""
    key = str(jd_path)
    if key not in cache:
        import tools
        try:
            jd = tools.parsed_jd(str(jd_path))
            items = [item for _, item in tools.jd_items(jd)]
            cache[key] = {"must": [i.value for i in items if i.requirement.value == "mandatory"],
                          "nice": [i.value for i in items if i.requirement.value != "mandatory"]}
        except Exception as e:   # not a JD / no JD found
            cache[key] = {"error": f"{type(e).__name__}: {e}"[:200]}
    return cache[key]


def requirement_met(requirement: str, text: str, min_share: float) -> bool:
    groups = key_term_groups(requirement)
    if not groups:
        return True
    hits = sum(any(term_in(a, text) for a in g) for g in groups)
    return hits >= min_share * len(groups) - 1e-9


def keyword_label(cv: Path, reqs: dict, min_share: float = 1.0) -> tuple[str, list[str]]:
    if "error" in reqs:
        return "invalid", []
    text = re.sub(r"\s+", " ", full_text(cv).lower())
    missing = [r for r in reqs["must"] if not requirement_met(r, text, min_share)]
    return ("0" if missing else "1"), missing


# ---------------------------------------------------------------- one-call baseline
ONECALL_PROMPT = """You screen ONE resume against ONE job description (JD).
Decide only on the JD's mandatory (must-have) requirements.
LABEL 1 if the resume meets every must-have requirement, LABEL 0 if it fails at least one.
End your reply with exactly one line: LABEL: 0 or LABEL: 1"""


def onecall_model():
    from dotenv import load_dotenv
    from langchain.chat_models import init_chat_model
    load_dotenv(HERE / ".env")
    # Same model and API as agent.py, plus a per-call timeout so a stuck request can't hang the run.
    return init_chat_model("openai:gpt-5-mini", use_responses_api=False, base_url=os.getenv("OPENAI_API_BASE_URL"),
                           timeout=180, max_retries=2)


def onecall_messages(cv: Path, jd: Path):
    from langchain_core.messages import HumanMessage, SystemMessage
    return [SystemMessage(ONECALL_PROMPT),
            HumanMessage(f"JOB DESCRIPTION:\n{full_text(jd)}\n\nRESUME:\n{full_text(cv)}")]


# ---------------------------------------------------------------- runners
FIELDS = ["system", "run", "group", "item", "cv_file", "jd_file", "expected", "label", "correct",
          "missing_must_haves", "tokens_in", "tokens_out", "cost_usd", "latency_s", "error"]


def new_run_dir(system: str) -> Path:
    d = DATA / "baselines" / f"{datetime.now():%Y%m%d-%H%M%S}_{system}"
    d.mkdir(parents=True)
    return d


def row(system, run, it, label, **kw) -> dict:
    exp = it["expected"]
    return {"system": system, "run": run, "group": it["group"], "item": it["item"], "cv_file": it["cv"].name,
            "jd_file": it["jd"].name, "expected": exp, "label": label,
            "correct": "" if not exp else int(label == exp), **kw}


def run_keyword(items, min_share: float = 1.0, reqs_file: str = "") -> Path:
    system = "keyword" if min_share >= 1 else f"keyword{round(min_share * 100)}"
    out = new_run_dir(system)
    # --reqs reuses a saved jd_requirements.json, so variants are checked against the SAME requirement list
    cache: dict = ({str(DATA / "_" / k): v for k, v in json.loads(Path(reqs_file).read_text()).items()}
                   if reqs_file else {})
    by_name = {Path(k).name: v for k, v in cache.items()}
    rows = []
    for it in items:
        reqs = by_name.get(it["jd"].name) or jd_requirements(it["jd"], cache)
        label, missing = keyword_label(it["cv"], reqs, min_share)
        rows.append(row(system, 1, it, label, missing_must_haves=len(missing), error=reqs.get("error", "")))
    (out / "jd_requirements.json").write_text(json.dumps({Path(k).name: v for k, v in cache.items()}, indent=2))
    write(out, rows)
    return out


def run_onecall(items, runs: int, workers: int) -> Path:
    out = new_run_dir("onecall")
    model = onecall_model()
    stop = threading.Event()

    def one(job):
        run, it = job
        if stop.is_set():
            return row("onecall", run, it, "quota", error="not run: quota")
        t0 = time.perf_counter()
        try:
            reply = model.invoke(onecall_messages(it["cv"], it["jd"]))
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if "insufficient_quota" in msg:
                stop.set()
                return row("onecall", run, it, "quota", error="insufficient_quota")
            return row("onecall", run, it, "error", error=msg[:200], latency_s=round(time.perf_counter() - t0, 1))
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        labels = re.findall(r"LABEL:\s*([01])\b", text)
        u = reply.usage_metadata or {}
        tin, tout = u.get("input_tokens", 0), u.get("output_tokens", 0)
        return row("onecall", run, it, labels[-1] if labels else "none", tokens_in=tin, tokens_out=tout,
                   cost_usd=round((tin * PRICE_IN_PER_M + tout * PRICE_OUT_PER_M) / 1e6, 5),
                   latency_s=round(time.perf_counter() - t0, 1))

    jobs = [(run, it) for run in range(1, runs + 1) for it in items]
    rows = []
    with ThreadPoolExecutor(workers) as pool:
        for i, r in enumerate(pool.map(one, jobs), start=1):
            rows.append(r)
            print(f"[{i}/{len(jobs)}] run{r['run']} {r['group']} {r['item']}: {r['label']}", flush=True)
            write(out, rows)   # rewrite after every pair: a crash keeps what finished
    if stop.is_set():
        print("STOPPED: insufficient_quota - remaining pairs marked 'quota'", flush=True)
    return out


def estimate(items, runs: int):
    """Prompt tokens from text length (~4 chars/token); output tokens assumed (reasoning model)."""
    chars = sum(len(ONECALL_PROMPT) + len(full_text(it["jd"])) + len(full_text(it["cv"])) + 40 for it in items)
    tin = chars / 4
    for tout_per in (1500, 3000):
        tout = tout_per * len(items)
        cost = runs * (tin * PRICE_IN_PER_M + tout * PRICE_OUT_PER_M) / 1e6
        print(f"{len(items)} pairs x {runs} runs: ~{tin / len(items):.0f} input tokens/pair, "
              f"assuming {tout_per} output tokens/pair -> ~${cost:.2f}")


def write(out: Path, rows):
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("system", choices=["keyword", "onecall", "estimate"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-share", type=float, default=1.0, help="keyword: share of a must-have's key terms needed")
    ap.add_argument("--reqs", default="", help="keyword: reuse this jd_requirements.json instead of parsing")
    args = ap.parse_args()
    items = build_inputs()
    missing = [str(it["cv"]) for it in items if not it["cv"].is_file()] + \
              [str(it["jd"]) for it in items if not it["jd"].is_file()]
    if missing:
        print("missing inputs:", sorted(set(missing)))
        return 1
    if args.system == "estimate":
        estimate(items, args.runs)
        return 0
    out = run_keyword(items, args.min_share, args.reqs) if args.system == "keyword" else run_onecall(items, args.runs, args.workers)
    rows = list(csv.DictReader(open(out / "labels.csv", encoding="utf-8")))
    scored = [r for r in rows if r["correct"] != ""]
    print(f"{len(rows)} rows, {sum(r['correct'] == '1' for r in scored)}/{len(scored)} correct where a label "
          f"is expected | written: {out / 'labels.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
