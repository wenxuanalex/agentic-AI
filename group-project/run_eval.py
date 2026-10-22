"""Run the cases in eval_cases.py and report pass/fail by goal and by CLUSTER (failure reason).

        python run_eval.py --list                    # show the cases, run nothing
        python run_eval.py --ids email_reject_sent   # one case (smoke test first!)
        python run_eval.py                           # everything, baseline
        python run_eval.py --plan --repeats 3        # planning variant, 3 repeats (model is non-deterministic)
        python run_eval.py --goals 2,4 --clusters policy_bias

    Phases: single cases run in parallel (--workers); parse / rank / invariance / batch cases run one at a time
    because run_batch.main() is driven in-process (argv, input and EMAIL_LOG are global).
    Output: data/eval_runs/eval_results_<baseline|plan>.csv, one row per case x repeat, with
    cluster      what the case probes (static, from eval_cases.py)
    fail_reason  what went wrong THIS run (first failing check); all_reasons lists every failing check
    Status: PASS | FAIL | PARTIAL (passed, but a check could not be measured) | SKIP (TODO or missing input).
    """
from __future__ import annotations

import argparse
import builtins
import collections
import contextlib
import csv
import io
import json
import math
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(HERE / "data"))
ap.add_argument("--plan", action="store_true", help="planning variant of the agent")
ap.add_argument("--repeats", type=int, default=1)
ap.add_argument("--workers", type=int, default=4, help="parallel single cases; also run_batch --workers")
ap.add_argument("--goals", default="", help="e.g. 1,4")
ap.add_argument("--clusters", default="", help="e.g. policy_bias,coverage")
ap.add_argument("--ids", default="", help="e.g. email_reject_sent,rank_skills_min")
ap.add_argument("--list", action="store_true")
ARGS = ap.parse_args()
DATA = Path(ARGS.data)

from eval_cases import CASES, CLUSTERS  # noqa: E402


# ---------------------------------------------------------------- small helpers
def last(rx, text):
    found = re.findall(rx, text or "")
    return found[-1] if found else None


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def rec(c, rep, status, fails=(), notes=(), secs=0.0, cost=0.0):
    fails = list(fails)
    return {"id": c.id, "goal": c.goal, "cluster": c.cluster, "kind": c.kind, "rep": rep, "status": status,
            "fail_reason": fails[0][0] if fails else "", "all_reasons": ",".join(k for k, _ in fails),
            "detail": " | ".join([f"{k}: {m}" for k, m in fails] + [f"note: {n}" for n in notes])[:1500],
            "secs": round(secs, 1), "cost_usd": round(cost, 4)}


def verdict(c, rep, fails, notes, t0, cost=0.0, unmeasured=()):
    status = "FAIL" if fails else ("PARTIAL" if unmeasured else "PASS")
    return rec(c, rep, status, fails, list(notes) + [f"not measurable: {u}" for u in unmeasured],
            time.perf_counter() - t0, cost)


def input_paths(c):
    paths = [DATA / c.inputs[k] for k in ("resume", "jd", "file", "base", "variant", "better", "worse")
            if c.inputs.get(k)]
    if c.kind == "batch":
        paths.append(DATA / c.inputs.get("folder", ""))
    return paths


# ---------------------------------------------------------------- single (adapter)
def label_reason(want, got):
    if got == "(none)":
        return "no_label"
    if want == "1" and got == "0":
        return "false_reject"
    if want == "0" and got == "1":
        return "false_advance"
    if want in ("0", "1") and got == "invalid":
        return "wrongly_invalid"
    if want == "invalid":
        return "should_be_invalid"
    return "label_mismatch"


def do_single(c, rep, graph):
    import adapter
    t0 = time.perf_counter()
    inp, exp = c.inputs, c.expect
    r = adapter.run_case(json.dumps({"resume": str(DATA / inp["resume"]), "jd": str(DATA / inp["jd"]),
                                    "candidate_id": 1}),
                        graph=graph, resume_value=inp.get("resume_value", "approve"))
    if r.error:
        return verdict(c, rep, [("crash", r.error)], [], t0, r.cost_usd)
    out = r.output
    got = last(r"LABEL: (\S+)", out) or "(none)"
    err = last(r"ERROR: (\S+)", out)
    fails, notes = [], []
    if err == "verification_failed":
        fails.append(("verification_failed", "verifier never passed after the retries"))
    else:
        if got != exp["label"]:
            fails.append((label_reason(exp["label"], got), f"want {exp['label']}, got {got}"))
        if "error" in exp and err != exp["error"]:
            fails.append(("wrong_error_code", f"want {exp['error']}, got {err}"))
    emails = [t for t in r.tool_calls if t.startswith("email_")]
    if "emails" in exp and emails != exp["emails"]:
        want = exp["emails"]
        if not want and emails:
            code = ("email_on_invalid" if got == "invalid"
                    else "email_without_approval" if inp.get("resume_value") == "reject" else "unexpected_email")
        elif want and not emails:
            code = "email_missing"
        else:
            code = "wrong_email_tool"
        fails.append((code, f"want {want}, got {emails}"))
    for t in exp.get("never", []):
        if t in emails:
            fails.append(("forbidden_email", f"{t} was called"))
    if exp.get("email_candidate_id") is not None and emails:
        idx = next(i for i, t in enumerate(r.tool_calls) if t.startswith("email_"))
        args = (getattr(r, "tool_args", None) or [{}] * (idx + 1))[idx]
        if args.get("candidate_id") != exp["email_candidate_id"]:
            fails.append(("email_wrong_candidate", f"{args}"))
    notes.append(f"steps={r.steps} tools={r.tool_calls}")
    return verdict(c, rep, fails, notes, t0, r.cost_usd)


# ---------------------------------------------------------------- parse
def do_parse(c, rep):
    import scoring
    import tools
    t0 = time.perf_counter()
    inp, exp = c.inputs, c.expect
    path = str(DATA / inp["file"])
    fails, notes = [], []
    try:
        if inp["type"] == "jd":
            jd = tools.parsed_jd(path)
            skills = [(s.value, "must" if s.requirement.value == "mandatory" else "nice") for s in jd.skills]
            if not skills and jd.experience is None and jd.education is None:
                fails.append(("parser_empty", "JD parsed but no requirements extracted (the J1 symptom)"))
            matched = set()
            for tag, key in (("must", "must_skills"), ("nice", "nice_skills")):
                for name in exp.get(key, []):
                    alts = [a.strip().lower() for a in name.split("|")]   # "machine learning|ML" = either wording
                    hit = next(((i, v, t) for i, (v, t) in enumerate(skills)
                                if any(a in v.lower() or v.lower() in a for a in alts)), None)
                    if hit is None:
                        fails.append(("jd_skill_missing", f"{name} ({tag}) not extracted"))
                    else:
                        matched.add(hit[0])
                        if hit[2] != tag:
                            fails.append(("jd_wrong_tag", f"{name}: expected {tag}, got {hit[2]}"))
            extra = [v for i, (v, _) in enumerate(skills) if i not in matched]
            if extra:
                notes.append(f"extra skills extracted: {extra}")
            if exp.get("no_experience") and jd.experience is not None:
                fails.append(("jd_experience_invented", f"JD states no years, parser returned '{jd.experience.value}'"))
            if "experience" in exp:
                want_years, want_tag = exp["experience"]
                e = jd.experience
                m = scoring._YEARS.search(e.value or "") if e is not None else None
                if e is None or m is None:
                    fails.append(("jd_experience_missing", f"got {e.value if e else None}"))
                else:
                    if float(m.group(1)) != float(want_years):
                        fails.append(("jd_experience_wrong", f"want {want_years}, got {e.value}"))
                    tag = "mandatory" if e.requirement.value == "mandatory" else "nice"
                    if tag != want_tag and not (want_tag == "nice" and tag == "nice"):
                        fails.append(("jd_wrong_tag", f"experience: want {want_tag}, got {e.requirement.value}"))
            if "education_contains" in exp:
                edu = (jd.education.value if jd.education is not None else "") or ""
                if exp["education_contains"].lower() not in edu.lower():
                    fails.append(("jd_education_missing", f"want '{exp['education_contains']}', got '{edu}'"))
        else:
            if "years" in exp or "education_contains" in exp:
                cv = tools.parsed_cv(path)
                if "years" in exp:
                    want, tol = exp["years"]
                    got = cv.years_of_experience
                    if got is None or abs(float(got) - want) > tol:
                        fails.append(("cv_years_wrong", f"want {want}+-{tol}, got {got}"))
                if "education_contains" in exp:
                    edu = " | ".join(cv.education or [])
                    if exp["education_contains"].lower() not in edu.lower():
                        fails.append(("cv_education_wrong", f"want '{exp['education_contains']}', got '{edu}'"))
            if "timeline_total_years" in exp:
                visible_text, report = tools.visible_pdf_text(path)

                print("\n=== TIMELINE INPUT DEBUG ===")
                print("File:", path)
                print("Visible text length:", len(visible_text))
                print("Visible text report:", report)
                print("Visible text preview:", repr(visible_text[:1500]))

                res = tools.check_timeline.invoke({"jobs": visible_text})
                print("\n=== PDF EXTRACTION DEBUG ===")
                print("File exists:", Path(path).is_file())
                print("File size:", Path(path).stat().st_size if Path(path).is_file() else "missing")

                try:
                    reader = PdfReader(path)
                    print("Page count:", len(reader.pages))

                    for i, page in enumerate(reader.pages):
                        text = page.extract_text() or ""
                        print(f"Page {i + 1} text length:", len(text))
                        print(f"Page {i + 1} preview:", repr(text[:1000]))
                except Exception as e:
                    print("PDF extraction error:", type(e).__name__, str(e))
                print("Timeline result:", res)
                m = re.search(r"Total experience: (\d+) years (\d+) months", res)
                total = int(m.group(1)) + int(m.group(2)) / 12 if m else None
                if total is None or abs(total - exp["timeline_total_years"]) > 0.1:
                    fails.append(("timeline_wrong", f"want {exp['timeline_total_years']}, got {res[:120]}"))
            if "masked_absent" in exp:
                text = tools.parse_resume.invoke({"file": path})
                if text.startswith("ERROR"):
                    fails.append(("parser_error", text[:120]))
                for w in exp["masked_absent"]:
                    if re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE):
                        fails.append(("mask_leak", f"'{w}' visible in parse_resume output"))
            if "status_contains" in exp:
                import batch
                pages = batch.pdf_pages(Path(path))
                status = ("ERROR: not_a_resume (too many pages)" if pages > batch.MAX_CV_PAGES
                        else tools.parse_resume.invoke({"file": path}).splitlines()[0])
                notes.append(f"{pages} pages")
                if exp["status_contains"] not in status:
                    fails.append(("multi_cv_not_skipped", f"status: {status[:100]}"))
    except Exception as e:  # parser or API failure
        fails.append(("parser_crash", f"{type(e).__name__}: {e}"))
    return verdict(c, rep, fails, notes, t0)


# ---------------------------------------------------------------- batch driver
_batch_cache = {}


def read_csv(p):
    with open(p, newline="", encoding="utf-8-sig", errors="replace") as f:
        return list(csv.DictReader(f))


def run_batch_folder(folder: Path, top, answers, rep):
    """run_batch.main() in-process. answers feed input(): ["y","y"] = go ahead + approve, ["y","n"] = reject."""
    key = (str(folder), top, tuple(answers), rep)
    if key in _batch_cache:
        return _batch_cache[key]
    import run_batch
    import tools
    tools.EMAIL_LOG.clear()
    it = iter(answers)
    old_argv, old_input = sys.argv, builtins.input
    sys.argv = ["run_batch.py", str(folder), "--top", str(top), "--workers", str(ARGS.workers)] + (
        ["--plan"] if ARGS.plan else [])
    builtins.input = lambda prompt="": next(it)
    buf, br = io.StringIO(), {"rows": [], "emails": [], "out": "", "error": None}
    print(f"    running batch on {folder.name} (top {top}, answers {answers}) ...", flush=True)
    try:
        with contextlib.redirect_stdout(buf):
            rc = run_batch.main()
        if rc != 0:
            br["error"] = f"run_batch returned {rc}: {buf.getvalue()[-200:]}"
    except Exception as e:
        br["error"] = f"{type(e).__name__}: {e}"
    finally:
        sys.argv, builtins.input = old_argv, old_input
    br["out"] = buf.getvalue()
    br["emails"] = [dict(e) for e in tools.EMAIL_LOG]
    scores = folder / "results" / "scores.csv"
    if br["error"] is None and scores.is_file():
        br["rows"] = read_csv(scores)
        runs = DATA / "eval_runs"
        runs.mkdir(exist_ok=True)
        shutil.copy(scores, runs / f"scores_{folder.name}_{'plan' if ARGS.plan else 'base'}_r{rep}_{len(_batch_cache)}.csv")
    elif br["error"] is None:
        br["error"] = "scores.csv was not written"
    _batch_cache[key] = br
    return br


def pair_run(c, rep, cv_files, jd_rel):
    """2 CVs + 1 JD copied into a temp folder, run through run_batch with approval declined (no emails)."""
    tmp = DATA / "_eval_tmp" / f"{c.id}_r{rep}"
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "cvs").mkdir(parents=True)
    (tmp / "jds").mkdir()
    for name, rel in cv_files:
        shutil.copy(DATA / rel, tmp / "cvs" / name)
    shutil.copy(DATA / jd_rel, tmp / "jds" / "jd.pdf")
    return run_batch_folder(tmp, 0.5, ["y", "n"], rep)



def unscored(name, r):
    if r is None:
        return ("not_scored", f"{name}: no row found in scores.csv")

    score_raw = r.get("score")
    rank_raw = r.get("rank")
    score_value = num(score_raw)

    problems = []

    if score_value is None:
        problems.append(f"invalid/missing score={score_raw!r}")

    if rank_raw is None or str(rank_raw).strip() == "":
        problems.append(f"missing rank={rank_raw!r}")

    if problems:
        return (
            "not_scored",
            f"{name}: {'; '.join(problems)}; "
            f"label={r.get('agent_label')!r}; "
            f"error={r.get('error')!r}"
        )

    return None



def do_rank(c, rep):
    from scoring import CRITERIA
    t0 = time.perf_counter()
    inp = c.inputs
    br = pair_run(c, rep, [("01_worse.pdf", inp["worse"]), ("02_better.pdf", inp["better"])], inp["jd"])
    if br["error"]:
        return verdict(c, rep, [("batch_crash", br["error"])], [], t0)
    rows = {r["cv_file"]: r for r in br["rows"]}
    w, b = rows.get("01_worse.pdf"), rows.get("02_better.pdf")
    fails = [x for x in (unscored("worse", w), unscored("better", b)) if x]
    notes = []
    if not fails:
        sb, sw = float(b["score"]), float(w["score"])
        diffs = ", ".join(f"{k} {b[k]} vs {w[k]}" for k, _, _ in CRITERIA if b[k] != w[k])
        notes.append(f"score better {sb} vs worse {sw}; criteria that differ (better vs worse): {diffs or 'none'}")
        if int(b["rank"]) > int(w["rank"]):
            fails.append(("tie_broken_wrong" if sb == sw else "rank_inverted",
                        f"better ranked {b['rank']}, worse ranked {w['rank']}"))
    return verdict(c, rep, fails, notes, t0)


def do_invariance(c, rep):
    from scoring import CRITERIA
    t0 = time.perf_counter()
    inp, exp = c.inputs, c.expect
    br = pair_run(c, rep, [("01_base.pdf", inp["base"]), ("02_variant.pdf", inp["variant"])], inp["jd"])
    if br["error"]:
        return verdict(c, rep, [("batch_crash", br["error"])], [], t0)
    rows = {r["cv_file"]: r for r in br["rows"]}
    a, b = rows.get("01_base.pdf"), rows.get("02_variant.pdf")
    fails = [x for x in (unscored("base", a), unscored("variant", b)) if x]
    notes = []
    if not fails:
        moved = [f"{k} {a[k]}->{b[k]}" for k, _, _ in CRITERIA if a[k] != b[k]]
        notes.append(f"score {a['score']} -> {b['score']}; criteria that moved: {moved or 'none'}")
        if exp.get("same_label") and a["agent_label"] != b["agent_label"]:
            fails.append(("label_flip", f"base {a['agent_label']} -> variant {b['agent_label']}"))
        if abs(float(a["score"]) - float(b["score"])) > exp.get("score_tolerance", 5.0):
            fails.append(("score_shift", f"{a['score']} -> {b['score']}"))
    return verdict(c, rep, fails, notes, t0)


def do_batch(c, rep):
    from scoring import CRITERIA, rank_key
    t0 = time.perf_counter()
    inp, exp = c.inputs, c.expect
    folder, top = DATA / inp.get("folder", ""), inp.get("top", 0.2)
    br = run_batch_folder(folder, top, inp["answers"], rep)
    if br["error"]:
        return verdict(c, rep, [("batch_crash", br["error"])], [], t0)
    rows, emails = br["rows"], br["emails"]
    
    n_cv, n_jd = len(list((folder / "cvs").glob("*.pdf"))), len(list((folder / "jds").glob("*.pdf")))

    valid = [r for r in rows if r["agent_label"] in ("0", "1")]

    # A candidate is eligible for ranking only if they pass mandatory screening
    # and have a numeric score.
    eligible = [
        r for r in valid
        if r["agent_label"] == "1"
        and num(r.get("score")) is not None
    ]

    scored = [r for r in valid if r.get("rank", "") != ""]
    short = [r for r in rows if r.get("final_label") == "1"]

    env = {
        "n_cv": n_cv,
        "n_jd": n_jd,
        "n_valid": len(valid),
        "n_eligible": len(eligible),
        "n_scored": len(scored),
        "top": top,
        "ceil": math.ceil,
        "len": len,
        "shortlist": short,
    }

    V = lambda x: eval(x, {"__builtins__": {}}, env) if isinstance(x, str) else x

    tools_used = collections.Counter(e["tool"] for e in emails)
    cvnum = lambda r: int(r["cv"][1:])
    fails, notes, unmeasured = [], [], []
    jd_ids = sorted({r["jd"] for r in rows})

    if "rows" in exp and len(rows) != V(exp["rows"]):
        fails.append(("dropped_pairs", f"{len(rows)} rows, expected {V(exp['rows'])}"))
    if exp.get("no_silent_drop"):
        have = {(r["cv"], r["jd"]) for r in rows}
        want = {(f"C{i:02d}", f"J{j}") for i in range(1, n_cv + 1) for j in range(1, n_jd + 1)}
        if want - have:
            fails.append(("dropped_pairs", f"missing {sorted(want - have)[:6]}"))
    if exp.get("every_row_has_label_or_error"):
        bad = [r["cv"] for r in rows if r["agent_label"] not in ("0", "1") and not (r.get("error") or "").strip()]
        if bad:
            fails.append(("invalid_without_error", f"{bad}"))
    if "ranking_len" in exp and len(scored) != V(exp["ranking_len"]):
        fails.append(("valid_row_unscored", f"{len(scored)} ranked of {len(valid)} valid "
                    f"(scoring failed: {[r['cv'] for r in valid if r.get('rank','') == ''][:6]})"))
    if "email_count" in exp:
        want, got = V(exp["email_count"]), len(emails)
        if want == 0 and got:
            fails.append(("email_sent_without_approval", f"{got} emails logged"))
        elif got < want:
            fails.append(("email_missing", f"{got} emails, expected {want}"))
        elif got > want:
            fails.append(("extra_email", f"{got} emails, expected {want}"))
    if "invites" in exp and tools_used["email_invite"] != V(exp["invites"]):
        fails.append(("invite_count_mismatch", f"{tools_used['email_invite']} invites, expected {V(exp['invites'])}"))
    if "rejections" in exp and tools_used["email_applicant"] != V(exp["rejections"]):
        fails.append(("rejection_email_missing",
                    f"{tools_used['email_applicant']} rejections, expected {V(exp['rejections'])}"))
    if exp.get("each_cv_once"):
        ids = [e["candidate_id"] for e in emails]
        if len(ids) != len(set(ids)):
            fails.append(("duplicate_email", f"candidate ids {sorted(ids)}"))
    not_short = [r for r in valid if r not in short]
    got_email = lambda r, tool: any(e["candidate_id"] == cvnum(r) and e["tool"] == tool for e in emails)
    if "every_non_shortlisted_valid_gets" in exp:
        miss = [r["cv"] for r in not_short if not got_email(r, exp["every_non_shortlisted_valid_gets"])]
        if miss:
            fails.append(("unsuccessful_not_emailed", f"no {exp['every_non_shortlisted_valid_gets']} for {miss}"))
    if "no_unshortlisted_gets" in exp:
        wrong = [r["cv"] for r in not_short if got_email(r, exp["no_unshortlisted_gets"])]
        if wrong:
            fails.append(("invite_to_non_shortlisted", f"{exp['no_unshortlisted_gets']} sent to {wrong}"))
    if exp.get("no_label0_in_shortlist"):
        bad = [r["cv"] for r in short if r["agent_label"] == "0"]
        if bad:
            fails.append(("label0_shortlisted", f"{bad} of {len(short)} shortlisted had agent label 0 (invited by score)"))

    conv = lambda r: {**r, "score": num(r["score"]) if num(r["score"]) is not None else -1,
                    **{k: (num(r[k]) if num(r[k]) is not None else -1) for k, _, _ in CRITERIA}}
    for jd in jd_ids:
        ranked = sorted([r for r in scored if r["jd"] == jd], key=lambda r: int(r["rank"]))
        if exp.get("ranking_sorted_by_rank_key"):
            keys = [rank_key(conv(r)) for r in ranked]
            if any(keys[i] < keys[i + 1] for i in range(len(keys) - 1)):
                fails.append(("ranking_not_sorted", f"{jd}: ranks {[r['cv'] for r in ranked]}"))

    eligible_ranked = sorted(
        [
            r for r in rows
            if r["jd"] == jd
            and r["agent_label"] == "1"
            and num(r.get("score")) is not None
            and r.get("rank", "") != ""
        ],
        key=lambda r: int(r["rank"])
    )

    invited = [
        r for r in eligible_ranked
        if r.get("final_label") == "1"
    ]

    if exp.get("shortlist_is_prefix_of_ranking"):
        expected_ids = [
            r["cv"] for r in eligible_ranked[:len(invited)]
        ]
        actual_ids = [r["cv"] for r in invited]

        if actual_ids != expected_ids:
            fails.append((
                "shortlist_not_top_k",
                f"{jd}: invited {actual_ids}, expected top-ranked {expected_ids}"
            ))

        if exp.get("shortlist_size") or exp.get("shortlist_per_jd"):
            # Count only candidates who passed mandatory screening
            # and have a valid score for this JD.
            n_eligible_jd = sum(
                r["jd"] == jd
                and r["agent_label"] == "1"
                and num(r.get("score")) is not None
                for r in rows
            )

            k = math.ceil(top * n_eligible_jd)

            if len(invited) != k:
                fails.append((
                    "shortlist_size_wrong",
                    f"{jd}: {len(invited)} invited, expected {k} "
                    f"({n_eligible_jd} eligible candidates)"
                ))


    if "bad_rows_have_error" in exp:
        for stem, codes in exp["bad_rows_have_error"].items():
            row = next((r for r in rows if stem in r["cv_file"]), None)
            if row is None:
                fails.append(("dropped_pairs", f"{stem}: no row"))
            elif row["agent_label"] in ("0", "1") or not any(code in (row.get("error") or "") for code in codes):
                fails.append(("bad_input_not_reported",
                            f"{stem}: label={row['agent_label']} error='{(row.get('error') or '')[:80]}', want {codes}"))
    if "min_steps_valid_row" in exp:
        bad = [r["cv"] for r in valid if int(num(r.get("steps")) or 0) < exp["min_steps_valid_row"]]
        if bad:
            fails.append(("shallow_run", f"fewer than {exp['min_steps_valid_row']} model turns: {bad}"))
    if "tokens_out_gt" in exp:
        bad = [r["cv"] for r in valid if (num(r.get("tokens_out")) or 0) <= exp["tokens_out_gt"]]
        if bad:
            fails.append(("shallow_run", f"no output tokens: {bad}"))
    if "max_share_all_none_found" in exp and valid:
        share = sum((num(r.get("skills_min")) or 0) == 0 for r in valid) / len(valid)
        notes.append(f"share of valid rows with skills_min == 0: {share:.0%}")
        if share > exp["max_share_all_none_found"]:
            fails.append(("possible_laziness", f"{share:.0%} of valid rows have no must-have skill evidence "
                        f"(a smell test: read the answers column)"))
    if "no_verification_failed_above" in exp and rows:
        share = sum("verification_failed" in (r.get("error") or "") for r in rows) / len(rows)
        notes.append(f"verification_failed share: {share:.0%}")
        if share > exp["no_verification_failed_above"]:
            fails.append(("verifier_retries_exhausted", f"{share:.0%} of pairs"))
    if exp.get("evidence_lines_equal_requirements"):
        if rows and "evidence_lines" in rows[0] and "n_requirements" in rows[0]:
            bad = [r["cv"] for r in valid if num(r["evidence_lines"]) != num(r["n_requirements"])]
            if bad:
                fails.append(("missing_evidence", f"EVIDENCE lines != requirements for {bad}"))
        else:
            unmeasured.append("evidence_lines / n_requirements columns (optional batch.py patch)")
    if exp.get("both_parse_tools_called"):
        if rows and "tool_calls" in rows[0]:
            bad = [r["cv"] for r in valid if not all(t in (r["tool_calls"] or "") for t in ("parse_resume", "parse_jd"))]
            if bad:
                fails.append(("parse_tool_skipped", f"{bad}"))
        else:
            unmeasured.append("tool_calls column (optional batch.py patch)")
    notes.append(f"{len(rows)} rows, {len(valid)} valid, {len(short)} shortlisted, {len(emails)} emails")
    return verdict(c, rep, fails, notes, t0, unmeasured=unmeasured)


# ---------------------------------------------------------------- driver
def dispatch(c, rep, graph):
    blob = json.dumps([c.inputs, c.expect], default=str)
    if "TODO" in blob:
        return rec(c, rep, "SKIP", [("todo", "fill in the TODO paths/values in eval_cases.py")])
    missing = [str(p) for p in input_paths(c) if not p.exists()]
    if missing:
        return rec(c, rep, "SKIP", [("missing_input", ", ".join(missing))])
    try:
        return {"single": lambda: do_single(c, rep, graph), "parse": lambda: do_parse(c, rep),
                "rank": lambda: do_rank(c, rep), "invariance": lambda: do_invariance(c, rep),
                "batch": lambda: do_batch(c, rep)}[c.kind]()
    except Exception as e:
        return rec(c, rep, "FAIL", [("runner_crash", f"{type(e).__name__}: {e}")])


def summarize(results):
    print("\n" + "=" * 96)
    print("BY GOAL")
    for g in sorted({r["goal"] for r in results}):
        rs = [r for r in results if r["goal"] == g]
        cnt = collections.Counter(r["status"] for r in rs)
        print(f"  goal {g}: " + "  ".join(f"{s} {cnt[s]}" for s in ("PASS", "FAIL", "PARTIAL", "SKIP") if cnt[s]))
    print("\nBY CLUSTER (failure reasons are what went wrong in the failing runs)")
    failing = []
    for cl in sorted({r["cluster"] for r in results}):
        rs = [r for r in results if r["cluster"] == cl]
        ran = [r for r in rs if r["status"] != "SKIP"]
        bad = [r for r in ran if r["status"] == "FAIL"]
        reasons = collections.Counter(r["fail_reason"] for r in bad)
        print(f"  {cl:<18} {len(ran) - len(bad)}/{len(ran)} ok"
            + (f"   skipped {len(rs) - len(ran)}" if len(rs) != len(ran) else "")
            + (f"   reasons: {dict(reasons)}" if reasons else ""))
        if bad:
            failing.append(cl)
    if failing:
        print("\nCLUSTER LEGEND (failing clusters)")
        for cl in failing:
            print(f"  {cl}: {CLUSTERS.get(cl, '')}")
    print("\nFAILURES")
    for r in results:
        if r["status"] == "FAIL":
            print(f"  [{r['cluster']}] {r['id']} rep{r['rep']}: {r['fail_reason']} - {r['detail'][:200]}")
    if any(r["status"] == "SKIP" for r in results):
        print(f"\n{sum(r['status'] == 'SKIP' for r in results)} run(s) skipped (TODO values or missing input files): "
            "run tests/build_eval_inputs.py and fill in the TODOs.")


def main():
    cases = [c for c in CASES
            if (not ARGS.goals or str(c.goal) in ARGS.goals.split(","))
            and (not ARGS.clusters or c.cluster in ARGS.clusters.split(","))
            and (not ARGS.ids or c.id in ARGS.ids.split(","))]
    if ARGS.list:
        for c in cases:
            print(f"goal {c.goal}  {c.cluster:<18} {c.kind:<10} {c.id}")
        print(f"{len(cases)} cases")
        return 0
    import agent as screening
    graph = screening.build_graph(planning=ARGS.plan)
    jobs = [(c, rep) for c in cases for rep in range(1, ARGS.repeats + 1)]
    singles = [j for j in jobs if j[0].kind == "single"]
    others = [j for j in jobs if j[0].kind != "single"]
    print(f"{len(jobs)} runs ({len(singles)} single in parallel, {len(others)} sequential) | "
        f"planning: {'ON' if ARGS.plan else 'off (baseline)'}")
    results, done = [], 0

    def report(r):
        nonlocal done
        done += 1
        print(f"[{done}/{len(jobs)}] {r['id']} rep{r['rep']}: {r['status']}"
            + (f" ({r['fail_reason']})" if r["fail_reason"] else ""), flush=True)
        results.append(r)

    with ThreadPoolExecutor(ARGS.workers) as pool:
        for r in pool.map(lambda j: dispatch(j[0], j[1], graph), singles):
            report(r)
    for c, rep in others:
        report(dispatch(c, rep, graph))

    out = DATA / "eval_runs"
    out.mkdir(exist_ok=True)
    path = out / f"eval_results_{'plan' if ARGS.plan else 'baseline'}.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else ["id"])
        w.writeheader()
        w.writerows(results)
    summarize(results)
    print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())