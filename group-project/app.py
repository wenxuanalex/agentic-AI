"""
Resume Screening Agent - Reviewer UI

Upload CVs and JDs, run the screening batch, review every candidate, adjust the
invite share, then approve. Nothing is emailed before approval.

Run:
    pip install -r requirements.txt
    python -m streamlit run app.py

Works from the repo root or from group-project/. With no scores.csv it falls back
to built-in mock rows, so the screen opens without a batch run.
"""

import csv
import math
import re
import sys
import time
import uuid
from pathlib import Path

import streamlit as st

# Works whether app.py sits in the repo root or inside group-project/.
GROUP = Path("group-project") if (Path("group-project") / "batch.py").exists() else Path(".")
DATA = GROUP / "data"
CV_DIR = DATA / "cvs"
JD_DIR = DATA / "jds"
RESULTS = DATA / "results"
SCORES_PATHS = [RESULTS / "scores.csv", Path("sample_scores.csv")]

sys.path.insert(0, str(GROUP.resolve()))

# Single source of truth: the criteria and the ranking rule come from the batch code itself,
# so this screen always matches who actually gets emailed.
import batch
import scoring

CRITERIA = [(k, label[:1].upper() + label[1:], w) for k, label, w in scoring.CRITERIA]

EST_COST_PER_PAIR = 0.023
EST_COST_PER_PAIR_PLAN = 0.019
HIDDEN_FLAG_CHARS = 20

POLICY_TEXT = {
    "P1": "Merit — skills, experience and ability to do the job only",
    "P2": "Job-related criteria applied consistently to all candidates",
    "P3": "Disability — consider fairly if able to perform the job",
    "P4": "Age — not a requirement unless legally required",
    "P5": "Race — not a selection criterion",
    "P6": "Language — only if the job genuinely needs it",
    "P7": "Gender — only if the job's practical requirements dictate it",
    "P8": "Marital status and family responsibilities — irrelevant",
    "P9": "Religion — unacceptable except for religious functions",
    "P10": "Irrelevant personal information must not be used",
    "P11": "Unsuccessful candidates should be informed of the result",
}

MOCK_ROWS = [
    {"jd": "J1", "jd_file": "JD_Data Scientist.pdf", "cv": "C01", "cv_file": "cv_1.pdf",
     "agent_label": "1", "score": "88.1", "skills_min": "1.0", "years_min": "1.0",
     "years_above_min": "0.8", "university": "0.75", "soft_skills": "0.67",
     "skills_pref": "0.5", "years_pref": "0.8", "evidence_lines": "2",
     "why": "skills: must-have 3/3, nice-to-have 1/1; years: 5.0 vs min 3",
     "hidden_text_chars": "0", "hidden_text_sample": "", "error": "", "verifier_notes": "",
     "answers": 'EVIDENCE: [R1: 3+ years Python] "Built data pipelines in Python, 2021-2024" | '
                'EVIDENCE: [R2: Degree in CS] "BSc Computer Science" | POLICY: P2 | LABEL: 1'},
    {"jd": "J1", "jd_file": "JD_Data Scientist.pdf", "cv": "C02", "cv_file": "cv_2.pdf",
     "agent_label": "1", "score": "71.4", "skills_min": "0.95", "years_min": "0.8",
     "years_above_min": "0.4", "university": "0.5", "soft_skills": "0.33",
     "skills_pref": "0.5", "years_pref": "0.4", "evidence_lines": "2",
     "why": "skills: must-have 19/20, nice-to-have 1/2; years: 2.4 vs min 3",
     "hidden_text_chars": "0", "hidden_text_sample": "", "error": "", "verifier_notes": "",
     "answers": 'EVIDENCE: [R1: 3+ years Python] "Python and SQL for analytics, 2022-2024" | '
                'EVIDENCE: [R2: Degree in CS] "BSc Statistics" | POLICY: P2 | LABEL: 1'},
    {"jd": "J1", "jd_file": "JD_Data Scientist.pdf", "cv": "C03", "cv_file": "cv_3.pdf",
     "agent_label": "0", "score": "41.9", "skills_min": "0.33", "years_min": "0.5",
     "years_above_min": "0.0", "university": "0.5", "soft_skills": "0.33",
     "skills_pref": "0.0", "years_pref": "0.2", "evidence_lines": "2",
     "why": "skills: must-have 7/20, nice-to-have 0/2; years: 1.5 vs min 3",
     "hidden_text_chars": "0", "hidden_text_sample": "", "error": "", "verifier_notes": "",
     "answers": 'EVIDENCE: [R1: 3+ years Python] "Python scripting, 2023-2024" | '
                'EVIDENCE: [R2: Degree in CS] none found | POLICY: P1 | LABEL: 0'},
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_csv_rows():
    for p in SCORES_PATHS:
        if p.exists():
            with open(p, newline="", encoding="utf-8-sig") as f:
                return list(csv.DictReader(f)), str(p)
    return MOCK_ROWS, None


def live_rows(out):
    """Graph results as display rows (in memory they use 'label', the CSV uses 'agent_label')."""
    return [{**r, "agent_label": r.get("label", "")} for r in out["results"]]


def rank_results(rows, invite_share):
    """Calls batch.rank_results so the screen always matches who gets emailed:
    only eligible candidates (label 1, a score, no error, no hidden text) are ranked,
    and the top share of THOSE is invited."""
    def as_batch_row(r):
        return {**r, "label": r.get("agent_label"), "score": to_float(r.get("score")),
                "error": r.get("error") or ""}

    jd_ids = sorted({r["jd"] for r in rows})
    shortlist, ranked_ids = batch.rank_results(
        [as_batch_row(r) for r in rows], [{"id": j} for j in jd_ids], invite_share)
    by_key = {(r["jd"], r["cv"]): r for r in rows}
    ranking = {j: [by_key[(j, cv)] for cv in ranked_ids[j]] for j in jd_ids}
    invited = {(j, cv) for j in jd_ids for cv in shortlist[j]}
    return invited, ranking


def parse_answer(text):
    out = {"evidence": [], "policy": None, "error": None, "label": None}
    for part in re.split(r"\s*\|\|\s*|\s*\|\s*|\n", text or ""):
        part = part.strip()
        if part.startswith("EVIDENCE:"):
            body = part[len("EVIDENCE:"):].strip()
            req, quote = body, None
            if body.startswith("["):
                close = body.find("]")
                if close != -1:
                    req = body[1:close].strip()
                    rest = body[close + 1:].strip()
                    quote = None if rest.lower().startswith("none") else rest.strip('"')
            out["evidence"].append({"requirement": req, "quote": quote})
        elif part.startswith("POLICY:"):
            out["policy"] = part[len("POLICY:"):].strip()
        elif part.startswith("ERROR:"):
            out["error"] = part[len("ERROR:"):].strip()
        elif part.startswith("LABEL:"):
            out["label"] = part[len("LABEL:"):].strip()
    return out


# ---------------------------------------------------------------------------
# Running the batch graph
# ---------------------------------------------------------------------------

def start_run(planning, invite_share, cv_ids, jd_ids, workers=8):
    graph = batch.build_batch_graph()
    cfg = {"configurable": {"thread_id": f"ui-{uuid.uuid4().hex[:8]}"},
           "max_concurrency": workers, "recursion_limit": 50}
    out = graph.invoke(
        {"data_dir": str(DATA), "planning": planning, "results": [],
         "cv_ids": cv_ids, "jd_ids": jd_ids, "invite_share": invite_share}, cfg)
    return graph, cfg, out


def resume_run(resume_value):
    from langgraph.types import Command
    return st.session_state.graph.invoke(Command(resume=resume_value), st.session_state.cfg)


def write_results(out):
    RESULTS.mkdir(parents=True, exist_ok=True)
    rank_of = {(jd, cv): i for jd, ids in out["ranking"].items() for i, cv in enumerate(ids, start=1)}
    invited = {(jd, cv) for jd, ids in out["shortlist"].items() for cv in ids}
    cols = (["jd", "jd_file", "cv", "cv_file", "final_label", "score", "rank", "agent_label",
             "tool_calls", "evidence_lines", "n_requirements"] + [k for k, _, _ in scoring.CRITERIA]
            + ["why", "hidden_text_chars", "hidden_text_sample", "error", "verifier_notes",
               "answers", "steps", "tokens_in", "tokens_out", "seconds"])
    rows = sorted(out["results"], key=lambda r: (r["jd"], rank_of.get((r["jd"], r["cv"]), 999), r["cv"]))
    with open(RESULTS / "scores.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            final = "invalid" if r["label"] not in ("0", "1") else ("1" if (r["jd"], r["cv"]) in invited else "0")
            w.writerow({**r, "rank": rank_of.get((r["jd"], r["cv"]), ""),
                        "final_label": final, "agent_label": r["label"]})
    with open(RESULTS / "shortlist.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["jd", "jd_file", "rank", "cv", "cv_file", "score"])
        for r in rows:
            if (r["jd"], r["cv"]) in invited:
                w.writerow([r["jd"], r["jd_file"], rank_of[(r["jd"], r["cv"])],
                            r["cv"], r["cv_file"], r["score"]])


def save_uploads(files, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for f in files:
        (folder / f.name).write_bytes(f.getbuffer())
    return len(files)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Resume Screening Review", layout="wide")
st.markdown("<style>[data-testid='stStatusWidget']{display:none}</style>", unsafe_allow_html=True)

for key, default in [("decisions", {}), ("graph", None), ("cfg", None),
                     ("out", None), ("emails", None), ("run_secs", 0.0)]:
    if key not in st.session_state:
        st.session_state[key] = default

paused = st.session_state.out is not None and "__interrupt__" in st.session_state.out
finished = st.session_state.emails is not None

st.title("Resume Screening - Review and Approval")

# --- Batch ------------------------------------------------------------------
with st.expander("Batch", expanded=not (paused or finished)):
    u1, u2 = st.columns(2)
    cvs = u1.file_uploader("CV files (PDF)", type="pdf", accept_multiple_files=True, key="cv_up")
    jds_up = u2.file_uploader("JD files (PDF)", type="pdf", accept_multiple_files=True, key="jd_up")

    # Uploads are saved straight into data/. Screen the uploaded files, or everything in data/ if none uploaded.
    save_uploads(cvs or [], CV_DIR)
    save_uploads(jds_up or [], JD_DIR)
    have_cv = sorted(CV_DIR.glob("*.pdf")) if CV_DIR.exists() else []
    have_jd = sorted(JD_DIR.glob("*.pdf")) if JD_DIR.exists() else []
    up_cv = {f.name for f in cvs or []}
    up_jd = {f.name for f in jds_up or []}
    cv_ids = [f"C{i:02d}" for i, f in enumerate(have_cv, 1) if not up_cv or f.name in up_cv]
    jd_ids = [f"J{i}" for i, f in enumerate(have_jd, 1) if not up_jd or f.name in up_jd]

    planning = st.checkbox("Planning mode (cheaper)", value=False)
    pairs = len(cv_ids) * len(jd_ids)
    per_pair = EST_COST_PER_PAIR_PLAN if planning else EST_COST_PER_PAIR
    st.caption(f"{len(cv_ids)} CV(s) x {len(jd_ids)} JD(s) = {pairs} pair(s), "
               f"about ${pairs * per_pair:.2f} and {pairs * 1.5:.0f} minutes")

    if st.button("Run screening", type="primary", disabled=not pairs):
        with st.spinner(f"Screening {pairs} pair(s). Do not refresh this page."):
            t0 = time.perf_counter()
            try:
                graph, cfg, out = start_run(planning, 0.20, cv_ids, jd_ids)
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
                st.stop()
            st.session_state.graph, st.session_state.cfg = graph, cfg
            st.session_state.out = out
            st.session_state.run_secs = time.perf_counter() - t0
            st.session_state.decisions, st.session_state.emails = {}, None
        st.rerun()

# --- Rows -------------------------------------------------------------------
if paused or finished:
    rows = live_rows(st.session_state.out)
    source = f"this run — {len(rows)} CV x JD pairs, {st.session_state.run_secs / 60:.1f} min"
else:
    rows, path = load_csv_rows()
    source = f"{path} — {len(rows)} CV x JD pairs" if path else "sample rows"

st.caption(f"Source: {source}")
if not rows:
    st.stop()

# --- Invite share -----------------------------------------------------------
st.subheader("Invite share")
current = float((st.session_state.out or {}).get("invite_share", 0.20)) if paused else 0.20
share = st.slider("Top share of eligible applicants (agent label 1) invited, per role",
                  0.05, 1.0, current, 0.05, format="%.2f")

if paused and abs(share - current) > 1e-9:
    if st.button(f"Re-rank at {share:.0%}"):
        with st.spinner("Re-ranking"):
            st.session_state.out = resume_run({"invite_share": share})
        st.rerun()

invited, ranking = rank_results(rows, share)

# --- Role -------------------------------------------------------------------
jds = sorted({r["jd"] for r in rows})
jd_labels = {j: next(r["jd_file"] for r in rows if r["jd"] == j) for j in jds}
jd = st.selectbox("Role", jds, format_func=lambda j: f"{j} - {Path(str(jd_labels[j])).name}")

jd_rows = [r for r in rows if r["jd"] == jd]
invalid_rows = [r for r in jd_rows if r["agent_label"] not in ("0", "1")]
eligible_ids = {r["cv"] for r in ranking[jd]}
other_rows = [r for r in jd_rows if r["agent_label"] in ("0", "1") and r["cv"] not in eligible_ids]
jd_invited = [r for r in ranking[jd] if (jd, r["cv"]) in invited]

st.divider()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Screened", len(jd_rows))
m2.metric("Eligible (label 1)", len(ranking[jd]))
m3.metric("Invited", len(jd_invited))
m4.metric("Reviewed", sum(1 for r in jd_rows if r["cv"] in st.session_state.decisions))

if invalid_rows:
    st.info(f"{len(invalid_rows)} file(s) could not be screened "
            f"({', '.join(sorted({r.get('error') or 'unknown' for r in invalid_rows}))}). "
            "They are not auto-rejected.")

hidden = [r for r in jd_rows if (to_float(r.get("hidden_text_chars")) or 0) > HIDDEN_FLAG_CHARS]
if hidden:
    st.error(f"HIDDEN TEXT detected in {len(hidden)} resume(s): {', '.join(r['cv'] for r in hidden)}. "
             "It was excluded from evidence and scoring, but review these before approving.")

st.divider()

# --- Candidates -------------------------------------------------------------
st.subheader(f"Ranked candidates - {Path(str(jd_labels[jd])).name}")


def decision_buttons(cid, prefix):
    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("Agree", key=f"ok_{prefix}_{jd}_{cid}"):
        st.session_state.decisions[cid] = "agree"
        st.rerun()
    if b2.button("Override", key=f"no_{prefix}_{jd}_{cid}"):
        st.session_state.decisions[cid] = "override"
        st.rerun()


def mark_of(cid):
    return {"agree": "Agreed", "override": "Overridden"}.get(st.session_state.decisions.get(cid), "Pending")


def show_evidence(r, parsed):
    if parsed["evidence"]:
        total = to_float(r.get("evidence_lines"))
        shown = len(parsed["evidence"])
        head = "**Evidence against the JD requirements**"
        st.markdown(f"{head} — showing {shown} of {int(total)}" if total and shown < total else head)
        for e in parsed["evidence"]:
            if e["quote"]:
                st.markdown(f"- ✅ **{e['requirement']}** — resume: *\"{e['quote']}\"*")
            else:
                st.markdown(f"- ❌ **{e['requirement']}** — no supporting quote found")
    if parsed["policy"]:
        st.markdown(f"**Policy clause**: `{parsed['policy']}` — "
                    f"{POLICY_TEXT.get(parsed['policy'], 'unknown clause')}")


for pos, r in enumerate(ranking[jd], start=1):
    cid = r["cv"]
    is_in = (jd, cid) in invited
    hid = (to_float(r.get("hidden_text_chars")) or 0) > HIDDEN_FLAG_CHARS
    score = to_float(r["score"])
    flag = "  HIDDEN TEXT" if hid else ""

    with st.expander(f"#{pos}  {cid}  ·  score {score:.1f}  ·  "
                     f"{'INVITE' if is_in else 'not invited'}  ·  {mark_of(cid)}{flag}"):
        parsed = parse_answer(r.get("answers", ""))

        if not is_in:
            st.warning(f"Agent decision: pass — no mandatory technical requirement unmet. "
                       f"Not invited only because {pos - 1} candidate(s) ranked higher "
                       f"at a {share:.0%} share.")
        else:
            st.caption("Agent decision: pass — no mandatory technical requirement unmet")

        st.markdown("**Score breakdown** (ranked on these in order, not on the total)")
        for key, label, weight in CRITERIA:
            v = to_float(r.get(key))
            if v is None:
                st.markdown(f"- {label}: —")
            else:
                st.markdown(f"- {label}: `{v:.2f}`")
                st.progress(min(max(v, 0.0), 1.0))

        show_evidence(r, parsed)

        if r.get("hidden_text_sample"):
            st.markdown(f"**Hidden text sample**: `{r['hidden_text_sample']}`")

        with st.popover("Scoring detail"):
            st.write(r.get("why", ""))
            if r.get("verifier_notes"):
                st.write(r["verifier_notes"])
            st.code(r.get("answers", ""), language="text")

        decision_buttons(cid, "el")

for r in other_rows:
    cid = r["cv"]
    score = to_float(r.get("score"))
    reason = ("hidden text" if (to_float(r.get("hidden_text_chars")) or 0) > 0
              else "no score" if score is None
              else r.get("error") or "a mandatory requirement is unmet")
    head = f"—  {cid}  ·  " + (f"score {score:.1f}  ·  " if score is not None else "")
    with st.expander(head + f"not eligible ({reason})  ·  {mark_of(cid)}"):
        st.caption(f"Not ranked: {reason}. Shown so nobody disappears from the review.")
        show_evidence(r, parse_answer(r.get("answers", "")))
        decision_buttons(cid, "ne")

for r in invalid_rows:
    cid = r["cv"]
    with st.expander(f"—  {cid}  ·  INVALID ({r.get('error') or '?'})  ·  {mark_of(cid)}"):
        st.warning(f"`{r.get('error')}` — not screened and not auto-rejected.")
        st.caption(f"File: {Path(str(r.get('cv_file', ''))).name}")
        decision_buttons(cid, "inv")

st.divider()

# --- Approval ---------------------------------------------------------------
st.subheader("Approval")

all_cvs = {r["cv"] for r in rows}
pending = sorted(all_cvs - set(st.session_state.decisions))

if finished:
    sent = st.session_state.emails
    if sent:
        st.success(f"Approved at {st.session_state.out.get('invite_share', share):.0%}. {len(sent)} email(s) sent (mock).")
        for e in sent:
            st.text(f"  {e}")
    else:
        st.info("Rejected at the approval gate. No emails sent.")
    if st.button("Start a new batch"):
        for k in ("graph", "cfg", "out", "emails"):
            st.session_state[k] = None
        st.session_state.decisions = {}
        st.rerun()

elif paused:
    shortlist = st.session_state.out["shortlist"]
    n_invite = sum(len(v) for v in shortlist.values())
    n_reject = sum(1 for r in rows if r["agent_label"] in ("0", "1")) - n_invite
    stale = abs(share - current) > 1e-9

    @st.dialog("Send these emails?")
    def confirm_send():
        st.write(f"Invite share: **{current:.0%}**")
        for j in sorted(shortlist):
            names = ", ".join(shortlist[j]) or "nobody"
            st.write(f"**{j}** — invite: {names}")
        st.write(f"Rejection emails to the remaining {n_reject} candidate(s).")
        if pending:
            st.warning(f"{len(pending)} candidate(s) not reviewed yet: {', '.join(pending)}")
        st.caption("Mock emails: logged only, nothing leaves the system.")
        d1, d2 = st.columns(2)
        if d1.button("Confirm and send", type="primary", key="dlg_ok"):
            with st.spinner("Sending"):
                out = resume_run("approve")
                st.session_state.out = out
                st.session_state.emails = out.get("emails", [])
                write_results(out)
            st.rerun()
        if d2.button("Cancel", key="dlg_no"):
            st.rerun()

    st.write(f"Ready to send at {current:.0%}: {n_invite} invite(s), {n_reject} rejection(s).")
    if stale:
        st.warning(f"Slider is at {share:.0%} but the ranking is still at {current:.0%}. Click Re-rank first.")
    if pending:
        st.caption(f"{len(pending)} candidate(s) not reviewed yet: {', '.join(pending)}")

    a1, a2, _ = st.columns([2, 1, 4])
    if a1.button("Approve and send", type="primary", disabled=stale):
        confirm_send()
    if a2.button("Reject"):
        with st.spinner("Cancelling"):
            out = resume_run("reject")
            st.session_state.out = out
            st.session_state.emails = []
            write_results(out)
        st.rerun()

else:
    st.caption("Run a batch above to reach the approval step. "
               "The rows above are the last saved results.")
