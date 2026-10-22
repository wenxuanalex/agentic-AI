"""Batch screening: every CV in <data>/cvs against every JD in <data>/jds.

    START → load_files → screen_pair (one per CV x JD, run in parallel) → rank → approve → send_emails → END

- load_files:  lists the PDFs, skips CV files that look like several CVs (> MAX_CV_PAGES pages),
               and parses every file ONCE (cached), so every pair sees the same parse.
- screen_pair: the normal screening graph (agent + tools + verifier, same output contract,
               planning on/off) with emails switched off, then scoring.score_pair() -> 0-100.
               Sent out with LangGraph's Send (map-reduce): one task per pair, results are
               collected in `results` by the operator.add reducer.
- rank:        per JD, the top `invite_share` (default 20%, set by the user) of that JD's valid applicants by score,
               rounded up, get final label 1 (invited); everyone else gets 0. The agent's own
               LABEL is kept for reference only - the decision comes from the score (agreed 20 Oct).
               Ties are broken by the criteria in priority order.
- approve:     human-in-the-loop pause (interrupt) showing the shortlist. Nothing is sent before it.
               Resume with "approve", "reject", or {"invite_share": 0.3} to re-rank with a new share
               and pause again - no new screening, so changing the share costs nothing.
- send_emails: if approved, email_invite to the shortlist; email_applicant (unsuccessful, policy P11)
               to every other valid applicant of that JD. All emails are mocks.

Run it with run_batch.py.
"""
from __future__ import annotations

import math
import operator
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

import tools
from scoring import CRITERIA, rank_key, score_pair

INVITE_SHARE = 0.20   # DEFAULT share of valid applicants per JD who get final label 1 (user can change it)
MAX_CV_PAGES = 8      # the single CVs are 1-4 pages; applicant-system exports are 18-101
PARSE_WORKERS = 8

_print_lock = threading.Lock()


def log(*a):
    with _print_lock:
        print(*a, flush=True)


class BatchState(TypedDict, total=False):
    data_dir: str
    planning: bool
    cv_ids: list         # optional: only these CVs (e.g. ["C07", "C12"]); empty = all
    jd_ids: list         # optional: only these JDs
    invite_share: float  # share of valid applicants per JD to invite (0-1); default INVITE_SHARE
    cvs: list            # [{"id": "C01", "num": 1, "file": ..., "status": "ok" | "ERROR: ..."}]
    jds: list            # [{"id": "J1", "file": ..., "role": ..., "status": ...}]
    results: Annotated[list, operator.add]   # one row per CV x JD pair (filled in parallel)
    shortlist: dict      # jd id -> [cv ids] to invite, best first
    ranking: dict        # jd id -> all scored cv ids, best first
    approved: bool
    emails: list


# ---------------------------------------------------------------- helpers
def pdf_pages(path: Path) -> int:
    """Rough page count without extra libraries (counts /Type /Page objects)."""
    data = path.read_bytes()
    return len(re.findall(rb"/Type\s*/Page(?![s\w])", data))


def _label(output: str) -> str:
    labels = re.findall(r"LABEL: (\S+)", output)
    return labels[-1].strip() if labels else "invalid"


_graphs: dict[bool, object] = {}
_graphs_lock = threading.Lock()


def screening_graph(planning: bool):
    import agent
    with _graphs_lock:
        if planning not in _graphs:
            _graphs[planning] = agent.build_graph(planning=planning, emails=False)
        return _graphs[planning]


# ---------------------------------------------------------------- nodes
def load_files(state: BatchState):
    tools.enable_parse_cache()
    data = Path(state["data_dir"])
    cv_files = sorted((data / "cvs").glob("*.pdf"))
    jd_files = sorted((data / "jds").glob("*.pdf"))
    if not cv_files or not jd_files:
        raise FileNotFoundError(f"need PDFs in {data/'cvs'} and {data/'jds'}")

    cvs = []
    for i, f in enumerate(cv_files, start=1):
        pages = pdf_pages(f)
        status = (f"ERROR: not_a_resume ({pages} pages - looks like several CVs; skipped)"
                  if pages > MAX_CV_PAGES else "ok")
        cvs.append({"id": f"C{i:02d}", "num": i, "file": str(f), "status": status})
    jds = [{"id": f"J{i}", "file": str(f), "status": "ok"} for i, f in enumerate(jd_files, start=1)]
    # Optional subset (cheap runs). IDs are given BEFORE filtering, so C07 is always the same CV.
    if state.get("cv_ids"):
        cvs = [c for c in cvs if c["id"] in state["cv_ids"]]
    if state.get("jd_ids"):
        jds = [j for j in jds if j["id"] in state["jd_ids"]]

    # Parse every usable file once, in parallel. The cache keeps the result for screen_pair.
    def parse(item, kind):
        tool = tools.parse_resume if kind == "cv" else tools.parse_jd
        out = tool.invoke({"file": item["file"]})
        if out.startswith("ERROR"):
            item["status"] = out.splitlines()[0]
        elif kind == "jd":
            item["role"] = tools.parsed_jd(item["file"]).role or ""
        log(f"parsed {item['id']}: {item['status'][:60]}")

    jobs = [(c, "cv") for c in cvs if c["status"] == "ok"] + [(j, "jd") for j in jds]
    with ThreadPoolExecutor(PARSE_WORKERS) as pool:
        list(pool.map(lambda a: parse(*a), jobs))
    # Debug: save what the agent sees (parser output) - <data>/results/parsed/ (gitignored, private).
    out_dir = data / "results" / "parsed"
    out_dir.mkdir(parents=True, exist_ok=True)
    for item, kind in jobs:
        if item["status"] == "ok":
            tool = tools.parse_resume if kind == "cv" else tools.parse_jd
            (out_dir / f"{item['id']}.txt").write_text(tool.invoke({"file": item["file"]}),encoding="utf-8",)  # cached
    log(f"{sum(c['status'] == 'ok' for c in cvs)}/{len(cvs)} CVs and "
        f"{sum(j['status'] == 'ok' for j in jds)}/{len(jds)} JDs usable")
    return {"cvs": cvs, "jds": jds}


def fan_out(state: BatchState):
    """One Send per CV x JD pair -> screen_pair runs them in parallel."""
    return [Send("screen_pair", {"cv": cv, "jd": jd, "planning": state.get("planning", False)})
            for jd in state["jds"] for cv in state["cvs"]]


def screen_pair(p: dict):
    import adapter
    cv, jd = p["cv"], p["jd"]
    row = {"jd": jd["id"], "cv": cv["id"], "cv_num": cv["num"], "jd_file": Path(jd["file"]).name,
           "cv_file": Path(cv["file"]).name, "label": "invalid", "score": None, "error": "",
           "steps": 0, "tokens_in": 0, "tokens_out": 0, "seconds": 0.0, "verifier_notes": "", "answers": ""}
    for k, _, _ in CRITERIA:
        row[k] = None
    # Guardrail 1: hidden text in the CV is flagged for the human at approval (not auto-rejected).
    hidden = tools.visible_pdf_text(cv["file"])[1]   # cached, no model call
    row["hidden_text_chars"] = hidden["hidden_chars"]
    row["hidden_text_sample"] = hidden["hidden_sample"] if hidden["hidden_chars"] > tools.HIDDEN_FLAG_CHARS else ""
    if cv["status"] != "ok" or jd["status"] != "ok":  # known bad input: no model call
        row["error"] = cv["status"] if cv["status"] != "ok" else jd["status"]
        log(f"{cv['id']} x {jd['id']}: skipped ({row['error'][:50]})")
        return {"results": [row]}

    task = {"resume": cv["file"], "jd": jd["file"], "candidate_id": cv["num"]}
    trace = {}
    r = adapter.run_case(task, graph=screening_graph(p["planning"]), trace=trace)
    # The verifier's feedback notes (HumanMessages after the first task message), for debugging.
    notes = [m.content for m in trace.get("messages", []) if isinstance(m, HumanMessage)][1:]
    row["verifier_notes"] = " || ".join(n.replace("\n", " | ") for n in notes)[:2000]
    # Debug: every answer the model wrote (length + ending), to see e.g. an empty or cut-off answer.
    from langchain_core.messages import AIMessage
    answers = [m for m in trace.get("messages", []) if isinstance(m, AIMessage) and not m.tool_calls
               and m.name not in ("planner", "plan_executor")]
    row["answers"] = " || ".join(
        f"[{len(adapter._text(m.content))} chars, stop={m.response_metadata.get('finish_reason')}] "
        + adapter._text(m.content)[-300:].replace("\n", " | ") for m in answers)[:3000]
    row.update(label=_label(r.output), steps=r.steps, tokens_in=r.input_tokens,
               tokens_out=r.output_tokens, seconds=round(r.latency_s, 1), error=r.error or "")
    row["tool_calls"] = ",".join(r.tool_calls)
    row["evidence_lines"] = len(re.findall(r"^EVIDENCE:", r.output, re.MULTILINE))
    try:
        row["n_requirements"] = len(tools.jd_items(tools.parsed_jd(jd["file"])))
    except Exception:
        row["n_requirements"] = ""
    errs = re.findall(r"ERROR: (\S+)", r.output)
    if errs:
        row["error"] = errs[-1]
    if row["label"] in ("0", "1"):
        try:
            resume_text = tools.parse_resume.invoke({"file": cv["file"]})  # cached
            s = score_pair(tools.parsed_cv(cv["file"]), tools.parsed_jd(jd["file"]), resume_text, r.output,
                           resume_file=cv["file"])
            row["tokens_in"] += s.pop("score_tokens_in", 0)
            row["tokens_out"] += s.pop("score_tokens_out", 0)
            row.update(s)
        except Exception as e:
            import traceback

            row["error"] = f"scoring failed: {type(e).__name__}: {e}"
            traceback.print_exc()
    log(f"{cv['id']} x {jd['id']}: label {row['label']}, score {row['score']}")
    return {"results": [row]}


def check_share(share) -> float:
    """invite_share must be a number in (0, 1]. Raises ValueError otherwise (for the UI to show)."""
    share = float(share)
    if not 0 < share <= 1:
        raise ValueError(f"invite_share must be between 0 and 1 (e.g. 0.2 for 20%), got {share}")
    return share




def rank_results(
    results: list,
    jds: list,
    invite_share: float = INVITE_SHARE,
) -> tuple[dict, dict]:
    """
    Stage 1: Reject candidates who fail mandatory screening.
    Stage 2: Rank eligible candidates by score.
    Stage 3: Invite the top invite_share of eligible candidates,
    rounded up.
    """
    share = check_share(invite_share)
    shortlist = {}
    ranking = {}

    for jd in jds:
        jd_id = jd["id"]

        # Stage 1: Only candidates who pass mandatory screening
        # and have a valid score are eligible.
        eligible = [
            r for r in results
            if r["jd"] == jd_id
            and r["label"] == "1"
            and r["score"] is not None
            and r.get("hidden_text_chars", 0) == 0
        ]

        # Stage 2: Rank eligible candidates by score and tie-break rules.
        eligible.sort(key=rank_key, reverse=True)

        # Store the complete ranking of eligible candidates.
        ranking[jd_id] = [r["cv"] for r in eligible]

        # Stage 3: Calculate the invitation quota from eligible candidates.
        k = math.ceil(share * len(eligible))
        shortlist[jd_id] = [
            r["cv"] for r in eligible[:k]
        ]

    return shortlist, ranking



def rank(state: BatchState):
    share = state.get("invite_share") or INVITE_SHARE
    shortlist, ranking = rank_results(state["results"], state["jds"], share)
    return {"shortlist": shortlist, "ranking": ranking, "invite_share": share}


def approve(state: BatchState):
    """Human approval before ANY email goes out. The pause shows the shortlist and the share.
    Resume with "approve" / "reject", or {"invite_share": x} to re-rank and pause again."""
    decision = interrupt({"shortlist": state["shortlist"], "invite_share": state["invite_share"]})
    if isinstance(decision, dict) and "invite_share" in decision:
        return Command(goto="rank", update={"invite_share": check_share(decision["invite_share"])})
    return Command(goto="send_emails", update={"approved": decision == "approve"})



def send_emails(state: BatchState):
    if not state.get("approved"):
        return {"emails": []}

    sent = []
    roles = {
        j["id"]: j.get("role") or Path(j["file"]).stem
        for j in state["jds"]
    }

    for jd_id, invited in state["shortlist"].items():
        for r in (
            x for x in state["results"]
            if x["jd"] == jd_id and x["label"] in ("0", "1")
        ):
            # Rule 1: Disqualify CVs containing detected hidden text.
            if r.get("hidden_text_chars", 0) > 0:
                log(
                    f"{r['cv']} x {jd_id}: disqualified due to "
                    f"{r['hidden_text_chars']} hidden-text characters; "
                    "no email sent."
                )
                continue

            # Rule 2: Invite shortlisted candidates.
            if r["cv"] in invited:
                tools.email_invite.invoke({
                    "candidate_id": r["cv_num"],
                    "message": (
                        f"We'd like to invite you to interview "
                        f"for {roles[jd_id]}."
                    ),
                })
                sent.append((r["cv"], jd_id, "email_invite"))

            # Rule 3: Reject eligible candidates who were not shortlisted.
            else:
                tools.email_applicant.invoke({
                    "candidate_id": r["cv_num"],
                    "message": (
                        f"Thank you for applying for {roles[jd_id]}. "
                        "We won't be taking your application further."
                    ),
                })
                sent.append((r["cv"], jd_id, "email_applicant"))

    return {"emails": sent}



# ---------------------------------------------------------------- graph
def build_batch_graph():
    b = StateGraph(BatchState)
    b.add_node("load_files", load_files)
    b.add_node("screen_pair", screen_pair)
    b.add_node("rank", rank)
    b.add_node("approve", approve, destinations=("rank", "send_emails"))
    b.add_node("send_emails", send_emails)
    b.add_edge(START, "load_files")
    b.add_conditional_edges("load_files", fan_out, ["screen_pair"])
    b.add_edge("screen_pair", "rank")
    b.add_edge("rank", "approve")   # approve -> send_emails or back to rank: chosen by its Command
    b.add_edge("send_emails", END)
    return b.compile(checkpointer=InMemorySaver())
