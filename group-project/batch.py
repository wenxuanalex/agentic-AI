"""Batch screening: every CV in <data>/cvs against every JD in <data>/jds.

    START → load_files → screen_pair (one per CV x JD, run in parallel) → rank → approve → send_emails → END

- load_files:  lists the PDFs, skips CV files that look like several CVs (> MAX_CV_PAGES pages),
               and parses every file ONCE (cached), so every pair sees the same parse.
- screen_pair: the normal screening graph (agent + tools + verifier, same output contract,
               planning on/off) with emails switched off, then scoring.score_pair() -> 0-100.
               Sent out with LangGraph's Send (map-reduce): one task per pair, results are
               collected in `results` by the operator.add reducer.
- rank:        per JD, invite the top INVITE_SHARE (10%) of that JD's valid applicants,
               rounded up, chosen only from LABEL 1 candidates (must-haves are a hard rule;
               the score only ranks). Ties are broken by the criteria in priority order.
- approve:     human-in-the-loop pause (interrupt) showing the shortlist. Nothing is sent before it.
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
from langgraph.types import Send, interrupt

import tools
from scoring import CRITERIA, rank_key, score_pair

INVITE_SHARE = 0.10
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
    cvs: list            # [{"id": "C01", "num": 1, "file": ..., "status": "ok" | "ERROR: ..."}]
    jds: list            # [{"id": "J1", "file": ..., "role": ..., "status": ...}]
    results: Annotated[list, operator.add]   # one row per CV x JD pair (filled in parallel)
    shortlist: dict      # jd id -> [cv ids] to invite, best first
    ranking: dict        # jd id -> all LABEL 1 cv ids, best first
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
           "steps": 0, "tokens_in": 0, "tokens_out": 0, "seconds": 0.0, "verifier_notes": ""}
    for k, _, _ in CRITERIA:
        row[k] = None
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
    row.update(label=_label(r.output), steps=r.steps, tokens_in=r.input_tokens,
               tokens_out=r.output_tokens, seconds=round(r.latency_s, 1), error=r.error or "")
    errs = re.findall(r"ERROR: (\S+)", r.output)
    if errs:
        row["error"] = errs[-1]
    if row["label"] in ("0", "1"):
        try:
            resume_text = tools.parse_resume.invoke({"file": cv["file"]})  # cached
            s = score_pair(tools.parsed_cv(cv["file"]), tools.parsed_jd(jd["file"]), resume_text, r.output)
            row["tokens_in"] += s.pop("score_tokens_in", 0)
            row["tokens_out"] += s.pop("score_tokens_out", 0)
            row.update(s)
        except Exception as e:  # scoring failed: keep the label, no score
            row["error"] = f"scoring failed: {type(e).__name__}: {e}"
    log(f"{cv['id']} x {jd['id']}: label {row['label']}, score {row['score']}")
    return {"results": [row]}


def rank(state: BatchState):
    shortlist, ranking = {}, {}
    for jd in state["jds"]:
        valid = [r for r in state["results"] if r["jd"] == jd["id"] and r["label"] in ("0", "1")]
        k = math.ceil(INVITE_SHARE * len(valid)) if valid else 0
        eligible = sorted([r for r in valid if r["label"] == "1" and r["score"] is not None],
                          key=rank_key, reverse=True)
        ranking[jd["id"]] = [r["cv"] for r in eligible]   # LABEL 1 candidates, best first
        shortlist[jd["id"]] = ranking[jd["id"]][:k]
    return {"shortlist": shortlist, "ranking": ranking}


def approve(state: BatchState):
    """Human approval before ANY email goes out."""
    decision = interrupt({"shortlist": state["shortlist"]})
    return {"approved": decision == "approve"}


def send_emails(state: BatchState):
    if not state.get("approved"):
        return {"emails": []}
    sent = []
    roles = {j["id"]: j.get("role") or Path(j["file"]).stem for j in state["jds"]}
    for jd_id, invited in state["shortlist"].items():
        for r in (x for x in state["results"] if x["jd"] == jd_id and x["label"] in ("0", "1")):
            if r["cv"] in invited:
                tools.email_invite.invoke({"candidate_id": r["cv_num"],
                                           "message": f"We'd like to invite you to interview for {roles[jd_id]}."})
                sent.append((r["cv"], jd_id, "email_invite"))
            else:
                tools.email_applicant.invoke({"candidate_id": r["cv_num"],
                                              "message": f"Thank you for applying for {roles[jd_id]}. We won't be taking your application further."})
                sent.append((r["cv"], jd_id, "email_applicant"))
    return {"emails": sent}


# ---------------------------------------------------------------- graph
def build_batch_graph():
    b = StateGraph(BatchState)
    b.add_node("load_files", load_files)
    b.add_node("screen_pair", screen_pair)
    b.add_node("rank", rank)
    b.add_node("approve", approve)
    b.add_node("send_emails", send_emails)
    b.add_edge(START, "load_files")
    b.add_conditional_edges("load_files", fan_out, ["screen_pair"])
    b.add_edge("screen_pair", "rank")
    b.add_edge("rank", "approve")
    b.add_edge("approve", "send_emails")
    b.add_edge("send_emails", END)
    return b.compile(checkpointer=InMemorySaver())
