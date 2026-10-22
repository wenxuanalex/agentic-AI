
"""
Batch screening: every CV in <data>/cvs against every JD in <data>/jds.

START -> load_files -> screen_pair (one per CV x JD, parallel)
      -> rank -> approve -> send_emails -> END

- load_files:
    Lists PDFs, skips CV files that look like multiple CVs (> MAX_CV_PAGES
    pages), and parses each usable file once using the parse cache.

- screen_pair:
    Checks for hidden text BEFORE model calls. Any detected hidden text
    disqualifies the CV. Otherwise runs the screening graph and scoring.

- rank:
    Ranks only candidates who pass screening, have a valid score, have
    no recorded errors, and have zero detected hidden-text characters.

- approve:
    Human-in-the-loop pause before any emails are sent.

- send_emails:
    Sends mock invitations to shortlisted candidates and mock rejection
    emails to other valid applicants. Disqualified candidates receive
    neither email.

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


INVITE_SHARE = 0.20
MAX_CV_PAGES = 8
PARSE_WORKERS = 8

_print_lock = threading.Lock()


def log(*a):
    with _print_lock:
        print(*a, flush=True)


class BatchState(TypedDict, total=False):
    data_dir: str
    planning: bool
    cv_ids: list
    jd_ids: list
    invite_share: float
    cvs: list
    jds: list
    results: Annotated[list, operator.add]
    shortlist: dict
    ranking: dict
    approved: bool
    emails: list


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def pdf_pages(path: Path) -> int:
    """Rough page count without an additional PDF library."""
    data = path.read_bytes()
    return len(re.findall(rb"/Type\s*/Page(?![s\w])", data))


def _label(output: str) -> str:
    """Extract the final LABEL field from the agent output."""
    labels = re.findall(r"LABEL: (\S+)", output)
    return labels[-1].strip() if labels else "invalid"


def hidden_text_count(row: dict) -> int:
    """Safely normalize a row's hidden-text character count."""
    try:
        return int(row.get("hidden_text_chars", 0) or 0)
    except (TypeError, ValueError):
        # Fail closed if a recorded count is malformed.
        return -1


_graphs: dict[bool, object] = {}
_graphs_lock = threading.Lock()


def screening_graph(planning: bool):
    """Build and cache the screening graph for each planning mode."""
    import agent

    with _graphs_lock:
        if planning not in _graphs:
            _graphs[planning] = agent.build_graph(
                planning=planning,
                emails=False,
            )
        return _graphs[planning]


# ---------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------

def load_files(state: BatchState):
    """List files, identify usable inputs, and populate the parse cache."""
    tools.enable_parse_cache()

    data = Path(state["data_dir"])
    cv_files = sorted((data / "cvs").glob("*.pdf"))
    jd_files = sorted((data / "jds").glob("*.pdf"))

    if not cv_files or not jd_files:
        raise FileNotFoundError(
            f"need PDFs in {data / 'cvs'} and {data / 'jds'}"
        )

    cvs = []

    for i, f in enumerate(cv_files, start=1):
        try:
            pages = pdf_pages(f)
            status = (
                f"ERROR: not_a_resume ({pages} pages - "
                "looks like several CVs; skipped)"
                if pages > MAX_CV_PAGES
                else "ok"
            )
        except Exception as e:
            status = (
                f"ERROR: pdf_page_count_failed: "
                f"{type(e).__name__}: {e}"
            )

        cvs.append({
            "id": f"C{i:02d}",
            "num": i,
            "file": str(f),
            "status": status,
        })

    jds = [
        {
            "id": f"J{i}",
            "file": str(f),
            "status": "ok",
        }
        for i, f in enumerate(jd_files, start=1)
    ]

    # IDs are assigned before filtering, so subset IDs remain stable.
    if state.get("cv_ids"):
        cvs = [
            c for c in cvs
            if c["id"] in state["cv_ids"]
        ]

    if state.get("jd_ids"):
        jds = [
            j for j in jds
            if j["id"] in state["jd_ids"]
        ]

    def parse(item, kind):
        parser = (
            tools.parse_resume
            if kind == "cv"
            else tools.parse_jd
        )

        try:
            out = parser.invoke({"file": item["file"]})

            if out.startswith("ERROR"):
                item["status"] = out.splitlines()[0]
            elif kind == "jd":
                item["role"] = (
                    tools.parsed_jd(item["file"]).role or ""
                )

        except Exception as e:
            item["status"] = (
                f"ERROR: parse_failed: {type(e).__name__}: {e}"
            )

        log(f"parsed {item['id']}: {item['status'][:100]}")

    # Parse valid CVs and all selected JDs in parallel.
    jobs = (
        [
            (c, "cv")
            for c in cvs
            if c["status"] == "ok"
        ]
        + [
            (j, "jd")
            for j in jds
            if j["status"] == "ok"
        ]
    )

    with ThreadPoolExecutor(max_workers=PARSE_WORKERS) as pool:
        list(pool.map(lambda args: parse(*args), jobs))

    # Save parsed text for debugging.
    out_dir = data / "results" / "parsed"
    out_dir.mkdir(parents=True, exist_ok=True)

    for item, kind in jobs:
        if item["status"] != "ok":
            continue

        parser = (
            tools.parse_resume
            if kind == "cv"
            else tools.parse_jd
        )

        try:
            parsed_text = parser.invoke({"file": item["file"]})
            (out_dir / f"{item['id']}.txt").write_text(
                parsed_text,
                encoding="utf-8",
            )
        except Exception as e:
            log(
                f"WARNING: could not save parsed text for "
                f"{item['id']}: {type(e).__name__}: {e}"
            )

    log(
        f"{sum(c['status'] == 'ok' for c in cvs)}/{len(cvs)} CVs "
        f"and {sum(j['status'] == 'ok' for j in jds)}/{len(jds)} "
        "JDs usable"
    )

    return {"cvs": cvs, "jds": jds}


def fan_out(state: BatchState):
    """Create one parallel screening task for every CV x JD pair."""
    return [
        Send(
            "screen_pair",
            {
                "cv": cv,
                "jd": jd,
                "planning": state.get("planning", False),
            },
        )
        for jd in state["jds"]
        for cv in state["cvs"]
    ]


def screen_pair(p: dict):
    """
    Screen one CV/JD pair.

    Security policy:
    Any detected hidden text disqualifies the CV before model calls.
    If hidden-text detection fails, do not proceed with screening.
    """
    import adapter

    cv = p["cv"]
    jd = p["jd"]

    row = {
        "jd": jd["id"],
        "cv": cv["id"],
        "cv_num": cv["num"],
        "jd_file": Path(jd["file"]).name,
        "cv_file": Path(cv["file"]).name,
        "label": "invalid",
        "score": None,
        "error": "",
        "steps": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "seconds": 0.0,
        "verifier_notes": "",
        "answers": "",
        "hidden_text_chars": 0,
        "hidden_text_sample": "",
    }

    for criterion, _, _ in CRITERIA:
        row[criterion] = None

    # Check hidden text before calling the LLM or scoring model.
    try:
        hidden = tools.visible_pdf_text(cv["file"])[1]
        count = int(hidden.get("hidden_chars", 0) or 0)

        if count < 0:
            raise ValueError("negative hidden-text count")

        row["hidden_text_chars"] = count

        # Preserve the sample for any positive detection, not only
        # detections exceeding the display warning threshold.
        row["hidden_text_sample"] = (
            hidden.get("hidden_sample", "")
            if count > 0
            else ""
        )

    except Exception as e:
        row["label"] = "invalid"
        row["error"] = (
            f"hidden_text_detection_failed: "
            f"{type(e).__name__}: {e}"
        )

        # Negative sentinel ensures ranking fails closed.
        row["hidden_text_chars"] = -1

        log(f"{cv['id']} x {jd['id']}: {row['error']}")
        return {"results": [row]}

    # Any positive hidden-text count means disqualification.
    if row["hidden_text_chars"] > 0:
        row["label"] = "0"
        row["error"] = "disqualified_hidden_text"

        log(
            f"{cv['id']} x {jd['id']}: DISQUALIFIED before model call "
            f"({row['hidden_text_chars']} hidden-text characters)"
        )

        return {"results": [row]}

    # Skip known invalid inputs.
    if cv["status"] != "ok" or jd["status"] != "ok":
        row["error"] = (
            cv["status"]
            if cv["status"] != "ok"
            else jd["status"]
        )

        log(
            f"{cv['id']} x {jd['id']}: "
            f"skipped ({row['error'][:80]})"
        )

        return {"results": [row]}

    task = {
        "resume": cv["file"],
        "jd": jd["file"],
        "candidate_id": cv["num"],
    }

    trace = {}

    try:
        r = adapter.run_case(
            task,
            graph=screening_graph(p["planning"]),
            trace=trace,
        )
    except Exception as e:
        row["error"] = (
            f"screening_failed: {type(e).__name__}: {e}"
        )
        log(f"{cv['id']} x {jd['id']}: {row['error']}")
        return {"results": [row]}

    # Verifier feedback notes.
    notes = [
        m.content
        for m in trace.get("messages", [])
        if isinstance(m, HumanMessage)
    ][1:]

    row["verifier_notes"] = " || ".join(
        n.replace("\n", " | ")
        for n in notes
    )[:2000]

    # Capture model answers for debugging.
    from langchain_core.messages import AIMessage

    answers = [
        m
        for m in trace.get("messages", [])
        if isinstance(m, AIMessage)
        and not m.tool_calls
        and m.name not in ("planner", "plan_executor")
    ]

    row["answers"] = " || ".join(
        (
            f"[{len(adapter._text(m.content))} chars, "
            f"stop={m.response_metadata.get('finish_reason')}] "
            + adapter._text(m.content)[-300:].replace("\n", " | ")
        )
        for m in answers
    )[:3000]

    row.update(
        label=_label(r.output),
        steps=r.steps,
        tokens_in=r.input_tokens,
        tokens_out=r.output_tokens,
        seconds=round(r.latency_s, 1),
        error=r.error or "",
    )

    row["tool_calls"] = ",".join(r.tool_calls)
    row["evidence_lines"] = len(
        re.findall(r"^EVIDENCE:", r.output, re.MULTILINE)
    )

    try:
        row["n_requirements"] = len(
            tools.jd_items(tools.parsed_jd(jd["file"]))
        )
    except Exception:
        row["n_requirements"] = ""

    errors = re.findall(r"ERROR: (\S+)", r.output)
    if errors:
        row["error"] = errors[-1]

    # Score only valid screening labels.
    if row["label"] in ("0", "1") and not row["error"]:
        try:
            resume_text = tools.parse_resume.invoke(
                {"file": cv["file"]}
            )

            score_result = score_pair(
                tools.parsed_cv(cv["file"]),
                tools.parsed_jd(jd["file"]),
                resume_text,
                r.output,
                resume_file=cv["file"],
            )

            row["tokens_in"] += score_result.pop(
                "score_tokens_in", 0
            )
            row["tokens_out"] += score_result.pop(
                "score_tokens_out", 0
            )
            row.update(score_result)

        except Exception as e:
            row["error"] = (
                f"scoring_failed: {type(e).__name__}: {e}"
            )

            import traceback
            traceback.print_exc()

    log(
        f"{cv['id']} x {jd['id']}: "
        f"label {row['label']}, score {row['score']}"
    )

    return {"results": [row]}


def check_share(share) -> float:
    """Validate invitation share."""
    share = float(share)

    if not 0 < share <= 1:
        raise ValueError(
            "invite_share must be between 0 and 1 "
            f"(e.g. 0.2 for 20%), got {share}"
        )

    return share


def rank_results(
    results: list,
    jds: list,
    invite_share: float = INVITE_SHARE,
) -> tuple[dict, dict]:
    """
    Rank only candidates who pass screening, have a valid score,
    have no screening/scoring errors, and contain no hidden text.
    """
    share = check_share(invite_share)

    shortlist = {}
    ranking = {}

    for jd in jds:
        jd_id = jd["id"]

        eligible = [
            r
            for r in results
            if r.get("jd") == jd_id
            and str(r.get("label")) == "1"
            and r.get("score") is not None
            and not r.get("error")
            and hidden_text_count(r) == 0
        ]

        # Establish deterministic tie order before stable score sorting.
        eligible.sort(key=lambda r: str(r.get("cv", "")))
        eligible.sort(key=rank_key, reverse=True)

        ranking[jd_id] = [
            r["cv"] for r in eligible
        ]

        # Calculate the quota only from eligible candidates.
        k = math.ceil(share * len(eligible))

        shortlist[jd_id] = [
            r["cv"] for r in eligible[:k]
        ]

        log(
            f"{jd_id}: {len(eligible)} eligible, "
            f"{k} shortlisted at {share:.0%}"
        )

    return shortlist, ranking


def rank(state: BatchState):
    share = state.get("invite_share", INVITE_SHARE)

    if share is None:
        share = INVITE_SHARE

    shortlist, ranking = rank_results(
        state["results"],
        state["jds"],
        share,
    )

    return {
        "shortlist": shortlist,
        "ranking": ranking,
        "invite_share": share,
    }


def approve(state: BatchState):
    """
    Pause for human approval before any email is sent.

    Resume with:
      "approve"
      "reject"
      {"invite_share": 0.3}
    """
    decision = interrupt({
        "shortlist": state["shortlist"],
        "invite_share": state["invite_share"],
    })

    if isinstance(decision, dict) and "invite_share" in decision:
        return Command(
            goto="rank",
            update={
                "invite_share": check_share(
                    decision["invite_share"]
                )
            },
        )

    return Command(
        goto="send_emails",
        update={"approved": decision == "approve"},
    )


def send_emails(state: BatchState):
    """Send mock emails only after explicit human approval."""
    if not state.get("approved"):
        log("Email dispatch rejected or not approved; no emails sent.")
        return {"emails": []}

    sent = []

    roles = {
        j["id"]: j.get("role") or Path(j["file"]).stem
        for j in state["jds"]
    }

    for jd_id, invited in state["shortlist"].items():
        candidates = [
            row
            for row in state["results"]
            if row.get("jd") == jd_id
        ]

        for row in candidates:
            # Never email disqualified or invalid candidates.
            hidden_count = hidden_text_count(row)

            if (
                hidden_count != 0
                or row.get("error") == "disqualified_hidden_text"
                or row.get("label") == "invalid"
            ):
                log(
                    f"{row.get('cv')} x {jd_id}: "
                    "not emailed (disqualified or invalid)"
                )
                continue

            # Do not email candidates whose screening/scoring failed.
            if row.get("error"):
                log(
                    f"{row.get('cv')} x {jd_id}: "
                    f"not emailed (error: {row['error']})"
                )
                continue

            if row.get("score") is None:
                log(
                    f"{row.get('cv')} x {jd_id}: "
                    "not emailed (missing score)"
                )
                continue

            candidate_id = row["cv_num"]

            if row["cv"] in invited:
                tools.email_invite.invoke({
                    "candidate_id": candidate_id,
                    "message": (
                        f"We'd like to invite you to interview "
                        f"for {roles[jd_id]}."
                    ),
                })

                sent.append((
                    row["cv"],
                    jd_id,
                    "email_invite",
                ))

                log(f"{row['cv']} x {jd_id}: mock invitation sent")

            else:
                tools.email_applicant.invoke({
                    "candidate_id": candidate_id,
                    "message": (
                        f"Thank you for applying for {roles[jd_id]}. "
                        "We won't be taking your application further."
                    ),
                })

                sent.append((
                    row["cv"],
                    jd_id,
                    "email_applicant",
                ))

                log(f"{row['cv']} x {jd_id}: mock rejection sent")

    log(f"Mock emails sent: {len(sent)}")

    return {"emails": sent}


# ---------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------

def build_batch_graph():
    b = StateGraph(BatchState)

    b.add_node("load_files", load_files)
    b.add_node("screen_pair", screen_pair)
    b.add_node("rank", rank)
    b.add_node(
        "approve",
        approve,
        destinations=("rank", "send_emails"),
    )
    b.add_node("send_emails", send_emails)

    b.add_edge(START, "load_files")

    b.add_conditional_edges(
        "load_files",
        fan_out,
        ["screen_pair"],
    )

    b.add_edge("screen_pair", "rank")
    b.add_edge("rank", "approve")
    b.add_edge("send_emails", END)

    return b.compile(
        checkpointer=InMemorySaver()
    )
