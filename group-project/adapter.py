"""Evaluation adapter: runs the screening graph for one test case and returns an
AgentResult, as promised in the output contract (section 4).

For the evaluator:
    from adapter import make_agent_fn
    agent_fn = make_agent_fn()                 # build the graph once, reuse it
    report = run_eval(agent_fn, cases, grader) # week 3 / week 6 harness

Case input (case.input) is a JSON string (or a dict) naming the two PDFs:
    {"resume": "data/cv/r12.pdf", "jd": "data/jd/j3.pdf", "candidate_id": 12}
candidate_id is optional (a running number is used if missing).

If the verifier still fails after all retries, output is
    ERROR: verification_failed
    LABEL: invalid
(no decision, no email). AgentResult.error is only for real crashes.

Recruiter emails pause for human approval (interrupt). This adapter AUTO-APPROVES
every pause by default (resume_value="approve"), like week 6's make_agent_fn.
Say so in the write-up: the measured runs never exercise a human rejection.
"""
from __future__ import annotations

import itertools
import json
import time
import uuid
from dataclasses import dataclass, field, fields

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

import agent as screening

# ---------------------------------------------------------------- AgentResult
# Use the evaluator's class if available, else an identical stand-in (week 6 fields).
try:
    from agentlab import AgentResult  # week 3 harness
except ImportError:
    try:
        from lab_support import AgentResult  # week 6 stand-in
    except ImportError:
        @dataclass
        class AgentResult:
            output: str
            steps: int = 0
            tool_calls: list = field(default_factory=list)   # tool NAMES, in order
            tool_args: list = field(default_factory=list)    # their arguments, same order
            input_tokens: int = 0
            output_tokens: int = 0
            cost_usd: float = 0.0
            latency_s: float = 0.0
            error: str | None = None


def _make_result(**kw) -> AgentResult:
    """Build AgentResult with only the fields this harness version has
    (week 3's AgentResult has no tool_args; week 6's does)."""
    allowed = {f.name for f in fields(AgentResult)}
    return AgentResult(**{k: v for k, v in kw.items() if k in allowed})


# ---------------------------------------------------------------- settings
# USD per 1M tokens. CHECK these against your provider before reporting cost:
# gpt-5-mini launched at $0.25 in / $2.00 out (one tracker lists $0.125 / $1.00, Oct 2026),
# and a proxy set via OPENAI_API_BASE_URL may charge differently.
PRICE_IN_PER_M = 0.25
PRICE_OUT_PER_M = 2.00
RECURSION_LIMIT = 40  # max node runs per case; a safety net, not the retry rule
# Output when the verifier still fails after all retries (agreed contract addition).
RETRIES_EXHAUSTED_OUTPUT = "ERROR: verification_failed\nLABEL: invalid"

_ids = itertools.count(1)


# ---------------------------------------------------------------- helpers
def _parse_input(task) -> tuple[str, str, int]:
    data = json.loads(task) if isinstance(task, str) else dict(task)
    return data["resume"], data["jd"], int(data.get("candidate_id") or next(_ids))


def _text(content) -> str:
    """Message content as plain text (handles list-of-blocks content too)."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


def _to_result(messages: list, candidate_id: int, latency: float) -> AgentResult:
    ai_all = [m for m in messages if isinstance(m, AIMessage)]
    ai = [m for m in ai_all if m.name != "plan_executor"]  # model turns only

    # output = the agent's LAST answer (an AIMessage with no tool requests),
    # not messages[-1], which is the email ToolMessage when an email was sent.
    finals = [m for m in ai if not m.tool_calls]
    output = _text(finals[-1].content) if finals else ""

    # tool_calls / tool_args: one entry per tool that actually ran, in order.
    args_by_id = {tc["id"]: tc["args"] for m in ai_all for tc in m.tool_calls}
    tool_calls, tool_args = [], []
    for m in messages:
        if isinstance(m, ToolMessage):
            tool_calls.append(m.name)
            if m.name in ("email_applicant", "email_recruiter"):
                tool_args.append({"candidate_id": candidate_id})  # called by the email node, not the model
            else:
                tool_args.append(args_by_id.get(m.tool_call_id, {}))

    tin = sum((m.usage_metadata or {}).get("input_tokens", 0) for m in ai)
    tout = sum((m.usage_metadata or {}).get("output_tokens", 0) for m in ai)
    return _make_result(
        output=output,
        steps=len(ai),  # model turns
        tool_calls=tool_calls,
        tool_args=tool_args,
        input_tokens=tin,
        output_tokens=tout,
        cost_usd=(tin * PRICE_IN_PER_M + tout * PRICE_OUT_PER_M) / 1_000_000,
        latency_s=latency,
    )


# ---------------------------------------------------------------- public API
def run_case(task, graph=None, resume_value="approve", planning=False, trace: dict | None = None) -> AgentResult:
    """Run one case end to end and return its AgentResult. Crashes go in .error.
    trace: optional dict; if given, the final messages are stored in trace["messages"] (for debugging)."""
    t0 = time.perf_counter()
    graph = graph or screening.build_graph(planning=planning)
    cfg = {"configurable": {"thread_id": f"eval-{uuid.uuid4().hex[:8]}"},  # fresh thread per run
           "recursion_limit": RECURSION_LIMIT}
    try:
        resume, jd, candidate_id = _parse_input(task)
        start = HumanMessage(f"Screen the resume at {resume} against the job description at {jd}.")
        out = graph.invoke({"messages": [start], "candidate_id": candidate_id,
                            "resume_file": resume, "jd_file": jd}, cfg)
        while "__interrupt__" in out:  # recruiter-email approval pause
            out = graph.invoke(Command(resume=resume_value), cfg)
    except Exception as e:
        return _make_result(output="", error=f"{type(e).__name__}: {e}",
                           latency_s=time.perf_counter() - t0)
    if trace is not None:
        trace["messages"] = out["messages"]
    result = _to_result(out["messages"], candidate_id, time.perf_counter() - t0)
    if out.get("pass_fail") is False:
        # Retries ran out and the verifier never passed: no decision was made and no
        # email was sent. Report it as its own ERROR code (not a crash in .error).
        result.output = RETRIES_EXHAUSTED_OUTPUT
    return result


def make_agent_fn(resume_value="approve", planning=False):
    """agent_fn for run_eval: builds the graph once and reuses it across cases.
    planning=True runs the planning version (see planning.py) - use it for the baseline comparison."""
    graph = screening.build_graph(planning=planning)
    return lambda task: run_case(task, graph=graph, resume_value=resume_value)
