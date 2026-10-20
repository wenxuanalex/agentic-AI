"""Optional planning mode (week 5: plan first, check the plan in code, then run it).

Switch it on with build_graph(planning=True) / make_agent_fn(planning=True).

    START → planner → execute_plan → agent → verifier → (email | END | back to agent)

- planner:      ONE small model call (task + tool list only, no policy text) writes a plan:
                which tools to run, in order.
- check_plan:   a hard constraint IN CODE. Only the four loop tools, both parse tools,
                at most 4 steps, no email tools. A rejected plan falls back to DEFAULT_PLAN.
- execute_plan: plain code runs the planned tools - no model call per tool. This is where
                tokens are saved: without a plan, the model re-reads the whole (growing)
                conversation every time it picks a tool.
                If a parse tool returns ERROR, the plan cannot work, so the remaining steps
                are skipped (week 5: "a missing order needs a new plan", here: no plan at all).
- agent:        ONE model call writes the final answer (same format, same verifier).
"""
from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import agent as A

PLAN_TOOLS = ("parse_resume", "parse_jd", "normalize_skills", "check_timeline")
REQUIRED = ("parse_resume", "parse_jd")
DEFAULT_PLAN = ["parse_resume", "parse_jd"]
MAX_STEPS = 4
EXECUTOR = "plan_executor"  # name on the executor's messages (not a model turn)

PLANNER_PROMPT = """You plan tool calls for a resume screening assistant that compares ONE resume with ONE job description (JD).

Tools you may plan (each runs once):
- parse_resume: reads the resume file. Always needed.
- parse_jd: reads the JD and lists its requirements (R1, R2, ...). Always needed.
- check_timeline: works out years of experience, gaps and overlaps from the resume's job dates. Plan it when the JD is likely to ask for years of experience.
- normalize_skills: maps different wordings of skills to standard names. Plan it when skills may be worded differently.

Reply with the tool names to run, in order, one per line. Nothing else."""

ANSWER_NOTE = """

## Planning mode
The tools in your plan have already been run for you. Their results are in the conversation above.
Do not call any tools. Write your final answer now, in the output format above."""


# ---------------------------------------------------------------- plan text -> checked plan
def parse_plan(text: str) -> list[str]:
    """Tool names in the order they first appear in the planner's reply."""
    found = []
    for name in re.findall(r"\b(parse_resume|parse_jd|normalize_skills|check_timeline|email_applicant|email_recruiter)\b", text):
        if name not in found:
            found.append(name)
    return found


def check_plan(plan: list[str]) -> str | None:
    """Hard constraint, in code. Returns the reason a plan is rejected, or None if it may run."""
    if not plan:
        return "empty plan"
    bad = [t for t in plan if t not in PLAN_TOOLS]
    if bad:
        return f"tools not allowed in a plan: {bad}"  # e.g. an email tool
    missing = [t for t in REQUIRED if t not in plan]
    if missing:
        return f"missing required tools: {missing}"
    if len(plan) > MAX_STEPS:
        return f"too many steps ({len(plan)} > {MAX_STEPS})"
    return None


# ---------------------------------------------------------------- nodes
def planner(state: A.State):
    task = state["messages"][0].content
    reply = A.model.invoke([SystemMessage(PLANNER_PROMPT), HumanMessage(task)])
    reply.name = "planner"
    plan = parse_plan(reply.content if isinstance(reply.content, str) else str(reply.content))
    reason = check_plan(plan)
    if reason:
        plan = list(DEFAULT_PLAN)
    # Parse tools always run first, whatever order the planner wrote.
    plan = [t for t in REQUIRED] + [t for t in plan if t not in REQUIRED]
    return {"messages": [reply], "plan": plan, "plan_status": f"rejected ({reason}); used default" if reason else "ok"}


def _requests(calls: list[tuple[str, dict]], step: int) -> AIMessage:
    """Tool requests written by code, in the same shape the model uses."""
    return AIMessage(content="", name=EXECUTOR, tool_calls=[
        {"name": n, "args": a, "id": f"plan-{step}-{i}", "type": "tool_call"} for i, (n, a) in enumerate(calls)])


def execute_plan(state: A.State):
    plan = state["plan"]
    out_messages, updates = [], {}

    # Phase 1: the parse tools (their inputs are the two files).
    phase1 = [(t, {"file": state["resume_file"] if t == "parse_resume" else state["jd_file"]})
              for t in plan if t in REQUIRED]
    req = _requests(phase1, 1)
    res = A.run_tools({"messages": [req]})
    out_messages += [req] + res["messages"]
    updates.update({k: v for k, v in res.items() if k != "messages"})

    # A parse error means the plan cannot work: skip the rest.
    if any(m.content.startswith("ERROR") for m in res["messages"]):
        return {"messages": out_messages, **updates}

    # Phase 2: tools whose input comes from the parse results.
    resume_text = updates.get("resume", "")
    req_values = [re.sub(r"^R\d+:\s*|\s*\((must-have|nice-to-have)\)$", "", line).strip()
                  for line in updates.get("requirement", "").splitlines()]
    phase2 = []
    for t in plan:
        if t == "check_timeline":
            phase2.append((t, {"jobs": resume_text}))
        elif t == "normalize_skills":
            phase2.append((t, {"skills": ", ".join(v for v in req_values if v)}))
    if phase2:
        req2 = _requests(phase2, 2)
        res2 = A.run_tools({"messages": [req2]})
        out_messages += [req2] + res2["messages"]
    return {"messages": out_messages, **updates}


# The answer model keeps the tool definitions (OpenAI expects them when the conversation
# contains tool calls) but is not allowed to call any.
_answer_model = A.model.bind_tools(list(A.tools_by_name[t] for t in PLAN_TOOLS), tool_choice="none")


def answer(state: A.State):
    reply = _answer_model.invoke([SystemMessage(A.SYSTEM + ANSWER_NOTE)] + state["messages"])
    return {"messages": [reply]}
