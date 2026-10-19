"""Resume Screening Agent — LangGraph nodes (group project, LangGraph portion).

Status: State, run_tools and verifier are written and tested.
To do: edge function after the verifier, email node, agent node + system prompt,
graph wiring, adapter to AgentResult, the six tools.
"""
import re
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

# ---------------------------------------------------------------- constants
POLICY_CLAUSES = ("P1", "P2", "P3", "P4", "P5", "P6")
RETRY_LIMIT = 2
tools_by_name = {}  # TODO: map each tool's name to the tool (all six tools)


# ---------------------------------------------------------------- state
class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    resume: str
    requirement: str
    retry: int
    pass_fail: bool
    label: str


# ---------------------------------------------------------------- nodes
def run_tools(state: State):
    # 0. create an empty list called message for the ToolMessages
    message = []  # for ToolMessages
    resume_content = None
    jd_requirements = None
    # 1. Extract the model's tool requests from the last message's .tool_calls
    requests = state["messages"][-1].tool_calls
    # 2. For each tool call in the requests, run the tool with its args using .invoke
    for request in requests:
        tool = tools_by_name[request["name"]]
        result = tool.invoke(request["args"])
        # 3. Wrap the result in a ToolMessage, labelled with the tool's name and the request's id
        tool_result = ToolMessage(content=str(result), name=request["name"], tool_call_id=request["id"])
        # 4. store the ToolMessage in message
        message.append(tool_result)
        # 5. if the tool's name is parse_resume, store the resume content in a variable called resume_content
        if request["name"] == "parse_resume":
            resume_content = str(result)
        # 6. if the tool's name is parse_jd, store the jd requirements in a variable called jd_requirements
        if request["name"] == "parse_jd":
            jd_requirements = str(tool_result.content)
    # 7. return messages: message
    #    also include resume: resume_content, only if parse_resume ran in this run
    #    also include requirement: jd_requirements, only if parse_jd ran in this run
    dict = {"messages": message}
    if resume_content is not None:
        dict["resume"] = resume_content
    if jd_requirements is not None:
        dict["requirement"] = jd_requirements
    return dict


def verifier(state: State):
    # 1. grab the last message from state
    message = state["messages"][-1]
    # 2. check the tool call record that both parse_resume and parse_jd were called
    errors = []
    tools = []
    label = None
    for call in state["messages"]:
        tools.append(call.name)
    if "parse_resume" in tools and "parse_jd" in tools:
        pass
    else:
        errors.append("parse_resume and/or parse_jd tool not called!")
    # 3. if parse_resume tool was called, check if every evidence quote is actually in the resume text
    if "parse_resume" in tools:
        resume_content = re.findall('EVIDENCE.*"(.*)"', message.content)
        for line in resume_content:
            if line not in state["resume"]:
                errors.append("Extracted resume content " + line + " not actually found in resume!")
    # 4. check that the last message in messages contains 'LABEL:' with a valid label value
    labels = re.findall("LABEL: (.*)", message.content)
    if not labels:
        errors.append("No label values found!")
    else:
        label = labels[0].strip()
        valid_labels = ("0", "1", "invalid")
        if label not in valid_labels:
            errors.append("Label value (" + label + ") is not valid!")
    # 5. if parse_jd tool was called, check that the requirement numbers match the requirement list
    if "parse_jd" in tools and label != 'invalid':
        evidence_r = re.findall(r"EVIDENCE: \[R(\d+)", message.content)
        state_requirements = re.findall(r"R(\d+)", state["requirement"])
        if sorted(evidence_r) != sorted(state_requirements):
            errors.append("Evidence list " + str(evidence_r) + " does not match with requirements list " + str(state_requirements))
    # 6. check that 'ERROR:' does not exist, unless the label value is 'invalid'; check that 'ERROR:' exists if the label value is 'invalid'
    error_lines = re.findall(r"ERROR:", message.content)
    if label:
        if label == 'invalid' and not error_lines:
            errors.append("Label is invalid but there is no error message, please double check both label and if an error message should exist")
        elif error_lines and label != 'invalid':
            errors.append('Error messages found but label value is not invalid, please double check both')
    # 7. if label is not invalid, check that the 'POLICY:' line exists in the last message and that the clause it states falls within the legal range
    if label and label != 'invalid':
        policies = re.findall(r"POLICY: (P\d+)", message.content)
        if not policies:
            errors.append("Company policy lines not found!")
        else:
            policy = policies[0].strip()
            if policy not in POLICY_CLAUSES:
                errors.append("You mentioned a policy not actually found in the company's official policy, please double check")
    # 8. the verifier updates two keys, pass/fail and label accordingly
    if not errors:
        return {"pass_fail": True, "label": label}
    # 9. if one or more checks fail, the verifier updates the state's pass/fail field to fail
    # 10. if one or more checks fail, increment retry count by 1; if the retry count was below the limit before incrementing, attach error message(s) to messages
    retry_limit = state.get("retry", 0)
    updates = {"pass_fail": False, "retry": retry_limit + 1}
    if retry_limit < RETRY_LIMIT:
        note = "\n".join(errors)
        updates["messages"] = [HumanMessage(content=note)]
    return updates
