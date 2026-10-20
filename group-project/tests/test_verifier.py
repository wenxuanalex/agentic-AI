"""Guardrail 1 in the verifier: quotes must come from the VISIBLE text."""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agent

REQS = "R1: Python (must-have)\nR2: Spark (must-have)"


def _state(resume_file, r1_quote, r2_quote, parser_text):
    answer = (f'EVIDENCE: [R1: Python] "{r1_quote}"\n'
              f'EVIDENCE: [R2: Spark] "{r2_quote}"\n'
              "POLICY: P2\nLABEL: 1")
    return {"messages": [HumanMessage("Screen the resume"),
                         ToolMessage(parser_text, name="parse_resume", tool_call_id="1"),
                         ToolMessage(REQS, name="parse_jd", tool_call_id="2"),
                         AIMessage(answer)],
            "resume": parser_text, "requirement": REQS, "resume_file": str(resume_file), "retry": 0}


def test_white_text_quote_is_rejected(pdfs):
    # Even if the parser's text contains the hidden words, they are not visible on the page.
    parser_text = "Built dashboards and data pipelines in Python and SQL\nPython SQL Spark"
    out = agent.verifier(_state(pdfs["white"], "Built dashboards and data pipelines in Python and SQL",
                                "Python SQL Spark", parser_text))
    assert out["pass_fail"] is False
    note = out["messages"][0].content
    assert "Python SQL Spark not actually found" in note
    assert "Built dashboards" not in note  # the visible quote was accepted


def test_visible_quotes_pass(pdfs):
    out = agent.verifier(_state(pdfs["white"], "Built dashboards and data pipelines in Python and SQL",
                                "Skills: Python, SQL, Excel", "anything"))
    assert out == {"pass_fail": True, "label": "1"}
