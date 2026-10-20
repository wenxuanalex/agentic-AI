# Resume Screening Agent — LangGraph part

Screens ONE resume against ONE job description (JD) and returns a decision in the format agreed in
*Resume Screening Agent: Output Contract for Evaluation*. Owner of this folder: Alex.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in this folder (never commit it):

```
OPENAI_API_KEY=sk-...
# OPENAI_API_BASE_URL=...   (optional, only if the group uses a proxy)
```

## Run it

```bash
python run_one.py <resume.pdf> <jd.pdf> [candidate_id]   # one case, prints the AgentResult
python run_checks.py [data_folder]                       # 3 real-run checks (reject / advance / invalid)
```

## For the evaluator: how to plug into `run_eval`

```python
from adapter import make_agent_fn
agent_fn = make_agent_fn()          # builds the graph once; reuse across cases and runs
report = run_eval(agent_fn, cases, grader, n_runs=3)
```

**`case.input`** is a JSON string naming the two PDFs:

```json
{"resume": "data/cv/kaggle_1.pdf", "jd": "data/jd/03. JD_Data Scientist.pdf", "candidate_id": 1}
```

**`AgentResult`** (contract section 4) — filled by `adapter.py`:

| Field | What you get |
|---|---|
| `output` | The agent's final answer, ending with the fixed lines (see below). Never the email message. |
| `tool_calls` | Tool names in the order they ran, e.g. `["parse_resume", "parse_jd", "email_applicant"]` |
| `tool_args` | Arguments for each call (only if your `AgentResult` has this field — week 6 does, week 3 doesn't) |
| `steps` | Model turns. More than 2 means the verifier sent the agent back at least once |
| `input_tokens` / `output_tokens` / `cost_usd` | Summed over the run. **Check the prices** in `adapter.py` (`PRICE_IN_PER_M`, `PRICE_OUT_PER_M`) before reporting cost |
| `latency_s` | Wall-clock seconds (about 30–90 s per case on gpt-5-mini) |
| `error` | **Only real crashes.** A bad input or a failed verification is NOT an error — see below |

**Fixed lines at the end of `output`:**

```
EVIDENCE: [R<n>: <requirement>] "<exact quote from resume>"   (or: none found)
...one line per requirement from parse_jd...
POLICY: P<n>
LABEL: 0 | 1
```

or, when there is no decision:

```
ERROR: <code>
LABEL: invalid
```

| `ERROR` code | When |
|---|---|
| `not_a_resume` | The resume file has no resume, or more than one |
| `not_a_jd` | The JD file has no JD, more than one, or no requirements |
| `unreadable` | File missing, not a PDF, empty, or the parser failed |
| `verification_failed` | **New.** The agent's answer still failed the built-in checks after 2 retries. No decision, no email. |

**Emails** (mocks — nothing is sent):

| Label | Email tool in `tool_calls` |
|---|---|
| `0` | `email_applicant` (sent automatically) |
| `1` | `email_recruiter` — the run **pauses for human approval** first. `make_agent_fn()` **auto-approves every pause**, so measured runs never include a human rejection. Mention this in the write-up. |
| `invalid` | none |

**Company policy clauses** the agent may cite in `POLICY:` are **P1–P11**, summarised from the TAFEP
*Tripartite Guidelines on Fair Employment Practices* (Apr 2024). The full text is in `SYSTEM` in `agent.py`.

**Not deterministic:** gpt-5-mini ignores `temperature`, so repeat runs can differ — that is what your
pass rate ± SD measures.

## How it works

```
START → agent ⇄ tools → verifier ─┬→ email → END        (pass, label 0 or 1)
          ↑                       ├→ END                (pass, invalid  |  fail, no retries left)
          └──── retry (≤ 2) ──────┘                     (fail: verifier tells the agent what to fix)
```

| File | What it is |
|---|---|
| `agent.py` | State, nodes (agent, tools, verifier, email), edges, system prompt + policy, `build_graph()` |
| `tools.py` | `parse_resume` / `parse_jd` (wrap `parser.py`), `normalize_skills`, `check_timeline`, email mocks |
| `adapter.py` | `run_case()` / `make_agent_fn()` → `AgentResult` for the harness |
| `parser.py`, `paths.py` | Group mate's PDF parser (CV/JD extraction + sensitive-word masking) |
| `run_one.py`, `run_checks.py` | Command-line runners |

**The verifier** (code, not the model) checks every final answer before any email: both parse tools
ran; one `EVIDENCE` line per requirement with quotes that really appear in the resume; a valid `LABEL`;
`ERROR` only when `invalid`; a real `POLICY` clause. A failed check sends the agent back with a note
saying exactly what to fix.

## Known limits

- The parser's masking is word-based: it also masks skills such as "Chinese" (language) or
  "Black" (Black-Scholes), and it does not mask names or emails.
- The JD parser marks unclear requirements as must-have, which makes rejections easier to trigger.
- One resume and one JD per run; PDFs only.
