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

## Planning mode (on/off switch) and the baseline

```python
make_agent_fn(planning=False)   # baseline: the agent picks tools step by step
make_agent_fn(planning=True)    # planning: plan first, check it in code, run it, answer once
```

```
START → planner → execute_plan → agent → verifier → (email | END | back to agent)
```

- **planner** — one small model call (task + tool list, no policy text) lists the tools to run.
- **check_plan** — a hard rule in code (week 5): only the four loop tools, both parse tools,
  at most 4 steps, never an email tool. A rejected plan falls back to `parse_resume, parse_jd`.
- **execute_plan** — code runs the plan, no model call per tool. If a parse tool returns
  `ERROR`, the rest of the plan is skipped.
- **agent** — one model call writes the final answer. Same output format, verifier, retries
  and emails as the baseline, so graders don't change.

Compare the two (tokens, cost, model turns, time, labels):

```bash
python compare_planning.py --runs 3        # writes planning_comparison.csv
python run_one.py <resume.pdf> <jd.pdf> --plan
python run_checks.py --plan
```

The token saving comes from fewer full-context model calls; check the labels match between
modes so the saving doesn't come from worse decisions.

## Batch job: score every CV against every JD, invite the top 20%

```bash
python run_batch.py --limit 2                   # first 2 CVs x all JDs
python run_batch.py --cvs C07,C12 --jds J1,J4   # only these CVs x these JDs (IDs = sorted file order)
python run_batch.py                             # full run: data/cvs x data/jds (add --plan for planning mode)
```

It prints the number of pairs and an estimated cost (~$0.02 per pair) and asks before spending anything.

Put CVs in `data/cvs/` and JDs in `data/jds/`. **`data/` is gitignored** (private applicant data);
results go to `data/results/scores.csv` and `data/results/shortlist.csv`.

```
START → load_files → screen_pair × (CVs × JDs, in parallel via Send) → rank → approve (human) → send_emails → END
```

1. **load_files** skips CV files over 8 pages (applicant-system exports holding several CVs) and parses
   every file **once**, so every candidate is judged against the same JD requirement list.
2. **screen_pair** runs the normal screening graph (same output contract, same verifier) with emails off,
   then scores the pair 0–100 (`scoring.py`):

   | # | Criterion (priority order) | Weight | Scored by |
   |---|---|---|---|
   | 1 | Relevant skills (minimum) + certifications | 6 | share of must-have skills with a verified quote |
   | 2 | Years of experience (minimum) | 5 | candidate years ÷ minimum (max 1) |
   | 3 | University ranking | 4 | QS 2027 tier table in code (top 50 / 51–200 / 201–500 / other degree / none) |
   | 4 | Soft skills | 3 | share of the JD's soft skills with a quote found in the resume (1 model call) |
   | 5 | Relevant skills (preferred) | 2 | share of nice-to-have skills with a verified quote |
   | 6 | Years of experience (preferred) | 1 | candidate years ÷ preferred (max 1) |

3. **rank**, per JD: the top **20% of that JD's valid applicants by score, rounded up**, get final label
   **1** (invited); everyone else gets **0**. The agent's own `LABEL` (its must-have check) is kept in
   `scores.csv` as `agent_label` for reference but does not decide (agreed 20 Oct). A CV can be invited
   to more than one role. Ties are broken by criterion 1, then 2, and so on.
4. **approve** pauses for a human (`--yes` skips the question). Nothing is sent before approval.
5. **send_emails** (mocks): `email_invite` to final label 1, `email_applicant` to final label 0 (policy P11).

The single-case contract and `make_agent_fn` for the evaluator are unchanged; the batch is a layer on top.

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
| `batch.py`, `scoring.py`, `run_batch.py` | Batch job: fan-out graph, the six-criterion score + university table, command-line runner |
| `planning.py` | Planning mode: planner, `check_plan`, plan executor, answer node |
| `run_one.py`, `run_checks.py`, `compare_planning.py` | Command-line runners (all take `--plan`; compare runs both) |

**The verifier** (code, not the model) checks every final answer before any email: both parse tools
ran; one `EVIDENCE` line per requirement with quotes that really appear in the resume; a valid `LABEL`;
`ERROR` only when `invalid`; a real `POLICY` clause. A failed check sends the agent back with a note
saying exactly what to fix.

## Known limits

- The parser's masking is word-based: it also masks skills such as "Chinese" (language) or
  "Black" (Black-Scholes), and it does not mask names or emails.
- The JD parser marks unclear requirements as must-have, which makes rejections easier to trigger.
- One resume and one JD per run; PDFs only.
