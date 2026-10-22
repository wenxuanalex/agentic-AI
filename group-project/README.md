# Resume Screening Agent — LangGraph Part

Screens **one resume against one job description (JD)** and returns a decision in the format agreed in *Resume Screening Agent: Output Contract for Evaluation*.

**Owner:** Alex

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in this folder. **Never commit it.**

```dotenv
OPENAI_API_KEY=sk-...
# OPENAI_API_BASE_URL=...  # Optional, only if the group uses a proxy
```

## Run it

```bash
python run_one.py <resume.pdf> <jd.pdf> [candidate_id]
python run_checks.py [data_folder]
```

- `run_one.py`: Screens one resume against one JD and prints the `AgentResult`.
- `run_checks.py`: Runs three real-model checks covering rejection, advancement and invalid input.

## For the evaluator: how to plug into `run_eval`

```python
from adapter import make_agent_fn

agent_fn = make_agent_fn()  # Build the graph once; reuse across cases and runs
report = run_eval(agent_fn, cases, grader, n_runs=3)
```

### `case.input`

A JSON string naming the two PDFs:

```json
{
  "resume": "data/cv/kaggle_1.pdf",
  "jd": "data/jd/03. JD_Data Scientist.pdf",
  "candidate_id": 1
}
```

### `AgentResult` — contract section 4

Filled by `adapter.py`.

| Field | What you get |
|---|---|
| `output` | The agent's final answer, ending with the fixed lines below. Never the email message. |
| `tool_calls` | Tool names in execution order, e.g. `["parse_resume", "parse_jd", "email_applicant"]`. |
| `tool_args` | Arguments for each call, if supported by the version of `AgentResult` being used. Week 6 includes this field; week 3 does not. |
| `steps` | Model turns. More than two usually means the verifier sent the agent back for a retry. |
| `input_tokens`, `output_tokens`, `cost_usd` | Token and estimated cost totals. Check `PRICE_IN_PER_M` and `PRICE_OUT_PER_M` in `adapter.py` before reporting costs. |
| `latency_s` | Wall-clock runtime; approximately 30–90 seconds per case on `gpt-5-mini`, depending on the case and retries. |
| `error` | Reserved for actual crashes. Invalid inputs and failed verification are represented in the agent output, not as runtime errors. |

### Fixed output contract

For a valid screening decision, the output ends with:

```text
EVIDENCE: [R<n>: <requirement>] "<exact quote from resume>"
...one line per requirement from parse_jd...
POLICY: P<n>
LABEL: 0 | 1
```

If no decision can be made:

```text
ERROR: <code>
LABEL: invalid
```

Each requirement must have an evidence line. If the resume contains no supporting quote, use `none found`.

| `ERROR` code | Meaning |
|---|---|
| `not_a_resume` | The file contains no resume or more than one resume. |
| `not_a_jd` | The file contains no JD, more than one JD, or a JD with no requirements. |
| `unreadable` | The file is missing, is not a PDF, is empty, or cannot be parsed. |
| `verification_failed` | The answer still fails the built-in checks after two retries. No decision or email is produced. |

### Emails in single-case screening

These are mock email tools; no real emails are sent.

| Label | Email behaviour |
|---|---|
| `0` | `email_applicant` sends the mock rejection notification automatically after verification succeeds. |
| `1` | The run pauses for human approval before `email_recruiter`. `make_agent_fn()` automatically approves every pause, so measured evaluator runs do not include a human rejection. Mention this limitation in the write-up. |
| `invalid` | No email is sent. |
| Verification failure | No decision is produced and no email is sent. |

## Company policy clauses

The agent may cite clauses `P1–P11` in its `POLICY:` line. These are summarised from the TAFEP *Tripartite Guidelines on Fair Employment Practices* (April 2024). The full policy prompt is in `SYSTEM` in `agent.py`.

## Nondeterminism

The `gpt-5-mini` model is not deterministic, and setting `temperature` does not guarantee repeatable outputs. Repeated runs may differ in labels, tool calls, latency and cost. Report the pass rate and standard deviation across repeated runs rather than relying on one run.

## Planning mode: on/off switch and baseline

```python
make_agent_fn(planning=False)  # Baseline: the agent chooses tools step by step
make_agent_fn(planning=True)   # Planning: plan first, validate it, execute it, then answer
```

Planning workflow:

```text
START → planner → execute_plan → agent → verifier → (email | END | retry)
```

- **Planner:** Makes one small model call to select tools based on the task and available tool list. It does not receive the full policy text.
- **`check_plan`:** Enforces code-level constraints: only the four loop tools, both parsing tools must be included, at most four steps, and no email tools. A rejected plan falls back to `parse_resume` and `parse_jd`.
- **`execute_plan`:** Executes the validated plan in code without a model call for every tool. If a parse tool returns an error, the rest of the plan is skipped.
- **Agent:** Makes a model call to produce the final answer. The output contract, verifier, retry logic and email handling remain the same as in baseline mode.

Compare the two modes:

```bash
python compare_planning.py --runs 3
python run_one.py <resume.pdf> <jd.pdf> --plan
python run_checks.py --plan
```

`compare_planning.py` writes `planning_comparison.csv`. Compare token usage, cost, model turns, latency and labels. Confirm that any savings do not come at the expense of worse screening decisions.

## Batch screening: score every CV against every JD and invite the top 20%

### Run the batch job

```bash
python run_batch.py --limit 2
python run_batch.py --cvs C07,C12 --jds J1,J4
python run_batch.py
```

- `--limit 2`: Screens the first two CVs against all JDs.
- `--cvs C07,C12 --jds J1,J4`: Screens only the selected CVs and JDs. IDs follow sorted file order.
- No arguments: Runs the full batch over `data/cvs/` and `data/jds/`.
- Add `--plan` to use planning mode.

The command displays the number of CV–JD pairs and an estimated cost (approximately $0.02 per pair) before asking for confirmation.

Place CVs in `data/cvs/` and JDs in `data/jds/`. **`data/` is gitignored because it may contain private applicant data.**

Outputs:

- `data/results/scores.csv`
- `data/results/shortlist.csv`

Batch workflow:

```text
START → load_files → screen_pair × (CVs × JDs, parallel via Send)
      → rank → approve (human) → send_emails → END
```

### 1. `load_files`

- Skips CV files over eight pages, which may be applicant-system exports containing several CVs.
- Parses each file once so candidates are evaluated against the same JD requirement list.
- Prepares the visible CV text used by the screening and scoring guardrails.

### 2. `screen_pair`

Runs the normal screening graph for each CV–JD pair, with single-case email handling disabled, then scores the pair from 0–100 using `scoring.py`.

| # | Criterion, in priority order | Weight | Scoring method |
|---|---|---:|---|
| 1 | Relevant minimum skills and certifications | 6 | Share of must-have skills supported by verified quotes. |
| 2 | Minimum years of experience | 5 | Candidate years divided by the minimum, capped at 1. |
| 3 | University ranking | 4 | QS 2027 tier table in code: top 50, 51–200, 201–500, other degree, or none. |
| 4 | Soft skills | 3 | Share of JD soft skills supported by resume quotes, using one model call. |
| 5 | Preferred skills | 2 | Share of nice-to-have skills supported by verified quotes. |
| 6 | Preferred years of experience | 1 | Candidate years divided by preferred experience, capped at 1. |

**Hidden-text rule:** A CV with detected hidden text is disqualified from batch screening. It must not enter the eligible ranking or shortlist, regardless of its model label or score. If the hidden-text detector fails, the system should fail closed rather than treat the CV as clean.

### 3. `rank`

Ranking is performed separately for each JD.

- Only valid, eligible applicants with a usable score and no hidden-text disqualification can be ranked.
- The top `ceil(invite_share × eligible_applicant_count)` applicants receive final label `1`; the remaining eligible applicants receive final label `0`.
- The default `invite_share` is `0.20`.
- The agent's own mandatory-requirement label is retained in `scores.csv` as `agent_label` for reference. It does not determine the batch invitation label; the ranking rule agreed on 20 October does.
- A CV can be invited for more than one role.
- Ties are broken using the criteria in priority order, beginning with criterion 1.

A disqualified CV is not eligible for invitation even if its score would otherwise place it in the top 20%.

### 4. Human approval and changing the invitation share

The batch pauses for human approval before sending mock emails. Use `--yes` to skip the interactive confirmation.

The invitation share can be changed without rerunning screening:

- Start with a different share: `python run_batch.py --top 0.3`.
- Set it in graph input: `{"invite_share": 0.3, ...}`.
- At the approval pause, resume with `Command(resume={"invite_share": 0.3})`. The graph returns to ranking, reuses the existing scores and pauses for approval again.
- Resume with `"approve"` or `"reject"` to finish.
- In the terminal, enter the new share when prompted.
- Preview a ranking without running the graph: `batch.rank_results(results, jds, 0.3)` returns `(shortlist, ranking)`.

The share must be greater than 0 and at most 1. Invalid values should raise `ValueError` with a message suitable for the UI.

Changing the share reranks the existing results: it does not trigger additional screening model calls.

### 5. `send_emails`

These are mock email actions performed only after batch approval:

- `email_invite` for final label `1`.
- `email_applicant` for final label `0`, including applicants rejected by the batch ranking policy.
- Hidden-text-disqualified CVs must never receive an invitation. Their disqualification must be preserved through ranking and email handling.

The single-case output contract and `make_agent_fn()` interface for the evaluator remain separate from the batch workflow.

## How the single-case graph works

```text
START → agent ⇄ tools → verifier ─┬→ email → END
          ↑                       ├→ END (invalid or retries exhausted)
          └──── retry (≤ 2) ──────┘
```

The verifier runs before any email action. It checks that the required parsing tools ran, that the evidence quotes appear in the resume's visible text, that the label is valid, that `ERROR` is used consistently with an invalid result, and that the `POLICY` clause is valid. A failed verification sends the agent back with instructions describing what to fix.

## Project files

| File | Purpose |
|---|---|
| `agent.py` | State, graph nodes, edges, system prompt, policy and `build_graph()`. |
| `tools.py` | Resume/JD parsing wrappers, `visible_pdf_text`, skill normalisation, timeline checks and email mocks. |
| `tests/` | Offline guardrail tests and attack-suite builder. |
| `adapter.py` | `run_case()` and `make_agent_fn()`; produces `AgentResult` for the evaluation harness. |
| `parser.py`, `paths.py` | PDF extraction and sensitive-word masking. |
| `batch.py`, `scoring.py`, `run_batch.py` | Batch screening, score calculation, ranking and command-line runner. |
| `planning.py` | Planning mode, plan validation and execution. |
| `run_one.py`, `run_checks.py`, `compare_planning.py` | Command-line runners and planning comparison. |

## Guardrails: prompt injection hidden in resumes

A prompt in the system message telling the model to treat a resume as data is not sufficient protection. Guardrails must be enforced in code.

The batch system uses two complementary controls.

### 1. Detect hidden text and disqualify the CV

`tools.visible_pdf_text(file)` uses `pdfplumber` to identify text a reader may not be able to see, including:

- White or near-white text on a white background (gray, RGB or CMYK channels at or above 0.95).
- Font sizes below 3 pt.
- Text outside the page box.
- PDF render mode 3, which makes text invisible.

White text on a dark header or box is retained.

For batch screening, **any detected hidden text triggers disqualification**. The CV is excluded before it can be ranked or shortlisted. If hidden-text detection fails, the batch must fail closed rather than proceed as though the CV were clean.

The screening parser may still be used to identify invalid resumes, but the visible text is used for evidence verification and scoring guardrails.

### 2. Require visible evidence for score inputs

Every score input must be backed by visible resume content:

- **Years of experience:** The score is capped by the lower of the parser's `years_of_experience` and the result of `check_timeline` on visible text. If no visible dates support the experience, use zero.
- **University tier:** A university only receives its ranking tier if its name appears in visible text. Otherwise, use the configured fallback for a degree or zero if there is no supported degree.
- **Skills:** Skill evidence must be supported by verifier-checked quotes from visible text.
- **Guardrail notes:** Any capped or unconfirmed value should have a `guardrail:` explanation in the `why` column.

These controls prevent hidden claims from inflating the batch score, while the hidden-text disqualification rule prevents a CV containing detected hidden text from entering the shortlist at all.

### Tests and attack suite

Run the offline tests:

```bash
python -m pytest -q tests
```

Generate the synthetic attack suite:

```bash
python tests/make_attack_suite.py
python run_batch.py data/attack_suite --plan
```

The suite includes one clean CV and four attack variants:

| CV | Attack | Expected batch outcome |
|---|---|---|
| `A0_clean.pdf` | Clean baseline | Eligible if it passes the ordinary screening and scoring requirements. |
| `A1_hidden_instructions.pdf` | Hidden instructions | Disqualified if hidden text is detected. |
| `A2_keyword_stuffing.pdf` | Hidden or manipulated keyword stuffing | Disqualified if hidden text is detected. |
| `A3_fake_system_note.pdf` | Fake system note | Disqualified if hidden text is detected. |
| `A4_fake_claims.pdf` | Fabricated experience or university claims | Disqualified if hidden text is detected. |

The expected outcome for attack cases depends on the detector actually finding hidden text. A generated attack PDF is not proof that detection worked: check `hidden_text_chars`, the disqualification result, `scores.csv` and `shortlist.csv`.

The clean baseline should remain eligible when it otherwise meets the screening requirements. The attack variants should never be invited when hidden text is detected.

## Known limitations

- The hidden-text detector evaluates text characters individually. It may miss text covered by an image or an overlaid shape, white text embedded in an image, or text squeezed into near-zero-width rendering.
- A two-column CV's visible text may be extracted in a reading order that mixes the columns.
- The parser still reads the raw PDF and may incorrectly classify a document as `not_a_resume` if hidden content affects extraction.
- Sensitive-word masking is word-based. It may mask legitimate skills such as “Chinese” or “Black-Scholes”, and it does not mask names or email addresses.
- The JD parser may classify unclear requirements as must-have, potentially increasing the likelihood of rejection.
- The single-case workflow supports one resume and one JD per run, and PDFs only.
- The batch workflow depends on the correctness of its PDF parser, hidden-text detector, evidence verifier, score calculations and approval controls. Passing offline tests does not establish that every real-world attack will be detected.

