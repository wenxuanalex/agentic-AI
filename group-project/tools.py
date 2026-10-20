"""Tools for the resume screening agent.

- parse_resume / parse_jd: wrappers around the group's PDF parser (parser.py:
  cv_graph / jd_graph). They reshape its output into the formats the verifier
  expects and turn problems into the contract's ERROR codes.
- normalize_skills / check_timeline: simple STAND-INS, to be replaced.
- email_applicant / email_recruiter: MOCKS. They only log; nothing is sent.

parser.py is imported only when a parse tool is first used, because importing it
needs OPENAI_API_KEY and its `paths` module.
"""
from __future__ import annotations

import asyncio
import base64
import re
import threading
from pathlib import Path

from langchain_core.tools import tool


# ---------------------------------------------------------------- helpers
# One event loop for ALL parse calls, running in a background thread. The parser's
# OpenAI client keeps connections tied to the loop of its first call, so a fresh loop
# per call (asyncio.run) fails on the second call with "Event loop is closed".
_LOOP: asyncio.AbstractEventLoop | None = None
_LOOP_LOCK = threading.Lock()


def _background_loop() -> asyncio.AbstractEventLoop:
    global _LOOP
    with _LOOP_LOCK:
        if _LOOP is None or _LOOP.is_closed():
            _LOOP = asyncio.new_event_loop()
            threading.Thread(target=_LOOP.run_forever, daemon=True, name="parser-loop").start()
        return _LOOP


def _run_graph(graph, file: str):
    """Read a PDF and run one of the parser graphs on it. Returns its 'result'."""
    import parser  # group mate's module (needs OPENAI_API_KEY and paths.py)

    path = Path(file)
    parser.validate_pdf_size(path)  # raises ValueError if too large
    pdf_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    args = {"pdf_base64": pdf_b64, "filename": path.name}
    # Works whether or not the caller is already inside an event loop (e.g. Jupyter).
    future = asyncio.run_coroutine_threadsafe(graph.ainvoke(args), _background_loop())
    return future.result()["result"]


def _check_file(file: str) -> str | None:
    """Return an ERROR string if the file can't be used, else None."""
    path = Path(file)
    if not path.is_file():
        return f"ERROR: unreadable (file not found: {file})"
    if path.suffix.lower() != ".pdf":
        return f"ERROR: unreadable (only PDF files are supported, got {path.suffix or 'no extension'})"
    if path.stat().st_size == 0:
        return "ERROR: unreadable (file is empty)"
    return None


# ---------------------------------------------------------------- parse tools
@tool
def parse_resume(file: str) -> str:
    """Read a resume PDF and return its full text, with sensitive personal details
    (sex, age, race, religion) masked as [MASKED]. Quote EVIDENCE word for word from
    this text. Returns a line starting with 'ERROR:' if the file is not a usable resume."""
    import parser

    problem = _check_file(file)
    if problem:
        return problem
    try:
        result = _run_graph(parser.cv_graph, file)
    except Exception as e:  # unreadable PDF, API failure, size limit, ...
        return f"ERROR: unreadable ({type(e).__name__}: {e})"

    cvs = result.cvs if result is not None else []
    if len(cvs) == 0:
        return "ERROR: not_a_resume (no resume found in the file)"
    if len(cvs) > 1:
        return f"ERROR: not_a_resume (file contains {len(cvs)} resumes; expected exactly one)"
    text = (cvs[0].full_text or "").strip()
    if not text:
        return "ERROR: unreadable (no text could be extracted)"
    return text


@tool
def parse_jd(file: str) -> str:
    """Read a job description PDF and return its requirements as a numbered list,
    one per line, e.g. 'R1: Python (must-have)'. Write one EVIDENCE line per R-number.
    Returns a line starting with 'ERROR:' if the file is not a usable job description."""
    import parser

    problem = _check_file(file)
    if problem:
        return problem
    try:
        result = _run_graph(parser.jd_graph, file)
    except Exception as e:
        return f"ERROR: unreadable ({type(e).__name__}: {e})"

    jds = result.jds if result is not None else []
    if len(jds) == 0:
        return "ERROR: not_a_jd (no job description found in the file)"
    if len(jds) > 1:
        return f"ERROR: not_a_jd (file contains {len(jds)} job descriptions; expected exactly one)"
    jd = jds[0]

    # Fixed order: skills, then experience, then education.
    items = list(jd.skills)
    if jd.experience is not None:
        items.append(jd.experience)
    if jd.education is not None:
        items.append(jd.education)
    if not items:
        return "ERROR: not_a_jd (no requirements found)"

    lines = []
    for n, item in enumerate(items, start=1):
        tag = "must-have" if item.requirement.value == "mandatory" else "nice-to-have"
        lines.append(f"R{n}: {item.value} ({tag})")
    return "\n".join(lines)


# ---------------------------------------------------------------- stand-ins (replace later)
_SYNONYMS = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "js": "javascript",
    "py": "python",
    "k8s": "kubernetes",
    "gcp": "google cloud platform",
    "aws": "amazon web services",
}


@tool
def normalize_skills(skills: str) -> str:
    """STAND-IN. Map different wordings of skills to one standard name.
    Input: comma-separated skills, e.g. 'ML, Py, SQL'. Output: comma-separated standard names."""
    out = []
    for s in skills.split(","):
        key = s.strip().lower()
        if key:
            out.append(_SYNONYMS.get(key, key))
    return ", ".join(out)


_YEAR_RANGE = re.compile(r"((?:19|20)\d{2})\s*[-–—to]+\s*((?:19|20)\d{2}|present|current|now)", re.IGNORECASE)


@tool
def check_timeline(jobs: str) -> str:
    """STAND-IN. Work out total years of experience, gaps and overlaps from job date
    ranges such as '2019-2021' or '2021 - present'. Input: the jobs section as text."""
    import datetime

    this_year = datetime.date.today().year
    spans = []
    for start, end in _YEAR_RANGE.findall(jobs):
        e = this_year if not end[:2].isdigit() else int(end)
        spans.append((int(start), e))
    if not spans:
        return "No date ranges found."
    spans.sort()
    years = sum(e - s for s, e in spans)
    gaps, overlaps = [], []
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        if s2 > e1:
            gaps.append(f"{e1}-{s2}")
        elif s2 < e1:
            overlaps.append(f"{s2}-{min(e1, e2)}")
    return (f"Total years (sum of ranges): {years}. "
            f"Gaps: {', '.join(gaps) or 'none'}. Overlaps: {', '.join(overlaps) or 'none'}.")


# ---------------------------------------------------------------- email mocks
EMAIL_LOG: list[dict] = []


@tool
def email_applicant(candidate_id: int, message: str) -> str:
    """MOCK. Log that a rejection email would be sent to the applicant. Nothing is sent."""
    EMAIL_LOG.append({"tool": "email_applicant", "candidate_id": candidate_id, "message": message})
    return "logged"


@tool
def email_recruiter(candidate_id: int, message: str) -> str:
    """MOCK. Log that HR would be notified about the candidate. Nothing is sent."""
    EMAIL_LOG.append({"tool": "email_recruiter", "candidate_id": candidate_id, "message": message})
    return "logged"
