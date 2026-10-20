"""Tools for the resume screening agent.

- parse_resume / parse_jd: wrappers around the group's PDF parser (parser.py:
  cv_graph / jd_graph). They reshape its output into the formats the verifier
  expects and turn problems into the contract's ERROR codes.
- normalize_skills / check_timeline: deterministic code (no LLM).
- email_applicant / email_recruiter / email_invite: MOCKS. They only log; nothing is sent.

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


# Batch mode turns this on (enable_parse_cache), so each file is parsed ONCE and every
# CV x JD pair sees the same parse - same JD requirement list for every candidate, which
# keeps the ranking fair. Off by default, so eval repeats still measure parser variation.
PARSE_CACHE: dict | None = None


def enable_parse_cache() -> None:
    global PARSE_CACHE
    if PARSE_CACHE is None:
        PARSE_CACHE = {}


def _run_graph(graph, file: str):
    """Read a PDF and run one of the parser graphs on it. Returns its 'result'."""
    import parser  # group mate's module (needs OPENAI_API_KEY and paths.py)

    path = Path(file)
    key = (id(graph), str(path.resolve()), path.stat().st_mtime)
    if PARSE_CACHE is not None and key in PARSE_CACHE:
        return PARSE_CACHE[key]
    parser.validate_pdf_size(path)  # raises ValueError if too large
    pdf_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    args = {"pdf_base64": pdf_b64, "filename": path.name}
    # Works whether or not the caller is already inside an event loop (e.g. Jupyter).
    future = asyncio.run_coroutine_threadsafe(graph.ainvoke(args), _background_loop())
    result = future.result()["result"]
    if PARSE_CACHE is not None:  # failures raise above, so they are never cached
        PARSE_CACHE[key] = result
    return result


def parsed_cv(file: str):
    """The parser's CVEntry for a single-CV file (uses the cache in batch mode)."""
    import parser
    return _run_graph(parser.cv_graph, file).cvs[0]


def parsed_jd(file: str):
    """The parser's JDEntry for a single-JD file (uses the cache in batch mode)."""
    import parser
    return _run_graph(parser.jd_graph, file).jds[0]


def jd_items(jd) -> list[tuple[str, object]]:
    """(kind, TaggedItem) in R-number order: skills, then experience, then education.
    parse_jd and the batch scorer both use this, so R1, R2, ... always mean the same thing."""
    items = [("skill", s) for s in jd.skills]
    if jd.experience is not None:
        items.append(("experience", jd.experience))
    if jd.education is not None:
        items.append(("education", jd.education))
    return items


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


# ---------------------------------------------------------------- PDF text layer (for the verifier)
_PDF_TEXT: dict = {}


def pdf_text(file: str | None) -> str:
    """The PDF's own text layer (pypdf), no model involved. The verifier also accepts quotes
    found here, because the parser's model-written text can mix up two-column CVs.
    Returns "" if pypdf is not installed or the file can't be read."""
    if not file:
        return ""
    try:
        path = Path(file)
        key = (str(path.resolve()), path.stat().st_mtime)
        if key not in _PDF_TEXT:
            from pypdf import PdfReader
            _PDF_TEXT[key] = "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
        return _PDF_TEXT[key]
    except Exception:
        return ""


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
    items = [item for _, item in jd_items(jd)]
    if not items:
        return "ERROR: not_a_jd (no requirements found)"

    lines = []
    for n, item in enumerate(items, start=1):
        tag = "must-have" if item.requirement.value == "mandatory" else "nice-to-have"
        lines.append(f"R{n}: {item.value} ({tag})")
    return "\n".join(lines)


# ---------------------------------------------------------------- normalize_skills
# Deterministic: same input, same output. Keys are lower-case, punctuation-free forms.
_SYNONYMS = {
    # languages
    "py": "python", "python3": "python", "python 3": "python",
    "js": "javascript", "ecmascript": "javascript", "node": "node.js", "nodejs": "node.js",
    "ts": "typescript", "golang": "go", "c sharp": "c#", "csharp": "c#", "cpp": "c++",
    "r programming": "r", "r language": "r",
    "postgres": "postgresql", "psql": "postgresql", "mssql": "sql server", "ms sql": "sql server",
    "t sql": "sql", "tsql": "sql", "structured query language": "sql",
    # data / ML
    "ml": "machine learning", "dl": "deep learning", "ai": "artificial intelligence",
    "nlp": "natural language processing", "cv": "computer vision", "llm": "large language models",
    "llms": "large language models", "genai": "generative ai", "gen ai": "generative ai",
    "rl": "reinforcement learning", "stats": "statistics", "statistical analysis": "statistics",
    "eda": "exploratory data analysis", "etl": "extract transform load",
    "sklearn": "scikit-learn", "scikit learn": "scikit-learn", "tf": "tensorflow",
    "torch": "pytorch", "hf": "hugging face", "huggingface": "hugging face",
    "xgb": "xgboost", "lgbm": "lightgbm", "pyspark": "spark", "apache spark": "spark",
    "ms excel": "excel", "microsoft excel": "excel", "powerbi": "power bi", "ms power bi": "power bi",
    "a b testing": "ab testing", "a/b testing": "ab testing",
    # cloud / engineering
    "aws": "amazon web services", "gcp": "google cloud platform", "google cloud": "google cloud platform",
    "azure": "microsoft azure", "k8s": "kubernetes", "ci cd": "ci/cd", "cicd": "ci/cd",
    "rest api": "rest", "restful": "rest", "restful api": "rest", "git hub": "github",
    # business
    "pm": "project management", "hr": "human resources", "ui ux": "ui/ux", "ux ui": "ui/ux",
    "ms office": "microsoft office",
}
_VERSION = re.compile(r"\s*(v?\d+(\.\d+)*)$")


def _skill_key(raw: str) -> str:
    s = raw.strip().lower()
    s = re.sub(r"[‐-―]", "-", s)            # unify dashes
    s = re.sub(r"[()\[\]\"'.]", " ", s)      # drop brackets, quotes, full stops
    s = re.sub(r"[_\-]", " ", s)             # snake/kebab -> spaces (scikit-learn handled by synonyms)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@tool
def normalize_skills(skills: str) -> str:
    """Map different wordings of skills to one standard name, so the resume and the JD can be
    compared like for like (e.g. 'ML' and 'Machine-Learning' -> 'machine learning').
    Input: skills separated by commas, semicolons or new lines. Output: comma-separated standard
    names, duplicates removed, in first-seen order."""
    out, seen = [], set()
    for part in re.split(r"[,;\n]|\s/\s", skills):
        key = _skill_key(part)
        if not key:
            continue
        name = _SYNONYMS.get(key)
        if name is None:  # drop a trailing version number: 'Python 3.11' -> 'python'
            base = _VERSION.sub("", key).strip() or key
            name = _SYNONYMS.get(base, base)
        if name not in seen:
            seen.add(name)
            out.append(name)
    return ", ".join(out)


# ---------------------------------------------------------------- check_timeline
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_MON = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DATE = rf"(?:{_MON}\s*'?\d{{2,4}}|\d{{1,2}}\s*/\s*(?:19|20)\d{{2}}|(?:19|20)\d{{2}}\s*/\s*\d{{1,2}}|(?:19|20)\d{{2}})"
_END = rf"(?:{_DATE}|present|current|now|to date|till date|ongoing)"
_RANGE = re.compile(rf"({_DATE})\s*(?:-|–|—|to|till|until)\s*({_END})", re.IGNORECASE)
GAP_MONTHS = 3  # report gaps longer than this


def _to_month(text: str, today: tuple[int, int], is_end: bool) -> int | None:
    """Date text -> months since year 0 (year*12 + month-1)."""
    t = text.strip().lower()
    if t in ("present", "current", "now", "to date", "till date", "ongoing"):
        return today[0] * 12 + today[1] - 1
    m = re.match(rf"({_MON})\s*'?(\d{{2,4}})$", t)
    if m:
        mon = _MONTHS[m.group(1)[:3]]
        yr = int(m.group(2)); yr = yr + 2000 if yr < 100 and yr <= today[0] % 100 else (yr + 1900 if yr < 100 else yr)
        return yr * 12 + mon - 1
    m = re.match(r"(\d{1,2})\s*/\s*((?:19|20)\d{2})$", t)
    if m:
        return int(m.group(2)) * 12 + int(m.group(1)) - 1
    m = re.match(r"((?:19|20)\d{2})\s*/\s*(\d{1,2})$", t)
    if m:
        return int(m.group(1)) * 12 + int(m.group(2)) - 1
    m = re.match(r"((?:19|20)\d{2})$", t)
    if m:  # year only: from January / to December
        return int(m.group(1)) * 12 + (11 if is_end else 0)
    return None


def _fmt(month_index: int) -> str:
    return f"{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][month_index % 12]} {month_index // 12}"


@tool
def check_timeline(jobs: str) -> str:
    """Work out total years of experience, gaps and overlaps from job date ranges in the
    resume's work history (e.g. 'Jan 2019 - Mar 2021', '2019-2021', '03/2021 - present').
    Overlapping jobs are counted once. Input: the work-history text. Use the result for
    'years of experience' requirements."""
    import datetime
    d = datetime.date.today()
    today = (d.year, d.month)

    spans = []
    for start_txt, end_txt in _RANGE.findall(jobs):
        s = _to_month(start_txt, today, is_end=False)
        e = _to_month(end_txt, today, is_end=True)
        if s is not None and e is not None and e >= s:
            spans.append((s, e + 1))  # end month inclusive -> half-open
    if not spans:
        return "No date ranges found in the text."

    spans.sort()
    merged, overlaps, gaps = [list(spans[0])], [], []
    for s, e in spans[1:]:
        last = merged[-1]
        if s < last[1]:  # overlaps the previous block
            overlaps.append(f"{_fmt(s)} to {_fmt(min(e, last[1]) - 1)}")
            last[1] = max(last[1], e)
        else:
            if s - last[1] > GAP_MONTHS:
                gaps.append(f"{_fmt(last[1])} to {_fmt(s - 1)} ({s - last[1]} months)")
            merged.append([s, e])
    total_months = sum(e - s for s, e in merged)
    return (f"Total experience: {total_months // 12} years {total_months % 12} months "
            f"(overlapping jobs counted once; {len(spans)} date range(s) found). "
            f"Gaps over {GAP_MONTHS} months: {'; '.join(gaps) or 'none'}. "
            f"Overlaps: {'; '.join(overlaps) or 'none'}.")


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


@tool
def email_invite(candidate_id: int, message: str) -> str:
    """MOCK. Log that an interview invitation would be sent to the applicant. Nothing is sent."""
    EMAIL_LOG.append({"tool": "email_invite", "candidate_id": candidate_id, "message": message})
    return "logged"
