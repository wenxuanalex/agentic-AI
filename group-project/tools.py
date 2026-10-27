"""Tools for the resume screening agent.

- parse_resume / parse_jd: wrappers around the group's PDF parser (parser.py:
  cv_graph / jd_graph). They reshape its output into the formats the verifier
  expects and turn problems into the contract's ERROR codes.
- visible_pdf_text: guardrail 1 - only text a human can see reaches the agent and the quote checks.
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
    """The PDF's own text layer (pypdf), no model involved. It includes HIDDEN text, so it is
    only a quote source when visible_pdf_text finds no text layer (see quote_sources).
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


# ---------------------------------------------------------------- visible text (guardrail 1)
# Prompt injection can hide in a resume as text a human never sees (white-on-white, tiny font,
# off the page, invisible render mode). The parser reads the raw PDF, so it sees that text too.
# visible_pdf_text keeps only characters a reader can see; the agent quotes from this text and
# the verifier / soft-skills check accept quotes only from it. Code, not a prompt rule.
WHITE_MIN = 0.95        # a fill colour with every RGB channel >= this counts as white
                        # (white text is hidden only on a white background, not on a dark box)
MIN_FONT_PT = 3         # smaller than this is unreadable
HIDDEN_FLAG_CHARS = 20  # more hidden characters than this -> "HIDDEN TEXT" flag for the human
_VISIBLE: dict = {}


def _patch_render_mode() -> None:
    """pdfminer knows each string's text render mode (Tr) but pdfplumber drops it.
    Copy it onto every char as char["render_mode"] (mode 3 = invisible)."""
    import pdfplumber.page as pp
    agg = pp.PDFPageAggregatorWithMarkedContent
    if getattr(agg, "_render_mode_patched", False):
        return
    pp.ALL_ATTRS.add("render_mode")
    original = agg.render_string

    def render_string(self, textstate, *args, **kwargs):
        start = len(self.cur_item._objs)
        original(self, textstate, *args, **kwargs)
        for obj in self.cur_item._objs[start:]:
            obj.render_mode = textstate.render

    agg.render_string = render_string
    agg._render_mode_patched = True


def _is_white(color) -> bool:
    """Fill colour as pdfplumber gives it: (gray,), (r, g, b) or (c, m, y, k). None = black."""
    if isinstance(color, (int, float)):
        color = (color,)
    if not isinstance(color, (tuple, list)) or not all(isinstance(c, (int, float)) for c in color):
        return False  # no colour or a pattern: can't tell, treat as visible
    if len(color) == 4:  # CMYK -> RGB
        c, m, y, k = color
        color = ((1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k))
    return len(color) in (1, 3) and all(ch >= WHITE_MIN for ch in color)


def _backgrounds(page) -> list[tuple[dict, bool]]:
    """Filled shapes and images on the page, in drawing order: (object, is it white?).
    White text on a dark box (a CV header or sidebar) is visible, so it must not be dropped."""
    shapes = [(o, _is_white(o.get("non_stroking_color"))) for o in page.rects + page.curves if o.get("fill")]
    return shapes + [(img, False) for img in page.images]  # an image: colour unknown, assume dark


def _on_white(ch: dict, backgrounds) -> bool:
    """Is the char on a white background? Uses the last (topmost) shape under its centre;
    no shape = the white page."""
    x, y = (ch["x0"] + ch["x1"]) / 2, (ch["top"] + ch["bottom"]) / 2
    under = [white for o, white in backgrounds if o["x0"] <= x <= o["x1"] and o["top"] <= y <= o["bottom"]]
    return under[-1] if under else True


def _hidden_reasons(ch: dict, page, backgrounds=()) -> list[str]:
    reasons = []
    if _is_white(ch.get("non_stroking_color")) and _on_white(ch, backgrounds):
        reasons.append("white")
    if (ch.get("size") or 0) < MIN_FONT_PT:
        reasons.append("tiny")
    x, y = (ch["x0"] + ch["x1"]) / 2, (ch["top"] + ch["bottom"]) / 2
    x0, top, x1, bottom = page.bbox
    if not (x0 <= x <= x1 and top <= y <= bottom):
        reasons.append("offpage")
    if ch.get("render_mode") == 3:
        reasons.append("invisible")
    return reasons


def visible_pdf_text(file: str | None) -> tuple[str, dict]:
    """(text a reader can see, report) for a PDF, no model involved.
    report = {"hidden_chars": n, "hidden_sample": first ~80 hidden chars, "reasons": {reason: n}}.
    Returns ("", report with 0 hidden chars) if the file can't be read (e.g. no text layer)."""
    empty = {"hidden_chars": 0, "hidden_sample": "", "reasons": {}}
    if not file:
        return "", empty
    try:
        path = Path(file)
        key = (str(path.resolve()), path.stat().st_mtime)
        if key not in _VISIBLE:
            import pdfplumber
            from pdfplumber.utils import extract_text
            _patch_render_mode()
            pages, samples, n_hidden, reasons = [], [], 0, {}
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    hidden, backgrounds = [], _backgrounds(page)
                    def keep(obj, page=page, backgrounds=backgrounds):
                        if obj.get("object_type") != "char":
                            return True
                        why = _hidden_reasons(obj, page, backgrounds)
                        if why:
                            hidden.append(obj)
                            if obj.get("text", "").strip():  # count real characters, not spaces
                                for r in why:
                                    reasons[r] = reasons.get(r, 0) + 1
                        return not why
                    filtered_page = page.filter(keep)
                    page_text = filtered_page.extract_text() or ""

                    print("VISIBLE DEBUG: page chars =", len(page.chars))
                    print("VISIBLE DEBUG: extracted chars =", len(page_text))
                    print("VISIBLE DEBUG: preview =", repr(page_text[:500]))

                    pages.append(page_text)
                    n_hidden += sum(bool(c.get("text", "").strip()) for c in hidden)
                    if hidden:
                        samples.append(extract_text(hidden))
            sample = re.sub(r"\s+", " ", " ".join(samples)).strip()[:80]
            _VISIBLE[key] = ("\n".join(pages).strip(),
                             {"hidden_chars": n_hidden, "hidden_sample": sample, "reasons": reasons})
        text, report = _VISIBLE[key]
        return text, dict(report)
    except Exception as e:
        print("VISIBLE PDF ERROR:", type(e).__name__, str(e))
        raise


def has_text_layer(file: str | None) -> bool:
    """True if the PDF has any text characters (visible or hidden). False e.g. for a scanned PDF:
    then the old behaviour (parser text + pypdf) is used. A PDF whose text is ALL hidden still
    counts as having a text layer, so hiding everything can't switch the guardrail off."""
    text, report = visible_pdf_text(file)
    return bool(text.strip()) or report["hidden_chars"] > 0


def _mask(text: str) -> str:
    """The parser's sensitive-word masking (sex, age, race, religion), applied to visible text."""
    import parser
    return parser.mask_sensitive_text(text) or ""


def resume_text_for_agent(file: str, parser_text: str) -> str:
    """What parse_resume returns: the masked VISIBLE text if the PDF has a text layer,
    else the parser's full_text (fallback, e.g. a scanned PDF)."""
    if has_text_layer(file):
        return _mask(visible_pdf_text(file)[0])
    return parser_text


def quote_sources(file: str | None, parser_text: str | None) -> list[str]:
    """Texts a quote may come from (verifier and soft-skills check).
    With a text layer: ONLY the visible text (raw and masked - the agent sees the masked one).
    Without (scanned PDF): the old sources, the parser's text and pypdf's text layer."""
    if has_text_layer(file):
        visible = visible_pdf_text(file)[0]
        return [visible, _mask(visible)]
    return [parser_text or "", pdf_text(file)]




# ---------------------------------------------------------------- parse tools

def _looks_like_jd_not_resume(text: str) -> bool:
    t = text.lower()

    jd_signals = [
        "job description",
        "key responsibilities",
        "job responsibilities",
        "what you'll do",
        "what you will do",
        "requirements for the role",
        "qualifications required",
        "about the role",
    ]

    resume_signals = [
        "work experience",
        "professional experience",
        "employment history",
        "education",
        "resume",
        "curriculum vitae",
    ]

    jd_hits = sum(signal in t for signal in jd_signals)
    resume_hits = sum(signal in t for signal in resume_signals)

    return jd_hits >= 2 and resume_hits == 0

@tool
def parse_resume(file: str) -> str:
    """Read a resume PDF and return its full text, with sensitive personal details
    (sex, age, race, religion) masked as [MASKED]. Quote EVIDENCE word for word from
    this text. Returns a line starting with 'ERROR:' if the file is not a usable resume."""
    import parser

    problem = _check_file(file)
    if problem:  # missing / not a PDF / empty: report it before trying to read the file
        return problem

    visible_text, _ = visible_pdf_text(file)

    if visible_text.strip() and _looks_like_jd_not_resume(visible_text):
        return "ERROR: not_a_resume (document appears to be a job description)"
    try:
        result = _run_graph(parser.cv_graph, file)
    except Exception as e:  # unreadable PDF, API failure, size limit, ...
        return f"ERROR: unreadable ({type(e).__name__}: {e})"

    cvs = result.cvs if result is not None else []
    if len(cvs) == 0:
        return "ERROR: not_a_resume (no resume found in the file)"
    if len(cvs) > 1:
        return f"ERROR: not_a_resume (file contains {len(cvs)} resumes; expected exactly one)"
    # The parser decides not_a_resume / several CVs; the agent gets the VISIBLE text (guardrail 1),
    # so it quotes from the same text the verifier checks.
    text = resume_text_for_agent(file, (cvs[0].full_text or "").strip()).strip()
    if not text:
        if has_text_layer(file):
            return "ERROR: unreadable (all text in the PDF is hidden)"
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


import re
from datetime import datetime



# ---------------------------------------------------------------- check_timeline
_MONTHS = {
    name.lower(): i
    for i, name in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        start=1,
    )
}

_MONTH_NAMES = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)

_DATE = (
    rf"(?:"
    rf"(?:{_MONTH_NAMES})\.?\s*['’]?\s*\d{{2,4}}"
    rf"|(?:0?[1-9]|1[0-2])\s*/\s*(?:19|20)\d{{2}}"
    rf"|(?:19|20)\d{{2}}\s*/\s*(?:0?[1-9]|1[0-2])"
    rf"|(?:19|20)\d{{2}}"
    rf")"
)

_END = rf"(?:{_DATE}|present|current|now|to date|till date|ongoing)"

_RANGE = re.compile(
    rf"({_DATE})\s*(?:-|–|—|−|\bto\b|\btill\b|\buntil\b)\s*({_END})",
    re.IGNORECASE,
)


def _to_month(value: str, is_end: bool = False) -> int | None:
    """Convert a supported date into an integer month index."""
    value = value.strip().lower().replace("’", "'").replace(".", "")
    
    if value in {
        "present", "current", "now", "to date", "till date", "ongoing"
    }:
        today = datetime.now()
        return today.year * 12 + today.month - 1

    match = re.fullmatch(
        rf"({_MONTH_NAMES})\s*'?\s*(\d{{2,4}})",
        value,
        re.IGNORECASE,
    )
    if match:
        month = _MONTHS[match.group(1)[:3].lower()]
        year = int(match.group(2))
        if year < 100:
            year += 2000 if year < 50 else 1900
        return year * 12 + month - 1

    match = re.fullmatch(
        r"(0?[1-9]|1[0-2])\s*/\s*((?:19|20)\d{2})",
        value,
    )
    if match:
        return int(match.group(2)) * 12 + int(match.group(1)) - 1

    match = re.fullmatch(
        r"((?:19|20)\d{2})\s*/\s*(0?[1-9]|1[0-2])",
        value,
    )
    if match:
        return int(match.group(1)) * 12 + int(match.group(2)) - 1

    match = re.fullmatch(r"(19|20)\d{2}", value)
    if match:
        year = int(value)
        # A year-only start covers January; an end covers December.
        return year * 12 + (11 if is_end else 0)

    return None


def extract_work_experience_section(text: str) -> str:
    """
    Extract the work experience section to avoid counting education,
    projects, certifications, or other unrelated date ranges.
    """
    start_pattern = re.compile(
        r"\b(?:work experience|professional experience|employment history|"
        r"work history|career history)\b",
        re.IGNORECASE,
    )

    end_pattern = re.compile(
        r"\b(?:education|academic background|technical skills|skills|"
        r"projects|certifications|certificates|awards|interests)\b",
        re.IGNORECASE,
    )

    start = start_pattern.search(text)
    if not start:
        return text

    remainder = text[start.end():]
    end = end_pattern.search(remainder)

    if end:
        return remainder[:end.start()]

    return remainder


@tool
def check_timeline(jobs: str) -> str:
    """
    Calculate total professional experience from employment date ranges.
    Merge overlapping periods so concurrent jobs are not double-counted.
    Supports month-year, numeric month-year, and year-only dates.
    """
    print(">>> CHECK_TIMELINE CALLED <<<")
    print(">>> INPUT TYPE:", type(jobs).__name__)
    print(">>> INPUT:", repr(jobs)[:3000])
    print("check_timeline input:", repr(jobs))
    print("check_timeline input type:", type(jobs))

    if not isinstance(jobs, str) or not jobs.strip():
        return "No date ranges found in the text."

    text = (
        jobs.replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("\u00a0", " ")
        .replace("\u200b", "")
    )

    text = extract_work_experience_section(text)

    matches = list(_RANGE.finditer(text))
    print(">>> MATCHES:", [m.groups() for m in matches])

    spans = []
    for match in matches:
        start = _to_month(match.group(1), is_end=False)
        end = _to_month(match.group(2), is_end=True)

        if start is None or end is None or end < start:
            continue

        spans.append((start, end))

    if not spans:
        return "No date ranges found in the text."

    spans.sort()
    merged = []

    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    total_months = sum(end - start + 1 for start, end in merged)
    years = total_months // 12
    months = total_months % 12

    return f"Total experience: {years} years {months} months"


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
