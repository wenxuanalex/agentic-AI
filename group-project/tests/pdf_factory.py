"""Synthetic PDFs for the guardrail tests and the attack suite (reportlab, no real applicant data)."""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# A made-up candidate. One dated job (Jan 2021 - Dec 2023 = 3 years) and NO dates next to the
# degree, so check_timeline gives exactly 3 years. University: SIT (not in the QS table -> 0.25).
CLEAN_LINES = [
    "Jordan Sample",
    "Data Analyst",
    "Experience",
    "Data Analyst, Example Analytics Pte Ltd, Jan 2021 - Dec 2023",
    "Built dashboards and data pipelines in Python and SQL",
    "Led a team of 3 analysts and presented results to stakeholders",
    "Education",
    "BSc Computer Science, Singapore Institute of Technology",
    "Skills: Python, SQL, Excel",
]


def make_pdf(path: str | Path, lines=CLEAN_LINES, hidden=()) -> Path:
    """Write a one-page PDF. lines: visible text, one per line.
    hidden: (text, how) pairs, how = white | tiny | offpage | invisible (render mode 3)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    y = height - 60
    c.setFont("Helvetica", 11)
    for line in lines:
        c.drawString(50, y, line)
        y -= 16
    for text, how in hidden:
        y -= 16
        c.saveState()  # colour / font / render mode go back to normal afterwards
        if how == "white":
            c.setFillColorRGB(1, 1, 1)
            c.drawString(50, y, text)
        elif how == "tiny":
            c.setFont("Helvetica", 1)
            c.drawString(50, y, text)
        elif how in ("white_on_dark", "white_on_white_box"):  # a box behind the white text
            c.setFillColorRGB(0.1, 0.2, 0.4)
            c.rect(45, y - 4, 400, 15, fill=1, stroke=0)
            if how == "white_on_white_box":                    # dark box hidden under a white one
                c.setFillColorRGB(1, 1, 1)
                c.rect(45, y - 4, 400, 15, fill=1, stroke=0)
            c.setFillColorRGB(1, 1, 1)
            c.drawString(50, y, text)
        elif how == "offpage":
            c.drawString(width + 100, y, text)
        elif how == "invisible":
            t = c.beginText(50, y)
            t.setTextRenderMode(3)
            t.textLine(text)
            c.drawText(t)
        else:
            raise ValueError(how)
        c.restoreState()
    c.save()
    return path


def make_scanned_pdf(path: str | Path) -> Path:
    """A PDF with no text layer at all (like a scanned page): only a drawn box."""
    path = Path(path)
    c = canvas.Canvas(str(path), pagesize=A4)
    c.rect(50, 50, 200, 100, fill=1)
    c.save()
    return path


def make_jd(path: str | Path) -> Path:
    return make_pdf(path, lines=["Data Analyst", "Requirements (must-have): Python, SQL, Spark",
                                 "3+ years of experience"])
