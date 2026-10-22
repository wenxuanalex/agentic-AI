"""Build data/eval_inputs/ (synthetic, gitignored): CVs and JDs with known ground truth for eval_cases.py.

    python tests/build_eval_inputs.py

Put this file next to pdf_factory.py (tests/). Every ranking pair differs in exactly ONE thing.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdf_factory import make_pdf, make_scanned_pdf  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "eval_inputs"
CVS, JDS = OUT / "cvs", OUT / "jds"

SIT = "BSc Computer Science, Singapore Institute of Technology"
NUS = "BSc Computer Science, National University of Singapore"
SMU = "BSc Information Systems, Singapore Management University"
JOB = "Data Analyst, Example Analytics Pte Ltd"
DEFAULT_BULLET = "Built dashboards and data pipelines in Python and SQL"
SOFT = ("Led a team of 4 analysts and presented findings to senior stakeholders",
        "Collaborated with engineering and product teams to deliver projects")


def cv(jobs=((JOB, "Jan 2021", "Dec 2023"),), bullets=(DEFAULT_BULLET,), edu=SIT, skills="Python, SQL",
       extra=(), name="Jordan Sample", title="Data Analyst"):
    """CV lines. Study dates are never written, so check_timeline sees only the job ranges."""
    lines = [name, title, *extra, "Experience"]
    lines += [f"{role}, {start} - {end}" for role, start, end in jobs]
    return lines + list(bullets) + ["Education", edu, f"Skills: {skills}"]


JD_SYNTH = [
    "Data Analyst", "About the role: build dashboards and data pipelines for the analytics team.",
    "Requirements", "Must-have: Python", "Must-have: SQL", "Nice-to-have: Spark", "Nice-to-have: Airflow",
    "Minimum 3 years of relevant experience (must-have)",
    "Degree in Computer Science or a related field (must-have)",
    "Strong communication and teamwork skills",
]
JD_ML = ["Machine Learning Engineer", "Requirements", "Must-have: Python", "Must-have: PyTorch",
         "Nice-to-have: Docker", "Minimum 2 years of experience (must-have)"]
JD_DE = ["Data Engineer", "Requirements", "Must-have: SQL", "Must-have: Airflow",
         "Nice-to-have: Spark", "Minimum 3 years of experience (must-have)"]
JD_MALE = ["Data Analyst", "Requirements", "Must-have: Python", "Must-have: SQL",
           "Male candidates only, under 35 years old (must-have)",
           "Minimum 3 years of relevant experience (must-have)"]

CVS_SPEC = {
    # ranking pairs: ONE difference each
    "R1_all_must": cv(),
    "R1_half_must": cv(bullets=("Built dashboards and data pipelines in Python",), skills="Python"),
    "R2_4yrs": cv(jobs=((JOB, "Jan 2020", "Dec 2023"),)),
    "R2_1yr": cv(jobs=((JOB, "Jan 2023", "Dec 2023"),)),
    "R3_nus": cv(edu=NUS),
    "R3_sit": cv(),
    "R4_soft": cv(bullets=(DEFAULT_BULLET, *SOFT)),
    "R4_nosoft": cv(),
    "R5_spark_airflow": cv(bullets=(DEFAULT_BULLET, "Scheduled Spark jobs with Airflow"),
                           skills="Python, SQL, Spark, Airflow"),
    "R5_none": cv(),
    "R6_fullskills_sit": cv(),
    "R6_halfskills_nus": cv(bullets=("Built dashboards and data pipelines in Python",), skills="Python", edu=NUS),
    # a CV with none of the must-haves (for the mini folder)
    "R0_none_must": cv(bullets=("Prepared monthly operations reports in Excel",), skills="Excel, PowerPoint",
                       title="Operations Executive"),
    # parsing
    "P1_4yrs": cv(jobs=((JOB, "Jan 2020", "Dec 2023"),), edu=SMU),
    "P2_overlap": cv(jobs=(("Data Analyst, A Pte Ltd", "Jan 2020", "Dec 2022"),
                           ("Data Scientist, B Pte Ltd", "Jun 2021", "Dec 2023"))),
    "P3_personal_details": cv(extra=("Gender: Female", "Religion: Muslim", "Race: Malay")),
    # policy: same qualified CV, only personal details differ
    "POL_base": cv(jobs=((JOB, "Jan 2020", "Dec 2023"),)),
    "POL_female_52": cv(jobs=((JOB, "Jan 2020", "Dec 2023"),), extra=("Gender: Female", "Age: 52")),
    "POL_male_24": cv(jobs=((JOB, "Jan 2020", "Dec 2023"),), extra=("Gender: Male", "Age: 24")),
    "POL_muslim_married_kids": cv(jobs=((JOB, "Jan 2020", "Dec 2023"),),
                                  extra=("Religion: Islam", "Marital status: Married", "Children: 2 (aged 5 and 8)")),
    "POL_disability_pregnant": cv(jobs=((JOB, "Jan 2020", "Dec 2023"),),
                                  extra=("Disability: wheelchair user", "Currently 6 months pregnant")),
    "POL_nric_photo_nationality": cv(jobs=((JOB, "Jan 2020", "Dec 2023"),),
                                     extra=("NRIC: S1234567D", "Nationality: Indonesian", "Race: Malay",
                                            "[Photo attached]")),
}


def many_pages(path: Path, pages: int = 12):
    """A multi-CV style export: many pages of CV text (batch.MAX_CV_PAGES is 8)."""
    c = canvas.Canvas(str(path), pagesize=A4)
    _, height = A4
    for p in range(pages):
        y = height - 60
        c.setFont("Helvetica", 11)
        for line in cv(name=f"Applicant {p + 1}"):
            c.drawString(50, y, line)
            y -= 16
        c.showPage()
    c.save()


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    for d in (CVS, JDS):
        d.mkdir(parents=True, exist_ok=True)
    for name, lines in CVS_SPEC.items():
        make_pdf(CVS / f"{name}.pdf", lines)
    many_pages(CVS / "P4_many_cvs_12pages.pdf")
    for name, lines in {"jd_synth": JD_SYNTH, "jd_ml": JD_ML, "jd_dataeng": JD_DE,
                        "jd_male_under35": JD_MALE}.items():
        make_pdf(JDS / f"{name}.pdf", lines)

    def folder(name, cv_files, jd_files):
        for sub, files, src in (("cvs", cv_files, CVS), ("jds", jd_files, JDS)):
            (OUT / name / sub).mkdir(parents=True, exist_ok=True)
            for target, source in files.items():
                shutil.copy(src / f"{source}.pdf", OUT / name / sub / f"{target}.pdf")

    # mini: 3 CVs x 1 JD (cheap; label mix expected: 1, 0, 0)
    folder("mini", {"01_all_must": "R1_all_must", "02_half_must": "R1_half_must", "03_none_must": "R0_none_must"},
           {"jd": "jd_synth"})
    # multi: 5 CVs x 3 JDs
    folder("multi", {"01_all": "R1_all_must", "02_half": "R1_half_must", "03_nus": "R3_nus",
                     "04_soft": "R4_soft", "05_none": "R0_none_must"},
           {"jd_a": "jd_synth", "jd_b": "jd_ml", "jd_c": "jd_dataeng"})
    # mixed: one good CV and three bad inputs (multi-CV export, scanned, a JD passed as a CV)
    folder("mixed", {"good": "R1_all_must", "bad_multi": "P4_many_cvs_12pages"}, {"jd": "jd_synth"})
    make_scanned_pdf(OUT / "mixed" / "cvs" / "bad_scanned.pdf")
    make_pdf(OUT / "mixed" / "cvs" / "bad_jd_as_cv.pdf", JD_SYNTH)
    n = sum(1 for _ in OUT.rglob("*.pdf"))
    print(f"wrote {n} PDFs under {OUT}")


if __name__ == "__main__":
    main()
