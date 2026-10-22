from __future__ import annotations

import os
import asyncio
import base64
import re
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Optional, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field, SecretStr
from enum import Enum

try:
    from app import paths
except:
    import paths

load_dotenv(override=True)

MAX_CONCURRENT = 10  # cap simultaneous OpenAI requests
MASK = "[MASKED]"

# ---------- Structured output schema ----------
class Requirement(str, Enum):
    mandatory = "mandatory"
    optional = "optional"

class TaggedItem(BaseModel):
    value: str = Field(description="The requirement text as stated.")
    requirement: Requirement = Field(description="Whether this requirement is mandatory or optional/preferred.")

class CVEntry(BaseModel):
    candidate_name: Optional[str] = Field(None, description="Candidate full name if identifiable")
    years_of_experience: Optional[float] = Field(None, description="Total years of professional experience, if inferable")
    skills: List[str] = Field(default_factory=list, description="Technical and soft skills explicitly listed")
    education: List[str] = Field(default_factory=list, description="Degrees, institutions, and graduation info")
    full_text: str = Field(description="The complete raw text content of this CV")

class CVExtractionResult(BaseModel):
    cvs: List[CVEntry] = Field(description="All CVs found in the PDF, in order of appearance")

class JDEntry(BaseModel):
    role: Optional[str] = Field(None, description="Job title / role")
    job_scope_description: Optional[str] = Field(None, description="Overall scope and description of the job")
    skills: List[TaggedItem] = Field(default_factory=list, description="Skills required or mentioned")
    experience: Optional[TaggedItem] = Field(None, description="Experience requirement as stated, e.g. '5+ years'")
    education: Optional[TaggedItem] = Field(None, description="Education requirement as stated, e.g. 'BSc in CS'")
    # full_text: str = Field(description="The complete raw text content of this JD")

class JDExtractionResult(BaseModel):
    jds: List[JDEntry] = Field(description="All JDs found in the PDF, in order of appearance")

# ---------- State ----------
class CVAgentState(TypedDict):
    pdf_base64: str
    filename: str
    result: Optional[CVExtractionResult]

class JDAgentState(TypedDict):
    pdf_base64: str
    filename: str
    result: Optional[JDExtractionResult]


# ---------- LLM ----------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")

llm = ChatOpenAI(
    base_url = os.getenv("OPENAI_API_BASE_URL"),
    api_key = SecretStr(api_key),
    model = "gpt-5-mini", temperature=0, use_responses_api=True)

structured_cv_llm = llm.with_structured_output(CVExtractionResult, method="json_schema")
structured_jd_llm = llm.with_structured_output(JDExtractionResult, method="json_schema")

# ---------- Node ----------
async def extract_cvs(state: CVAgentState) -> dict:
    file_block = {
        "type": "input_file",
        "filename": state["filename"],
        "file_data": f"data:application/pdf;base64,{state['pdf_base64']}",
    }
    text_block = {
        "type": "text",
        "text": (
            "This PDF may contain one or more CVs/resumes. "
            "Identify every distinct CV and extract the requested fields, "
            "including the full text of each CV.\n\n"
            "DOCUMENT-TYPE VALIDATION:\n"
            "- Only extract documents that are actually candidate CVs/resumes.\n"
            "- A job description, job advertisement, role specification, or "
            "list of hiring requirements is not a CV.\n"
            "- Do not convert a job description into a CV just because it "
            "mentions skills, qualifications, education, or experience.\n"
            "- If the PDF contains no CVs, return an empty 'cvs' list.\n"
            "- If the PDF contains multiple distinct CVs, return each CV "
            "separately.\n"
            "- Do not invent candidate information that is absent from the PDF."
        )
    }
    message = {
        "role": "user",
        "content": [
            file_block,
            text_block
        ]
    }
    result = await structured_cv_llm.ainvoke([message])
    result = mask_cv_result(result)
    return {"result": result}

async def extract_jds(state: JDAgentState) -> dict:
    file_block = {
        "type": "input_file",
        "filename": state["filename"],
        "file_data": f"data:application/pdf;base64,{state['pdf_base64']}",
    }    
    text_block = {
        "type": "text",
        "text": ( """
            This PDF may contain one or more job descriptions (JDs).
            Identify every distinct JD and extract the requested fields.

            For each skill, experience requirement, and education requirement,
            tag it as 'mandatory' if the JD presents it as required, or
            'optional' if presented as preferred, nice-to-have, a plus, or optional.
            If the requirement's status is unclear, default to 'mandatory'.

            EXPERIENCE EXTRACTION RULES:
            1. Only populate the experience field when the JD explicitly states
            an experience requirement.
            2. Do not classify a statement as an experience requirement merely
            because it contains phrases such as 'experience building',
            'experience with', 'familiarity with', 'proficiency in', or
            'ability to'. These statements usually describe skills or
            capabilities and should be extracted under skills instead.
            3. Statements describing technical activities, responsibilities,
            or capabilities must not be converted into years-of-experience
            requirements.
            4. Never infer a minimum number of years from the job title,
            seniority, responsibilities, qualifications, or other context.
            5. Do not invent or expand requirements beyond what the JD states.
            6. If the JD does not explicitly state an experience requirement,
            return experience as null.
            7. If a qualitative experience requirement is explicitly stated,
            preserve its original meaning without inventing a number of years.
            8. Before returning the result, verify that every non-null
            experience value is supported by the source text.

            For example, 'Experience building and evaluating ML/AI models'
            describes a skill or capability and should normally be extracted
            under skills, not experience.

            Return the extracted information using the specified output schema.
            """
        )
    }
    message = {
        "role": "user",
        "content": [
            file_block,
            text_block
        ]
    }
    result = await structured_jd_llm.ainvoke([message])
    result = mask_jd_result(result)
    return {"result": result}


# ---------- Graph ----------
cv_builder = StateGraph(CVAgentState)
cv_builder.add_node("extract_cvs", extract_cvs)
cv_builder.add_edge(START, "extract_cvs")
cv_builder.add_edge("extract_cvs", END)
cv_graph = cv_builder.compile()

jd_builder = StateGraph(JDAgentState)
jd_builder.add_node("extract_jds", extract_jds)
jd_builder.add_edge(START, "extract_jds")
jd_builder.add_edge("extract_jds", END)
jd_graph = jd_builder.compile()


# ---------- Validation & Masking ----------
MAX_PDF_SIZE_MB = 40

def validate_pdf_size(pdf_path: Path, max_mb: int = MAX_PDF_SIZE_MB) -> None:
    size_mb = pdf_path.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        raise ValueError(f"{pdf_path.name} is {size_mb:.1f} MB, exceeds {max_mb} MB")

SEX_PATTERN = re.compile(
    r"\b(male|female|man|woman|men|women|boy|girl|non-?binary|genderqueer|transgender|trans|he/him|she/her|they/them)\b",
    re.IGNORECASE,
)
AGE_PATTERN = re.compile(
    r"\b(\d{1,2})\s*(years?|yrs?)\s*old\b|"
    r"\b(age[d]?\s*:?\s*\d{1,2})\b|"
    r"\b(early|mid|late)[- ](twenties|thirties|forties|fifties|sixties)\b",
    re.IGNORECASE,
)
RACE_PATTERN = re.compile(
    r"\b(Chinese|Malay|Indian|Eurasian|Sikh|Asian|African|African[- ]American|Caucasian|Hispanic|Latino|Latina|White|Black|Indigenous|Native American|Pacific Islander|Middle Eastern|South Asian|East Asian)\b",
    re.IGNORECASE,
)
RELIGION_PATTERN = re.compile(
    r"\b(Christian|Christianity|Catholic|Protestant|Muslim|Islamic|Islam|Jewish|Judaism|Hindu|Hinduism|Buddhist|Buddhism|Sikh|Sikhism|Atheist|Agnostic|Mormon|Evangelical)\b",
    re.IGNORECASE,
)
SENSITIVE_PATTERNS = [SEX_PATTERN, AGE_PATTERN, RACE_PATTERN, RELIGION_PATTERN]

def mask_sensitive_text(text: Optional[str]) -> Optional[str]:
    """Mask sex, age, race, and religion references in a single string."""
    if not text:
        return text
    for pattern in SENSITIVE_PATTERNS:
        text = pattern.sub(MASK, text)
    return text

def mask_tagged_item(item: Optional[TaggedItem]) -> Optional[TaggedItem]:
    if item is not None:
        item.value = mask_sensitive_text(item.value)
    return item

def mask_tagged_items(items: List[TaggedItem]) -> List[TaggedItem]:
    for item in items:
        item.value = mask_sensitive_text(item.value)
    return items

def mask_jd_entry(jd: JDEntry) -> JDEntry:
    jd.role = mask_sensitive_text(jd.role)
    jd.job_scope_description = mask_sensitive_text(jd.job_scope_description)
    jd.experience = mask_tagged_item(jd.experience)
    jd.education = mask_tagged_item(jd.education)
    jd.skills = mask_tagged_items(jd.skills)
    return jd

def mask_jd_result(result: JDExtractionResult) -> JDExtractionResult:
    """Apply masking to every JD in the extraction result."""
    for jd in result.jds:
        mask_jd_entry(jd)
    return result

def mask_cv_entry(cv: CVEntry) -> CVEntry:
    """Mask all free-text fields of a single CV entry in place."""
    cv.candidate_name = mask_sensitive_text(cv.candidate_name)
    cv.skills = [mask_sensitive_text(s) for s in cv.skills]
    cv.education = [mask_sensitive_text(e) for e in cv.education]
    cv.full_text = mask_sensitive_text(cv.full_text)
    return cv

def mask_cv_result(result: CVExtractionResult) -> CVExtractionResult:
    """Apply masking to every CV in the extraction result."""
    for cv in result.cvs:
        mask_cv_entry(cv)
    return result


# ---------- Async batch processing with semaphore ----------
async def process_one(pdf_path: Path, sem: asyncio.Semaphore, graph, state_key: str = 'result') -> tuple[str, object]:
    """Process a single PDF while respecting the concurrency limit."""
    async with sem:  # blocks here if MAX_CONCURRENT requests are already in flight
        try:
            with open(pdf_path, "rb") as f:
                pdf_b64 = base64.b64encode(f.read()).decode("utf-8")

            output = await graph.ainvoke({
                "pdf_base64": pdf_b64,
                "filename": pdf_path.name,
            })
            return pdf_path.name, output[state_key]
        except Exception as e:
            return pdf_path.name, e


async def process_folder(folder_path: str, graph, sem: asyncio.Semaphore) -> dict:
    pdf_files = sorted(Path(folder_path).glob("*.pdf"))
    valid_files = []
    for p in pdf_files:
        try:
            validate_pdf_size(p)
            valid_files.append(p)
        except ValueError as e:
            print(f"Skipping {p.name}: {e}")

    results = {}
    # Fire all tasks at once; semaphore throttles actual concurrency
    tasks = [process_one(p, sem, graph) for p in valid_files]
    for name, result in await asyncio.gather(*tasks):
        if isinstance(result, Exception):
            print(f"Failed {name}: {result}")
        else:
            results[name] = result
            # print(f"Processed {name}")
    return results


# ---------- Main running ----------
async def main():
    CV_FOLDER = str(paths.CV_DIR)
    JD_FOLDER = str(paths.JD_DIR)

    sem = asyncio.Semaphore(MAX_CONCURRENT)  # shared across both pipelines
    cv_results, jd_results = await asyncio.gather(
        process_folder(CV_FOLDER, cv_graph, sem),
        process_folder(JD_FOLDER, jd_graph, sem),
    )
    return cv_results, jd_results


if __name__ == "__main__":
    cv_results, jd_results = asyncio.run(main())

    print("\n=== CVs ===")
    for filename, result in cv_results.items():
        print(f"\n{filename}: {len(result.cvs)} CV(s)")
        for i, cv in enumerate(result.cvs):
            print(f"  CV #{i+1}: {cv.candidate_name} | "
                  f"{cv.years_of_experience} yrs | {len(cv.skills)} skills | Skills:{cv.skills} | Education:{cv.education}")
            # print(f"Full: {cv.full_text}\n\n")

    print("\n=== JDs ===")
    for filename, result in jd_results.items():
        print(f"{filename}: {len(result.jds)} JD(s)")
        for i, jd in enumerate(result.jds):
            print(f"  JD #{i+1}: {jd.role}")
            print(f"    Description: {jd.job_scope_description}")
            print("    Skills:")
            for s in jd.skills:
                print(f"      - [{s.requirement.value}] {s.value}")
            if jd.experience is not None:
                print(f"    Experience: [{jd.experience.requirement.value}] {jd.experience.value}")
            else:
                print("    Experience: (none)")
            if jd.education is not None:
                print(f"    Education: [{jd.education.requirement.value}] {jd.education.value}")
            else:
                print("    Education: (none)")
