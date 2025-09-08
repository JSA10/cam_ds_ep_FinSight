#!/usr/bin/env python3
"""
JPM (and similar) earnings call PDF → tidy CSV parser.

Includes:
- Robust text extraction (PyMuPDF → pdfminer.six → PyPDF2 → pdftotext, with optional OCR)
- Q&A parsing with speaker/role/company attribution
- Presentation extraction before the Q&A:
    --include-presentation {none|single_row|per_speaker}  (default: single_row)
    * Recognizes headings: "PRESENTATION", "PREPARED REMARKS", and
      JPM's "MANAGEMENT DISCUSSION SECTION"
- Pleasantry labeling + resequencing so pleasantries never create new Qs:
    --pleasantry-mode {keep_raw|label_only|label_and_resequence}  (default: label_and_resequence)

Outputs columns:
    section, question_number, answer_number, speaker_name, role, company, content, year, quarter,
    is_pleasantry, is_intro  (the last two depend on pleasantry-mode)

Batch mode:
    Write one CSV per PDF (and optional combined CSV with a source_pdf column).

Install:
    pip install pymupdf pdfminer.six PyPDF2 pandas pillow pytesseract
    # If using --ocr, also install the Tesseract binary (brew/apt/choco).

Examples:
    # Single file (keep presentation + label/resequence pleasantries)
    python jpm_earnings_qa_parser.py ./jpm-2q25-earnings-call-transcript.pdf

    # Batch a folder recursively with combined CSV
    python jpm_earnings_qa_parser.py ./transcripts \
      --batch --recursive --outdir ./qa_csvs --combined-out ./all_qna.csv \
      --prefer pymupdf --ocr --max-ocr-pages 6 \
      --include-presentation single_row \
      --pleasantry-mode label_and_resequence
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd

# --- Canonical labels for prepared remarks (presentation) ---
PRESENTATION_SPEAKER_NAME = "Jeremy Barnum"
PRESENTATION_ROLE = "Chief Financial Officer"

# ------------------------------------------------------------
# Text extraction
# ------------------------------------------------------------
def _extract_with_pymupdf(pdf_path: Path) -> str:
    try:
        import fitz  # PyMuPDF
    except Exception:
        return ""
    try:
        doc = fitz.open(str(pdf_path))
        parts = []
        for page in doc:
            t = page.get_text("text") or page.get_text("block") or ""
            parts.append(t)
        return "\n".join(parts)
    except Exception:
        return ""


def _extract_with_pdfminer(pdf_path: Path) -> str:
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text
        return pdfminer_extract_text(str(pdf_path)) or ""
    except Exception:
        return ""


def _extract_with_pypdf2(pdf_path: Path) -> str:
    try:
        import PyPDF2
    except Exception:
        return ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            parts = []
            for p in reader.pages:
                try:
                    parts.append(p.extract_text() or "")
                except Exception:
                    parts.append("")
        return "\n".join(parts)
    except Exception:
        return ""


def _extract_with_pdftotext(pdf_path: Path) -> str:
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.decode("utf-8", errors="ignore")
        return ""
    except Exception:
        return ""


def _ocr_with_tesseract(pdf_path: Path, max_pages: int = 6, lang: str = "eng") -> str:
    try:
        import fitz  # PyMuPDF
        from PIL import Image
        import pytesseract
    except Exception:
        return ""
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return ""

    texts = []
    pages_to_ocr = min(len(doc), max_pages)
    for i in range(pages_to_ocr):
        page = doc[i]
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        try:
            t = pytesseract.image_to_string(img, lang=lang)
        except Exception:
            t = ""
        texts.append(t)
    return "\n".join(texts)


def extract_text_from_pdf(
    pdf_path: Path,
    prefer: Optional[str] = None,
    ocr: bool = False,
    ocr_lang: str = "eng",
    max_ocr_pages: int = 6,
) -> Tuple[str, str]:
    """
    Return (text, method_used). prefer ∈ {'pymupdf','pdfminer','pypdf2','pdftotext'}.
    """
    methods = [
        ("pymupdf", _extract_with_pymupdf),
        ("pdfminer", _extract_with_pdfminer),
        ("pypdf2", _extract_with_pypdf2),
        ("pdftotext", _extract_with_pdftotext),
    ]
    if prefer:
        methods.sort(key=lambda x: 0 if x[0] == (prefer or "").lower() else 1)

    for name, fn in methods:
        text = fn(pdf_path)
        if text and text.strip():
            return text, name

    if ocr:
        text = _ocr_with_tesseract(pdf_path, max_pages=max_ocr_pages, lang=ocr_lang)
        if text and text.strip():
            return text, f"ocr({ocr_lang})"

    return "", "none"


# ------------------------------------------------------------
# Common helpers
# ------------------------------------------------------------
def clean_lines(raw_text: str) -> List[str]:
    text = raw_text.replace("\r", "\n")
    text = re.sub(r"[.\u2026]{10,}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [ln.strip() for ln in text.split("\n")]
    cleaned: List[str] = []
    for ln in lines:
        if not ln:
            cleaned.append("")
            continue
        if re.fullmatch(r"\d{1,3}", ln):
            cleaned.append("")  # drop bare page numbers
            continue
        cleaned.append(ln)
    return cleaned


def slice_to_qa_section(lines: List[str]) -> List[str]:
    start_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().upper() == "QUESTION AND ANSWER SECTION":
            start_idx = i
            break
    return lines[start_idx:] if start_idx is not None else lines


def is_label_line_eol(ln: str) -> bool:
    return bool(re.search(r"\s[QA]$", ln.strip()))


def is_label_line_solo(ln: str) -> bool:
    return ln.strip() in {"Q", "A"}


def is_any_label_line(ln: str) -> bool:
    return is_label_line_solo(ln) or is_label_line_eol(ln)


def looks_like_name(s: str) -> bool:
    s = s.strip()
    if not s or ":" in s or len(s.split()) < 2 or len(s.split()) > 6:
        return False
    return all(tok and tok[0].isupper() for tok in s.split())


def prev_nonempty(lines: List[str], idx: int) -> Tuple[Optional[int], Optional[str]]:
    j = idx
    while j >= 0:
        s = lines[j].strip()
        if s:
            return j, s
        j -= 1
    return None, None


def next_nonempty(lines: List[str], start: int) -> Optional[int]:
    j = start
    while j < len(lines):
        if lines[j].strip():
            return j
        j += 1
    return None


def parse_company_and_role(rc_line: str, qa_flag: str) -> Tuple[Optional[str], Optional[str]]:
    core = rc_line.strip()
    if qa_flag == "Q":
        role = "analyst"
        company = None
        if "," in core:
            parts = core.split(",", 1)
            company = parts[1].strip() if len(parts) > 1 else None
        return company, role
    else:
        company = None
        role = core
        if "," in core:
            role_part, company_part = core.split(",", 1)
            role = role_part.strip()
            company = company_part.strip()
        return company, role


def header_triple_detected(lines: List[str], j: int) -> bool:
    j_name = next_nonempty(lines, j)
    if j_name is None:
        return False
    j_rc = next_nonempty(lines, j_name + 1)
    if j_rc is None:
        return False
    j_flag = next_nonempty(lines, j_rc + 1)
    if j_flag is None:
        return False
    name_ok = looks_like_name(lines[j_name].strip())
    rc_ok = ("," in lines[j_rc])
    flag_ok = is_label_line_solo(lines[j_flag]) or is_label_line_eol(lines[j_flag])
    return bool(name_ok and rc_ok and flag_ok)

def postprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize output:
      - Force speaker_name + role for presentation section
      - Remove deprecated columns: is_intro, speaker, speaker_title
      - Light normalization of key columns
    """
    # Normalize section
    if "section" in df.columns:
        df["section"] = df["section"].astype(str).str.strip().str.lower()

    # Ensure the canonical columns exist
    if "speaker_name" not in df.columns and "speaker" in df.columns:
        df = df.rename(columns={"speaker": "speaker_name"})
    if "role" not in df.columns and "speaker_title" in df.columns:
        df = df.rename(columns={"speaker_title": "role"})

    # Clean strings
    if "speaker_name" in df.columns:
        df["speaker_name"] = df["speaker_name"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    if "role" in df.columns:
        df["role"] = df["role"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

    # Canonicalize prepared remarks
    if "section" in df.columns:
        pres_mask = df["section"].eq("presentation")
        if pres_mask.any():
            df.loc[pres_mask, "speaker_name"] = PRESENTATION_SPEAKER_NAME
            df.loc[pres_mask, "role"] = PRESENTATION_ROLE

    # Drop unused columns if they still exist
    drop_cols = [c for c in ["is_intro", "speaker", "speaker_title"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


# ------------------------------------------------------------
# Presentation extraction (before Q&A)
# ------------------------------------------------------------
PRESENTATION_HEADINGS = {
    "PRESENTATION",
    "PREPARED REMARKS",
    "PREPARED REMARKS:",
    "MANAGEMENT DISCUSSION SECTION",
    "MANAGEMENT DISCUSSION SECTION:",
}

def _find_qa_start_idx(lines: List[str]) -> Optional[int]:
    for i, ln in enumerate(lines):
        if ln.strip().upper() == "QUESTION AND ANSWER SECTION":
            return i
    return None


def _find_presentation_start_idx(lines: List[str]) -> int:
    # Prefer explicit headings (now includes JPM's "MANAGEMENT DISCUSSION SECTION")
    for i, ln in enumerate(lines[:300]):
        if ln.strip().upper() in PRESENTATION_HEADINGS:
            return i + 1
    # Otherwise: first Name + "Role, Company" header pair
    n = len(lines)
    for i in range(min(300, n - 2)):
        name = lines[i].strip()
        role = lines[i + 1].strip()
        if looks_like_name(name) and ("," in role):
            return i + 2
    return 0


def _parse_presentation_blocks(lines_before_qa: List[str], default_company="JPMorganChase") -> List[Dict[str, str]]:
    SKIP = {
        "COMPANY PARTICIPANTS",
        "CONFERENCE CALL PARTICIPANTS",
        "PRESENTATION",
        "PREPARED REMARKS",
        "PREPARED REMARKS:",
        "MANAGEMENT DISCUSSION SECTION",
        "MANAGEMENT DISCUSSION SECTION:",
    }
    clean = [ln for ln in lines_before_qa if ln and ln.strip().upper() not in SKIP and not ln.startswith("Operator:")]

    blocks: List[Dict[str, str]] = []
    i, n = 0, len(clean)
    while i < n - 2:
        name = clean[i].strip()
        role_line = clean[i + 1].strip()
        if looks_like_name(name) and ("," in role_line):
            # role/company split
            role, company = role_line.split(",", 1)
            role, company = role.strip(), (company.strip() or default_company)
            j = i + 2
            content_lines: List[str] = []
            # collect until next header (name+role) or end
            while j < n - 1:
                if looks_like_name(clean[j].strip()) and ("," in clean[j + 1].strip()):
                    break
                content_lines.append(clean[j])
                j += 1
            content = " ".join(x.strip() for x in content_lines if x.strip())
            if content:
                blocks.append({"speaker_name": name, "role": role, "company": company, "content": content})
            i = j
        else:
            i += 1

    if not blocks:
        body = " ".join(clean).strip()
        if body:
            blocks = [{
                "speaker_name": "Management",
                "role": "prepared_remarks",
                "company": default_company,
                "content": body
            }]
    return blocks


def extract_presentation_blocks_from_text(all_lines: List[str], default_company="JPMorganChase") -> List[Dict[str, str]]:
    qa_idx = _find_qa_start_idx(all_lines)
    before_qa = all_lines[:qa_idx] if qa_idx is not None else all_lines
    start_idx = _find_presentation_start_idx(before_qa)
    pres_slice = before_qa[start_idx:]
    return _parse_presentation_blocks(pres_slice, default_company=default_company)


# ------------------------------------------------------------
# Pleasantry labeling + resequencing
# ------------------------------------------------------------
QUESTION_CUES = re.compile(
    r"\?|(^|\b)(what|how|why|when|where|which|could you|can you|would you|"
    r"give us|talk about|walk us|update on|help us|color on|outlook|guidance|"
    r"drivers?|puts and takes|bridge|framework)\b", re.I
)

PLEASANTRY_START = re.compile(
    r"^\s*(thanks( very much)?|thank you|appreciate|welcome|"
    r"(good\s+)?(morning|afternoon|evening)|hi|hello|hey|"
    r"congrats|congratulations|great\.?|okay\.?|ok\.?|sure\.?|yeah\.?|right\.)\b", re.I
)

def _looks_substantive_question(text: str) -> bool:
    return isinstance(text, str) and bool(QUESTION_CUES.search(text))


def _looks_pleasantry(text: str, max_words: int = 18) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    t = text.strip()
    if "?" in t:
        return False
    if not PLEASANTRY_START.search(t):
        return False
    word_count = len(re.findall(r"\w+", t))
    return word_count <= max_words


def label_pleasantries(df: pd.DataFrame, max_words: int = 18) -> pd.DataFrame:
    df = df.copy()
    df["is_pleasantry"] = df["content"].apply(lambda s: _looks_pleasantry(s, max_words))
    return df


def resequence_qa_ignoring_pleasantries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute question_number / answer_number for section=='qa' rows while ignoring pleasantries:
      - pleasantries never start a new question
      - analyst pleasantries after answers are tagged with the current question but don't bump counters
      - internal pleasantries inherit current a_num without bumping
    """
    df = df.copy()
    mask_qa = (df["section"] == "qa")
    q = a = 0
    # reset
    df.loc[mask_qa, "question_number"] = pd.NA
    df.loc[mask_qa, "answer_number"] = pd.NA

    for i, r in df[mask_qa].iterrows():
        role = str(r["role"]).lower()
        text = r.get("content", "")
        is_p = bool(r.get("is_pleasantry"))
        if role == "analyst":
            if _looks_substantive_question(text) and not is_p:
                q += 1; a = 0
                df.at[i, "question_number"] = q
            else:
                if q > 0:
                    df.at[i, "question_number"] = q
                    if a > 0:
                        df.at[i, "answer_number"] = a
        else:
            if q > 0:
                if not is_p:
                    a += 1
                df.at[i, "question_number"] = q
                if a > 0:
                    df.at[i, "answer_number"] = a
    # Cast to nullable ints
    df["question_number"] = df["question_number"].astype("Int64")
    df["answer_number"] = df["answer_number"].astype("Int64")
    return df


# ------------------------------------------------------------
# Q&A core parsing
# ------------------------------------------------------------
def collect_blocks(lines: List[str]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i].strip()

        # Case 1: 'Q' or 'A' on its own line
        if is_label_line_solo(ln):
            qa_flag = ln
            rc_idx, rc_line = prev_nonempty(lines, i - 1)
            name_idx, name_line = prev_nonempty(lines, (rc_idx - 1) if rc_idx is not None else -1)
            if not rc_line or not name_line or name_line.startswith("Operator:"):
                i += 1
                continue

            company, role = parse_company_and_role(rc_line, qa_flag)

            content_lines: List[str] = []
            j = i + 1
            while j < n:
                if is_any_label_line(lines[j]):
                    break
                if header_triple_detected(lines, j):
                    break
                if lines[j].strip() in {name_line.strip(), rc_line.strip()}:
                    j += 1
                    continue
                content_lines.append(lines[j])
                j += 1

            content = " ".join(x.strip() for x in content_lines if x.strip())
            content = re.sub(r"\s{2,}", " ", content).strip()

            items.append(
                {
                    "type": qa_flag,
                    "speaker_name": name_line.strip(),
                    "role": "analyst" if qa_flag == "Q" else (role or ""),
                    "company": (company or ("JPMorganChase" if qa_flag == "A" else None)),
                    "content": content,
                }
            )
            i = j
            continue

        # Case 2: ' Q' or ' A' at end of line
        if is_label_line_eol(ln):
            qa_flag = ln[-1]
            name_idx, name_line = prev_nonempty(lines, i - 1)
            rc_line = ln.rsplit(" ", 1)[0]
            if not name_line or name_line.startswith("Operator:"):
                i += 1
                continue
            company, role = parse_company_and_role(rc_line, qa_flag)

            content_lines: List[str] = []
            j = i + 1
            while j < n and not is_any_label_line(lines[j]) and not header_triple_detected(lines, j):
                if lines[j].strip() in {name_line.strip(), rc_line.strip()}:
                    j += 1
                    continue
                content_lines.append(lines[j])
                j += 1
            content = " ".join(x.strip() for x in content_lines if x.strip())
            content = re.sub(r"\s{2,}", " ", content).strip()

            items.append(
                {
                    "type": qa_flag,
                    "speaker_name": name_line.strip(),
                    "role": "analyst" if qa_flag == "Q" else (role or ""),
                    "company": (company or ("JPMorganChase" if qa_flag == "A" else None)),
                    "content": content,
                }
            )
            i = j
            continue

        i += 1

    return items


def merge_consecutive_questions(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for it in items:
        if it["type"] == "Q" and merged and merged[-1]["type"] == "Q":
            if merged[-1]["speaker_name"] == it["speaker_name"]:
                merged[-1]["content"] = (merged[-1]["content"] + " " + it["content"]).strip()
            else:
                merged.append(it)
        else:
            merged.append(it)
    return merged


def to_dataframe(items: List[Dict[str, Any]], year: int, quarter: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    q_counter = 0
    a_counter = 0
    for it in items:
        if it["type"] == "Q":
            q_counter += 1
            a_counter = 0
            rows.append(
                {
                    "section": "qa",
                    "question_number": q_counter,
                    "answer_number": pd.NA,
                    "speaker_name": it.get("speaker_name", ""),
                    "role": "analyst",
                    "company": it.get("company"),
                    "content": it.get("content", ""),
                    "year": year,
                    "quarter": quarter,
                }
            )
        else:
            a_counter += 1
            rows.append(
                {
                    "section": "qa",
                    "question_number": q_counter if q_counter > 0 else pd.NA,
                    "answer_number": a_counter,
                    "speaker_name": it.get("speaker_name", ""),
                    "role": it.get("role", ""),
                    "company": it.get("company") or "JPMorganChase",
                    "content": it.get("content", ""),
                    "year": year,
                    "quarter": quarter,
                }
            )
    df = pd.DataFrame(rows)
    df["question_number"] = df["question_number"].astype("Int64")
    df["answer_number"] = df["answer_number"].astype("Int64")
    return df


def infer_year_quarter_from_filename(path: Path) -> Tuple[Optional[int], Optional[str]]:
    name = path.name.lower()
    m = re.search(r'(\d)q(\d{2,4})', name)
    if not m:
        return None, None
    qnum = int(m.group(1))
    yy = m.group(2)
    year = 2000 + int(yy) if len(yy) == 2 else int(yy)
    quarter = f"Q{qnum}"
    return year, quarter


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def parse_pdf_to_df(
    pdf_path: Path,
    year: Optional[int] = None,
    quarter: Optional[str] = None,
    prefer: Optional[str] = None,
    ocr: bool = False,
    ocr_lang: str = "eng",
    max_ocr_pages: int = 6,
    include_presentation: str = "single_row",   # none|single_row|per_speaker
    pleasantry_mode: str = "label_and_resequence",  # keep_raw|label_only|label_and_resequence
    verbose: bool = True,
) -> pd.DataFrame:
    # 1) Extract text once
    text, method = extract_text_from_pdf(pdf_path, prefer=prefer, ocr=ocr, ocr_lang=ocr_lang, max_ocr_pages=max_ocr_pages)
    if verbose:
        print(f"Extraction method for {pdf_path.name}: {method}")
    if not text or not text.strip():
        raise RuntimeError(f"Unable to extract text from PDF: {pdf_path}")

    # 2) Preprocess lines
    lines = clean_lines(text)

    # 3) Q&A parse
    qa_lines = slice_to_qa_section(lines)
    items = collect_blocks(qa_lines)
    items = merge_consecutive_questions(items)

    inf_year, inf_quarter = infer_year_quarter_from_filename(pdf_path)
    year = year if year is not None else (inf_year if inf_year is not None else pd.NA)
    quarter = quarter if quarter is not None else (inf_quarter if inf_quarter is not None else pd.NA)

    df_qa = to_dataframe(items, year=year, quarter=quarter)

    # 4) Presentation extraction (optional)
    pres_rows: List[Dict[str, Any]] = []
    if include_presentation and include_presentation.lower() != "none":
        blocks = extract_presentation_blocks_from_text(lines, default_company="JPMorganChase")
        if include_presentation.lower() == "per_speaker" and len(blocks) > 1:
            for b in blocks:
                if not b.get("content"):
                    continue
                pres_rows.append({
                    "section": "presentation",
                    "question_number": pd.NA,
                    "answer_number": pd.NA,
                    "speaker_name": b["speaker_name"],
                    "role": b.get("role") or "prepared_remarks",
                    "company": b.get("company") or "JPMorganChase",
                    "content": b["content"],
                    "year": year,
                    "quarter": quarter,
                })
        else:
            # single_row: collapse all content
            combined = " ".join([b["content"] for b in blocks if b.get("content")]).strip()
            spk = ", ".join([b["speaker_name"] for b in blocks]) if blocks else "Management"
            pres_rows.append({
                "section": "presentation",
                "question_number": pd.NA,
                "answer_number": pd.NA,
                "speaker_name": spk or "Management",
                "role": "prepared_remarks",
                "company": "JPMorganChase",
                "content": combined,
                "year": year,
                "quarter": quarter,
            })

    if pres_rows:
        pres_df = pd.DataFrame(pres_rows)
    
        # Ensure the same columns exist as in df_qa
        for col in df_qa.columns:
            if col not in pres_df.columns:
                pres_df[col] = pd.NA
    
        # Enforce dtypes for integer-ish columns to avoid dtype inference surprises
        for col in ["question_number", "answer_number", "year"]:
            if col in pres_df.columns and str(df_qa[col].dtype) == "Int64":
                pres_df[col] = pres_df[col].astype("Int64")
    
        # Match column order, then concat
        pres_df = pres_df[df_qa.columns]
        df = pd.concat([pres_df, df_qa], ignore_index=True)
    else:
        df = df_qa


    # 5) Pleasantry handling
    mode = (pleasantry_mode or "keep_raw").lower()
    if mode in {"label_only", "label_and_resequence"}:
        df = label_pleasantries(df, max_words=18)
        # Mark intro pleasantries (those before first QA question starts)
        first_q_idx = df.index[(df["section"] == "qa") & df["question_number"].notna()].min() \
                      if ((df["section"] == "qa") & df["question_number"].notna()).any() else None
        df["is_intro"] = False
        if first_q_idx is not None:
            intro_mask = df.index < first_q_idx
            df.loc[intro_mask & (df["section"] == "qa") & df["is_pleasantry"].fillna(False), "is_intro"] = True

        if mode == "label_and_resequence":
            df = resequence_qa_ignoring_pleasantries(df)
    else:
        # ensure columns exist for schema consistency
        if "is_pleasantry" not in df.columns:
            df["is_pleasantry"] = False
        if "is_intro" not in df.columns:
            df["is_intro"] = False

    # Final dtypes
    if "question_number" in df.columns:
        df["question_number"] = df["question_number"].astype("Int64")
    if "answer_number" in df.columns:
        df["answer_number"] = df["answer_number"].astype("Int64")

    # Final tidy-up
    df = postprocess_df(df)

    return df


# ------------------------------------------------------------
# Batch mode utilities
# ------------------------------------------------------------
def iter_pdfs(folder: Path, recursive: bool, pattern: Optional[str]) -> List[Path]:
    if recursive:
        files = list(folder.rglob("*.pdf"))
    else:
        files = list(folder.glob("*.pdf"))
    if pattern:
        import fnmatch
        files = [f for f in files if fnmatch.fnmatch(f.name.lower(), pattern.lower())]
    return sorted(files)


def batch_process(
    input_path: Path,
    outdir: Optional[Path],
    recursive: bool,
    pattern: Optional[str],
    combined_out: Optional[Path],
    prefer: Optional[str] = None,
    ocr: bool = False,
    ocr_lang: str = "eng",
    max_ocr_pages: int = 6,
    include_presentation: str = "single_row",   # none|single_row|per_speaker
    pleasantry_mode: str = "label_and_resequence",  # keep_raw|label_only|label_and_resequence
) -> Path:
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    all_rows: List[pd.DataFrame] = []
    if input_path.is_file():
        pdfs = [input_path]
    elif input_path.is_dir():
        pdfs = iter_pdfs(input_path, recursive=recursive, pattern=pattern)
    else:
        raise SystemExit(f"Path not found: {input_path}")

    if not pdfs:
        raise SystemExit("No PDF files found to process.")

    for pdf in pdfs:
        try:
            print(f"Parsing: {pdf}")
            df = parse_pdf_to_df(
                pdf,
                prefer=prefer,
                ocr=ocr,
                ocr_lang=ocr_lang,
                max_ocr_pages=max_ocr_pages,
                include_presentation=include_presentation,
                pleasantry_mode=pleasantry_mode,
                verbose=True,
            )
            if outdir:
                out_path = outdir / f"{pdf.stem}_qa.csv"
            else:
                out_path = pdf.with_suffix("").with_name(pdf.stem + "_qa.csv")
            df.to_csv(out_path, index=False)
            print(f"  -> {len(df)} rows written to {out_path}")
            all_rows.append(df.assign(source_pdf=str(pdf)))
        except Exception as e:
            print(f"  !! Failed: {pdf} -> {e}")
            

    if combined_out:
        combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
        # Final tidy-up
        combined = postprocess_df(combined)
        combined.to_csv(combined_out, index=False)
        print(f"Wrote combined CSV ({len(combined)} rows) -> {combined_out}")
        return combined_out
    return Path("")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Parse JPM earnings-call PDFs (single file or batch folder) into CSV (+ presentation, pleasantries).")
    p.add_argument("input", type=str, help="Path to a PDF or a folder of PDFs")
    p.add_argument("--out", type=str, default=None, help="For single-file mode: CSV output path")
    p.add_argument("--year", type=int, default=None, help="Override inferred year (e.g., 2025)")
    p.add_argument("--quarter", type=str, default=None, help="Override inferred quarter (e.g., Q2)")

    # Extractor options
    p.add_argument("--prefer", type=str, default=None, choices=["pymupdf", "pdfminer", "pypdf2", "pdftotext"], help="Preferred extractor to try first")
    p.add_argument("--ocr", action="store_true", help="Enable OCR fallback via pytesseract if text extraction fails")
    p.add_argument("--ocr-lang", type=str, default="eng", help="OCR language code (default 'eng')")
    p.add_argument("--max-ocr-pages", type=int, default=6, help="Max pages to OCR for fallback (default 6)")

    # Presentation options
    p.add_argument("--include-presentation", type=str, default="single_row", choices=["none", "single_row", "per_speaker"],
                   help="Attach opening statement before Q&A (default: single_row)")

    # Pleasantry options
    p.add_argument("--pleasantry-mode", type=str, default="label_and_resequence",
                   choices=["keep_raw", "label_only", "label_and_resequence"],
                   help="Pleasantry handling (default: label_and_resequence)")

    # Batch options
    p.add_argument("--batch", action="store_true", help="Treat input as a folder and process multiple PDFs")
    p.add_argument("--recursive", action="store_true", help="Recurse into subfolders when batch processing")
    p.add_argument("--outdir", type=str, default=None, help="Directory to write per-file CSVs in batch mode")
    p.add_argument("--pattern", type=str, default=None, help="Filename pattern filter (e.g., 'jpm*earnings*pdf')")
    p.add_argument("--combined-out", type=str, default=None, help="Path to write a single combined CSV across PDFs")

    args = p.parse_args()
    in_path = Path(args.input)

    if not args.batch and in_path.is_file():
        df = parse_pdf_to_df(
            in_path,
            year=args.year,
            quarter=args.quarter,
            prefer=args.prefer,
            ocr=args.ocr,
            ocr_lang=args.ocr_lang,
            max_ocr_pages=args.max_ocr_pages,
            include_presentation=args.include_presentation,
            pleasantry_mode=args.pleasantry_mode,
            verbose=True,
        )
        out_path = Path(args.out) if args.out else in_path.with_suffix("").with_name(in_path.stem + "_qa.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Wrote {len(df)} rows -> {out_path}")
    else:
        outdir = Path(args.outdir) if args.outdir else None
        combined_out = Path(args.combined_out) if args.combined_out else None
        batch_process(
            in_path,
            outdir=outdir,
            recursive=args.recursive,
            pattern=args.pattern,
            combined_out=combined_out,
            prefer=args.prefer,
            ocr=args.ocr,
            ocr_lang=args.ocr_lang,
            max_ocr_pages=args.max_ocr_pages,
            include_presentation=args.include_presentation,
            pleasantry_mode=args.pleasantry_mode,
        )


if __name__ == "__main__":
    main()
