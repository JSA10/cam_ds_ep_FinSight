#!/usr/bin/env python3
"""
JPM (and similar) earnings call Q&A PDF parser — with batch mode.

Features:
- Single PDF or entire folder (optionally recursive) processing
- Per-PDF CSVs and an optional combined CSV
- Robust PDF text extraction (pdfminer.six, PyPDF2 fallback)
- Q&A section detection; supports 'Q'/'A' as separate lines or trailing ' Q'/' A'
- Output columns: question_number, answer_number, speaker_name, role, company, content, year, quarter

Examples:
  # Single file
  python jpm_earnings_qa_parser.py ./jpm-2q25-earnings-call-transcript.pdf --out ./jpm_q2_2025_qa.csv

  # Batch a folder (non-recursive), write per-file CSVs next to source PDFs
  python jpm_earnings_qa_parser.py ./transcripts --batch

  # Batch a folder recursively, send all outputs to a folder and also write a combined CSV
  python jpm_earnings_qa_parser.py ./transcripts --batch --recursive --outdir ./qa_csvs --combined-out ./all_qna.csv

Dependencies:
    pip install pdfminer.six PyPDF2 pandas
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd


# ---------------- PDF extraction ----------------
def extract_text_from_pdf(pdf_path: Path) -> str:
    text = None
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text
        text = pdfminer_extract_text(str(pdf_path))
    except Exception:
        text = None
    if text is None or not text.strip():
        try:
            import PyPDF2
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                pages = []
                for p in reader.pages:
                    try:
                        pages.append(p.extract_text() or "")
                    except Exception:
                        pages.append("")
                text = "\n".join(pages)
        except Exception:
            text = ""
    if not text or not text.strip():
        raise RuntimeError(f"Unable to extract text from PDF: {pdf_path}")
    return text


# ---------------- Heuristics & helpers ----------------
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


# ---------------- Core parsing ----------------
def collect_blocks(lines: List[str]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i].strip()

        # Case 1: Q/A on its own line
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

        # Case 2: Q/A at end of line
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
    if len(yy) == 2:
        year = 2000 + int(yy)
    else:
        year = int(yy)
    quarter = f"Q{qnum}"
    return year, quarter


def parse_pdf_to_df(pdf_path: Path, year: Optional[int] = None, quarter: Optional[str] = None) -> pd.DataFrame:
    raw_text = extract_text_from_pdf(pdf_path)
    lines = clean_lines(raw_text)
    qa_lines = slice_to_qa_section(lines)
    items = collect_blocks(qa_lines)
    items = merge_consecutive_questions(items)

    inf_year, inf_quarter = infer_year_quarter_from_filename(pdf_path)
    year = year if year is not None else (inf_year if inf_year is not None else pd.NA)
    quarter = quarter if quarter is not None else (inf_quarter if inf_quarter is not None else pd.NA)

    df = to_dataframe(items, year=year, quarter=quarter)
    return df


# ---------------- Batch mode ----------------
def iter_pdfs(folder: Path, recursive: bool, pattern: Optional[str]) -> List[Path]:
    if recursive:
        files = list(folder.rglob("*.pdf"))
    else:
        files = list(folder.glob("*.pdf"))
    if pattern:
        import fnmatch
        files = [f for f in files if fnmatch.fnmatch(f.name.lower(), pattern.lower())]
    return sorted(files)


def batch_process(input_path: Path, outdir: Optional[Path], recursive: bool, pattern: Optional[str], combined_out: Optional[Path]) -> Path:
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    all_rows: List[pd.DataFrame] = []
    pdfs: List[Path] = []

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
            df = parse_pdf_to_df(pdf)
            # where to write per-file CSV
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
        combined.to_csv(combined_out, index=False)
        print(f"Wrote combined CSV ({len(combined)} rows) -> {combined_out}")
        return combined_out
    return Path("")


# ---------------- CLI ----------------
def main():
    p = argparse.ArgumentParser(description="Parse JPM earnings-call PDFs (single file or batch folder) into Q&A CSVs.")
    p.add_argument("input", type=str, help="Path to a PDF or a folder of PDFs")
    p.add_argument("--out", type=str, default=None, help="For single-file mode: CSV output path")
    p.add_argument("--year", type=int, default=None, help="Override inferred year (e.g., 2025)")
    p.add_argument("--quarter", type=str, default=None, help="Override inferred quarter (e.g., Q2)")

    # Batch options
    p.add_argument("--batch", action="store_true", help="Treat input as a folder and process multiple PDFs")
    p.add_argument("--recursive", action="store_true", help="Recurse into subfolders when batch processing")
    p.add_argument("--outdir", type=str, default=None, help="Directory to write per-file CSVs in batch mode")
    p.add_argument("--pattern", type=str, default=None, help="Filename pattern filter (e.g., 'jpm*earnings*pdf')")
    p.add_argument("--combined-out", type=str, default=None, help="Path to write a single combined CSV across PDFs")

    args = p.parse_args()

    in_path = Path(args.input)

    if not args.batch and in_path.is_file():
        # single-file mode
        df = parse_pdf_to_df(in_path, year=args.year, quarter=args.quarter)
        out_path = Path(args.out) if args.out else in_path.with_suffix("").with_name(in_path.stem + "_qa.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Wrote {len(df)} rows -> {out_path}")
    else:
        # batch mode (folder or single file but forced batch)
        outdir = Path(args.outdir) if args.outdir else None
        combined_out = Path(args.combined_out) if args.combined_out else None
        batch_process(in_path, outdir=outdir, recursive=args.recursive, pattern=args.pattern, combined_out=combined_out)


if __name__ == "__main__":
    main()
