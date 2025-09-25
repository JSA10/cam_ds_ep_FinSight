
"""
PRA Risk Summaries & Evasiveness Detector (2023–2025, JPM & HSBC)
Run as a script or copy cells into Jupyter. Designed for Apple Silicon (M3) with MPS.
"""

import os, platform, sys, re, json, warnings
from pathlib import Path
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

print("Python:", sys.version)
print("Platform:", platform.platform())
print("CWD:", os.getcwd())

# Apple Metal (MPS) accelerator check
try:
    import torch
    mps_ok = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    device = torch.device("mps" if mps_ok else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Torch version: {torch.__version__}")
    print("MPS built:", torch.backends.mps.is_built())
    print("MPS available (usable):", torch.backends.mps.is_available())
    print("Using device:", device)
except Exception as e:
    print("Torch not available, falling back to CPU-only:", e)
    device = "cpu"

# Imports that may need pip install
try:
    import matplotlib.pyplot as plt
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from sentence_transformers import SentenceTransformer, util
    import textstat
except Exception as e:
    print("Missing libs. Install first:", e)
    print("Try: pip install pandas numpy matplotlib transformers sentence-transformers textstat")
    raise

# ---------- Config ----------
DATA_DIR = Path("../data/processed")
JPM_PATH = DATA_DIR / "jpm" / "all_jpm_2023_2025.csv"
HSBC_PATH = DATA_DIR / "hsbc" / "all_hsbc_2023_2025.csv"

# Try both possible filenames for the PRA categories
PRA_PATHS = [
    DATA_DIR / "PRA Risk Categories.csv",
    DATA_DIR / "PRA Risk Categories - Sheet1.csv"
]

# Models (swap summarizer if you want higher quality and have time/VRAM)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SUMM_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"  # or "facebook/bart-large-cnn"
USE_NLI = False
NLI_MODEL_NAME = "typeform/distilroberta-base-uncased-mnli"

# Summaries
SUMMARY_MAX_TOKENS = 200  # for distilbart cnn
SUMMARY_TARGET_WORDS = 120

# Evasion thresholds (tweak to taste)
SIMILARITY_LOW = 0.38      # cosine sim below this suggests low alignment to question
HEDGE_MIN_COUNT = 2        # minimum hedge/deflection cues to matter
VERBOSITY_RATIO_HIGH = 6.0 # answer-to-question char ratio
READABILITY_SIMPLE = 8.0   # Flesch-Kincaid grade; higher can indicate complexity
EVASION_SCORE_FLAG = 0.65  # composite score threshold to flag

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

HEDGE_PHRASES = [
    "i think","we think","i believe","we believe","we feel","i feel",
    "sort of","kind of","a bit","a little","roughly","approximately",
    "around","more or less","to some extent","somewhat",
    "we don't break out","we do not break out","we don't disclose","we do not disclose",
    "we won't comment","we will not comment","not going to comment",
    "too early to say","too soon to say","too soon to tell",
    "we'll have to see","we will have to see",
    "we'll come back","we will come back",
    "as we've said before","as we said before",
    "as previously mentioned","as mentioned",
    "let me step back","take a step back",
    "the way i would frame","i would frame it",
    "i'm not sure","we're not sure",
    "it's complicated","it's complex",
    "moving parts",
    "as you know","as you can appreciate",
    "that's a great question","good question",
    "let me answer a different","let me start somewhere else",
]

def count_hedges(text: str) -> int:
    t = " " + text.lower() + " "
    return sum(1 for p in HEDGE_PHRASES if f" {p} " in t)

def fk_grade(text: str) -> float:
    try:
        return textstat.flesch_kincaid_grade(text)
    except Exception:
        return np.nan

class LoaderAgent:
    REQ_COLS = ["section","question_number","answer_number","speaker_name","role",
                "company","content","year","quarter","is_pleasantry","source_pdf"]
    def __init__(self, jpm_path: Path, hsbc_path: Path):
        self.jpm_path = jpm_path
        self.hsbc_path = hsbc_path

    def load_df(self) -> pd.DataFrame:
        frames = []
        if self.jpm_path.exists():
            df_j = pd.read_csv(self.jpm_path)
            df_j["bank"] = "JPM"
            frames.append(df_j)
        if self.hsbc_path.exists():
            df_h = pd.read_csv(self.hsbc_path)
            df_h["bank"] = "HSBC"
            frames.append(df_h)
        assert frames, "No input CSVs found. Check paths."
        df = pd.concat(frames, ignore_index=True)
        # hygiene
        for c in ["content","speaker_name","role","company","section","source_pdf"]:
            if c in df.columns:
                df[c] = df[c].astype(str).map(normalize_text)
        for c in ["year"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "quarter" in df.columns:
            df["quarter"] = df["quarter"].astype(str).str.upper().str.replace(" ", "")
            df["quarter"] = df["quarter"].str.replace("Q0","Q4")
        if "section" in df.columns:
            df = df[df["section"].str.contains("QUESTION|Q&A", case=False, na=False) | (df["answer_number"].notna())]
        if "is_pleasantry" in df.columns:
            df = df[df["is_pleasantry"] != True]
        df = df.dropna(subset=["content"]).reset_index(drop=True)
        return df

def load_pra_categories(paths):
    for p in paths:
        if p.exists():
            cat = pd.read_csv(p)
            # Expect columns: category, description (flexible)
            cols = {c.lower().strip(): c for c in cat.columns}
            if "category" not in cols:
                first_col = cat.columns[0]
                cat = cat.rename(columns={first_col: "category"})
            else:
                cat = cat.rename(columns={cols["category"]:"category"})
            if "description" not in [c.lower() for c in cat.columns]:
                cat["description"] = ""
            return cat[["category","description"]].dropna().reset_index(drop=True)
    raise FileNotFoundError("PRA categories file not found.")

def map_to_pra_categories_builder(pra_df, embedder):
    from sentence_transformers import util
    pra_texts = (pra_df["category"] + ". " + pra_df["description"].fillna("")).tolist()
    pra_embs = embedder.encode(pra_texts, convert_to_tensor=True, normalize_embeddings=True)

    KEYWORDS = {
        "Credit risk": ["credit","NPL","non-performing","loan loss","default","provision","counterparty"],
        "Market risk": ["trading","VaR","volatility","rates","FX","equities","derivatives","market risk"],
        "Liquidity risk": ["liquidity","LCR","NSFR","funding","deposits","outflows","liquidity coverage"],
        "Capital risk": ["capital","CET1","RWA","leverage ratio","buffers","Pillar 2","dividends","buybacks"],
        "Operational risk": ["operational","ops","cyber","fraud","conduct","model risk","technology"],
        "IRRBB": ["interest rate risk in the banking book","IRRBB","ALM","duration","asset-liability"],
        "Climate & ESG": ["climate","ESG","sustainability","transition risk","physical risk","emissions"],
        "Model risk": ["model risk","validation","challenge","backtesting","stress test"],
        "Conduct risk": ["conduct","mis-selling","complaints","whistleblowing","FCA"],
    }
    kw2cat = {kw.lower(): cat for cat, kws in KEYWORDS.items() for kw in kws}

    def map_to_pra_categories(text: str, top_k=2):
        text_norm = (text or "").lower()
        hits = {kw2cat[kw] for kw in kw2cat if f" {kw} " in f" {text_norm} "}
        q_emb = embedder.encode([text or ""], convert_to_tensor=True, normalize_embeddings=True)
        sims = util.cos_sim(q_emb, pra_embs).cpu().numpy().ravel()
        nn_idx = sims.argsort()[::-1][:top_k]
        nn_cats = [pra_df.iloc[i]["category"] for i in nn_idx]
        cats = list(dict.fromkeys(list(hits) + nn_cats))
        return cats, float(sims[nn_idx[0]])
    return map_to_pra_categories

def build_pairs(df: pd.DataFrame, map_to_pra_categories) -> pd.DataFrame:
    def is_analyst(role, speaker):
        role = (role or "").lower(); speaker = (speaker or "").lower()
        return ("analyst" in role) or ("analyst" in speaker)
    def is_banker(role, speaker):
        role = (role or "").lower(); speaker = (speaker or "").lower()
        mgmt = ["chief","ceo","cfo","coo","treasurer","head","president","vice","managing director"]
        return any(m in role for m in mgmt) or ("jpmorgan" in speaker) or ("hsbc" in speaker) or ("executive" in role)

    df = df.copy()
    df["is_analyst_row"] = df.apply(lambda r: is_analyst(r.get("role",""), r.get("speaker_name","")), axis=1)
    df["is_banker_row"]  = df.apply(lambda r: is_banker(r.get("role",""), r.get("speaker_name","")), axis=1)

    if "question_number" not in df.columns:
        df["question_number"] = df.groupby(["bank","year","quarter"]).cumcount()+1

    gcols = ["bank","year","quarter","question_number"]
    pairs = []
    for key, g in df.groupby(gcols, dropna=False):
        g = g.sort_index()
        qtxt = " ".join(g.loc[g["is_analyst_row"], "content"].tolist())
        atxt = " ".join(g.loc[g["is_banker_row"], "content"].tolist())
        if not qtxt and not atxt:
            continue
        cats, cat_sim = map_to_pra_categories((qtxt or "") + " " + (atxt or ""))
        pairs.append({
            "bank": key[0], "year": key[1], "quarter": key[2], "question_number": key[3],
            "question_text": normalize_text(qtxt), "answer_text": normalize_text(atxt),
            "pra_categories": cats, "pra_sim": cat_sim
        })
    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df[(pairs_df["question_text"].str.len()>0) | (pairs_df["answer_text"].str.len()>0)].reset_index(drop=True)
    return pairs_df

def compute_evasion(pairs_df, embedder,
                    SIMILARITY_LOW=0.38, HEDGE_MIN_COUNT=2,
                    VERBOSITY_RATIO_HIGH=6.0, READABILITY_SIMPLE=8.0, EVASION_SCORE_FLAG=0.65):
    from sentence_transformers import util as st_util
    q_embs = embedder.encode(pairs_df["question_text"].tolist(), convert_to_tensor=True, normalize_embeddings=True)
    a_embs = embedder.encode(pairs_df["answer_text"].tolist(), convert_to_tensor=True, normalize_embeddings=True)
    cos_sims = st_util.cos_sim(q_embs, a_embs).diagonal().cpu().numpy()

    def safe_len(x): return 0 if x is None else len(str(x))

    def evasion_score(row, sim):
        q = row["question_text"]; a = row["answer_text"]
        if not a: 
            return 1.0
        hedges = count_hedges(a)
        ratio = (safe_len(a)+1)/(safe_len(q)+1)
        grade = fk_grade(a)
        sim_comp   = max(0.0, min(1.0, (SIMILARITY_LOW - sim) / SIMILARITY_LOW))
        hedge_comp = min(1.0, hedges / max(HEDGE_MIN_COUNT, 1))
        ratio_comp = min(1.0, max(0.0, (ratio - 1.5) / (VERBOSITY_RATIO_HIGH - 1.5)))
        grade_comp = min(1.0, max(0.0, (grade - READABILITY_SIMPLE)/10.0))
        w = dict(sim=0.45, hedge=0.25, ratio=0.20, grade=0.10)
        score = w["sim"]*sim_comp + w["hedge"]*hedge_comp + w["ratio"]*ratio_comp + w["grade"]*grade_comp
        return float(round(score, 4))

    pairs_df = pairs_df.copy()
    pairs_df["qa_similarity"] = cos_sims
    pairs_df["hedge_count"] = pairs_df["answer_text"].map(count_hedges)
    pairs_df["ans_to_q_len_ratio"] = (pairs_df["answer_text"].str.len()+1)/(pairs_df["question_text"].str.len()+1)
    pairs_df["fk_grade_answer"] = pairs_df["answer_text"].map(fk_grade)
    pairs_df["evasion_score"] = [evasion_score(r, s) for r, s in zip(pairs_df.to_dict("records"), cos_sims)]
    pairs_df["evasive_flag"] = pairs_df["evasion_score"] >= EVASION_SCORE_FLAG
    return pairs_df

def build_summaries(pairs: pd.DataFrame, by_cols=("bank","year","quarter"),
                    model_name="sshleifer/distilbart-cnn-12-6", device="cpu",
                    target_words=120):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch, re
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    try:
        model = model.to(device)
    except Exception:
        pass

    def chunk_text(s, max_chars=2500):
        s = normalize_text(s)
        if len(s) <= max_chars:
            return [s]
        parts = re.split(r'(?<=[\.\!\?])\s+', s)
        chunks, buf = [], ""
        for p in parts:
            if len(buf) + len(p) + 1 < max_chars:
                buf += (" " if buf else "") + p
            else:
                chunks.append(buf); buf = p
        if buf: chunks.append(buf)
        return chunks

    def summarise_chunks(chunks, max_new_tokens=200):
        out_texts = []
        for ch in chunks:
            inputs = tokenizer(ch, return_tensors="pt", truncation=True, max_length=1024).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            out_texts.append(tokenizer.decode(out[0], skip_special_tokens=True).strip())
        return " ".join(out_texts)

    rows = []
    for key, g in pairs.groupby(list(by_cols)):
        gg = g.explode("pra_categories").dropna(subset=["pra_categories"])
        for cat, gcat in gg.groupby("pra_categories"):
            text = " ".join(gcat["answer_text"].tolist())
            if not text.strip():
                summary = ""
            else:
                chunks = chunk_text(text)
                summary = summarise_chunks(chunks, max_new_tokens=200)
                if len(summary.split()) > target_words*1.5:
                    chunks2 = chunk_text(summary, max_chars=1200)
                    summary = summarise_chunks(chunks2, max_new_tokens=120)
            rows.append({**{c:k for c,k in zip(by_cols,key)}, "pra_category": cat,
                         "summary": summary, "n_pairs": len(gcat),
                         "median_evasion": float(np.median(gcat["evasion_score"]))})
    return pd.DataFrame(rows).sort_values(list(by_cols)+["pra_category"]).reset_index(drop=True)

def main():
    print("Loading data...")
    loader = LoaderAgent(JPM_PATH, HSBC_PATH)
    qa_df = loader.load_df()
    print("Rows:", len(qa_df))

    print("Loading PRA categories...")
    pra_df = load_pra_categories(PRA_PATHS)

    print("Loading embedding model:", EMBED_MODEL_NAME)
    embedder = SentenceTransformer(EMBED_MODEL_NAME, device=str(device))
    map_to_pra = map_to_pra_categories_builder(pra_df, embedder)

    print("Building Q/A pairs...")
    pairs_df = build_pairs(qa_df, map_to_pra)
    print("Pairs:", len(pairs_df))

    print("Scoring evasiveness...")
    pairs_df = compute_evasion(pairs_df, embedder,
                               SIMILARITY_LOW=SIMILARITY_LOW, HEDGE_MIN_COUNT=HEDGE_MIN_COUNT,
                               VERBOSITY_RATIO_HIGH=VERBOSITY_RATIO_HIGH, READABILITY_SIMPLE=READABILITY_SIMPLE,
                               EVASION_SCORE_FLAG=EVASION_SCORE_FLAG)

    print("Summarising per PRA category...")
    summ_df = build_summaries(pairs_df, by_cols=("bank","year","quarter"),
                              model_name=SUMM_MODEL_NAME, device=device, target_words=SUMMARY_TARGET_WORDS)

    out_dir = Path("./outputs"); out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "qa_pairs_with_evasion.csv"
    summ_path  = out_dir / "pra_category_summaries.csv"
    pairs_df.to_csv(pairs_path, index=False)
    summ_df.to_csv(summ_path, index=False)

    print("Saved:", pairs_path, "|", summ_path)

    # Quick report
    top = pairs_df.sort_values("evasion_score", ascending=False).head(10)
    print("\nTop 5 evasive examples (truncated):")
    for _, r in top.head(5).iterrows():
        print(f"- {r['bank']} {r['year']} {r['quarter']} Q#{r['question_number']} | score={r['evasion_score']} | cats={r['pra_categories']}")
        print("  Q:", r['question_text'][:180], "...")
        print("  A:", r['answer_text'][:220], "...\n")

if __name__ == "__main__":
    main()
