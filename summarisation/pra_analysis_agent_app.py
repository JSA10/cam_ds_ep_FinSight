"""
PRA Analysis Agent – Streamlit MVP
Single-file Streamlit app to generate a PRA-style quarterly report by bank
from DS CSV outputs + lightweight Q&A over the same material.

Run locally:
    pip install -U streamlit pandas numpy altair scikit-learn
    # (optional) sentence-transformers for semantic search in Q&A
    pip install sentence-transformers
    streamlit run app.py

CSV schema (minimum contract) expected across inputs (extra columns are fine):
    bank: str   # e.g., 'JPMorgan', 'HSBC'
    quarter: str  # e.g., '2025Q1', '2025Q2'
    section: str  # 'Q' or 'A' (or 'question'/'answer')
    question_number: int
    answer_number: int
    role: str   # 'Analyst', 'Executive', 'CFO', etc.
    speaker_name: str
    content: str  # transcript text
    pra_category: str  # mapped PRA risk category (if available)
    topic: str  # topic label (if available)
    sentiment_score: float  # -1..1 or 0..1 (we adapt)
    sentiment_label: str   # 'pos'/'neg'/'neu' (optional)
    evasion_score: float   # 0..1 (optional)
    evasion_label: str     # 'evasive'/'direct' (optional)

If no CSVs are uploaded, a small in-memory demo dataset is used.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Optional: enable semantic search if sentence-transformers is installed
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_EMBED = True
except Exception:
    _HAS_EMBED = False

from sklearn.feature_extraction.text import TfidfVectorizer

REQUIRED_COLS = [
    'bank','quarter','section','question_number','answer_number','role',
    'speaker_name','content','pra_category','topic','sentiment_score',
    'evasion_score'
]


# ------------------------------
# Utilities
# ------------------------------

def _normalize_quarter(q: str) -> str:
    if pd.isna(q):
        return q
    q = str(q).strip().upper().replace(" ", "")
    # Normalize common forms like 'Q1 2025' -> '2025Q1'
    if q.startswith('Q') and ' ' in q:
        # e.g. 'Q1 2025'
        parts = q.split()
        if len(parts) == 2 and parts[0].startswith('Q'):
            return f"{parts[1]}{parts[0]}"
    if 'Q' in q and q.index('Q') > 0:
        # likely already '2025Q1'
        return q
    return q


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return (len(missing) == 0, missing)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Normalize quarter
    if 'quarter' in df.columns:
        df['quarter'] = df['quarter'].map(_normalize_quarter)
    # Coerce numerics if present
    for col in ['question_number','answer_number','sentiment_score','evasion_score']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Fill basic NA
    for col in ['pra_category','topic','role','speaker_name','content','bank','quarter','section']:
        if col in df.columns:
            df[col] = df[col].fillna('')
    return df


def load_demo_df() -> pd.DataFrame:
    # Tiny illustrative dataset
    rows = [
        # JPM Q1
        dict(bank='JPMorgan', quarter='2025Q1', section='Q', question_number=1, answer_number=0,
             role='Analyst', speaker_name='Analyst A', pra_category='Capital Adequacy', topic='Capital',
             sentiment_score=-0.2, evasion_score=0.3, content='Can you comment on CET1 trajectory and buffers?'),
        dict(bank='JPMorgan', quarter='2025Q1', section='A', question_number=1, answer_number=1,
             role='Executive', speaker_name='CFO', pra_category='Capital Adequacy', topic='Capital',
             sentiment_score=0.3, evasion_score=0.1, content='Buffers remain comfortable; we target stability with modest upside.'),
        # JPM Q2
        dict(bank='JPMorgan', quarter='2025Q2', section='Q', question_number=5, answer_number=0,
             role='Analyst', speaker_name='Analyst B', pra_category='Operational Risk', topic='Ops Risk',
             sentiment_score=-0.1, evasion_score=0.6, content='Operational losses uptick: any remediation specifics?'),
        dict(bank='JPMorgan', quarter='2025Q2', section='A', question_number=5, answer_number=1,
             role='Executive', speaker_name='CFO', pra_category='Operational Risk', topic='Ops Risk',
             sentiment_score=0.2, evasion_score=0.42, content='We have robust controls; investigations are ongoing.'),
        # HSBC Q1
        dict(bank='HSBC', quarter='2025Q1', section='Q', question_number=3, answer_number=0,
             role='Analyst', speaker_name='Analyst C', pra_category='Credit Risk', topic='Credit',
             sentiment_score=-0.4, evasion_score=0.32, content='How exposed are you to commercial real estate?'),
        dict(bank='HSBC', quarter='2025Q1', section='A', question_number=3, answer_number=1,
             role='Executive', speaker_name='CFO', pra_category='Credit Risk', topic='Credit',
             sentiment_score=0.1, evasion_score=0.35, content='Exposure is prudent and diversified across regions.'),
        # HSBC Q2
        dict(bank='HSBC', quarter='2025Q2', section='Q', question_number=7, answer_number=0,
             role='Analyst', speaker_name='Analyst D', pra_category='Credit Risk', topic='Credit',
             sentiment_score=-0.3, evasion_score=0.42, content='Can you quantify expected Stage 3 charges near-term?'),
        dict(bank='HSBC', quarter='2025Q2', section='A', question_number=7, answer_number=1,
             role='Executive', speaker_name='CFO', pra_category='Credit Risk', topic='Credit',
             sentiment_score=0.15, evasion_score=0.45, content='We will not speculate; provisioning follows our standard process.'),
    ]
    return pd.DataFrame(rows)


# ------------------------------
# Analytics helpers used in Report
# ------------------------------

def role_polarity(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate average sentiment by role (Analyst vs Exec) and compute divergence."""
    if df.empty:
        return pd.DataFrame(columns=['role','sentiment_mean'])
    gb = df.groupby('role', dropna=False)['sentiment_score'].mean().reset_index(name='sentiment_mean')
    return gb


def top_themes(df: pd.DataFrame, by_col='pra_category', topn=5) -> pd.DataFrame:
    if df.empty or by_col not in df.columns:
        return pd.DataFrame(columns=[by_col,'share'])
    counts = df[by_col].value_counts(dropna=False).reset_index()
    counts.columns = [by_col, 'count']
    counts['share'] = counts['count'] / counts['count'].sum()
    return counts.head(topn)


def evasion_hotspots(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['pra_category','evasion_rate'])
    # threshold 0.5 as default; can be adjusted in UI
    ev = df.assign(is_evasive=(df['evasion_score'] >= 0.5))
    out = ev.groupby('pra_category')['is_evasive'].mean().reset_index(name='evasion_rate')
    return out.sort_values('evasion_rate', ascending=False)


def benchmarking_delta(df_a: pd.DataFrame, df_b: pd.DataFrame, metric='evasion_rate') -> pd.DataFrame:
    ea = evasion_hotspots(df_a).rename(columns={'evasion_rate':'ea'})
    eb = evasion_hotspots(df_b).rename(columns={'evasion_rate':'eb'})
    merged = pd.merge(ea, eb, on='pra_category', how='outer').fillna(0)
    merged['delta'] = merged['ea'] - merged['eb']
    return merged.sort_values('delta', ascending=False)


def build_report_md(df: pd.DataFrame, bank: str, quarter: str) -> str:
    scope = df[(df['bank']==bank) & (df['quarter']==quarter)]
    if scope.empty:
        return f"### No data found for {bank} — {quarter}. Upload CSVs or switch to Demo Mode."

    # Themes
    themes = top_themes(scope, 'pra_category', topn=5)
    theme_lines = [f"- **{r.pra_category}** ({r.share:.0%})" for r in themes.itertuples(index=False)]

    # Sentiment divergence (Analyst vs Exec)
    roles = role_polarity(scope)
    anal = roles[roles['role'].str.contains('Analyst', case=False, na=False)]['sentiment_mean']
    execs = roles[~roles['role'].str.contains('Analyst', case=False, na=False)]['sentiment_mean']
    anal_m = float(anal.iloc[0]) if len(anal)>0 else np.nan
    exec_m = float(execs.mean()) if len(execs)>0 else np.nan

    div_line = "Not enough data to compare roles."
    if not math.isnan(anal_m) and not math.isnan(exec_m):
        diff = exec_m - anal_m
        direction = "more positive" if diff>0 else "more negative"
        div_line = f"Executives appear **{abs(diff):.2f}** {direction} than analysts on average."

    # Evasion hotspots
    ev = evasion_hotspots(scope).head(5)
    ev_lines = [f"- **{r.pra_category}**: {r.evasion_rate:.0%}" for r in ev.itertuples(index=False)]

    # Q3 watch-outs (simple heuristic using rising evasion + negative sentiment themes)
    # For demo: pick top 2 by evasion_rate
    watch_lines = [f"- **{r.pra_category}**: sustain scrutiny; clarify metrics & remediation plans." for r in ev.head(2).itertuples(index=False)]

    md = [
        f"## PRA Quarterly Brief — {bank} {quarter}",
        "### Themes",
        *(theme_lines or ['- n/a']),
        "\n### Sentiment divergence",
        f"{div_line}",
        "\n### Evasion hotspots",
        *(ev_lines or ['- n/a']),
        "\n### Q3 watch-outs",
        *(watch_lines or ['- n/a']),
    ]
    return "\n".join(md)


# ------------------------------
# Q&A Retrieval (local-only)
# ------------------------------
@dataclass
class QAIndex:
    df: pd.DataFrame
    use_embeddings: bool
    tfidf: TfidfVectorizer | None
    tfidf_mat: any
    model: SentenceTransformer | None
    emb_mat: np.ndarray | None


def build_index(df: pd.DataFrame, use_embeddings: bool = False) -> QAIndex:
    scope = df.copy()
    scope['doc'] = scope.apply(lambda r: f"{r['quarter']} {r['bank']} | {r['role']} {r['speaker_name']} | Q{int(r['question_number'])} A{int(r['answer_number'])} | {r['pra_category']} | {r['topic']}\n{r['content']}", axis=1)

    tfidf = TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1,2))
    tfidf_mat = tfidf.fit_transform(scope['doc'])

    model = None
    emb_mat = None
    if use_embeddings and _HAS_EMBED:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        emb_mat = model.encode(scope['doc'].tolist(), normalize_embeddings=True)

    return QAIndex(df=scope, use_embeddings=(use_embeddings and _HAS_EMBED), tfidf=tfidf, tfidf_mat=tfidf_mat, model=model, emb_mat=emb_mat)


def query_index(index: QAIndex, question: str, topk: int = 5) -> pd.DataFrame:
    if not question.strip():
        return pd.DataFrame(columns=index.df.columns.tolist()+['score'])

    # TF-IDF baseline
    q_tfidf = index.tfidf.transform([question])
    tfidf_scores = (q_tfidf @ index.tfidf_mat.T).toarray().ravel()

    if index.use_embeddings and index.model is not None and index.emb_mat is not None:
        q_emb = index.model.encode([question], normalize_embeddings=True)
        emb_scores = cosine_similarity(q_emb, index.emb_mat)[0]
        # Blend scores 50/50
        scores = 0.5*tfidf_scores + 0.5*emb_scores
    else:
        scores = tfidf_scores

    best_idx = np.argsort(-scores)[:topk]
    out = index.df.iloc[best_idx].copy()
    out['score'] = scores[best_idx]
    return out


# ------------------------------
# Streamlit App
# ------------------------------
st.set_page_config(page_title="PRA Analysis Agent – MVP", layout="wide")

st.title("PRA Analysis Agent – MVP")

with st.sidebar:
    st.header("Data inputs")
    st.caption("Upload CSVs (can be combined or separate). The app will concatenate by columns.")
    uploaded = st.file_uploader("Upload one or more CSVs", type=["csv"], accept_multiple_files=True)
    demo = st.toggle("Use demo data if nothing uploaded", value=True)

    st.divider()
    st.header("Scope")
    selected_bank = st.selectbox("Bank", ["JPMorgan","HSBC"])  # can be derived after load
    selected_quarter = st.selectbox("Quarter", ["2025Q1","2025Q2"])  # can be derived after load

    st.divider()
    st.header("Q&A settings")
    use_embeddings = st.toggle("Use semantic search (if installed)", value=False, help="Falls back to TF-IDF if not available.")
    topk = st.slider("Results to retrieve", 3, 10, 5)


# Load data
if uploaded:
    dfs = []
    for f in uploaded:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            st.error(f"Failed to read {f.name}: {e}")
    if dfs:
        data = pd.concat(dfs, ignore_index=True)
    else:
        data = pd.DataFrame()
elif demo:
    data = load_demo_df()
else:
    data = pd.DataFrame()

if data.empty:
    st.info("No data loaded yet. Upload CSVs or enable demo data in the sidebar.")
    st.stop()

# clean and validate
data = coerce_types(data)
valid, missing = validate_schema(data)
if not valid:
    st.warning(f"Missing required columns: {missing}. The app may still work partially if equivalent fields exist.")

# Derive available banks/quarters dynamically
banks = sorted([b for b in data['bank'].dropna().unique().tolist() if b])
quarters = sorted([q for q in data['quarter'].dropna().unique().tolist() if q])
col1, col2, col3 = st.columns([1,1,2])
with col1:
    selected_bank = st.selectbox("Bank (from data)", banks or [selected_bank], index=0 if banks else 0)
with col2:
    selected_quarter = st.selectbox("Quarter (from data)", quarters or [selected_quarter], index=0 if quarters else 0)

scope_df = data[(data['bank']==selected_bank) & (data['quarter']==selected_quarter)]

# ------------------------------
# Report Section
# ------------------------------
st.subheader("PRA Report")

if scope_df.empty:
    st.info(f"No records for {selected_bank} – {selected_quarter}.")
else:
    report_md = build_report_md(data, selected_bank, selected_quarter)
    st.markdown(report_md)

    # Charts row
    c1, c2, c3 = st.columns(3)

    # Theme chart (PRA category share)
    with c1:
        themes = top_themes(scope_df, 'pra_category', topn=8)
        if not themes.empty:
            chart1 = alt.Chart(themes).mark_bar().encode(
                x=alt.X('share:Q', axis=alt.Axis(format='%')),
                y=alt.Y('pra_category:N', sort='-x')
            ).properties(title='Theme share (PRA category)')
            st.altair_chart(chart1, use_container_width=True)
        else:
            st.caption("No theme data")

    # Sentiment by role
    with c2:
        roles = role_polarity(scope_df)
        if not roles.empty:
            chart2 = alt.Chart(roles).mark_bar().encode(
                x=alt.X('sentiment_mean:Q'),
                y=alt.Y('role:N', sort='-x')
            ).properties(title='Average sentiment by role')
            st.altair_chart(chart2, use_container_width=True)
        else:
            st.caption("No sentiment data")

    # Evasion by PRA
    with c3:
        ev = evasion_hotspots(scope_df)
        if not ev.empty:
            chart3 = alt.Chart(ev).mark_bar().encode(
                x=alt.X('evasion_rate:Q', axis=alt.Axis(format='%')),
                y=alt.Y('pra_category:N', sort='-x')
            ).properties(title='Evasion rate by PRA category')
            st.altair_chart(chart3, use_container_width=True)
        else:
            st.caption("No evasion data")

    # Download button
    st.download_button(
        label="Download report (Markdown)",
        data=report_md,
        file_name=f"PRA_Brief_{selected_bank}_{selected_quarter}.md",
        mime="text/markdown"
    )

# ------------------------------
# Q&A Section
# ------------------------------
st.subheader("Ask a question about this quarter")

qa_prompt = st.text_input("Your question", placeholder="e.g., Where did evasion increase vs last quarter?")

if 'qa_index' not in st.session_state or st.session_state.get('qa_key') != (selected_bank, selected_quarter, use_embeddings):
    # Build index for the current scope
    idx_df = data[(data['bank']==selected_bank) & (data['quarter']==selected_quarter)].copy()
    st.session_state['qa_index'] = build_index(idx_df, use_embeddings=use_embeddings)
    st.session_state['qa_key'] = (selected_bank, selected_quarter, use_embeddings)

if st.button("Search", disabled=(scope_df.empty or not qa_prompt.strip())):
    idx: QAIndex = st.session_state['qa_index']
    hits = query_index(idx, qa_prompt, topk=topk)
    if hits.empty:
        st.info("No matching snippets found.")
    else:
        for i, row in hits.iterrows():
            with st.expander(f"Q{int(row['question_number'])} A{int(row['answer_number'])} — {row['role']} {row['speaker_name']} · {row['pra_category']} · score {row['score']:.2f}"):
                st.markdown(f"**Quarter/Bank:** {row['quarter']} / {row['bank']}")
                st.markdown(f"**Topic:** {row['topic']}")
                st.write(row['content'])

st.caption("All answers are grounded in the uploaded CSVs or demo data. No external sources are used.")
