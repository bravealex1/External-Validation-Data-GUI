# radiology_preference_app_4tabs.py
# Streamlit app for blinded A/B preference of radiology report impressions (AI vs Human)
# Four GUIs together (CT Head, CTPE, Ultrasound, General) as separate tabs,
# each with its own randomized case order & progress pointer, sharing ONE SQLite DB.
#
# How to run:
#   pip install streamlit pandas
#   streamlit run radiology_preference_app_4tabs.py
#
# CSVs expected in the working directory (or adjust paths in the sidebar):
#   enhanced_top_CTHead_generated_impressions.csv
#   enhanced_top_CTPE_generated_impressions.csv
#   enhanced_top_Ultrasound_generated_impressions.csv
#   enhanced_top_general_generated_impressions.csv
#
# Expected CSV columns (at minimum):
#   Findings, reference_impression, generated_impression
# Optional columns (if present, they will be recorded):
#   model_name, reward, mean_f1_score
#
# Blinding rules:
# - A/B order is randomized per dataset & case using a deterministic seed.
# - Admin-only "Reveal mapping" checkbox can show which side is AI after choices are stored.

import os
import hashlib
import random
import sqlite3
from datetime import datetime
import re
from typing import Dict, Tuple, Optional

import pandas as pd
import streamlit as st

APP_TITLE = "Blinded A/B Preference: Radiology Impressions (4-in-1)"
DB_PATH_DEFAULT = "preferences.db"

# ---------- Cleaning trailing non-technical text ----------

TRAILING_PATTERNS = [
    r'(?i)\bimages?\s*(?:and|&)\s*interpretations?\b.*$',
    r'(?i)\bplease\s+see\s+images?\b.*$',
    r'(?i)\bcorrelate\s+clinically\b.*$',
    r'(?i)\bcontact\s+radiology\b.*$',
    r'(?i)\bthis\s+report\b.*?(?:contact|call|page).*$',  # "This report ... contact ..."
    r'(?i)\bfindings\s+and\s+interpretations?\b.*$',
]
TRAILING_REGEXES = [re.compile(p, re.DOTALL) for p in TRAILING_PATTERNS]

def clean_impression(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = text.strip()
    for rx in TRAILING_REGEXES:
        s = rx.sub("", s).strip()
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s

# ---------- Datasets ----------

DEFAULT_DATASETS = {
    "CT Head": "enhanced_top_CTHead_generated_impressions.csv",
    "CT Pulmonary Embolism (CTPE)": "enhanced_top_CTPE_generated_impressions.csv",
    "Ultrasound": "enhanced_top_Ultrasound_generated_impressions.csv",
    "General": "enhanced_top_general_generated_impressions.csv",
}

REQUIRED_COLUMNS = {"Findings", "reference_impression", "generated_impression"}

# ---------- DB ----------

def init_db(db_path: str) -> None:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_utc TEXT NOT NULL,
                rater_id TEXT NOT NULL,
                dataset TEXT NOT NULL,
                case_index INTEGER NOT NULL,
                ai_is_a INTEGER NOT NULL,          -- 1 if AI shown as A; 0 if AI shown as B
                choice TEXT NOT NULL,              -- "A", "B", "Tie", "Skip"
                findings_hash TEXT NOT NULL,
                ref_hash TEXT NOT NULL,
                gen_hash TEXT NOT NULL,
                ref_model TEXT,
                gen_model TEXT,
                reward REAL,
                mean_f1 REAL,
                findings_preview TEXT,
                a_preview TEXT,
                b_preview TEXT
            )
            """
        )
        conn.commit()

def save_rating(db_path: str, row: dict) -> None:
    with sqlite3.connect(db_path) as conn:
        cols = [
            "ts_utc","rater_id","dataset","case_index","ai_is_a","choice",
            "findings_hash","ref_hash","gen_hash","ref_model","gen_model",
            "reward","mean_f1","findings_preview","a_preview","b_preview"
        ]
        values = [row.get(c) for c in cols]
        conn.execute(
            f"INSERT INTO ratings ({', '.join(cols)}) VALUES ({', '.join(['?']*len(cols))})",
            values
        )
        conn.commit()

def export_ratings(db_path: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM ratings ORDER BY id ASC", conn)

# ---------- Helpers ----------

def text_hash(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()

def dataset_salt(dataset_name: str) -> int:
    # Derive a small integer salt from dataset name for per-dataset randomization
    h = hashlib.sha256(dataset_name.encode("utf-8")).hexdigest()[:8]
    return int(h, 16) & 0x7FFFFFFF

def ab_mapping_for_case(base_seed: int, dataset_name: str, case_index: int) -> bool:
    """True if AI should be A, else False (AI is B)."""
    rng = random.Random(base_seed + dataset_salt(dataset_name) + case_index * 7919)
    return rng.random() < 0.5

def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df = df.copy()
    df["reference_impression_clean"] = df["reference_impression"].apply(clean_impression)
    df["generated_impression_clean"] = df["generated_impression"].apply(clean_impression)
    df = df[(df["reference_impression_clean"].str.len() > 0) & (df["generated_impression_clean"].str.len() > 0)]
    df.reset_index(drop=True, inplace=True)
    return df

def make_case_order(n: int, seed: int, dataset_name: str) -> list:
    order = list(range(n))
    rng = random.Random(seed + dataset_salt(dataset_name))
    rng.shuffle(order)
    return order

# ---------- Streamlit UI ----------

def ui_header():
    st.title(APP_TITLE)
    st.caption(
        "Evaluate **four datasets** side-by-side (tabs). Each tab presents clinical **Findings** and two blinded **Impressions (A & B)**—"
        "one human (`reference_impression`) and one AI (`generated_impression`). Choose which impression you prefer for each case."
    )

def ui_sidebar():
    st.sidebar.header("Global Setup")
    rater_id = st.sidebar.text_input("Your rater ID (required)", value="")
    db_path = st.sidebar.text_input("SQLite DB path", value=DB_PATH_DEFAULT)
    base_seed = st.sidebar.number_input("Randomization seed", min_value=0, value=42, step=1)
    n_cases = st.sidebar.number_input("Cases per dataset", min_value=1, value=10, step=1)

    st.sidebar.divider()
    st.sidebar.subheader("Data files")
    data_dir = st.sidebar.text_input("Data directory", value=".")
    paths = {}
    for name, fname in DEFAULT_DATASETS.items():
        paths[name] = st.sidebar.text_input(f"{name} CSV", value=os.path.join(data_dir, fname))

    st.sidebar.divider()
    start_all = st.sidebar.button("Start / Reset ALL datasets", type="primary")

    st.sidebar.divider()
    admin_code = st.sidebar.text_input("Admin code (optional)", value="")
    reveal = False
    if admin_code.strip():
        reveal = st.sidebar.checkbox("Reveal AI/Human mapping (admin)")

    return dict(
        rater_id=rater_id.strip(),
        db_path=db_path.strip(),
        base_seed=int(base_seed),
        n_cases=int(n_cases),
        paths=paths,
        start_all=start_all,
        reveal=reveal,
    )

def init_session_state():
    if "datasets_state" not in st.session_state:
        st.session_state.datasets_state = {}  # per dataset: {df, order, ptr}

def init_dataset_state(dataset_name: str, df: pd.DataFrame, n_cases: int, base_seed: int):
    dkey = dataset_name
    order = make_case_order(len(df), base_seed, dataset_name)[: min(n_cases, len(df))]
    st.session_state.datasets_state[dkey] = {
        "df": df,
        "order": order,
        "ptr": 0,
        "n_cases": len(order),
    }

def get_dataset_state(dataset_name: str):
    return st.session_state.datasets_state.get(dataset_name, None)

def ensure_dataset_loaded(cfg, dataset_name: str):
    state = get_dataset_state(dataset_name)
    if state is None:
        # Try to load lazily
        path = cfg["paths"][dataset_name]
        df = load_dataset(path)
        init_dataset_state(dataset_name, df, cfg["n_cases"], cfg["base_seed"])

def preview(txt: str, n=300) -> str:
    s = (txt or "").strip()
    return s[:n] + ("..." if len(s) > n else "")

def render_dataset_tab(cfg, dataset_name: str):
    st.subheader(f"{dataset_name}")
    # Lazy load this dataset if needed
    try:
        ensure_dataset_loaded(cfg, dataset_name)
    except Exception as e:
        st.error(f"Failed to load {dataset_name}: {e}")
        return

    state = get_dataset_state(dataset_name)
    df = state["df"]
    order = state["order"]
    ptr = state["ptr"]

    # Controls row
    cols_top = st.columns([1,1,1,2])
    with cols_top[0]:
        st.metric("Total cases", len(df))
    with cols_top[1]:
        st.metric("In session", state["n_cases"])
    with cols_top[2]:
        st.metric("Current index", ptr+1 if order else 0)

    # Reset this dataset
    with cols_top[3]:
        if st.button(f"Reset {dataset_name}", key=f"reset_{dataset_name}"):
            init_dataset_state(dataset_name, df, cfg["n_cases"], cfg["base_seed"])
            st.success("Reset complete.")
            state = get_dataset_state(dataset_name)
            order = state["order"]
            ptr = state["ptr"]

    if not order:
        st.warning("No cases selected for this dataset. Increase 'Cases per dataset' and click 'Start / Reset ALL datasets'.")
        return

    case_index = order[ptr]
    row = df.iloc[case_index].to_dict()

    # A/B mapping
    ai_is_a = ab_mapping_for_case(cfg["base_seed"], dataset_name, case_index)

    ref = row.get("reference_impression_clean", "")
    gen = row.get("generated_impression_clean", "")
    findings = row.get("Findings", "")

    A_text = gen if ai_is_a else ref
    B_text = ref if ai_is_a else gen

    with st.expander("Clinical Findings", expanded=True):
        st.write(findings)

    st.write("### Impressions (blinded)")
    colA, colB = st.columns(2, gap="large")
    with colA:
        st.markdown("**Impression A**")
        st.text_area(" ", A_text, height=260, label_visibility="collapsed",
                     key=f"{dataset_name}_A_{case_index}")
    with colB:
        st.markdown("**Impression B**")
        st.text_area("  ", B_text, height=260, label_visibility="collapsed",
                     key=f"{dataset_name}_B_{case_index}")

    if cfg["reveal"]:
        st.info(f"Admin view: In **{dataset_name}**, AI is shown as: **{'A' if ai_is_a else 'B'}**.")

    # Choice
    with st.form(f"choice_form_{dataset_name}_{case_index}", clear_on_submit=False):
        choice = st.radio(
            "Which impression do you prefer?",
            options=["A", "B", "Tie / No preference", "Skip"],
            index=0,
            horizontal=True,
            key=f"choice_{dataset_name}_{case_index}"
        )
        submitted = st.form_submit_button("Save & Next")

    if submitted:
        if not cfg["rater_id"]:
            st.error("Enter your rater ID in the sidebar first.")
        else:
            db_row = dict(
                ts_utc=datetime.utcnow().isoformat(timespec="seconds").replace("+00:00","Z"),
                rater_id=cfg["rater_id"],
                dataset=dataset_name,
                case_index=int(case_index),
                ai_is_a=1 if ai_is_a else 0,
                choice={"A":"A","B":"B","Tie / No preference":"Tie","Skip":"Skip"}.get(choice, "Skip"),
                findings_hash=text_hash(str(findings)),
                ref_hash=text_hash(ref),
                gen_hash=text_hash(gen),
                ref_model="",  # human reference
                gen_model=row.get("model_name", ""),
                reward=float(row["reward"]) if "reward" in row and pd.notna(row["reward"]) else None,
                mean_f1=float(row["mean_f1_score"]) if "mean_f1_score" in row and pd.notna(row["mean_f1_score"]) else None,
                findings_preview=preview(findings),
                a_preview=preview(A_text),
                b_preview=preview(B_text),
            )
            try:
                init_db(cfg["db_path"])
                save_rating(cfg["db_path"], db_row)
                # advance pointer
                if state["ptr"] < len(order) - 1:
                    state["ptr"] += 1
                st.success("Saved.")
            except Exception as e:
                st.error(f"Failed to save rating: {e}")

    # Navigation
    nav1, prog, nav2 = st.columns([1,6,1])
    with nav1:
        if st.button("⬅️ Previous", disabled=(state["ptr"] == 0), key=f"prev_{dataset_name}"):
            state["ptr"] = max(0, state["ptr"] - 1)
    with prog:
        st.progress((state["ptr"] + 1) / max(1, len(order)))
    with nav2:
        if st.button("Next ➡️", disabled=(state["ptr"] >= len(order) - 1), key=f"next_{dataset_name}"):
            state["ptr"] = min(len(order) - 1, state["ptr"] + 1)

def main():
    ui_header()
    cfg = ui_sidebar()
    init_session_state()

    # Initialize DB once early (if possible) so export works even before first save
    try:
        if cfg["db_path"]:
            init_db(cfg["db_path"])
    except Exception as e:
        st.error(f"Database init failed: {e}")
        st.stop()

    # Start/Reset ALL
    if cfg["start_all"]:
        st.session_state.datasets_state = {}
        for name, path in cfg["paths"].items():
            try:
                df = load_dataset(path)
                init_dataset_state(name, df, cfg["n_cases"], cfg["base_seed"])
            except Exception as e:
                st.warning(f"{name}: {e}")
        st.success("All datasets initialized.")

    # Tabs for four GUIs
    tab_names = list(DEFAULT_DATASETS.keys())
    tabs = st.tabs(tab_names)
    for tab, name in zip(tabs, tab_names):
        with tab:
            render_dataset_tab(cfg, name)

    st.divider()
    st.subheader("Export Results")
    if st.button("Export all ratings to CSV"):
        try:
            df_out = export_ratings(cfg["db_path"])
            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download ratings_export.csv",
                data=csv_bytes,
                file_name="ratings_export.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Export failed: {e}")

    st.caption("Note: Each tab is an independent GUI with its own randomized case order and progress. "
               "All ratings go to the same SQLite database for unified analysis.")

if __name__ == "__main__":
    main()
