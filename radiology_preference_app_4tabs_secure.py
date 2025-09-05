# radiology_preference_app_4tabs_secure.py
# Streamlit app for blinded A/B preference of radiology report impressions (AI vs Human)
# Four GUIs together (CT Head, CTPE, Ultrasound, General) as separate tabs,
# each with its own sequential case order & independent progress pointer, sharing ONE SQLite DB.
#
# CHANGES for this revision:
# - ✅ Voting buttons (A/B/Tie/Skip) now SAVE ONLY and DO NOT advance.
# - ✅ User must click "Next ➡️" to move to the next case.
# - ✅ Everything else remains the same (deterministic A/B, debiasing scrubber, auth, DB, etc.).
#
# How to run:
#   pip install streamlit pandas streamlit-authenticator pyyaml
#   streamlit run radiology_preference_app_4tabs_secure.py
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
# - A/B order is deterministic per dataset & case (via hash) to preserve blinding
#   with stability across users/sessions—no UI seed.
# - Admin-only "Reveal mapping" checkbox can show which side is AI after choices are stored.

import os
import hashlib
import random
import sqlite3
from datetime import datetime
import re
from typing import Optional

import pandas as pd
import streamlit as st

# --- Authentication libs ---
import json
import uuid
import logging
import time
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from streamlit_authenticator.utilities.hasher import Hasher

APP_TITLE = "Blinded A/B Preference: Radiology Impressions"
DB_PATH_DEFAULT = "preferences.db"

# ---------- Cleaning / Debiasing ----------

# Strip trailing boilerplate
TRAILING_PATTERNS = [
    r'(?i)\bimages?\s*(?:and|&)\s*interpretations?\b.*$',
    r'(?i)\bplease\s+see\s+images?\b.*$',
    r'(?i)\bcorrelate\s+clinically\b.*$',
    r'(?i)\bcontact\s+radiology\b.*$',
    r'(?i)\bthis\s+report\b.*?(?:contact|call|page).*$',  # "This report ... contact ..."
    r'(?i)\bfindings\s+and\s+interpretations?\b.*$',
]
TRAILING_REGEXES = [re.compile(p, re.DOTALL) for p in TRAILING_PATTERNS]

# Remove numbered/bulleted prefixes at line starts, e.g., "1. ", "2) ", "(3) ", "• ", "-", "a) ", "i."
LIST_PREFIX_RE = re.compile(
    r'(?im)^\s*(?:\(?\d+\)?[.)]|[ivxlcdm]+[.)]|[a-zA-Z][.)]|[-–—*•])\s+'
)

# Lines that can unblind/bias (critical alerts, escalation/communication meta, signatures, headings)
BIAS_LINE_PATTERNS = [
    r'(?im)^\s*(critical|urgent|important|significant)\s+(?:result|results|finding|findings|value|values|alert|notification|communication)\b.*$',
    r'(?im)^\s*(communication|contact|call|page|pager|phone)\s*[:].*$',
    r'(?im)^\s*(dictated by|signed by|attending|resident|fellow|radiologist)\b.*$',
    r'(?im)^\s*impression\s*[:\-]\s*$',
    r'(?im)^\s*final\s+report\s*[:\-]?.*$',
]
BIAS_LINE_REGEXES = [re.compile(p) for p in BIAS_LINE_PATTERNS]

def strip_list_markers(s: str) -> str:
    lines = (s or "").splitlines()
    cleaned = []
    for ln in lines:
        cleaned.append(LIST_PREFIX_RE.sub("", ln))
    return "\n".join(cleaned)

def remove_bias_lines(s: str) -> str:
    if not isinstance(s, str):
        return ""
    lines = s.splitlines()
    kept = []
    for ln in lines:
        if any(rx.search(ln) for rx in BIAS_LINE_REGEXES):
            continue
        kept.append(ln)
    out = "\n".join(kept)
    out = re.sub(r'\n{3,}', '\n\n', out).strip()
    return out

def clean_impression(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = text.strip()
    # Remove trailing boilerplate
    for rx in TRAILING_REGEXES:
        s = rx.sub("", s).strip()
    # Debias: remove headings/alerts/signatures, strip list markers
    s = remove_bias_lines(s)
    s = strip_list_markers(s)
    # Normalize whitespace
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

# ---------- Authentication ----------

def setup_authentication():
    """
    Uses streamlit-authenticator with a YAML config file (config.yaml).
    If config.yaml doesn't exist, create a default one with 20 demo testers:
      - usernames: tester1..tester20
      - passwords: pass1..pass20 (hashed in config)
    Replace config.yaml with your real credential set as needed.
    """
    try:
        if not os.path.exists("config.yaml"):
            plain_passwords = [f"pass{i+1}" for i in range(20)]
            hashed_passwords = Hasher(plain_passwords).generate()
            credentials_dict = {}
            for i in range(20):
                username = f"tester{i+1}"
                credentials_dict[username] = {
                    "email": f"{username}@example.com",
                    "name": f"Test User {i+1}",
                    "password": hashed_passwords[i],
                }
            config = {
                "credentials": {"usernames": credentials_dict},
                "cookie": {"expiry_days": 180, "key": "your_cookie_key", "name": "auth_cookie"},
                "preauthorized": [],
            }
            with open("config.yaml", "w") as f:
                yaml.dump(config, f)

        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=SafeLoader)

        return stauth.Authenticate(
            credentials=config["credentials"],
            cookie_name=config["cookie"]["name"],
            key=config["cookie"]["key"],
            cookie_expiry_days=config["cookie"]["expiry_days"],
            preauthorized=config.get("preauthorized", []),
        )
    except Exception as e:
        st.error(f"Authentication setup failed: {str(e)}")
        st.stop()

# ---------- Helpers ----------

def text_hash(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()

def deterministic_ai_is_a(dataset_name: str, case_index: int) -> bool:
    """
    Deterministic A/B mapping, no UI seed needed.
    Stable across users/sessions; preserves blinding (identity hidden).
    """
    seed_src = f"{dataset_name}|{case_index}|abmap_v1"
    seed_int = int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed_int ^ 0x9E3779B1)
    return rng.random() < 0.5

def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df = df.copy()

    # Clean & debias both impressions
    df["reference_impression_clean"] = df["reference_impression"].apply(clean_impression)
    df["generated_impression_clean"] = df["generated_impression"].apply(clean_impression)

    # Drop empty after cleaning
    df = df[
        (df["reference_impression_clean"].astype(str).str.len() > 0) &
        (df["generated_impression_clean"].astype(str).str.len() > 0)
    ].reset_index(drop=True)
    return df

# ---------- Streamlit UI ----------

def ui_header():
    st.title(APP_TITLE)
    st.caption(
        "Evaluate **four datasets** via tabs. Each tab shows clinical **Findings** and two blinded "
        "**Impressions (A & B)**—one human and one AI. Click a choice to save, then press **Next ➡️** to move on."
    )

def ui_sidebar(auth_user: Optional[str]):
    st.sidebar.header("Global Setup")

    # Rater ID defaults to authenticated username; user can override if desired.
    rater_id = st.sidebar.text_input("Your User ID (required)", value=(auth_user or ""))

    db_path = st.sidebar.text_input("SQLite DB path", value=DB_PATH_DEFAULT)

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
        paths=paths,
        start_all=start_all,
        reveal=reveal,
    )

def init_session_state():
    if "datasets_state" not in st.session_state:
        st.session_state.datasets_state = {}  # per dataset: {df, order (sequential), ptr}

def init_dataset_state(dataset_name: str, df: pd.DataFrame):
    # Sequential order from first to last for everyone.
    order = list(range(len(df)))
    st.session_state.datasets_state[dataset_name] = {
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
        path = cfg["paths"][dataset_name]
        df = load_dataset(path)
        init_dataset_state(dataset_name, df)

def preview(txt: str, n=300) -> str:
    s = (txt or "").strip()
    return s[:n] + ("..." if len(s) > n else "")

def record_choice(cfg, dataset_name, case_index, ai_is_a, choice, findings, ref, gen, row, state):
    """Save the rating ONLY. Do NOT advance pointer; user must press Next."""
    if not cfg["rater_id"]:
        st.error("Enter your User ID in the sidebar first.")
        return
    db_row = dict(
        ts_utc=datetime.utcnow().isoformat(timespec="seconds").replace("+00:00","Z"),
        rater_id=cfg["rater_id"],
        dataset=dataset_name,
        case_index=int(case_index),
        ai_is_a=1 if ai_is_a else 0,
        choice=choice,
        findings_hash=text_hash(str(findings)),
        ref_hash=text_hash(ref),
        gen_hash=text_hash(gen),
        ref_model="",  # human reference
        gen_model=row.get("model_name", ""),
        reward=float(row["reward"]) if "reward" in row and pd.notna(row["reward"]) else None,
        mean_f1=float(row["mean_f1_score"]) if "mean_f1_score" in row and pd.notna(row["mean_f1_score"]) else None,
        findings_preview=preview(findings),
        a_preview=preview(gen if ai_is_a else ref),
        b_preview=preview(ref if ai_is_a else gen),
    )
    try:
        init_db(cfg["db_path"])
        save_rating(cfg["db_path"], db_row)
        st.success(f"Saved: {choice}. Press **Next ➡️** to continue.")
    except Exception as e:
        st.error(f"Failed to save rating: {e}")

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
            init_dataset_state(dataset_name, df)
            st.success("Reset complete.")
            state = get_dataset_state(dataset_name)
            order = state["order"]
            ptr = state["ptr"]

    if not order:
        st.warning("No cases available in this dataset.")
        return

    case_index = order[ptr]
    row = df.iloc[case_index].to_dict()

    # A/B mapping (deterministic, no seed UI)
    ai_is_a = deterministic_ai_is_a(dataset_name, case_index)

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
                     key=f"{dataset_name}_A_{case_index}", disabled=True)
    with colB:
        st.markdown("**Impression B**")
        st.text_area("  ", B_text, height=260, label_visibility="collapsed",
                     key=f"{dataset_name}_B_{case_index}", disabled=True)

    if cfg["reveal"]:
        st.info(f"Admin view: In **{dataset_name}**, AI is shown as: **{'A' if ai_is_a else 'B'}**.")

    # --- Voting buttons: SAVE ONLY (no advance) ---
    st.write("#### Choose your preference")
    btn_cols = st.columns([1,1,1,1])
    choice_pressed = None
    with btn_cols[0]:
        if st.button("A is better", key=f"btnA_{dataset_name}_{case_index}"):
            choice_pressed = "A"
    with btn_cols[1]:
        if st.button("B is better", key=f"btnB_{dataset_name}_{case_index}"):
            choice_pressed = "B"
    with btn_cols[2]:
        if st.button("Tie / No preference", key=f"btnT_{dataset_name}_{case_index}"):
            choice_pressed = "Tie"
    with btn_cols[3]:
        if st.button("Skip", key=f"btnS_{dataset_name}_{case_index}"):
            choice_pressed = "Skip"

    if choice_pressed is not None:
        record_choice(
            cfg, dataset_name, case_index, ai_is_a,
            choice_pressed, findings, ref, gen, row, state
        )

    # Navigation: Previous / Progress / Next (Next advances pointer)
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
    # --- Auth barrier ---
    authenticator = setup_authentication()
    authenticator.login(location="sidebar", key="login")
    authentication_status = st.session_state.get("authentication_status", None)
    username = st.session_state.get("username", None)

    if authentication_status:
        authenticator.logout("Logout", "sidebar", key="logout_button")
    else:
        if authentication_status is False:
            st.error("❌ Username/password is incorrect")
        else:
            st.warning("⚠️ Please enter your username and password")
        st.stop()

    # --- App proper ---
    ui_header()
    cfg = ui_sidebar(auth_user=username)
    init_session_state()

    # Initialize DB early so export works even before first save
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
                init_dataset_state(name, df)
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

    st.caption("Note: Each tab is an independent GUI with its own sequential case order and progress. "
               "All ratings go to the same SQLite database for unified analysis.")

if __name__ == "__main__":
    main()
