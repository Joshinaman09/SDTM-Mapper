import streamlit as st
import json
import tempfile
import pandas as pd
import pyreadstat
import os
import re
from openai import OpenAI


# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="TrialMapper AI", layout="wide")
st.title("🧬 TrialMapper AI")
st.caption("Raw → SDTM Mapping with Preprocessing Layer")

# ===============================
# CONFIG
# ===============================
MODEL_NAME = "gpt-4o-mini"
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found. Add it in Streamlit secrets or environment.")
    st.stop()

client = OpenAI(api_key=api_key)


# ===============================
# HELPERS
# ===============================
def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON found in model response")

    return json.loads(match.group(0))


def read_sas_uploaded_file(uploaded_file):
    suffix = ".xpt" if uploaded_file.name.lower().endswith(".xpt") else ".sas7bdat"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if suffix == ".xpt":
        df, meta = pyreadstat.read_xport(tmp_path)
    else:
        df, meta = pyreadstat.read_sas7bdat(tmp_path)

    return df, meta


def columns_signature(df: pd.DataFrame):
    return tuple(df.columns.tolist())


def build_file_groups(file_dfs: dict):
    """
    Group files by identical column structure.
    """
    groups = {}
    for name, df in file_dfs.items():
        sig = columns_signature(df)
        groups.setdefault(sig, []).append(name)
    return groups


def append_same_structure_groups(file_dfs: dict):
    """
    Append files that have identical columns.
    Returns:
      combined_tables: dict[str, DataFrame]
      append_log: list[str]
    """
    groups = build_file_groups(file_dfs)
    combined_tables = {}
    append_log = []

    for idx, (sig, file_names) in enumerate(groups.items(), start=1):
        if len(file_names) == 1:
            table_name = file_names[0]
            combined_tables[table_name] = file_dfs[file_names[0]].copy()
            append_log.append(f"No append needed: {file_names[0]}")
        else:
            appended_df = pd.concat([file_dfs[f] for f in file_names], ignore_index=True)
            table_name = f"APPENDED_GROUP_{idx}"
            combined_tables[table_name] = appended_df
            append_log.append(f"Appended: {', '.join(file_names)} → {table_name}")

    return combined_tables, append_log


def safe_merge(left: pd.DataFrame, right: pd.DataFrame, key: str, how: str = "left"):
    """
    Merge while avoiding duplicate non-key column collisions.
    """
    overlapping = [c for c in right.columns if c in left.columns and c != key]
    if overlapping:
        right = right.rename(columns={c: f"{c}_REF" for c in overlapping})

    return pd.merge(left, right, on=key, how=how)


def add_usubjid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "USUBJID" in df.columns:
        return df

    if "STUDYID" in df.columns and "SUBJECT" in df.columns:
        df["USUBJID"] = df["STUDYID"].astype(str).str.strip() + "-" + df["SUBJECT"].astype(str).str.strip()
        return df

    if "STUDY" in df.columns and "SUBJECT" in df.columns:
        df["USUBJID"] = df["STUDY"].astype(str).str.strip() + "-" + df["SUBJECT"].astype(str).str.strip()
        return df

    if "STUDY" in df.columns and "PATIENT" in df.columns:
        df["USUBJID"] = df["STUDY"].astype(str).str.strip() + "-" + df["PATIENT"].astype(str).str.strip()
        return df

    return df


def add_studyid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "STUDYID" not in df.columns:
        if "STUDY" in df.columns:
            df["STUDYID"] = df["STUDY"]

    return df


def unpivot_dataframe(df: pd.DataFrame, id_vars: list, value_vars: list) -> pd.DataFrame:
    return pd.melt(
        df,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="RAW_TEST",
        value_name="RAW_VALUE"
    )


def get_sample_row(df: pd.DataFrame):
    for i in range(len(df)):
        if df.iloc[i].notna().any():
            return df.iloc[i]
    return None


def build_raw_metadata(df: pd.DataFrame, meta=None):
    raw_metadata = []
    sample_row = get_sample_row(df)

    label_dict = {}
    if meta is not None:
        try:
            label_dict = meta.column_names_to_labels or {}
        except Exception:
            label_dict = {}

    for col in df.columns:
        label = label_dict.get(col, "")
        if not label:
            label = col.replace("_", " ").title()

        sample_val = None
        if sample_row is not None:
            v = sample_row[col]
            if pd.notna(v):
                sample_val = str(v)

        dtype_name = str(df[col].dtype).lower()
        var_type = "Character" if ("object" in dtype_name or "string" in dtype_name) else "Numeric"

        raw_metadata.append({
            "raw": col,
            "raw_label": label,
            "type": var_type,
            "sample_value": sample_val
        })

    return raw_metadata


# ===============================
# LOAD DOMAIN CONFIG
# ===============================
with open("domain_config.json", encoding="utf-8") as f:
    DOMAIN_CONFIG = json.load(f)

domain = st.selectbox("Select SDTM Domain", sorted(DOMAIN_CONFIG.keys()))
cfg = DOMAIN_CONFIG[domain]

sdtm_meta = cfg["allowed_sdtm_vars"]
allowed_sdtm_vars = list(sdtm_meta.keys())
core_cols = cfg["core_columns"]
sequence_field = cfg.get("sequence_field")

# ===============================
# UPLOAD SECTION
# ===============================
st.subheader("1️⃣ Upload Raw SAS Files")

uploaded_files = st.file_uploader(
    "Upload 1 or more SAS files",
    type=["xpt", "sas7bdat"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.stop()

# ===============================
# READ FILES
# ===============================
file_dfs = {}
file_metas = {}

for uploaded in uploaded_files:
    try:
        df, meta = read_sas_uploaded_file(uploaded)
        file_dfs[uploaded.name] = df
        file_metas[uploaded.name] = meta
    except Exception as e:
        st.error(f"Failed to read {uploaded.name}: {e}")
        st.stop()

st.success(f"Loaded {len(file_dfs)} file(s)")

with st.expander("Preview uploaded file structures", expanded=False):
    for name, df in file_dfs.items():
        st.markdown(f"**{name}** — shape: {df.shape}")
        st.write(df.columns.tolist())

# ===============================
# PREPROCESSING
# ===============================
st.subheader("2️⃣ Preprocessing")

st.markdown("### Step A: Auto-append files with identical structure")

combined_tables, append_log = append_same_structure_groups(file_dfs)

for msg in append_log:
    st.write(f"- {msg}")

table_names = list(combined_tables.keys())

if len(table_names) == 1:
    selected_main_table = table_names[0]
else:
    selected_main_table = st.selectbox(
        "Select the primary/main table after append step",
        table_names
    )

df_processed = combined_tables[selected_main_table].copy()

other_tables = [t for t in table_names if t != selected_main_table]

st.markdown("### Step B: Optional merge with another table")

do_merge = False
merge_table = None
merge_key = None
merge_how = "left"

if other_tables:
    do_merge = st.checkbox("Merge main table with another table")
    if do_merge:
        merge_table = st.selectbox("Select reference / lookup table to merge", other_tables)
        candidate_keys = sorted(list(set(df_processed.columns).intersection(set(combined_tables[merge_table].columns)))))
        if candidate_keys:
            merge_key = st.selectbox("Select merge key", candidate_keys)
        else:
            st.warning("No common columns found between selected tables.")
            merge_key = None

        merge_how = st.selectbox("Merge type", ["left", "inner", "outer"], index=0)

if do_merge and merge_table and merge_key:
    df_processed = safe_merge(df_processed, combined_tables[merge_table], merge_key, how=merge_how)
    st.success(f"Merged {selected_main_table} with {merge_table} on {merge_key}")

st.markdown("### Step C: Optional add derived identifiers")

derive_ids = st.checkbox("Auto-create STUDYID / USUBJID if possible", value=True)

if derive_ids:
    df_processed = add_studyid(df_processed)
    df_processed = add_usubjid(df_processed)

st.markdown("### Step D: Optional unpivot")

do_unpivot = st.checkbox("Unpivot dataset", value=False)

if do_unpivot:
    all_cols = df_processed.columns.tolist()
    default_id_vars = [c for c in ["STUDYID", "USUBJID", "VISIT", "VISITNUM", "DOMAIN"] if c in all_cols]

    id_vars = st.multiselect(
        "Select ID variables to keep",
        options=all_cols,
        default=default_id_vars
    )

    value_vars = st.multiselect(
        "Select value columns to unpivot",
        options=[c for c in all_cols if c not in id_vars]
    )

    if st.button("Apply Unpivot"):
        if not id_vars or not value_vars:
            st.error("Select at least one ID variable and one value variable.")
            st.stop()

        df_processed = unpivot_dataframe(df_processed, id_vars=id_vars, value_vars=value_vars)
        st.success("Unpivot applied")

st.subheader("📦 Preprocessed Dataset Preview")
st.write(f"Shape: {df_processed.shape}")
st.dataframe(df_processed.head(20), use_container_width=True)

# ===============================
# BUILD RAW METADATA
# ===============================
st.subheader("3️⃣ Raw Metadata for Mapping")

# Use meta only when there is exactly one original file and no major reshape.
meta_for_labels = None
if len(uploaded_files) == 1 and not do_merge and not do_unpivot:
    meta_for_labels = file_metas[uploaded_files[0].name]

raw_metadata = build_raw_metadata(df_processed, meta=meta_for_labels)
st.dataframe(pd.DataFrame(raw_metadata), use_container_width=True)
