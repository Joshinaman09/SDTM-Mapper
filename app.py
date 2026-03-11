import streamlit as st
import pandas as pd
import pyreadstat
import tempfile
import json
import os
import re
from openai import OpenAI

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="TrialMapper AI",
    layout="wide"
)

# ===============================
# HEADER (LOGO + TITLE)
# ===============================
col1, col2 = st.columns([1, 6])

with col1:
    st.image("logo.png", width=120)

with col2:
    st.markdown("""
    # 🧬 TrialMapper AI

    **AI-Powered Raw → SDTM Mapping Engine**

    Automatically convert raw clinical datasets into **CDISC SDTM compliant datasets** using AI-assisted metadata interpretation.
    """)

st.markdown("---")

# ===============================
# SIDEBAR INFO
# ===============================
with st.sidebar:
    st.header("About TrialMapper AI")
    st.write("""
    AI-powered SDTM mapping assistant designed for:

    • Clinical Data Managers  
    • CDISC Programmers  
    • Biostatisticians  
    • Clinical Data Engineers  

    Key capabilities:

    • AI-driven SDTM variable mapping  
    • Raw metadata analysis  
    • MAIN + SUPP dataset generation  
    • Duplicate mapping detection  
    • Core variable validation
    """)

# ===============================
# WELCOME DESCRIPTION
# ===============================
st.info("""
### 🚀 How This Tool Works

1️⃣ Upload a **raw SAS dataset (.XPT or .sas7bdat)**  
2️⃣ AI analyzes **variable names, labels, and sample values**  
3️⃣ Suggested **SDTM mappings are generated automatically**  
4️⃣ You can manually adjust mappings  
5️⃣ The tool generates:

• **MAIN SDTM dataset**  
• **SUPP supplemental dataset**  
• **Mapping validation**

Designed to accelerate **clinical data standardization workflows**.
""")

# ===============================
# CONFIG
# ===============================
MODEL_NAME = "gpt-4o-mini"

api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found. Add it in Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=api_key)

# ===============================
# JSON CLEANER
# ===============================
def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON found")
    return json.loads(match.group(0))

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
# FILE UPLOAD
# ===============================
uploaded_files = [
    "/mnt/data/LAB.xpt", "/mnt/data/LAB1.xpt", "/mnt/data/LAB2.xpt", "/mnt/data/labcode.xpt", "/mnt/data/VS.xpt"
]

# ===============================
# Load SAS Files with Error Handling
# ===============================
def load_sas_file(file_path):
    """Function to load SAS files with error handling."""
    try:
        if file_path.lower().endswith(".xpt"):
            df, _ = pyreadstat.read_xport(file_path)
        else:
            df, _ = pyreadstat.read_sas7bdat(file_path)
        return df
    except pyreadstat.pyreadstat.PyreadstatError as e:
        st.error(f"Error reading {file_path}: {e}")
        return None

# Load the uploaded SAS files
dfs = {}
for file in uploaded_files:
    file_name = file.split("/")[-1]
    dfs[file_name] = load_sas_file(file)
    if dfs[file_name] is None:
        st.error(f"Failed to load {file_name}. Please check the file.")
        st.stop()  # Stop if any file failed to load

# Check that 'LAB' file is loaded
if 'LAB.xpt' not in dfs:
    st.error("The 'LAB.xpt' file is missing or could not be loaded.")
    st.stop()

# ===============================
# Check column names
# ===============================
def check_columns(dfs):
    """Function to print column names of each dataframe."""
    for key, df in dfs.items():
        st.write(f"Columns in {key}: {df.columns.tolist()}")

# Check columns of the loaded files
check_columns(dfs)

# ===============================
# Rename columns if necessary
# ===============================
# Assuming the common column for merging should be 'USUBJID'
# We will standardize column names before merging
def standardize_column_names(df, column_mapping):
    """Standardize column names to 'USUBJID' if they are named differently."""
    df.rename(columns=column_mapping, inplace=True)

# Define column mappings (adjust based on the column names you find)
column_mappings = {
    'SUBJECT': 'USUBJID',  # 'SUBJECT' column in the LAB dataset maps to 'USUBJID'
    'PATIENT': 'USUBJID'   # 'PATIENT' column in LAB, LAB1, LAB2 maps to 'USUBJID'
}

# Standardize columns in each dataset
for key, df in dfs.items():
    standardize_column_names(df, column_mappings)

# Merge the DataFrames based on a common identifier 'USUBJID'
merged_df = dfs['LAB.xpt'].merge(dfs['LAB1.xpt'], on="USUBJID", how="outer")
merged_df = merged_df.merge(dfs['LAB2.xpt'], on="USUBJID", how="outer")
merged_df = merged_df.merge(dfs['labcode.xpt'], on="USUBJID", how="outer")
merged_df = merged_df.merge(dfs['VS.xpt'], on="USUBJID", how="outer")

st.success(f"Merged dataset with {merged_df.shape[0]} rows and {merged_df.shape[1]} columns.")

# ===============================
# UNPIVOTING
# ===============================
def unpivot_df(df, id_vars, value_vars):
    """Unpivot the dataset, keeping 'id_vars' and transforming 'value_vars'."""
    return pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name="Variable", value_name="Value")

id_vars = ['USUBJID', 'STUDYID', 'RDOMAIN']
value_vars = [col for col in merged_df.columns if col not in id_vars]  # Example, unpivot all non-id columns

unpivoted_df = unpivot_df(merged_df, id_vars, value_vars)
st.success(f"Unpivoted dataset has {unpivoted_df.shape[0]} rows.")

# ===============================
# Preprocessing (clean, transform)
# ===============================
def preprocess_data(df):
    """Preprocessing the unpivoted data before mapping."""
    df = df.dropna(subset=["Value", "Variable"])
    df["Variable"] = df["Variable"].str.upper().str.replace(" ", "_")
    return df

processed_df = preprocess_data(unpivoted_df)

# ===============================
# DISPLAY DATAFRAME
# ===============================
st.dataframe(processed_df.head(20), use_container_width=True)

# ===============================
# LLM PROMPT
# ===============================
prompt = f"""
You are an SDTM mapping expert.

Target SDTM domain: {domain}

Raw variables with labels and sample values:

{json.dumps(processed_df.to_dict(orient="records"), indent=2)}

Allowed SDTM variables:
{allowed_sdtm_vars}

Return JSON only.

{{
"domain": "{domain}",
"mappings":[
{{
"raw":"<raw>",
"raw_label":"<label>",
"sample_value":"<sample>",
"sdtm":"<SDTM variable or null>",
"type":"<Character or Numeric>"
}}]
}}

Rules:
- Use only allowed SDTM variables
- Use sample_value to infer meaning
- Return raw JSON only
- No markdown
"""

# ===============================
# GENERATE MAPPING
# ===============================
if st.button("🚀 Generate AI Mapping"):
    with st.spinner("Calling OpenAI..."):
        resp = client.responses.create(
            model=MODEL_NAME,
            input=prompt
        )
        result = resp.output_text

    try:
        parsed = extract_json(result)
        st.session_state["mappings"] = parsed["mappings"]
    except Exception as e:
        st.error(f"LLM returned invalid JSON: {e}")
        st.code(result)
        st.stop()

# ===============================
# MAPPING UI
# ===============================
if "mappings" not in st.session_state:
    st.stop()

st.subheader("🔗 Raw → SDTM Mapping")

updated = []

header = st.columns([2, 4, 3, 2, 4])

header[0].markdown("**Raw**")
header[1].markdown("**Raw Label**")
header[2].markdown("**SDTM Variable**")
header[3].markdown("**Type**")
header[4].markdown("**Core**")

for i, m in enumerate(st.session_state["mappings"]):
    c1, c2, c3, c4, c5 = st.columns([2, 4, 3, 2, 4])
    c1.write(m["raw"])
    c2.write(m["raw_label"])

    guess = m["sdtm"] if m["sdtm"] in allowed_sdtm_vars else None

    sdtm_val = c3.selectbox(
        "",
        options=[None] + allowed_sdtm_vars,
        index=(allowed_sdtm_vars.index(guess) + 1 if guess else 0),
        key=f"sdtm_{i}"
    )

    c4.write(m["type"])

    core = sdtm_meta.get(sdtm_val, {}).get("core")
    c5.write(core if core else "-")

    updated.append({
        "raw": m["raw"],
        "raw_label": m["raw_label"],
        "sdtm": sdtm_val,
        "type": m["type"]
    })

# ===============================
# DUPLICATE CHECK
# ===============================
st.subheader("🚨 Duplicate SDTM Variables")

df_map = pd.DataFrame(updated)
dups = (
    df_map[df_map["sdtm"].notna()]
    .groupby("sdtm")["raw"]
    .nunique()
    .reset_index(name="count")
)

dups = dups[dups["count"] > 1]

if not dups.empty:
    st.warning("Duplicate SDTM mappings found")
    for _, r in dups.iterrows():
        raws = df_map[df_map["sdtm"] == r["sdtm"]]["raw"].tolist()
        st.write(f"{r['sdtm']} ← {', '.join(raws)}")
else:
    st.success("No duplicate mappings")

# ===============================
# DOWNLOAD MAIN DOMAIN
# ===============================
st.subheader("📊 MAIN Domain Preview")

main_df = pd.DataFrame()

for m in updated:
    if m["sdtm"]:
        main_df[m["sdtm"]] = merged_df[m["raw"]]

for col in core_cols:
    if col not in main_df.columns:
        main_df[col] = None

main_df["DOMAIN"] = domain

if sequence_field and sequence_field not in main_df.columns:
    main_df[sequence_field] = range(1, len(main_df) + 1)

st.dataframe(main_df.head(20), use_container_width=True)

# ===============================
# DOWNLOAD
# ===============================
st.download_button(
    "⬇ Download MAIN Domain",
    main_df.to_csv(index=False),
    file_name=f"{domain}.csv",
    mime="text/csv"
)
