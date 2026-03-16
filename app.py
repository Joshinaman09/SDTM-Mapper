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

# ===============================
# HEADER
# ===============================
col1, col2 = st.columns([1,6])

with col1:
    st.image("logo.png", width=120)

with col2:
    st.markdown("""
    # 🧬 TrialMapper AI

    **AI-Powered Raw → SDTM Mapping Engine**
    """)

st.markdown("---")

# ===============================
# CONFIG
# ===============================
MODEL_NAME = "gpt-4o-mini"

api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found")
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
with open("domain_config.json") as f:
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
uploaded = st.file_uploader(
    "Upload Raw SAS File",
    type=["xpt","sas7bdat"]
)

if not uploaded:
    st.stop()

# ===============================
# SAVE FILE
# ===============================
suffix = ".xpt" if uploaded.name.lower().endswith(".xpt") else ".sas7bdat"

with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

# ===============================
# READ SAS
# ===============================
if suffix == ".xpt":
    df_raw, meta = pyreadstat.read_xport(tmp_path)
else:
    df_raw, meta = pyreadstat.read_sas7bdat(tmp_path)

st.success(f"Loaded {df_raw.shape[1]} raw variables")

# ===============================
# FIRST NON EMPTY ROW
# ===============================
sample_row = None

for i in range(len(df_raw)):
    if df_raw.iloc[i].notna().any():
        sample_row = df_raw.iloc[i]
        break

# ===============================
# RAW METADATA
# ===============================
raw_metadata = []

for col in df_raw.columns:

    try:
        label = meta.column_labels[meta.column_names.index(col)]
    except:
        label = ""

    sample_val = None

    if sample_row is not None:
        v = sample_row[col]

        if not pd.isna(v):
            sample_val = str(v)

    raw_metadata.append({
        "raw": col,
        "raw_label": label,
        "type": "Character" if df_raw[col].dtype == object else "Numeric",
        "sample_value": sample_val
    })

st.subheader("📄 Raw Metadata")
st.dataframe(pd.DataFrame(raw_metadata), use_container_width=True)

# ===============================
# LLM PROMPT
# ===============================
prompt = f"""
You are an SDTM mapping expert.

Target SDTM domain: {domain}

Raw variables:

{json.dumps(raw_metadata, indent=2)}

Allowed SDTM variables:
{allowed_sdtm_vars}

Return JSON only.

{{
"domain":"{domain}",
"mappings":[
{{
"raw":"",
"raw_label":"",
"sample_value":"",
"sdtm":"",
"type":""
}}
]
}}

Rules:
Map raw variables to correct SDTM variables using CDISC standards.

Examples:

SUBJECT → USUBJID  
STUDY → STUDYID  
LABCODE → LBTESTCD  
LABVALUE → LBORRES  
LAB_UNIT → LBORRESU
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

if "mappings" not in st.session_state:
    st.stop()

# ===============================
# MAPPING UI
# ===============================
st.subheader("🔗 Raw → SDTM Mapping")

updated = []

header = st.columns([2,4,2])

header[0].markdown("**Raw**")
header[1].markdown("**SDTM Variable**")
header[2].markdown("**Type**")

for i, m in enumerate(st.session_state["mappings"]):

    c1,c2,c3 = st.columns([2,4,2])

    c1.write(m["raw"])

    # -----------------------------
    # FIXED PRESELECT LOGIC
    # -----------------------------
    guess = m.get("sdtm")

    if guess not in allowed_sdtm_vars:
        guess = None

    index_val = 0

    if guess:
        index_val = allowed_sdtm_vars.index(guess) + 1

    sdtm_val = c2.selectbox(
        "",
        options=[None] + allowed_sdtm_vars,
        index=index_val,
        key=f"sdtm_{i}"
    )

    c3.write(m["type"])

    updated.append({
        "raw": m["raw"],
        "sdtm": sdtm_val
    })

# ===============================
# BUILD MAIN DOMAIN
# ===============================
st.subheader("📊 MAIN Domain Preview")

main_df = pd.DataFrame()

for m in updated:

    if m["sdtm"]:
        main_df[m["sdtm"]] = df_raw[m["raw"]]

for col in core_cols:
    if col not in main_df.columns:
        main_df[col] = None

main_df["DOMAIN"] = domain

if sequence_field and sequence_field not in main_df.columns:
    main_df[sequence_field] = range(1,len(main_df)+1)

st.dataframe(main_df.head(20), use_container_width=True)

# ===============================
# BUILD SUPP DOMAIN
# ===============================
st.subheader("📦 SUPP Domain Preview")

supp_rows = []

unmapped = [m for m in updated if m["sdtm"] is None]

for idx,row in main_df.iterrows():

    for m in unmapped:

        val = df_raw.loc[idx, m["raw"]]

        if pd.isna(val):
            continue

        supp_rows.append({
            "STUDYID": row.get("STUDYID"),
            "RDOMAIN": domain,
            "USUBJID": row.get("USUBJID"),
            "IDVAR": sequence_field,
            "IDVARVAL": row.get(sequence_field),
            "QNAM": m["raw"].upper()[:8],
            "QLABEL": m["raw"],
            "QVAL": str(val),
            "QORIG": "CRF",
            "QEVAL": None
        })

supp_df = pd.DataFrame(supp_rows)

if supp_df.empty:
    st.info("No SUPP records generated")
else:
    st.dataframe(supp_df.head(20), use_container_width=True)

# ===============================
# DOWNLOAD
# ===============================
st.download_button(
    "⬇ Download MAIN Domain",
    main_df.to_csv(index=False),
    file_name=f"{domain}.csv",
    mime="text/csv"
)

if not supp_df.empty:

    st.download_button(
        "⬇ Download SUPP Domain",
        supp_df.to_csv(index=False),
        file_name=f"SUPP{domain}.csv",
        mime="text/csv"
    )
