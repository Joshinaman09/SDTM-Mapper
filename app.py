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

Automatically convert raw clinical datasets into **CDISC SDTM compliant datasets** using AI assisted metadata interpretation.
""")


st.markdown("---")


# ===============================
# SIDEBAR INFO
# ===============================
with st.sidebar:

    st.header("About TrialMapper AI")

    st.write("""
AI powered SDTM mapping assistant designed for:

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

The tool generates:

• **MAIN SDTM dataset**  
• **SUPP supplemental dataset**  
• **Mapping validation**
""")


# ===============================
# CONFIG
# ===============================
MODEL_NAME = "gpt-4o-mini"

api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found.")
    st.stop()

client = OpenAI(api_key=api_key)


# ===============================
# JSON CLEANER
# ===============================
def extract_json(text):

    text = text.strip()

    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)

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
uploaded = st.file_uploader(
    "Upload Raw SAS File",
    type=["xpt", "sas7bdat"]
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

        if pd.notna(v):
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

Raw variables with labels and sample values:

{json.dumps(raw_metadata, indent=2)}

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
}}
]
}}

Rules:
Use only allowed SDTM variables.
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
# STOP IF NO MAPPING
# ===============================
if "mappings" not in st.session_state:
    st.stop()


# ===============================
# MAPPING UI
# ===============================
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
