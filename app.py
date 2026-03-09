import streamlit as st
import json
import tempfile
import pandas as pd
import pyreadstat
from openai import OpenAI

# ===============================
# CONFIG
# ===============================
MODEL_NAME = "gpt-5.2"
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
st.set_page_config(layout="wide")
st.title("🧬 Raw → SDTM Mapping (MAIN + SUPP)")

# ===============================
# LOAD DOMAIN CONFIG (CORE-BASED)
# ===============================
with open("domain_config.json", encoding="utf-8") as f:
    DOMAIN_CONFIG = json.load(f)

domain = st.selectbox("Select SDTM Domain", sorted(DOMAIN_CONFIG.keys()))
cfg = DOMAIN_CONFIG[domain]

sdtm_meta = cfg["allowed_sdtm_vars"]              # {VAR: {core: Req/Exp/Perm}}
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
# SAVE & READ FILE
# ===============================
suffix = ".xpt" if uploaded.name.lower().endswith(".xpt") else ".sas7bdat"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

if suffix == ".xpt":
    df_raw, meta = pyreadstat.read_xport(tmp_path)
else:
    df_raw, meta = pyreadstat.read_sas7bdat(tmp_path)

st.success(f"Loaded {df_raw.shape[1]} raw variables from {uploaded.name}")

# ===============================
# RAW METADATA + FIRST NON-EMPTY ROW
# ===============================
sample_row = None
for i in range(len(df_raw)):
    if df_raw.iloc[i].notna().any():
        sample_row = df_raw.iloc[i]
        break

raw_metadata = []
for col in df_raw.columns:
    try:
        label = meta.column_labels[meta.column_names.index(col)]
    except Exception:
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

st.subheader("📄 Raw Metadata (Label + Sample Value)")
st.dataframe(pd.DataFrame(raw_metadata), use_container_width=True)

# ===============================
# LLM PROMPT (LABEL + FIRST ROW)
# ===============================
prompt = f"""
You are an SDTM mapping expert.

Target SDTM domain: {domain}

Raw variables with label and a sample value
(from the first non-empty data row):

{json.dumps(raw_metadata, indent=2)}

Allowed SDTM variables:
{allowed_sdtm_vars}

Return JSON ONLY:

{{
  "domain": "{domain}",
  "mappings": [
    {{
      "raw": "<raw>",
      "raw_label": "<label>",
      "sample_value": "<sample or null>",
      "sdtm": "<SDTM variable or null>",
      "type": "<Character or Numeric>"
    }}
  ]
}}

Rules:
- Use ONLY allowed SDTM variables
- Use sample_value to infer dates, flags, grades, identifiers
- If label and sample_value conflict, trust sample_value
- Use null if unsure
- No markdown
"""

# ===============================
# GENERATE INITIAL MAPPING
# ===============================
if st.button("🧠 Generate Mapping via LLM"):
    with st.spinner("Calling LLM..."):
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
    st.session_state["mappings"] = json.loads(
        resp.choices[0].message.content
    )["mappings"]

# ===============================
# MANUAL MAPPING UI (CORE ALIGNED)
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
header[4].markdown("**Core (SDTM IG)**")

for i, m in enumerate(st.session_state["mappings"]):
    c1, c2, c3, c4, c5 = st.columns([2, 4, 3, 2, 4])

    c1.markdown(f"<div style='padding-top:8px'>{m['raw']}</div>", unsafe_allow_html=True)
    c2.markdown(f"<div style='padding-top:8px'>{m['raw_label']}</div>", unsafe_allow_html=True)

    guess = m["sdtm"] if m["sdtm"] in allowed_sdtm_vars else None
    sdtm_val = c3.selectbox(
        "",
        options=[None] + allowed_sdtm_vars,
        index=(allowed_sdtm_vars.index(guess) + 1 if guess else 0),
        key=f"sdtm_{i}"
    )

    c4.markdown(f"<div style='padding-top:8px'>{m['type']}</div>", unsafe_allow_html=True)

    core = sdtm_meta.get(sdtm_val, {}).get("core")
    if core:
        c5.markdown(
            f"<div style='padding-top:8px; font-weight:600'>Core: {core}</div>",
            unsafe_allow_html=True
        )
    else:
        c5.markdown("<div style='padding-top:8px'>—</div>", unsafe_allow_html=True)

    updated.append({
        "raw": m["raw"],
        "raw_label": m["raw_label"],
        "sdtm": sdtm_val,
        "type": m["type"]
    })

# ===============================
# DUPLICATE SDTM WARNING
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
    st.warning("Same SDTM variable mapped from multiple raw variables")
    for _, r in dups.iterrows():
        raws = df_map[df_map["sdtm"] == r["sdtm"]]["raw"].tolist()
        st.write(f"**{r['sdtm']}** ← {', '.join(raws)}")
else:
    st.success("No duplicate SDTM mappings")

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
    main_df[sequence_field] = range(1, len(main_df) + 1)

st.dataframe(main_df.head(20), use_container_width=True)

# ===============================
# BUILD SUPP DOMAIN
# ===============================
st.subheader("📦 SUPP Domain Preview")

supp_rows = []
unmapped = [m for m in updated if m["sdtm"] is None]

for idx, row in main_df.iterrows():
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
            "QLABEL": m["raw_label"],
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
# DOWNLOADS
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

