import streamlit as st
import json
import tempfile
import pandas as pd
import pyreadstat
from openai import OpenAI
import re

# ======================================================
# CONFIG
# ======================================================
OPENAI_API_KEY = "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"
MODEL_NAME = "gpt-5.2"

st.set_page_config(layout="wide")
st.title("🧬 Raw → SDTM Mapping ")

if not OPENAI_API_KEY or "PASTE" in OPENAI_API_KEY:
    st.error("Please paste a valid OpenAI API key.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ======================================================
# HELPERS
# ======================================================
def safe_json(text):
    text = re.sub(r"```.*?```", "", text, flags=re.S).strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("No JSON found")
    return json.loads(match.group(0))

def first_non_empty_row(df):
    for i in range(len(df)):
        if df.iloc[i].notna().any():
            return df.iloc[i]
    return None

# ======================================================
# LOAD DOMAIN CONFIG
# ======================================================
with open("domain_config.json", encoding="utf-8") as f:
    DOMAIN_CONFIG = json.load(f)

domain = st.selectbox("Select SDTM Domain", sorted(DOMAIN_CONFIG.keys()))
cfg = DOMAIN_CONFIG[domain]

allowed_vars = list(cfg["allowed_sdtm_vars"].keys())
core_meta = cfg["allowed_sdtm_vars"]
core_cols = cfg.get("core_columns", [])
sequence_field = cfg.get("sequence_field")

# ======================================================
# FILE UPLOAD
# ======================================================
uploaded = st.file_uploader("Upload Raw SAS File", type=["xpt", "sas7bdat"])
if not uploaded:
    st.stop()

suffix = ".xpt" if uploaded.name.lower().endswith(".xpt") else ".sas7bdat"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

df_raw, meta = (
    pyreadstat.read_xport(tmp_path)
    if suffix == ".xpt"
    else pyreadstat.read_sas7bdat(tmp_path)
)

st.success(f"Loaded {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

# ======================================================
# METADATA + SAMPLE VALUE
# ======================================================
sample_row = first_non_empty_row(df_raw)

raw_metadata = []
for col in df_raw.columns:
    try:
        label = meta.column_labels[meta.column_names.index(col)]
    except Exception:
        label = ""

    sample = None
    if sample_row is not None and not pd.isna(sample_row[col]):
        sample = str(sample_row[col])

    raw_metadata.append({
        "raw": col,
        "raw_label": label,
        "type": "Character" if df_raw[col].dtype == object else "Numeric",
        "sample_value": sample
    })

with st.expander("📄 Raw Metadata", expanded=True):
    st.dataframe(pd.DataFrame(raw_metadata), use_container_width=True)

# ======================================================
# LLM PROMPT
# ======================================================
prompt = f"""
Return ONLY JSON.

You are an SDTM mapping expert.

Target domain: {domain}

Raw variables with labels and sample values:
{json.dumps(raw_metadata, indent=2)}

Allowed SDTM variables:
{allowed_vars}

Return:
{{
  "mappings": [
    {{
      "raw": "<raw>",
      "sdtm": "<SDTM var or null>",
      "reason": "<short reason>"
    }}
  ]
}}
"""

if st.button("🧠 Generate Mapping via LLM"):
    with st.spinner("Calling LLM..."):
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
    st.session_state["mappings"] = safe_json(
        resp.choices[0].message.content
    )["mappings"]

if "mappings" not in st.session_state:
    st.stop()

# ======================================================
# MAPPING UI (CLEAN, ALIGNED)
# ======================================================
st.subheader("🔗 Raw → SDTM Mapping")

header = st.columns([2, 4, 3, 2, 2, 5])
for col, name in zip(
    header,
    ["Raw", "Raw Label", "SDTM Variable", "Type", "Core", "Reason"]
):
    col.markdown(f"**{name}**")

updated = []

for i, m in enumerate(st.session_state["mappings"]):
    raw = m["raw"]
    meta_row = next(x for x in raw_metadata if x["raw"] == raw)

    c1, c2, c3, c4, c5, c6 = st.columns([2, 4, 3, 2, 2, 5])

    c1.write(raw)
    c2.write(f"{meta_row['raw_label']}\n\n*eg: {meta_row['sample_value']}*")

    guess = m["sdtm"] if m["sdtm"] in allowed_vars else None
    sdtm_val = c3.selectbox(
        "",
        options=[None] + allowed_vars,
        index=(allowed_vars.index(guess) + 1 if guess else 0),
        key=f"sdtm_{i}"
    )

    c4.write(meta_row["type"])
    c5.write(core_meta.get(sdtm_val, {}).get("core", "—"))

    reason_text = m.get("reason", "")
    c6.markdown(
        f"<div style='white-space:pre-wrap'>{reason_text}</div>",
        unsafe_allow_html=True
    )

    updated.append({
        "raw": raw,
        "raw_label": meta_row["raw_label"],
        "sdtm": sdtm_val
    })

# ======================================================
# DUPLICATE CHECK (ONLY QC LEFT)
# ======================================================
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
    st.warning("Duplicate SDTM mappings detected")
    st.dataframe(dups)
else:
    st.success("No duplicate SDTM mappings")

# ======================================================
# BUILD MAIN DOMAIN
# ======================================================
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

# ======================================================
# BUILD SUPP DOMAIN (ONLY UNMAPPED)
# ======================================================
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
            "QNAM": m["raw"][:8].upper(),
            "QLABEL": m["raw_label"],
            "QVAL": str(val),
            "QORIG": "CRF",
            "QEVAL": None
        })

supp_df = pd.DataFrame(supp_rows)

if supp_df.empty:
    st.info("No SUPP records")
else:
    st.dataframe(supp_df.head(20), use_container_width=True)

# ======================================================
# DOWNLOADS
# ======================================================
st.subheader("⬇ Downloads")

st.download_button(
    "Download MAIN",
    main_df.to_csv(index=False),
    file_name=f"{domain}.csv",
    mime="text/csv"
)

if not supp_df.empty:
    st.download_button(
        "Download SUPP",
        supp_df.to_csv(index=False),
        file_name=f"SUPP{domain}.csv",
        mime="text/csv"
    )
