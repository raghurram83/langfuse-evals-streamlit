"""
Conversation → FAQ Extractor

Features:
- Upload transcript (.txt, .csv, .json) or paste raw text
- Normalization with speaker detection
- PII redaction (remove names, business names, phone numbers; keep cities/locations/products)
- LLM-based FAQ extraction (OpenAI gpt-4.1-mini by default)
- Side-by-side pipeline view (raw/cleaned vs normalized/FAQ)
- Confidence-colored FAQs and CSV export

Running locally: streamlit run app.py
"""

from __future__ import annotations

import io
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI, OpenAIError


# -----------------------------
# Requirements generator section
# -----------------------------
REQUIREMENTS = ["streamlit", "pandas", "openai"]


# -----------------------------
# Parsing & normalization
# -----------------------------
def parse_transcript(content: bytes, filename: str) -> List[str]:
    """Convert uploaded content into a list of message lines."""
    name = filename.lower()
    text_lines: List[str] = []
    if name.endswith(".txt"):
        text = content.decode(errors="ignore")
        text_lines = [line.strip() for line in text.splitlines() if line.strip()]
    elif name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
        # heuristics: use first text-like column
        text_col = None
        for col in df.columns:
            if df[col].dtype == object:
                text_col = col
                break
        if text_col:
            text_lines = [str(x).strip() for x in df[text_col].dropna().tolist() if str(x).strip()]
    elif name.endswith(".json"):
        data = json.loads(content.decode(errors="ignore"))
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # pick first string value
                    for val in item.values():
                        if isinstance(val, str):
                            text_lines.append(val.strip())
                            break
                elif isinstance(item, str):
                    text_lines.append(item.strip())
        elif isinstance(data, dict):
            for val in data.values():
                if isinstance(val, str):
                    text_lines.append(val.strip())
    else:
        # fallback treat as text
        text = content.decode(errors="ignore")
        text_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return text_lines


def normalize_lines(lines: List[str]) -> pd.DataFrame:
    """Detect speaker labels and normalize to dataframe."""
    records = []
    for line in lines:
        speaker = "Unknown"
        raw_text = line
        if line.lower().startswith("agent:"):
            speaker = "Agent"
            raw_text = line.split(":", 1)[1].strip()
        elif line.lower().startswith("customer:") or line.lower().startswith("user:"):
            speaker = "Customer"
            raw_text = line.split(":", 1)[1].strip()
        records.append({"speaker": speaker, "raw_text": raw_text})
    return pd.DataFrame(records)


# -----------------------------
# PII Redaction
# -----------------------------
PHONE_REGEX = re.compile(r"(?:\+?\d[\d\-\s]{6,}\d)")
BUSINESS_REGEX = re.compile(r"\b(?:inc|llc|ltd|corp|company|co\.|store)\b", re.IGNORECASE)
NAME_REGEX = re.compile(r"\b(Mr\.?|Ms\.?|Mrs\.?|Dr\.?|Prof\.?)\s+[A-Z][a-z]+|\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b")


def redact_text(text: str) -> str:
    """Heuristic redaction: remove names, business names, phone numbers; keep locations/products."""
    redacted = PHONE_REGEX.sub("[REDACTED_PHONE]", text)
    redacted = BUSINESS_REGEX.sub("[REDACTED_BUSINESS]", redacted)
    redacted = NAME_REGEX.sub("[REDACTED_NAME]", redacted)
    return redacted


def redact_pii(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cleaned_text"] = df["raw_text"].apply(redact_text)
    return df


# -----------------------------
# LLM FAQ extraction
# -----------------------------
FAQ_SYSTEM_PROMPT = """Extract possible FAQs from this conversation.
Identify question intent even when no '?' is present.
Suggest FAQ only if agent response contains useful info.
Return structured JSON list:
[
  { "question": "...", "answer": "...", "confidence": 0-1 }
]
Score must reflect answer completeness.
0.8+ = strong, 0.5–0.79 = medium, <0.5 = weak.
"""


def extract_faq_via_llm(cleaned_transcript: str, model: str, api_key: str) -> List[Dict[str, Any]]:
    if not api_key:
        raise ValueError("OpenAI API key is required for FAQ extraction.")
    client = OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": FAQ_SYSTEM_PROMPT},
        {"role": "user", "content": cleaned_transcript},
    ]
    try:
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2, max_tokens=800)
        content = resp.choices[0].message.content
    except OpenAIError as exc:
        raise RuntimeError(f"OpenAI API error: {exc}") from exc

    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    # fallback: no JSON parsed
    return []


def color_by_confidence(conf: float) -> str:
    if conf >= 0.75:
        return "green"
    if conf >= 0.5:
        return "orange"
    return "red"


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Conversation → FAQ Extractor", layout="wide")
st.title("Conversation → FAQ Extractor")
st.caption("Upload a transcript, see normalization, PII redaction, and FAQ suggestions step by step.")

st.sidebar.header("Configuration")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Used for FAQ extraction.")
model = st.sidebar.text_input("Model", value="gpt-4.1-mini", help="Override if needed.")

uploaded = st.sidebar.file_uploader("Upload transcript (.txt, .csv, .json)", type=["txt", "csv", "json"])
manual_text = st.sidebar.text_area("Or paste raw transcript text")

if uploaded:
    raw_lines = parse_transcript(uploaded.read(), uploaded.name)
elif manual_text.strip():
    raw_lines = [line.strip() for line in manual_text.splitlines() if line.strip()]
else:
    raw_lines = []

if not raw_lines:
    st.info("Upload a transcript file or paste text to begin.")
    st.stop()

# Pipeline steps
normalized_df = normalize_lines(raw_lines)
redacted_df = redact_pii(normalized_df)

cleaned_transcript = "\n".join([f"{row.speaker}: {row.cleaned_text}" for row in redacted_df.itertuples()])

faq_results: List[Dict[str, Any]] = []
faq_error: Optional[str] = None
if st.sidebar.button("Run FAQ Extraction", type="primary"):
    with st.spinner("Extracting FAQs via OpenAI..."):
        try:
            faq_results = extract_faq_via_llm(cleaned_transcript, model, openai_api_key)
            if not faq_results:
                faq_error = "No FAQs returned or unable to parse JSON."
        except Exception as exc:  # pylint: disable=broad-except
            faq_error = str(exc)

# Layout
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Step 1: Raw Transcript")
    st.write("\n".join(raw_lines))

    st.subheader("Step 2: PII Redaction")
    st.dataframe(redacted_df[["speaker", "raw_text", "cleaned_text"]], use_container_width=True, hide_index=True)

with right_col:
    st.subheader("Step 1b: Normalized View")
    st.dataframe(normalized_df, use_container_width=True, hide_index=True)

    st.subheader("Step 3: FAQ Extraction")
    if faq_error:
        st.error(faq_error)
    elif faq_results:
        for item in faq_results:
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            conf = float(item.get("confidence", 0) or 0)
            color = color_by_confidence(conf)
            with st.expander(f"FAQ (confidence {conf:.2f})", expanded=False):
                st.markdown(f"**Question:** {q}")
                st.markdown(f"**Answer:** {a}")
                st.markdown(f"**Confidence:** <span style='color:{color}; font-weight:bold'>{conf:.2f}</span>", unsafe_allow_html=True)
    else:
        st.info("Run FAQ extraction from the sidebar to see results.")

# Download
if faq_results:
    faq_df = pd.DataFrame(faq_results)
    csv_buf = io.StringIO()
    faq_df.to_csv(csv_buf, index=False)
    st.download_button("Download FAQ CSV", data=csv_buf.getvalue(), file_name="faqs.csv", mime="text/csv")
