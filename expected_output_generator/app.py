"""
Expected Output Generator

Streamlit app to generate and refine `expected_output` values for rows in an uploaded
CSV/Excel file using OpenAI models. Upload a sheet with an `input` column, configure
model + prompts, generate row by row or fill empty rows, and export updated data.
"""

from __future__ import annotations

import io
import json
import os
from typing import Any, Dict, Literal, Optional, TypedDict

import pandas as pd
import streamlit as st
from openai import OpenAI


# -----------------------
# Session state helpers
# -----------------------
def init_state() -> None:
    if "df" not in st.session_state:
        st.session_state.df: Optional[pd.DataFrame] = None
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "api_key" not in st.session_state:
        st.session_state.api_key = os.environ.get("OPENAI_API_KEY", "")
    if "model" not in st.session_state:
        st.session_state.model = "gpt-4.1-mini"
    if "base_prompt" not in st.session_state:
        st.session_state.base_prompt = ""
    if "schema_instructions" not in st.session_state:
        st.session_state.schema_instructions = ""
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.2
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 1024


# -----------------------
# Types
# -----------------------
class InterfaceOption(TypedDict):
    id: str
    title: str
    description: Optional[str]


class InterfaceMedia(TypedDict):
    kind: Literal["image", "document", "video", "audio"]
    url: str
    caption: Optional[str]


class InterfaceUISpec(TypedDict):
    body: str
    options: list[InterfaceOption]
    media: Optional[InterfaceMedia]


def reset_for_new_file(df: pd.DataFrame) -> None:
    st.session_state.df = df
    st.session_state.current_index = 0


# -----------------------
# Mapping helpers
# -----------------------
def build_whatsapp_sequence(spec: InterfaceUISpec) -> Dict[str, Any]:
    """
    Map InterfaceUISpec -> WhatsAppMessageSequence (single message).
    Titles are truncated to schema limits.
    """
    body = (spec.get("body") or "").strip()
    options = spec.get("options") or []
    media = spec.get("media")
    n = len(options)
    has_media = media is not None

    def truncate(text: str, limit: int) -> str:
        return (text or "")[:limit]

    # Case 1: plain text
    if n == 0 and not has_media:
        return {"messages": [{"type": "text", "content": {"body": body}}]}

    # Case 2: pure media
    if n == 0 and has_media:
        return {
            "messages": [
                {
                    "type": "media",
                    "content": {
                        "media_link": media["url"],
                        "caption": media.get("caption") or body,
                    },
                }
            ]
        }

    # Case 3: 1-3 options, no media -> interactive_button
    if 1 <= n <= 3 and not has_media:
        return {
            "messages": [
                {
                    "type": "interactive_button",
                    "content": {
                        "header": None,
                        "body": body,
                        "footer": None,
                        "buttons": [{"title": truncate(opt.get("title", ""), 20) or "Option"} for opt in options],
                    },
                }
            ]
        }

    # Case 4: 1-3 options + media -> interactive_button with media header
    if 1 <= n <= 3 and has_media:
        return {
            "messages": [
                {
                    "type": "interactive_button",
                    "content": {
                        "header": {"text": None, "media_link": media["url"]},
                        "body": body,
                        "footer": None,
                        "buttons": [{"title": truncate(opt.get("title", ""), 20) or "Option"} for opt in options],
                    },
                }
            ]
        }

    # Case 5: 4-10 options -> interactive_list (single section)
    if 4 <= n <= 10:
        return {
            "messages": [
                {
                    "type": "interactive_list",
                    "content": {
                        "header": None,
                        "button_text": "Choose an option",
                        "body": body,
                        "footer": None,
                        "sections": [
                            {
                                "title": "Options",
                                "rows": [
                                    {
                                        "title": truncate(opt.get("title", ""), 24) or "Option",
                                        "description": truncate(opt.get("description") or "", 72) or None,
                                    }
                                    for opt in options
                                ],
                            }
                        ],
                    },
                }
            ]
        }

    # Fallback: too many options or weird data -> text
    return {"messages": [{"type": "text", "content": {"body": body}}]}


# -----------------------
# Data loading
# -----------------------
def load_uploaded_file(uploaded_file: Any) -> Optional[pd.DataFrame]:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Failed to read file: {exc}")
        return None

    if "input" not in df.columns:
        st.error("The uploaded file must contain an 'input' column.")
        return None

    if "expected_output" not in df.columns:
        df["expected_output"] = ""
    if "feedback" not in df.columns:
        df["feedback"] = ""

    return df


# -----------------------
# Sidebar configuration
# -----------------------
def render_config_sidebar() -> Dict[str, Any]:
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input(
        "OPENAI_API_KEY",
        value=st.session_state.api_key or "",
        type="password",
        placeholder="sk-...",
        help="Used to call OpenAI for generation.",
    )
    model = st.sidebar.selectbox(
        "Model",
        options=["gpt-4.1-mini", "gpt-5-nano", "gpt-5-mini", "gpt-5.1"],
        index=["gpt-4.1-mini", "gpt-5-nano", "gpt-5-mini", "gpt-5.1"].index(st.session_state.model)
        if st.session_state.model in ["gpt-4.1-mini", "gpt-5-nano", "gpt-5-mini", "gpt-5.1"]
        else 0,
    )
    base_prompt = st.sidebar.text_area(
        "Base Prompt (system-level instructions)",
        value=st.session_state.base_prompt,
        height=180,
    )
    schema_instructions = st.sidebar.text_area(
        "JSON Format / Schema Instructions",
        value=st.session_state.schema_instructions,
        height=140,
    )
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.05)
    max_tokens = st.sidebar.number_input("Max tokens", min_value=64, max_value=4096, value=st.session_state.max_tokens)

    # Persist values in session for later runs.
    st.session_state.api_key = api_key.strip()
    st.session_state.model = model
    st.session_state.base_prompt = base_prompt
    st.session_state.schema_instructions = schema_instructions
    st.session_state.temperature = temperature
    st.session_state.max_tokens = int(max_tokens)

    return {
        "api_key": st.session_state.api_key,
        "model": model,
        "base_prompt": base_prompt,
        "schema_instructions": schema_instructions,
        "temperature": temperature,
        "max_tokens": int(max_tokens),
    }


def has_required_config(cfg: Dict[str, Any]) -> bool:
    return bool(
        cfg["api_key"].strip()
        and cfg["model"]
        and cfg["base_prompt"].strip()
        and cfg["schema_instructions"].strip()
    )


# -----------------------
# Generation helpers
# -----------------------
def build_user_message(row: pd.Series, cfg: Dict[str, Any]) -> str:
    input_raw = str(row.get("input", "")).strip()
    try:
        input_obj = json.loads(input_raw)
        input_text = input_obj.get("content", "")
    except Exception:
        input_text = input_raw

    return f"""
You are an interface planner. Given a WhatsApp-style message, produce an InterfaceUISpec JSON with:
- body: concise body text
- options: list of {{id, title, description?}} (id can be slug/placeholder)
- media: optional {{kind, url, caption}}

Message:
{input_text}

Return ONLY JSON in this shape:
{{
  "body": "...",
  "options": [{{"id": "opt_1", "title": "...", "description": "..." }}],
  "media": null OR {{"kind": "image|document|video|audio", "url": "...", "caption": "..."}}
}}
"""


def call_openai_spec(row: pd.Series, cfg: Dict[str, Any]) -> InterfaceUISpec:
    client = OpenAI(api_key=cfg["api_key"])
    user_content = build_user_message(row, cfg)
    completion = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": "Generate InterfaceUISpec JSON only."},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
    )
    content = completion.choices[0].message.content or "{}"
    spec: InterfaceUISpec = json.loads(content)
    return spec


def generate_for_index(idx: int, cfg: Dict[str, Any]) -> None:
    df = st.session_state.df
    if df is None:
        st.error("No dataset loaded.")
        return
    try:
        row = df.iloc[idx]
    except IndexError:
        st.error("Row index out of range.")
        return
    try:
        spec = call_openai_spec(row, cfg)
    except Exception as exc:  # pylint: disable=broad-except
        st.session_state.df.at[df.index[idx], "expected_output"] = f"ERROR generating spec: {exc}"
        return

    sequence = build_whatsapp_sequence(spec)
    st.session_state.df.at[df.index[idx], "expected_output"] = json.dumps(sequence, ensure_ascii=False)


# -----------------------
# UI rendering
# -----------------------
def render_upload_section() -> None:
    st.header("Upload dataset")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if uploaded:
        df = load_uploaded_file(uploaded)
        if df is not None:
            reset_for_new_file(df)


def render_basic_info(df: pd.DataFrame) -> None:
    st.subheader("Dataset info")
    st.write(f"Rows: {len(df)}")
    st.write(f"Columns: {list(df.columns)}")
    preview_cols = [col for col in ["input", "expected_output", "feedback"] if col in df.columns]
    st.dataframe(df[preview_cols].head(), use_container_width=True)


def render_progress(df: pd.DataFrame) -> None:
    filled_mask = df["expected_output"].astype(str).str.strip() != ""
    filled = int(filled_mask.sum())
    total = len(df)
    st.markdown(f"**Row {st.session_state.current_index + 1} of {total}**")
    st.progress(filled / total if total else 0)
    st.caption(f"{filled} of {total} rows have expected_output filled.")


def render_navigation(df: pd.DataFrame) -> None:
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Previous", disabled=st.session_state.current_index <= 0):
            st.session_state.current_index = max(0, st.session_state.current_index - 1)
    with col2:
        if st.button("Next", disabled=st.session_state.current_index >= len(df) - 1):
            st.session_state.current_index = min(len(df) - 1, st.session_state.current_index + 1)
    with col3:
        target = st.number_input(
            "Jump to row (0-indexed)",
            min_value=0,
            max_value=max(len(df) - 1, 0),
            value=st.session_state.current_index,
            step=1,
        )
        if st.button("Go"):
            st.session_state.current_index = int(target)


def render_row_detail(df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    idx = st.session_state.current_index
    if idx < 0 or idx >= len(df):
        st.info("Select a valid row to begin.")
        return

    row = df.iloc[idx]
    st.subheader("Row detail")
    st.text_area("Input", value=str(row["input"]), height=140, disabled=True)

    expected_key = f"expected_output_{idx}"
    feedback_key = f"feedback_{idx}"
    st.session_state[expected_key] = row.get("expected_output", "")
    st.session_state[feedback_key] = row.get("feedback", "")

    expected_val = st.text_area("Existing expected_output (editable)", key=expected_key, height=160)
    feedback_val = st.text_area("Feedback (optional, editable)", key=feedback_key, height=120)

    # Sync back to DataFrame.
    st.session_state.df.at[df.index[idx], "expected_output"] = expected_val
    st.session_state.df.at[df.index[idx], "feedback"] = feedback_val

    disabled = not has_required_config(cfg)
    if st.button("Generate Expected Output for This Row", type="primary", disabled=disabled):
        if not has_required_config(cfg):
            st.warning("Set model, base prompt, and JSON format instructions first.")
        else:
            with st.spinner("Calling OpenAI..."):
                try:
                    generate_for_index(idx, cfg)
                    st.success("expected_output updated.")
                except Exception as exc:  # pylint: disable=broad-except
                    st.error(f"Generation failed: {exc}")


def render_bulk_generate(df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    st.subheader("Bulk assist")
    disabled = not has_required_config(cfg)
    if st.button("Generate for All Empty Rows (Sequential)", disabled=disabled):
        if not has_required_config(cfg):
            st.warning("Set model, base prompt, and JSON format instructions first.")
            return
        total = len(df)
        if total == 0:
            st.info("No rows to process.")
            return
        progress = st.progress(0.0)
        updated = 0
        for i in range(total):
            if str(df.iloc[i]["expected_output"]).strip():
                progress.progress((i + 1) / total)
                continue
            try:
                generate_for_index(i, cfg)
                updated += 1
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Row {i} failed: {exc}")
                break
            progress.progress((i + 1) / total)
        st.success(f"Bulk generation complete. Updated {updated} rows.")


def render_table(df: pd.DataFrame) -> None:
    st.subheader("Table view")
    table_df = df[["input", "expected_output", "feedback"]].copy()
    table_df.insert(0, "index", df.index)
    st.dataframe(table_df, use_container_width=True, hide_index=True)


def render_downloads(df: pd.DataFrame) -> None:
    st.subheader("Download updated sheet")
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_data, file_name="updated_expected_outputs.csv", mime="text/csv")

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="expected_outputs")
    st.download_button(
        "Download Excel (.xlsx)",
        data=buffer.getvalue(),
        file_name="updated_expected_outputs.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# -----------------------
# Main app
# -----------------------
def main() -> None:
    st.set_page_config(page_title="Expected Output Generator", layout="wide")
    init_state()
    cfg = render_config_sidebar()

    st.title("Expected Output Generator")
    st.caption("Upload a dataset with an 'input' column, generate and refine 'expected_output' row by row.")

    render_upload_section()

    df = st.session_state.df
    if df is None:
        st.info("Upload a CSV or Excel file to get started.")
        return

    render_basic_info(df)
    render_progress(df)
    render_navigation(df)
    render_row_detail(df, cfg)
    render_table(df)
    render_bulk_generate(df, cfg)
    render_downloads(df)


if __name__ == "__main__":
    main()
