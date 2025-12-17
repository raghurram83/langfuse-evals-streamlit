"""
Two-step WhatsApp interface evaluator.

Uploads a dataset (input, expected_output), runs an agent to produce an intermediate
InterfaceUISpec, then maps it via rule-based middleware to final WhatsApp payload.
Displays input, expected_output, agent_output, and actual_output side by side.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from openai import OpenAI

# -------- Defaults --------
DEFAULT_PROMPT = """You are an Interface Message Extractor for WhatsApp.
Your only job is to read ONE natural-language message and output a JSON interface spec with three fields: `body`, `options`, and `media`.
Do NOT output WhatsApp schema. Do NOT decide message type (buttons vs list vs text vs media). Only extract:

- `body`: main user-facing sentence (clean text, no numbering).
- `options`: array of actions. Each with `id` (UPPERCASE_SNAKE_CASE from the title), `title` (short label), `description` (or null).
- `media`: either null, or an object with `kind` (image/document/video/audio), `url` (media URL), and optional `caption`.

Return ONLY a JSON object matching the provided JSON schema."""

DEFAULT_SCHEMA = """{
  "type": "object",
  "title": "InterfaceUISpec",
  "properties": {
    "body": {
      "type": "string",
      "minLength": 1,
      "maxLength": 1024,
      "description": "Main message shown to the user. Clean text without numbering like 1., 2., -."
    },
    "options": {
      "type": "array",
      "description": "User-selectable choices inferred from the message.",
      "minItems": 0,
      "maxItems": 10,
      "items": {
        "type": "object",
        "properties": {
          "id": {
            "type": "string",
            "description": "Stable identifier, e.g. CALL_AGENT, VIEW_MENU"
          },
          "title": {
            "type": "string",
            "minLength": 1,
            "maxLength": 24,
            "description": "Short label shown on button or list row."
          },
          "description": {
            "type": ["string", "null"],
            "maxLength": 72,
            "description": "Optional secondary text for list rows."
          }
        },
        "required": ["id", "title"]
      }
    },
    "media": {
      "type": ["object", "null"],
      "description": "Optional media referenced in the message.",
      "properties": {
        "kind": {
          "type": "string",
          "enum": ["image", "document", "video", "audio"],
          "description": "Basic media type based on file extension."
        },
        "url": {
          "type": "string",
          "minLength": 1,
          "description": "Direct media URL (.jpg, .png, .pdf, .mp4, etc.)."
        },
        "caption": {
          "type": ["string", "null"],
          "maxLength": 1024,
          "description": "Optional caption or label for this media."
        }
      },
      "required": ["kind", "url"]
    }
  },
  "required": ["body", "options"]
}"""


# -------- State helpers --------
def init_state() -> None:
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame(columns=["input", "expected_output", "agent_output", "actual_output"])
    st.session_state.setdefault("api_key", os.environ.get("OPENAI_API_KEY", ""))
    st.session_state.setdefault("base_url", "")
    st.session_state.setdefault("model", "gpt-4.1-mini")
    st.session_state.setdefault("agent_prompt", DEFAULT_PROMPT)
    st.session_state.setdefault("agent_schema", DEFAULT_SCHEMA)
    st.session_state.setdefault("stop_on_error", False)


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["input", "expected_output", "agent_output", "actual_output"]:
        if col not in df.columns:
            df[col] = ""
    return df[["input", "expected_output", "agent_output", "actual_output"]]


# -------- Core logic --------
def build_whatsapp_sequence(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Map InterfaceUISpec -> WhatsAppMessageSequence (single message)."""
    body = (spec.get("body") or "").strip()
    options = spec.get("options") or []
    media = spec.get("media")
    n = len(options)
    has_media = media is not None

    # 1) Plain text
    if n == 0 and not has_media:
        return {"messages": [{"type": "text", "content": {"body": body}}]}

    # 2) Pure media
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

    # 3) 1–3 options, no media → interactive_button
    if 1 <= n <= 3 and not has_media:
        return {
            "messages": [
                {
                    "type": "interactive_button",
                    "content": {
                        "header": None,
                        "body": body,
                        "footer": None,
                        "buttons": [{"title": (o.get("title") or "")[:20]} for o in options],
                    },
                }
            ]
        }

    # 4) 1–3 options + media → interactive_button with header media
    if 1 <= n <= 3 and has_media:
        return {
            "messages": [
                {
                    "type": "interactive_button",
                    "content": {
                        "header": {"text": None, "media_link": media["url"]},
                        "body": body,
                        "footer": None,
                        "buttons": [{"title": (o.get("title") or "")[:20]} for o in options],
                    },
                }
            ]
        }

    # 5) 4–10 options → interactive_list
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
                                        "title": (o.get("title") or "")[:24],
                                        "description": o.get("description"),
                                    }
                                    for o in options
                                ],
                            }
                        ],
                    },
                }
            ]
        }

    # Fallback
    return {"messages": [{"type": "text", "content": {"body": body}}]}


def call_agent(text: str, prompt: str, schema: str, model: str, api_key: str, base_url: str) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key, base_url=base_url or None)
    parsed_schema = json.loads(schema)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": parsed_schema.get("title", "InterfaceUISpec"),
                "schema": parsed_schema,
            },
        },
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)


def run_agent_on_df(df: pd.DataFrame, stop_on_error: bool, cfg: Dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    progress = st.progress(0.0, text="Running agent...")
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        text = str(row.get("input", ""))
        try:
            spec = call_agent(text, cfg["prompt"], cfg["schema"], cfg["model"], cfg["api_key"], cfg["base_url"])
            df.at[row.name, "agent_output"] = json.dumps(spec, ensure_ascii=False)
            sequence = build_whatsapp_sequence(spec)
            df.at[row.name, "actual_output"] = json.dumps(sequence, ensure_ascii=False)
        except Exception as exc:  # pylint: disable=broad-except
            err = f"ERROR: {exc}"
            df.at[row.name, "agent_output"] = err
            df.at[row.name, "actual_output"] = ""
            if stop_on_error:
                progress.progress(i / total, text=f"Stopped at row {i} due to error.")
                st.error(err)
                break
        progress.progress(i / total, text=f"Processed {i}/{total}")
    progress.progress(1.0, text="Done.")
    return df


def run_middleware_only(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for idx, row in df.iterrows():
        raw_spec = row.get("agent_output", "")
        try:
            spec = json.loads(raw_spec)
            sequence = build_whatsapp_sequence(spec)
            df.at[idx, "actual_output"] = json.dumps(sequence, ensure_ascii=False)
        except Exception as exc:  # pylint: disable=broad-except
            df.at[idx, "actual_output"] = f"ERROR: {exc}"
    return df


# -------- UI helpers --------
def load_sample_df() -> pd.DataFrame:
    data = [
        {"input": '{"content": "Hi, I need help with my order. Options: Track order, Cancel, Talk to agent."}', "expected_output": ""},
        {"input": '{"content": "Please find the brochure attached: https://example.com/file.pdf"}', "expected_output": ""},
        {"input": '{"content": "Choose a plan: Basic, Standard, Premium"}', "expected_output": ""},
    ]
    return ensure_columns(pd.DataFrame(data))


def load_uploaded_file(uploaded) -> Optional[pd.DataFrame]:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            data = json.load(uploaded)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                st.error("JSON must be a list of objects.")
                return None
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Failed to load file: {exc}")
        return None

    if "input" not in df.columns:
        st.error("Dataset must include 'input' column.")
        return None
    if "expected_output" not in df.columns:
        st.warning("expected_output missing; initializing empty.")
        df["expected_output"] = ""
    return ensure_columns(df)


def truncate(text: str, limit: int = 120) -> str:
    t = str(text)
    return t if len(t) <= limit else t[: limit - 3] + "..."


def render_sidebar() -> Dict[str, Any]:
    st.sidebar.header("OpenAI config")
    api_key = st.sidebar.text_input("OpenAI API key", value=st.session_state.api_key, type="password")
    base_url = st.sidebar.text_input("Base URL (optional)", value=st.session_state.base_url)
    model_options = ["gpt-4.1-mini", "gpt-4.1", "gpt-5.1-mini", "gpt-5.1", "custom"]
    model_choice = st.sidebar.selectbox("Model", options=model_options, index=model_options.index(st.session_state.model) if st.session_state.model in model_options else 0)
    custom_model = ""
    if model_choice == "custom":
        custom_model = st.sidebar.text_input("Custom model name", value=st.session_state.model if st.session_state.model not in model_options else "")
    model = custom_model or model_choice

    st.sidebar.header("Agent config")
    prompt = st.sidebar.text_area("Agent system prompt", value=st.session_state.agent_prompt, height=200)
    schema_text = st.sidebar.text_area("Agent JSON schema", value=st.session_state.agent_schema, height=220)

    st.sidebar.header("Dataset controls")
    uploaded = st.sidebar.file_uploader("Upload CSV or JSON", type=["csv", "json"])
    load_sample = st.sidebar.button("Load sample dataset")

    st.sidebar.header("Run controls")
    stop_on_error = st.sidebar.checkbox("Stop on first error", value=st.session_state.stop_on_error)
    run_agent = st.sidebar.button("Run agent on all rows")
    run_mw = st.sidebar.button("Run middleware only (rebuild actual_output)")

    # Persist sidebar values
    st.session_state.api_key = api_key
    st.session_state.base_url = base_url
    st.session_state.model = model
    st.session_state.agent_prompt = prompt
    st.session_state.agent_schema = schema_text
    st.session_state.stop_on_error = stop_on_error

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "prompt": prompt,
        "schema": schema_text,
        "uploaded": uploaded,
        "load_sample": load_sample,
        "run_agent": run_agent,
        "run_mw": run_mw,
        "stop_on_error": stop_on_error,
    }


def render_table(df: pd.DataFrame) -> None:
    display_df = df.copy()
    display_df["agent_output"] = display_df["agent_output"].apply(truncate)
    display_df["actual_output"] = display_df["actual_output"].apply(truncate)
    st.dataframe(display_df, use_container_width=True)


def render_row_inspector(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No data loaded.")
        return
    idx = st.number_input("Row index", min_value=0, max_value=len(df) - 1, value=0, step=1)
    row = df.iloc[int(idx)]
    st.caption("input")
    st.code(str(row["input"]))
    st.caption("expected_output")
    try:
        st.code(json.dumps(json.loads(row["expected_output"]), indent=2, ensure_ascii=False))
    except Exception:
        st.code(str(row["expected_output"]))
    st.caption("agent_output")
    try:
        st.code(json.dumps(json.loads(row["agent_output"]), indent=2, ensure_ascii=False))
    except Exception:
        st.code(str(row["agent_output"]))
    st.caption("actual_output")
    try:
        st.code(json.dumps(json.loads(row["actual_output"]), indent=2, ensure_ascii=False))
    except Exception:
        st.code(str(row["actual_output"]))


def render_export(df: pd.DataFrame) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    json_bytes = df.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8")
    st.download_button("Download dataset as CSV", data=csv_bytes, file_name="dataset.csv", mime="text/csv")
    st.download_button("Download dataset as JSON", data=json_bytes, file_name="dataset.json", mime="application/json")


# -------- Main --------
def main() -> None:
    st.set_page_config(page_title="2-step WhatsApp Interface Evaluator", layout="wide")
    init_state()
    cfg = render_sidebar()

    # Dataset loading
    if cfg["load_sample"]:
        st.session_state.df = load_sample_df()
    if cfg["uploaded"] is not None:
        loaded = load_uploaded_file(cfg["uploaded"])
        if loaded is not None:
            st.session_state.df = loaded

    df = ensure_columns(st.session_state.df)

    # Run actions
    if cfg["run_agent"]:
        if not cfg["api_key"]:
            st.error("API key required.")
        elif not cfg["model"] or not cfg["prompt"] or not cfg["schema"]:
            st.error("Model, prompt, and schema are required.")
        else:
            with st.spinner("Running agent..."):
                st.session_state.df = run_agent_on_df(df, cfg["stop_on_error"], cfg)
            st.success("Agent run complete.")

    if cfg["run_mw"]:
        with st.spinner("Rebuilding actual_output from existing agent_output..."):
            st.session_state.df = run_middleware_only(df)
        st.success("Middleware run complete.")

    df = st.session_state.df

    # Tabs
    tab_table, tab_inspect, tab_export = st.tabs(["Table", "Row Inspector", "Export"])
    with tab_table:
        render_table(df)
    with tab_inspect:
        render_row_inspector(df)
    with tab_export:
        render_export(df)


if __name__ == "__main__":
    main()
