"""
Evaluation Runner: Streamlit app for configuring an agent, uploading a dataset,
running the agent to generate outputs, and scoring with an LLM-as-judge.

All state is in-memory via st.session_state; no persistence beyond the session.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import sys

import numpy as np
import pandas as pd
import streamlit as st
import requests

try:
    from openai import OpenAI

    client: Optional[OpenAI] = OpenAI()
except Exception:  # pragma: no cover - defensive; we keep app usable without SDK
    client = None

# Disk persistence to survive full page refresh within the same machine/session.
PERSIST_PATH = Path(__file__).parent / ".session_cache.json"
AGENT_API_URL = "https://n8n.myoperator.biz/webhook/df959123-eb69-43da-baa6-de4884c717a7"

# Ensure project root on sys.path for package imports when run via Streamlit.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ----- Session state helpers -----
def load_persisted_state() -> Dict[str, Any]:
    """Load cached state from disk if present."""
    if not PERSIST_PATH.exists():
        return {}
    try:
        with PERSIST_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def persist_state() -> None:
    """Persist key parts of session state to disk."""
    try:
        data = {
            "agent_config": st.session_state.get("agent_config"),
            "judge_config": st.session_state.get("judge_config"),
            "dataset": st.session_state.get("dataset", pd.DataFrame()).replace({np.nan: None}).to_dict(orient="records"),
            "test_runs": st.session_state.get("test_runs", []),
            "actual_output_backup": st.session_state.get("actual_output_backup", pd.Series(dtype=object))
            .replace({np.nan: None})
            .tolist(),
        }
        with PERSIST_PATH.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        # Persistence is best-effort; ignore errors to avoid breaking UI.
        pass


def init_state() -> None:
    """Ensure all expected session state keys exist."""
    persisted = load_persisted_state()
    st.session_state.setdefault(
        "agent_config",
        persisted.get("agent_config")
        or {"prompt": "", "json_schema": "", "model": "gpt-4.1-mini"},
    )
    st.session_state.setdefault(
        "judge_config",
        persisted.get("judge_config")
        or {"prompt": "", "model": "gpt-4.1-mini"},
    )
    persisted_dataset = pd.DataFrame(persisted.get("dataset", []))
    st.session_state.setdefault(
        "dataset",
        update_type_comparisons(
            persisted_dataset
            if not persisted_dataset.empty
            else pd.DataFrame(
                columns=[
                    "input",
                    "expected_output",
                    "actual_output",
                    "expected_type",
                    "actual_type",
                    "type_match",
                    "generation_info",
                    "judge_score",
                    "judge_reasoning",
                ]
            )
        ),
    )
    st.session_state.setdefault("test_runs", persisted.get("test_runs", []))
    backup_series = pd.Series(persisted.get("actual_output_backup", []))
    st.session_state.setdefault("actual_output_backup", backup_series if not backup_series.empty else pd.Series(dtype=object))
    # Agent run queue state for progressive per-row updates.
    st.session_state.setdefault("agent_queue", [])
    st.session_state.setdefault("agent_run_outputs", [])
    st.session_state.setdefault("agent_run_row_indices", [])
    st.session_state.setdefault("agent_run_meta", {})
    st.session_state.setdefault("agent_processing", False)
    st.session_state.setdefault("agent_stop_requested", False)


def ensure_dataset_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing required columns with default values."""
    required_defaults = {
        "input": "",
        "expected_output": "",
        "actual_output": "",
        "expected_type": "",
        "actual_type": "",
        "type_match": np.nan,
        "generation_info": "",
        "judge_score": np.nan,
        "judge_reasoning": "",
    }
    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default
    # Keep judge_score numeric where possible.
    df["judge_score"] = pd.to_numeric(df["judge_score"], errors="coerce")
    # Reorder columns for consistency
    return df[list(required_defaults.keys())]


def restore_actual_outputs_if_missing() -> None:
    """
    Defensively restore actual_output from backup if the column is unexpectedly cleared.
    This prevents accidental loss during reruns.
    """
    backup = st.session_state.get("actual_output_backup")
    df = st.session_state.get("dataset")
    if backup is None or df is None or df.empty:
        return
    try:
        current = df["actual_output"].fillna("").astype(str).reset_index(drop=True)
        backup_vals = pd.Series(backup).fillna("").astype(str).reset_index(drop=True)
        if len(backup_vals) == len(current):
            filled = current.where(current != "", backup_vals)
            df["actual_output"] = filled.values  # align by position
            st.session_state.dataset = df
    except Exception:
        # Best-effort only; do not raise.
        pass


# ----- Type helpers -----
def extract_primary_message_type(payload: Any) -> str:
    """
    Pull out the first message type from a nested payload.
    Supports both legacy {"messages": [...]} and WhatsAppUnifiedMessageSequence structures.
    """
    def _coerce(obj: Any) -> Any:
        if isinstance(obj, str):
            try:
                return json.loads(obj)
            except Exception:
                return obj
        return obj

    def _find_messages(obj: Any) -> Optional[List[Any]]:
        obj = _coerce(obj)
        if isinstance(obj, dict):
            if isinstance(obj.get("messages"), list):
                return obj["messages"]
            if "WhatsAppUnifiedMessageSequence" in obj:
                seq = obj.get("WhatsAppUnifiedMessageSequence", {})
                if isinstance(seq, dict) and isinstance(seq.get("messages"), list):
                    return seq["messages"]
            if "output" in obj:
                msgs = _find_messages(obj["output"])
                if msgs:
                    return msgs
            for val in obj.values():
                if isinstance(val, (dict, list)):
                    msgs = _find_messages(val)
                    if msgs:
                        return msgs
        if isinstance(obj, list):
            for item in obj:
                msgs = _find_messages(item)
                if msgs:
                    return msgs
        return None

    messages = _find_messages(payload)
    if not messages:
        return ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        msg_type = msg.get("message_type") or msg.get("type")
        if isinstance(msg_type, str) and msg_type.strip():
            return msg_type.strip()
    return ""


def update_type_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """Compute expected/actual types and a type_match flag."""
    df = ensure_dataset_columns(df.copy())
    df["expected_type"] = df["expected_output"].apply(extract_primary_message_type)
    df["actual_type"] = df["actual_output"].apply(extract_primary_message_type)

    matches: List[Any] = []
    for exp, act in zip(df["expected_type"], df["actual_type"]):
        if not exp or not act:
            matches.append(np.nan)
        else:
            matches.append(bool(exp == act))
    df["type_match"] = matches
    return df


# ----- OpenAI helpers -----
def hash_prompt(prompt: str) -> str:
    """Short hash for display in run metadata."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8] if prompt else "none"


def call_openai_model(model: str, system_prompt: str, user_content: str) -> str:
    """
    Invoke the OpenAI SDK, preferring the newer Responses API when available.
    Falls back to chat.completions for broader compatibility.
    """
    if client is None:
        raise RuntimeError("OpenAI client not initialized. Check your installation and API key.")

    # Try the Responses API (recommended for 4.1/5.1 family).
    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        output = response.output_text or ""
        if output:
            return output
    except Exception as exc:
        # Surface helpful message for model-not-found while still allowing fallback.
        if "model_not_found" in str(exc):
            raise RuntimeError(f"Model '{model}' is not available to this API key or region.") from exc
        # Fall back to chat completions for environments without Responses.
        pass

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return completion.choices[0].message["content"]  # type: ignore[index]


# ----- Agent and judge runners -----
def select_agent_rows(df: pd.DataFrame, n_rows: int) -> List[int]:
    """Pick the first N rows missing actual_output."""
    mask = df["actual_output"].fillna("") == ""
    indices = list(df[mask].index[:n_rows])
    return indices


def run_agent_on_rows(
    df: pd.DataFrame, row_indices: Sequence[int], agent_config: Dict[str, Any]
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Run the agent sequentially over selected rows using the external webhook."""
    updated_df = df.copy()
    outputs: List[str] = []
    infos: List[str] = []

    def call_agent_api(user_message: str) -> Tuple[str, str]:
        payload = {
            "role": "assistant",
            "content": user_message,
            "additional_kwargs": {"refusal": None},
        }
        try:
            resp = requests.post(AGENT_API_URL, json=payload, timeout=30)
            resp.raise_for_status()
            try:
                parsed = resp.json()
            except Exception:
                parsed = resp.text
            result = json.dumps(parsed, ensure_ascii=False) if not isinstance(parsed, str) else parsed
            msg_type = extract_primary_message_type(parsed)
            info = f"API {resp.status_code}; type={msg_type or 'unknown'}"
        except Exception as exc:  # pylint: disable=broad-except
            result = f"ERROR: {exc}"
            info = f"ERROR calling agent API: {exc}"
        return result, info

    for idx in row_indices:
        user_content = str(updated_df.at[idx, "input"])
        try:
            result, info = call_agent_api(user_content)
        except Exception as exc:  # pylint: disable=broad-except
            result = f"ERROR: {exc}"
            info = f"ERROR calling agent API: {exc}"
        updated_df.at[idx, "actual_output"] = result
        updated_df.at[idx, "generation_info"] = info
        outputs.append(result)
        infos.append(info)
    return update_type_comparisons(updated_df), outputs, infos


def run_judge_on_rows(
    df: pd.DataFrame, row_indices: Sequence[int], judge_config: Dict[str, Any]
) -> pd.DataFrame:
    """Score rows with the LLM-as-judge, storing score and reasoning in place."""
    updated_df = df.copy()

    # JSON schema to strongly steer judge output.
    judge_schema = {
        "type": "object",
        "properties": {
            "score": {"type": "number"},
            "reason_short": {"type": "string"},
            "details": {
                "type": "object",
                "properties": {
                    "type_match": {"type": "boolean"},
                    "structure_match": {"type": "string"},
                    "missing_or_extra_elements": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["type_match", "structure_match", "missing_or_extra_elements", "notes"],
            },
        },
        "required": ["score", "reason_short", "details"],
        "additionalProperties": False,
    }

    def call_judge(system_prompt: str, user_prompt: str) -> str:
        """Prefer Responses API with json schema; fall back to generic call."""
        if client is not None:
            try:
                response = client.responses.create(
                    model=judge_config["model"],
                    input=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    response_format={"type": "json_schema", "json_schema": {"name": "judge", "schema": judge_schema}},
                )
                if response.output_text:
                    return response.output_text
            except Exception:
                # If schema forcing not available, fall back to generic call.
                pass
        return call_openai_model(judge_config["model"], system_prompt, user_prompt)

    for idx in row_indices:
        row = updated_df.loc[idx]
        user_prompt = (
            f"INPUT:\n{row['input']}\n\n"
            f"EXPECTED_OUTPUT:\n{row['expected_output']}\n\n"
            f"ACTUAL_OUTPUT:\n{row['actual_output']}\n\n"
            "Return a JSON object with fields: score (float in [0,1]) and reasoning (string)."
        )
        try:
            raw = call_judge(judge_config["prompt"], user_prompt)
            parsed = json.loads(raw)
            if isinstance(parsed, list) and parsed:
                parsed = parsed[0]
            score = float(parsed.get("score", parsed.get("value", 0.0)))
            reasoning = (
                parsed.get("reason_short")
                or parsed.get("reasoning")
                or parsed.get("reason")
                or parsed.get("explanation")
                or parsed.get("rationale")
                or parsed.get("analysis")
                or ""
            )
            reasoning = str(reasoning)
            # Also capture structured details for potential debugging.
            details = parsed.get("details") or {}
            if details and not reasoning:
                reasoning = json.dumps(details, ensure_ascii=False)
        except Exception as exc:  # pylint: disable=broad-except
            score = 0.0
            reasoning = f"ERROR: {exc} | Raw: {raw if 'raw' in locals() else ''}"
        updated_df.at[idx, "judge_score"] = score
        updated_df.at[idx, "judge_reasoning"] = reasoning
    return updated_df


# ----- Styling and reporting -----
def style_dataset_df(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Apply color coding based on type_match first, then judge_score."""
    def color_row(row: pd.Series) -> List[str]:
        type_match = row.get("type_match", np.nan)
        if isinstance(type_match, bool):
            color = "#e6f4ea" if type_match else "#fdecea"
            return [f"background-color: {color}"] * len(row)

        score = row.get("judge_score", np.nan)
        if pd.isna(score):
            return [""] * len(row)
        if score >= 0.8:
            color = "background-color: #d5f5e3"
        elif score >= 0.5:
            color = "background-color: #fcf3cf"
        else:
            color = "background-color: #f5b7b1"
        return [color] * len(row)

    return df.style.apply(color_row, axis=1)


def compute_report_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute aggregate metrics and bucket counts for scored rows."""
    df = df.copy()
    df["judge_score"] = pd.to_numeric(df["judge_score"], errors="coerce")
    scored = df.dropna(subset=["judge_score"])
    if scored.empty:
        return {}

    scores = scored["judge_score"].astype(float)
    green = (scores >= 0.8).sum()
    yellow = ((scores >= 0.5) & (scores < 0.8)).sum()
    red = (scores < 0.5).sum()
    exact_one = (scores == 1.0).sum()
    exact_zero = (scores == 0.0).sum()
    exact_other = len(scores) - exact_one - exact_zero
    return {
        "count": len(scored),
        "mean": float(scores.mean()),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "green": int(green),
        "yellow": int(yellow),
        "red": int(red),
        "exact_one": int(exact_one),
        "exact_zero": int(exact_zero),
        "exact_other": int(exact_other),
    }


def compute_type_mismatch_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Summarize type alignment between expected_output and actual_output."""
    if df.empty:
        return {}
    expected = df.get("expected_type", pd.Series(dtype=str)).fillna("").astype(str)
    actual = df.get("actual_type", pd.Series(dtype=str)).fillna("").astype(str)
    comparable = (expected != "") & (actual != "")
    if not comparable.any():
        return {}

    subset = df[comparable].copy()
    subset["expected_type"] = expected[comparable]
    subset["actual_type"] = actual[comparable]
    matches = subset["expected_type"] == subset["actual_type"]

    def pct_for_expected(msg_type: str) -> float:
        mask = subset["expected_type"] == msg_type
        if not mask.any():
            return 0.0
        mismatches = (~matches) & mask
        return float(mismatches.sum() / mask.sum() * 100.0)

    return {
        "rows_with_types": int(len(subset)),
        "match_rate_pct": float(matches.mean() * 100.0),
        "mismatch_rate_pct": float((~matches).mean() * 100.0),
        "list_mismatch_pct": pct_for_expected("interactive_list"),
        "buttons_mismatch_pct": pct_for_expected("interactive_buttons"),
    }


# ----- UI helpers -----
def render_sidebar_agent() -> None:
    st.header("1. Agent Configuration")
    cfg = st.session_state.agent_config
    prompt = st.text_area("Agent System Prompt", value=cfg.get("prompt", ""), height=180)
    schema = st.text_area("JSON Schema", value=cfg.get("json_schema", ""), height=160)
    model_options = ["gpt-4.1-mini", "gpt-5-nano", "gpt-5-mini", "gpt-5.1"]
    model = st.selectbox(
        "Agent Model",
        model_options,
        index=model_options.index(cfg.get("model", "gpt-4.1-mini"))
        if cfg.get("model", "gpt-4.1-mini") in model_options
        else 0,
    )

    if st.button("Save Agent Config", type="primary"):
        try:
            json.loads(schema or "{}")
        except json.JSONDecodeError as exc:
            st.error(f"JSON schema is invalid: {exc}")
            return
        st.session_state.agent_config = {"prompt": prompt, "json_schema": schema, "model": model}
        st.success("Agent configuration saved.")
        persist_state()


def render_sidebar_judge() -> None:
    st.header("2. LLM-as-Judge Configuration")
    cfg = st.session_state.judge_config
    prompt = st.text_area("Judge Prompt", value=cfg.get("prompt", ""), height=160)
    model_options = ["gpt-4.1-mini", "gpt-5-nano", "gpt-5-mini", "gpt-5.1"]
    model = st.selectbox(
        "Judge Model",
        model_options,
        index=model_options.index(cfg.get("model", "gpt-4.1-mini"))
        if cfg.get("model", "gpt-4.1-mini") in model_options
        else 0,
        key="judge_model_select",
    )
    if st.button("Save Judge Config", type="primary", key="save_judge"):
        st.session_state.judge_config = {"prompt": prompt, "model": model}
        st.success("Judge configuration saved.")
        persist_state()


def render_sidebar_dataset() -> None:
    st.header("3. Dataset")
    uploader = st.file_uploader("Upload CSV with columns: input, expected_output", type=["csv"])
    if uploader is not None:
        try:
            df = pd.read_csv(uploader)
            if not {"input", "expected_output"}.issubset(df.columns):
                st.error('Dataset must include "input" and "expected_output" columns.')
                return
            df = update_type_comparisons(df)
            st.session_state.dataset = df
            st.success(f"Dataset loaded with {len(df)} rows.")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Failed to load dataset: {exc}")
        else:
            persist_state()

    st.warning("Actual outputs and scores will persist until you clear them.", icon="⚠️")
    clear_confirm = st.checkbox("I understand clearing will erase actual outputs and scores")
    if st.button("Clear actual outputs and scores", type="secondary", disabled=not clear_confirm):
        df = ensure_dataset_columns(st.session_state.dataset.copy())
        df["actual_output"] = ""
        df["judge_score"] = np.nan
        df["judge_reasoning"] = ""
        st.session_state.dataset = update_type_comparisons(df)
        st.success("Cleared actual outputs and judge results.")
        persist_state()


def render_sidebar_agent_runner() -> None:
    st.header("4. Run Agent (Generate Actual Output)")
    df = st.session_state.dataset
    if df.empty:
        st.info("Upload or create a dataset to run the agent.")
        return
    n_default = min(10, len(df))
    n_to_run = st.number_input("Number of rows to run", min_value=1, max_value=100, value=n_default, step=1)

    if st.button("Run Agent on Selected Rows", type="primary", disabled=st.session_state.agent_processing):
        row_indices = select_agent_rows(df, int(n_to_run))
        if not row_indices:
            st.warning("No rows with empty actual_output to process.")
            return

        # Initialize progressive queue processing so UI updates per row.
        st.session_state.agent_queue = list(row_indices)
        st.session_state.agent_run_outputs = []
        st.session_state.agent_run_row_indices = list(row_indices)
        st.session_state.agent_run_meta = {
            "model": st.session_state.agent_config["model"],
            "prompt_hash": hash_prompt(st.session_state.agent_config["prompt"]),
            "start_time": datetime.now().isoformat(timespec="seconds"),
        }
        st.session_state.agent_processing = True
        st.rerun()

    if st.session_state.agent_processing:
        if st.button("Stop current agent run", type="secondary"):
            st.session_state.agent_stop_requested = True
            st.rerun()


def render_sidebar_judge_runner() -> None:
    st.header("5. Run Judge (Score Rows)")
    df = st.session_state.dataset
    if df.empty:
        st.info("Upload or create a dataset to run the judge.")
        return

    scope = st.selectbox(
        "Rows to score",
        ["Only rows without judge_score", "All rows with actual_output"],
    )
    n_default = min(10, len(df))
    n_to_run = st.number_input("Number of rows to score", min_value=1, max_value=100, value=n_default, step=1)

    if st.button("Run Judge", type="primary"):
        if not st.session_state.judge_config.get("prompt"):
            st.error("Judge configuration is missing.")
            return

        has_actual = df["actual_output"].fillna("") != ""
        if scope == "Only rows without judge_score":
            to_score = df[has_actual & df["judge_score"].isna()]
        else:
            to_score = df[has_actual]

        if to_score.empty:
            st.warning("No rows available for judging.")
            return

        row_indices = list(to_score.index[: int(n_to_run)])
        with st.spinner("Running judge..."):
            updated_df = run_judge_on_rows(df, row_indices, st.session_state.judge_config)
        # Only judge_score and judge_reasoning may change. Keep actual_output as-is from state.
        updated_df["actual_output"] = st.session_state.dataset["actual_output"]
        st.session_state.dataset = update_type_comparisons(updated_df)
        st.session_state.actual_output_backup = st.session_state.dataset["actual_output"].copy()
        persist_state()
        st.success(f"Judge run completed for {len(row_indices)} rows.")


def render_sidebar_report_controls() -> Dict[str, Any]:
    st.header("6. Report Controls")
    min_score = st.slider("Minimum score filter", 0.0, 1.0, 0.0, 0.05)
    top_k = st.number_input("Top N worst rows", min_value=1, max_value=100, value=5, step=1)
    return {"min_score": min_score, "top_k": int(top_k)}


# ----- Main content -----
def render_dataset_editor() -> None:
    st.subheader("Dataset (editable)")
    prev_df = update_type_comparisons(st.session_state.dataset.copy())
    df = prev_df.copy()

    # Coerce complex types (lists/dicts) to JSON strings so the editor stays compatible.
    for col in ["input", "expected_output", "actual_output", "judge_reasoning"]:
        df[col] = df[col].apply(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
    df["expected_type"] = df["expected_type"].astype(str)
    df["actual_type"] = df["actual_type"].astype(str)
    df["type_match"] = df["type_match"].apply(lambda v: "" if pd.isna(v) else str(bool(v)))
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=False,
        column_config={
            "input": st.column_config.TextColumn("input", required=False, width="medium"),
            "expected_output": st.column_config.TextColumn("expected_output", required=False, width="medium"),
            "actual_output": st.column_config.TextColumn(
                "actual_output",
                required=False,
                width="medium",
                help="Filled by the webhook response for each input; editable if you need to override.",
            ),
            "expected_type": st.column_config.TextColumn("expected_type", help="Derived message type from expected_output"),
            "actual_type": st.column_config.TextColumn("actual_type", help="Derived message type from actual_output"),
            "type_match": st.column_config.TextColumn("type_match", help="True when expected_type matches actual_type"),
            "generation_info": st.column_config.TextColumn(
                "generation_info",
                required=False,
                width="large",
                help="Per-row trace: API status and detected message type from the webhook response.",
            ),
            "judge_reasoning": st.column_config.TextColumn("judge_reasoning", required=False, width="large"),
            "judge_score": st.column_config.NumberColumn("judge_score", format="%.2f"),
        },
    )
    edited = ensure_dataset_columns(edited)

    # Preserve existing actual_output if the edited cell is blank (helps avoid accidental loss on reruns).
    prev_actuals = prev_df["actual_output"].fillna("").astype(str)
    new_actuals = edited["actual_output"].fillna("").astype(str)
    merged_actuals = new_actuals.where(new_actuals != "", prev_actuals)
    edited["actual_output"] = merged_actuals

    st.session_state.dataset = update_type_comparisons(edited.reset_index(drop=True))
    # Keep backup in sync with any edits or merges (positional).
    st.session_state.actual_output_backup = pd.Series(merged_actuals.reset_index(drop=True).values)
    persist_state()


def render_styled_table() -> None:
    st.subheader("Evaluation Table")
    df = st.session_state.dataset.copy()
    df["judge_score"] = pd.to_numeric(df["judge_score"], errors="coerce")
    if st.session_state.agent_processing:
        st.info("Agent run in progress; rows update as each generation completes.")
    if df.empty:
        st.info("No data to show yet.")
        return

    show_full = st.checkbox("Show full text", value=False)
    display_df = df.copy()
    if not show_full:
        for col in ["input", "expected_output", "actual_output", "judge_reasoning"]:
            display_df[col] = display_df[col].astype(str).str.slice(0, 200)
        display_df["generation_info"] = display_df.get("generation_info", "").astype(str).str.slice(0, 200)

    styled = style_dataset_df(display_df)
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=False,
        column_config={
            "generation_info": st.column_config.TextColumn("generation_info", help="API trace for this row"),
            "expected_type": st.column_config.TextColumn("expected_type"),
            "actual_type": st.column_config.TextColumn("actual_type"),
            "type_match": st.column_config.TextColumn("type_match"),
        },
    )


def render_report(filters: Dict[str, Any]) -> None:
    st.subheader("Evaluation Report")
    df = st.session_state.dataset.copy()
    df["judge_score"] = pd.to_numeric(df["judge_score"], errors="coerce")
    type_metrics = compute_type_mismatch_metrics(df)
    if type_metrics:
        st.write("Type comparison")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows with types", type_metrics["rows_with_types"])
        c2.metric("Type match rate", f"{type_metrics['match_rate_pct']:.1f}%")
        c3.metric("Mismatch when list expected", f"{type_metrics['list_mismatch_pct']:.1f}%")
        c4.metric("Mismatch when buttons expected", f"{type_metrics['buttons_mismatch_pct']:.1f}%")
    else:
        st.info("Type comparison pending: add expected/actual outputs with message types to see stats.")

    metrics = compute_report_metrics(df)
    if not metrics:
        st.info("No judged rows yet. Run the judge to see scoring metrics.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows scored", metrics["count"])
    col2.metric("Avg score", f"{metrics['mean']:.2f}")
    col3.metric("Min score", f"{metrics['min']:.2f}")
    col4.metric("Max score", f"{metrics['max']:.2f}")

    st.write("Bucket distribution")
    buckets = pd.DataFrame(
        {
            "bucket": ["Green (>=0.8)", "Yellow (0.5-0.79)", "Red (<0.5)"],
            "count": [metrics["green"], metrics["yellow"], metrics["red"]],
        }
    )
    st.bar_chart(buckets, x="bucket", y="count", height=240)

    st.write("Score split (exact)")
    total = metrics["count"]
    split = pd.DataFrame(
        {
            "score": ["1.0", "0.0", "other"],
            "count": [metrics["exact_one"], metrics["exact_zero"], metrics["exact_other"]],
            "percent": [
                f"{(metrics['exact_one']/total*100):.1f}%",
                f"{(metrics['exact_zero']/total*100):.1f}%",
                f"{(metrics['exact_other']/total*100):.1f}%",
            ],
        }
    )
    st.dataframe(split, use_container_width=True, hide_index=True)

    scored = df.dropna(subset=["judge_score"])
    scored = scored[scored["judge_score"] >= filters["min_score"]]
    worst = scored.nsmallest(filters["top_k"], "judge_score")
    if not worst.empty:
        st.markdown("Top low-scoring rows")
        st.dataframe(
            worst[["input", "expected_output", "actual_output", "judge_score", "judge_reasoning"]],
            use_container_width=True,
            hide_index=False,
        )


def render_future_placeholder() -> None:
    st.subheader("Analysis Chat (Future scope)")
    st.info("Chat-based analysis will be added in a future version.")


def main() -> None:
    st.set_page_config(page_title="Evaluation Runner", layout="wide")
    init_state()
    restore_actual_outputs_if_missing()
    st.session_state.dataset = update_type_comparisons(st.session_state.dataset)
    # Process agent queue one row per run for per-row visibility.
    if st.session_state.agent_stop_requested:
        st.session_state.agent_queue = []
        st.session_state.agent_processing = False
        st.session_state.agent_run_meta = {}
        st.session_state.agent_run_row_indices = []
        st.session_state.agent_run_outputs = []
        st.session_state.agent_stop_requested = False
        persist_state()
    elif st.session_state.agent_queue:
        idx = st.session_state.agent_queue.pop(0)
        df = st.session_state.dataset
        try:
            updated_df, outputs, infos = run_agent_on_rows(df, [idx], st.session_state.agent_config)
            result = outputs[0] if outputs else ""
            info = infos[0] if infos else ""
            st.session_state.dataset = update_type_comparisons(updated_df)
            st.session_state.actual_output_backup = st.session_state.dataset["actual_output"].copy()
            st.session_state.agent_run_outputs.append(result)
            # Maintain info column sync.
            if "generation_info" in st.session_state.dataset.columns:
                st.session_state.actual_output_backup = st.session_state.dataset["actual_output"].copy()
            persist_state()
        except Exception as exc:  # pylint: disable=broad-except
            st.session_state.agent_run_outputs.append(f"ERROR: {exc}")

        if st.session_state.agent_queue:
            st.session_state.agent_processing = True
            st.rerun()
        else:
            # Finalize run record.
            if st.session_state.agent_run_row_indices:
                run_id = len(st.session_state.test_runs) + 1
                st.session_state.test_runs.append(
                    {
                        "id": run_id,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "agent_model": st.session_state.agent_run_meta.get("model"),
                        "agent_prompt_hash": st.session_state.agent_run_meta.get("prompt_hash"),
                        "row_indices": st.session_state.agent_run_row_indices,
                        "actual_outputs": st.session_state.agent_run_outputs,
                    }
                )
            st.session_state.agent_processing = False
            st.session_state.agent_run_meta = {}
            st.session_state.agent_run_row_indices = []
            st.session_state.agent_run_outputs = []
            persist_state()

    with st.sidebar:
        render_sidebar_agent()
        render_sidebar_judge()
        render_sidebar_dataset()
        render_sidebar_agent_runner()
        render_sidebar_judge_runner()
        report_filters = render_sidebar_report_controls()

    st.title("Evaluation Runner")
    st.caption("Configure an agent, upload a dataset, run generations, and score with an LLM-as-judge.")

    render_dataset_editor()
    # After editor, ensure backups are synced to the latest state in case the judge run happens immediately after.
    st.session_state.actual_output_backup = st.session_state.dataset["actual_output"].copy()
    render_styled_table()
    render_report(report_filters)
    render_future_placeholder()


if __name__ == "__main__":
    main()
