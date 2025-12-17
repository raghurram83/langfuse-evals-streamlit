from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from openai import OpenAI


MODEL_NAME = "gpt-4.1-mini"

ANALYSIS_PROMPT = """You are an Intent & Use-Case Discovery Engine for a SaaS product.

You will be given:

A raw conversation dump as a single text blob (calls + WhatsApp).
Each line may contain messages from customers and agents.
Conversations may be separated by blank lines or system markers.
Your job is to:

Discover the top 5–10 customer intents.
Give real examples for each intent.
Suggest Day-1 bot flows for each intent.
Generate simple bot-flow drafts in structured JSON.
You MUST follow the OUTPUT JSON SCHEMA exactly (see bottom).

HIGH-LEVEL TASK
From the conversation dump, focus ONLY on what customers want (their questions, requests, problems, objections, and tasks).

Identify recurring customer problems / requests.

Cluster them into intents.

Select the top intents by frequency (aim for 5–10).

For each top intent:

Give it a short, clear name (e.g., "Pricing Query", "Demo Request", "Order Status", "Refund").
Estimate its share of total conversations as a percentage.
Extract real example utterances as they appear in the text.
Suggest a Day-1 bot flow (step-by-step).
Generate a bot_flow_draft that is ready to be mapped into a Composer-style flow later.
If data is noisy, mixed, or sparse, still do your best and explain any limitations in the analysis_notes field.

HOW TO THINK ABOUT INTENTS
Definition of an intent:

A customer goal you can summarise as “The user is trying to ___”.
Examples: ask price, book a demo, check order status, cancel, change plan, get support, etc.
Rules:

Merge similar phrasings into one intent.
Prefer fewer, clearer intents over many tiny ones.
Ignore one-off or extremely rare requests for the MVP (long-tail).
OUTPUT FORMAT (IMPORTANT)
You MUST return a single JSON object using this schema:

{
"summary": {
"total_conversations_estimate": number,
"total_customer_messages_estimate": number,
"high_level_overview": string
},
"intents": [
{
"intent_id": string,
"display_name": string,
"description": string,
"volume_percent_estimate": number,
"example_utterances": [string, ...],
"sample_conversations": [
{
"short_id": string,
"snippet": string,
"reason_why_this_intent": string
}
],
"recommended_flow": {
"goal": string,
"when_to_trigger": string,
"steps": [
{
"step_order": number,
"type": "ask" | "inform" | "confirm" | "handover",
"message_template": string,
"variable_key": string | null,
"notes": string | null
}
]
},
"bot_flow_draft": {
"flow_name": string,
"entry_condition": string,
"steps": [
{
"step_id": string,
"step_order": number,
"type": "ask" | "inform" | "confirm" | "handover",
"message_template": string,
"capture_variable": string | null,
"validation_hint": string | null,
"next_step_if_success": string | null,
"next_step_if_failure": string | null
}
],
"handover_rules": [
{
"reason": string,
"condition_description": string
}
]
}
}
],
"analysis_notes": string
}

IMPORTANT BEHAVIOUR RULES
DO NOT invent product features or policies not implied by the dump.
Stay grounded in what customers actually asked.
Prefer simple English.
Always return valid JSON. No markdown, no extra text.
CALL PATTERN:
system: this prompt
user: a JSON string like { "conversation_dump": "<RAW_TEXT>" }
Return only the JSON object.
END ANALYSIS PROMPT"""


def parse_uploaded_file(uploaded_file: Any) -> Tuple[Optional[str], Optional[str], Optional[List[Dict[str, Any]]], Optional[str]]:
    """Return (conversation_dump_text, preview_text, preview_table, error_message)."""
    try:
        file_bytes = uploaded_file.read()
    except Exception as exc:  # pragma: no cover - UI surfaced error
        return None, None, None, f"Failed to read file: {exc}"

    suffix = uploaded_file.name.lower().rsplit(".", 1)
    ext = suffix[1] if len(suffix) == 2 else ""
    size_kb = len(file_bytes) / 1024

    if ext == "txt":
        conversation_text = file_bytes.decode("utf-8", errors="ignore")
        preview_lines = "\n".join(conversation_text.splitlines()[:20])
        return conversation_text, preview_lines, [{"type": "txt", "size_kb": f"{size_kb:.1f} KB"}], None

    if ext == "json":
        try:
            raw_text = file_bytes.decode("utf-8", errors="ignore")
            parsed = json.loads(raw_text)
        except Exception as exc:  # pragma: no cover - UI surfaced error
            return None, None, None, f"Failed to parse JSON: {exc}"

        # Shape 1: dict with conversations
        if isinstance(parsed, dict) and isinstance(parsed.get("conversations"), list):
            convs = parsed["conversations"]
            preview_rows = []
            blocks: List[str] = []
            for conv in convs:
                if not isinstance(conv, dict):
                    continue
                conv_id = conv.get("conversation_id", "unknown")
                channel = conv.get("channel", "unknown")
                messages = conv.get("messages") if isinstance(conv.get("messages"), list) else []
                preview_rows.append(
                    {
                        "conversation_id": conv_id,
                        "channel": channel,
                        "#messages": len(messages),
                    }
                )
                lines = []
                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    sender = str(msg.get("sender", "") or "Speaker").capitalize()
                    text = str(msg.get("text", "") or "").strip()
                    if text:
                        lines.append(f"{sender}: {text}")
                header = f"Conversation {conv_id} ({channel})"
                blocks.append(header if not lines else f"{header}\n" + "\n".join(lines))
            if not blocks:
                return None, None, None, "No valid conversations/messages found in JSON."
            conversation_text = "\n\n".join(blocks)
            preview_dump = "\n".join(conversation_text.splitlines()[:30])
            return conversation_text, preview_dump, preview_rows[:3], None

        # Shape 2: list of call objects
        if isinstance(parsed, list):
            transcripts: List[str] = []
            preview_rows: List[Dict[str, Any]] = []
            for call in parsed:
                if not isinstance(call, dict):
                    continue
                call_transcript = call.get("call_transcript")
                if isinstance(call_transcript, str):
                    transcripts.append(call_transcript)
                preview_rows.append(
                    {
                        "call_id": call.get("call_id", ""),
                        "caller": call.get("caller", ""),
                        "agent_name": call.get("agent_name", ""),
                        "start_time": call.get("start_time", ""),
                    }
                )
            if not transcripts:
                return None, None, None, "No call_transcript fields found in JSON."
            conversation_text = "\n\n".join(transcripts)
            preview_dump = "\n".join(conversation_text.splitlines()[:30])
            return conversation_text, preview_dump, preview_rows[:3], None

        return None, None, None, "Unsupported JSON shape. Provide the unified schema or an array of call objects."

    return None, None, None, "Unsupported file type. Please upload a .txt or .json file."


def run_intent_discovery(conversation_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """Call the OpenAI API and return parsed JSON, raw text, and error message if any."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None, None, "OPENAI_API_KEY environment variable is not set."

    client = OpenAI(api_key=api_key)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": ANALYSIS_PROMPT},
                {"role": "user", "content": json.dumps({"conversation_dump": conversation_text})},
            ],
        )
    except Exception as exc:  # pragma: no cover - UI surfaced error
        return None, None, f"Error calling OpenAI API: {exc}"

    raw_text = ""
    if completion.choices:
        message = completion.choices[0].message
        raw_text = (message.content or "").strip()

    if not raw_text:
        return None, None, "Empty response from model."

    try:
        parsed = json.loads(raw_text)
        return parsed, raw_text, None
    except json.JSONDecodeError:
        return None, raw_text, "Model response could not be parsed as JSON."


def render_summary(summary: Dict[str, Any]) -> None:
    st.subheader("Summary")
    total_conversations = summary.get("total_conversations_estimate", "—")
    total_messages = summary.get("total_customer_messages_estimate", "—")
    high_level_overview = summary.get("high_level_overview", "")

    col1, col2 = st.columns(2)
    col1.metric("Total conversations (est.)", total_conversations)
    col2.metric("Customer messages (est.)", total_messages)
    if high_level_overview:
        st.write(high_level_overview)


def render_recommended_flow(flow: Dict[str, Any]) -> None:
    st.markdown("**Recommended Day-1 Flow**")
    goal = flow.get("goal")
    trigger = flow.get("when_to_trigger")
    if goal or trigger:
        st.write(f"Goal: {goal or '—'}")
        st.write(f"When to trigger: {trigger or '—'}")

    steps = flow.get("steps") or []
    if steps:
        rows = []
        for step in steps:
            rows.append(
                {
                    "Order": step.get("step_order"),
                    "Type": step.get("type"),
                    "Message": step.get("message_template"),
                    "Variable": step.get("variable_key"),
                    "Notes": step.get("notes"),
                }
            )
        st.table(rows)
    else:
        st.write("No recommended flow steps provided.")


def render_bot_flow(bot_flow: Dict[str, Any]) -> None:
    st.markdown("**Bot Flow Draft**")
    st.write(f"Flow name: {bot_flow.get('flow_name', '—')}")
    st.write(f"Entry condition: {bot_flow.get('entry_condition', '—')}")

    steps = bot_flow.get("steps") or []
    if steps:
        rows = []
        for step in steps:
            rows.append(
                {
                    "Step ID": step.get("step_id"),
                    "Order": step.get("step_order"),
                    "Type": step.get("type"),
                    "Message": step.get("message_template"),
                    "Capture": step.get("capture_variable"),
                    "Validation": step.get("validation_hint"),
                    "Next if success": step.get("next_step_if_success"),
                    "Next if failure": step.get("next_step_if_failure"),
                }
            )
        st.table(rows)
    else:
        st.write("No bot flow steps provided.")

    handover_rules = bot_flow.get("handover_rules") or []
    if handover_rules:
        st.markdown("**Handover rules:**")
        for rule in handover_rules:
            reason = rule.get("reason", "Reason not provided")
            condition = rule.get("condition_description", "Condition not provided")
            st.markdown(f"- {reason}: {condition}")


def render_intents(intents: List[Dict[str, Any]]) -> None:
    def is_non_english(text: str) -> bool:
        ascii_ratio = sum(1 for ch in text if ord(ch) < 128) / max(len(text), 1)
        return ascii_ratio < 0.85

    def translate_text(text: str) -> Optional[str]:
        cache = st.session_state.setdefault("translation_cache", {})
        if text in cache:
            return cache[text]
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        client = OpenAI(api_key=api_key)
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Translate the user message to concise English. Return only the translation."},
                    {"role": "user", "content": text},
                ],
                max_tokens=120,
            )
            translation = (resp.choices[0].message.content or "").strip()
            cache[text] = translation
            return translation
        except Exception:
            return None

    st.subheader("Intent Catalogue")
    for intent in intents:
        display_name = intent.get("display_name", "Unknown intent")
        volume = intent.get("volume_percent_estimate")
        expander_title = f"Intent: {display_name}"
        if volume is not None:
            expander_title += f" (≈ {volume}% of conversations)"

        with st.expander(expander_title, expanded=False):
            st.markdown(f"**Intent ID:** {intent.get('intent_id', '—')}")
            st.write(intent.get("description", ""))

            examples = intent.get("example_utterances") or []
            if examples:
                st.markdown("**Example utterances:**")
                for utt in examples:
                    st.markdown(f"- {utt}")
                    if is_non_english(utt):
                        translated = translate_text(utt)
                        if translated:
                            st.markdown(
                                f"<div style='color:#6a5acd;font-style:italic;margin-left:12px;'>↪ {translated}</div>",
                                unsafe_allow_html=True,
                            )

            samples = intent.get("sample_conversations") or []
            if samples:
                st.markdown("**Sample conversations:**")
                for sample in samples:
                    snippet = sample.get("snippet", "")
                    st.markdown(f"- `{sample.get('short_id', 'sample')}`: {snippet}")
                    if is_non_english(snippet):
                        translated = translate_text(snippet)
                        if translated:
                            st.markdown(
                                f"<div style='color:#6a5acd;font-style:italic;margin-left:12px;'>↪ {translated}</div>",
                                unsafe_allow_html=True,
                            )
                    reason = sample.get("reason_why_this_intent")
                    if reason:
                        st.markdown(f"  - Why: {reason}")

            recommended_flow = intent.get("recommended_flow") or {}
            render_recommended_flow(recommended_flow)

            bot_flow = intent.get("bot_flow_draft") or {}
            render_bot_flow(bot_flow)


def render_pipeline_steps(result_json: Dict[str, Any]) -> None:
    intents = result_json.get("intents") or []
    st.subheader("Pipeline Steps")

    def render_intent_table() -> None:
        rows = []
        for intent in intents:
            rows.append(
                {
                    "Intent": intent.get("display_name", intent.get("intent_id", "")),
                    "Intent ID": intent.get("intent_id", ""),
                    "Volume %": intent.get("volume_percent_estimate"),
                }
            )
        if rows:
            st.table(rows)
        else:
            st.write("No intents found.")

    def render_examples() -> None:
        for intent in intents:
            examples = intent.get("example_utterances") or []
            if not examples:
                continue
            st.markdown(f"- **{intent.get('display_name', intent.get('intent_id', 'Intent'))}**")
            for utt in examples:
                st.markdown(f"  - {utt}")

    def render_recommended_flows() -> None:
        for intent in intents:
            flow = intent.get("recommended_flow") or {}
            st.markdown(f"**{intent.get('display_name', intent.get('intent_id', 'Intent'))}**")
            st.write(f"Goal: {flow.get('goal', '—')}")
            steps = flow.get("steps") or []
            if steps:
                rows = []
                for step in steps:
                    rows.append(
                        {
                            "Order": step.get("step_order"),
                            "Type": step.get("type"),
                            "Message": step.get("message_template"),
                        }
                    )
                st.table(rows)
            else:
                st.write("No steps provided.")

    def render_bot_flows() -> None:
        for intent in intents:
            bot_flow = intent.get("bot_flow_draft") or {}
            st.markdown(f"**{intent.get('display_name', intent.get('intent_id', 'Intent'))}**")
            st.write(f"Flow name: {bot_flow.get('flow_name', '—')}")
            steps = bot_flow.get("steps") or []
            if steps:
                rows = []
                for step in steps:
                    rows.append(
                        {
                            "Step ID": step.get("step_id"),
                            "Type": step.get("type"),
                            "Message": step.get("message_template"),
                        }
                    )
                st.table(rows)
            else:
                st.write("No bot flow steps provided.")

    step_items = [
        ("Intent Discovery", render_intent_table),
        ("Example Extraction", render_examples),
        ("Use-Case & Flow Suggestion", render_recommended_flows),
        ("Bot Flow Generator", render_bot_flows),
    ]

    for idx, (title, renderer) in enumerate(step_items, start=1):
        with st.expander(f"{idx}. {title} ✅", expanded=False):
            renderer()


def render_results(result_json: Dict[str, Any]) -> None:
    summary = result_json.get("summary")
    if isinstance(summary, dict):
        render_summary(summary)

    intents = result_json.get("intents") or []
    if intents:
        render_intents(intents)

    render_pipeline_steps(result_json)

    analysis_notes = result_json.get("analysis_notes")
    if analysis_notes:
        st.subheader("Analysis Notes")
        st.write(analysis_notes)

    st.subheader("Raw JSON Result")
    st.code(json.dumps(result_json, indent=2), language="json")
    st.download_button(
        label="Download JSON",
        file_name="result.json",
        mime="application/json",
        data=json.dumps(result_json, indent=2),
    )


def show_upload_preview(conversation_text: str, preview: Optional[str], preview_table: Optional[List[Dict[str, Any]]], uploaded_file: Any) -> None:
    file_label = uploaded_file.type or "file"
    size_kb = len(uploaded_file.getvalue()) / 1024 if hasattr(uploaded_file, "getvalue") else 0
    st.info(f"Uploaded: {uploaded_file.name} • {file_label} • {size_kb:.1f} KB")

    if preview_table:
        if "call_id" in preview_table[0]:
            st.markdown("Preview of calls (first 3):")
        else:
            st.markdown("Preview of conversations (first 3):")
        st.table(preview_table)

    st.text_area(
        "Conversation dump preview (first ~30 lines)",
        preview or "\n".join(conversation_text.splitlines()[:30]),
        height=360,
        disabled=True,
    )


def main() -> None:
    st.set_page_config(page_title="Intent & Use-Case Discovery PoC", layout="wide")
    st.title("Intent & Use-Case Discovery PoC")
    st.write(
        "Upload a conversation dump (.txt or .json). The system will analyze customer intents, "
        "show top intents with examples, and generate draft bot flows."
    )

    uploaded_file = st.file_uploader("Upload conversation dump (.txt or .json)", type=["txt", "json"])
    conversation_text: Optional[str] = None
    preview_text: Optional[str] = None
    preview_table: Optional[List[Dict[str, Any]]] = None
    parse_error: Optional[str] = None

    if uploaded_file is not None:
        conversation_text, preview_text, preview_table, parse_error = parse_uploaded_file(uploaded_file)
        if parse_error:
            st.error(parse_error)
        else:
            show_upload_preview(conversation_text, preview_text, preview_table, uploaded_file)

    run_clicked = st.button("Run Intent Discovery", type="primary")

    if run_clicked:
        if parse_error:
            st.error(parse_error)
            return

        if not uploaded_file or not conversation_text:
            st.error("Please upload a .txt or .json conversation dump first.")
            return

        if not os.environ.get("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY environment variable is not set.")
            return

        with st.spinner("Analyzing conversation dump..."):
            result_json, raw_text, error_message = run_intent_discovery(conversation_text)

        if error_message:
            st.error(error_message)
            if raw_text:
                with st.expander("Raw model response"):
                    st.code(raw_text)
            return

        if result_json:
            st.session_state.discovery_result = result_json

    discovery_result = st.session_state.get("discovery_result")
    if discovery_result:
        render_results(discovery_result)


if __name__ == "__main__":
    main()
