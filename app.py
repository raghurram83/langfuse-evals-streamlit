from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
import os

import pandas as pd
import streamlit as st

from config import DATA_DIR, EMBEDDING_MODEL, LLM_MODEL, OPENAI_API_KEY, load_json
from pipeline.cache_io import clear_pipeline_cache, load_pipeline_result, save_pipeline_result
from pipeline.runner import PipelineResult, run_pipeline
from pipeline import step_01_ingestion
from pipeline import step_02_ticketing_llm
from pipeline import step_04_embeddings
from pipeline import step_05_similarity_grouping
from pipeline import step_06_labeling_llm
from pipeline import step_07_compression
from pipeline import step_08_flow_viability


st.set_page_config(page_title="Intent & Flow Discovery Validator", layout="wide")


def save_uploaded_conversations(uploaded_file: Optional[Any]) -> Optional[Path]:
    if uploaded_file is None:
        return None
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / "_uploaded_conversations.json"
    target.write_bytes(uploaded_file.getvalue())
    return target


def format_usage_table(usage: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    rows = []
    for step, stats in usage.items():
        if not isinstance(stats, dict):
            continue
        has_token_keys = any(key in stats for key in ["prompt_tokens", "completion_tokens", "total_tokens"])
        if not has_token_keys:
            continue
        rows.append(
            {
                "step": step,
                "prompt_tokens": stats.get("prompt_tokens", 0),
                "completion_tokens": stats.get("completion_tokens", 0),
                "total_tokens": stats.get("total_tokens", 0),
            }
        )
    return pd.DataFrame(rows)


def render_overview(result: PipelineResult) -> None:
    st.markdown("### Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Conversations", len(result.conversations))
    col2.metric("Tickets", len(result.tickets))
    col3.metric("Similarity groups", result.grouped_tickets["similarity_group_id"].nunique())
    col4.metric("Compressed flows", len(result.flows))

    st.caption("Pipeline: Ingest → Ticket (LLM per conversation) → Embed → Similarity (analysis only) → Single-call flow labeling → Compression (≤20 flows) → Flow viability → Rank")

    st.markdown("#### Models")
    st.write(f"LLM model: `{result.llm_model}` • Embedding model: `{result.embedding_model}`")

    sim_metrics = result.usage.get("similarity_metrics", {})
    if sim_metrics:
        st.write(
            f"Similarity metrics (for audit, not intent): avg={sim_metrics.get('avg_similarity', 0):.2f}, "
            f"max={sim_metrics.get('max_similarity', 0):.2f}"
        )

    st.markdown("#### Token usage")
    usage_table = format_usage_table(result.usage)
    if not usage_table.empty:
        st.dataframe(usage_table, use_container_width=True)
    else:
        st.write("No token usage captured yet.")


def render_step_inspector(result: PipelineResult) -> None:
    st.markdown("### Step-by-Step Inspector")
    steps: Dict[str, pd.DataFrame] = {
        "1. Conversation ingestion": result.conversations,
        "2. Ticket extraction (LLM per conversation)": result.tickets,
        "3. Ticket embeddings (cached)": result.embedded_tickets,
        "4. Similarity grouping (analysis only)": result.grouped_tickets,
        "5. Single-call flow labeling per ticket": result.labeled_tickets,
        "6. Compression (<=20 flows)": result.compressed_tickets,
        "7. Flow viability + ranking": result.flows,
    }

    step_names = list(steps.keys())
    selected = st.selectbox("Select a step to inspect", step_names, index=1 if len(step_names) > 1 else 0)
    st.write("Inputs → Outputs")
    if "Similarity grouping" in selected:
        st.warning("Similarity ≠ Intent. Groups are only for nearest-neighbor review.")
    df = steps[selected]
    if not df.empty:
        st.dataframe(df.head(20), use_container_width=True)
    else:
        st.write("No data for this step yet.")

    st.divider()
    st.markdown("**Track a ticket end-to-end**")
    ticket_ids = result.labeled_tickets["ticket_id"].tolist()
    if ticket_ids:
        selected_ticket = st.selectbox("Ticket", ticket_ids)
        trace = {
            "ticket": result.tickets[result.tickets["ticket_id"] == selected_ticket].to_dict("records"),
            "label": result.labeled_tickets[result.labeled_tickets["ticket_id"] == selected_ticket].to_dict("records"),
        }
        for label, rows in trace.items():
            st.markdown(f"**{label.capitalize()}**")
            st.json(rows[0] if rows else {})
    else:
        st.write("No tickets to trace.")


def render_tickets_tab(result: PipelineResult) -> None:
    st.markdown("### Tickets")
    df = result.labeled_tickets.copy()
    flow_options = sorted(df["proposed_flow"].unique())
    category_options = sorted(df["category"].unique())

    col1, col2 = st.columns(2)
    selected_flows = col1.multiselect("Flow filter", flow_options, default=flow_options)
    selected_cats = col2.multiselect("Category filter", category_options, default=category_options)

    filtered = df[
        df["proposed_flow"].isin(selected_flows)
        & df["category"].isin(selected_cats)
    ]

    st.dataframe(
        filtered[
            [
                "ticket_id",
                "conversation_id",
                "ticket_text",
                "proposed_flow",
                "confidence",
                "category",
            ]
        ],
        use_container_width=True,
    )

    st.markdown("**Inspect ticket**")
    if not filtered.empty:
        selected_ticket = st.selectbox("Ticket to inspect", filtered["ticket_id"].tolist())
        ticket_row = result.labeled_tickets[result.labeled_tickets["ticket_id"] == selected_ticket].iloc[0]
        st.write(f"Ticket: {ticket_row.ticket_text}")
        st.write(f"Proposed flow: {ticket_row.get('proposed_flow', '')}")
        st.write(f"Reasoning: {ticket_row.get('reasoning', '')}")
        st.write(f"Confidence: {ticket_row.get('confidence', 0):.2f}")
        st.write(f"Category: {ticket_row.get('category', '')}")
    else:
        st.write("No tickets after filter.")


def render_intents_and_flows(result: PipelineResult, taxonomy_lookup: Dict[str, str]) -> None:
    st.markdown("### Flows (compressed)")
    df = result.flows.copy()
    if df.empty:
        st.write("No flows mapped yet.")
        return

    def split_category(flow_name: str) -> tuple[str, str]:
        if "." in flow_name:
            cat, rest = flow_name.split(".", 1)
            return cat, rest
        return "uncategorized", flow_name

    # Group by category prefix (topic)
    df[["category_topic", "subcategory"]] = df["compressed_flow"].apply(
        lambda x: pd.Series(split_category(str(x)))
    )

    st.markdown("#### Topics → Flows")
    for topic, topic_df in df.groupby("category_topic"):
        with st.expander(f"Topic: {topic} ({len(topic_df)} flows)", expanded=False):
            for _, row in topic_df.iterrows():
                sub = row["subcategory"]
                examples = row.get("example_tickets") or []
                example_snippet = examples[0][:200] if examples else ""
                desc = row.get("why") or ""
                if desc:
                    desc = desc.strip()
                agent_steps = (row.get("agent_resolutions") or [])
                agent_snippet = agent_steps[0] if agent_steps else ""
                if agent_snippet:
                    desc = f"Mirror agent handling: {agent_snippet}"
                    if desc and desc[-1] != ".":
                        desc += "."
                elif example_snippet:
                    desc = (
                        f"Mirror agent handling: ask for needed details referenced in '{example_snippet}', "
                        f"confirm, then execute the {sub} flow. {desc}"
                    )
                elif not desc:
                    desc = f"Mirror agent handling: ask clarifying questions, confirm intent, then execute the {sub} flow."
                st.markdown(
                    f"**{sub}** — freq {row['frequency']}, avg_conf {row['avg_confidence']:.2f}, viability {row.get('viability','')}"
                )
                st.markdown(f"*What to do:* {desc}")

    st.markdown("#### Top flows to build (exclude HUMAN_ONLY automatically)")
    build_df = df[df["viability"] != "HUMAN_ONLY"]
    st.dataframe(build_df[["compressed_flow", "frequency", "avg_confidence", "viability", "why"]])

    st.download_button(
        "Export flows CSV",
        data=df.to_csv(index=False),
        file_name="flows_ranked.csv",
        mime="text/csv",
    )


def ensure_openai_key() -> bool:
    if not (os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY):
        st.error("Set OPENAI_API_KEY environment variable or enter the key in the sidebar.")
        return False
    return True


def run_step_ticketing(result: PipelineResult, conversations_path: Optional[str], llm_model: str, progress_callback=None) -> PipelineResult:
    path_obj = Path(conversations_path) if conversations_path else DATA_DIR / "_uploaded_conversations.json"
    conversations = step_01_ingestion.ingest_conversations(path_obj if path_obj.exists() else None)
    tickets, ticket_usage = step_02_ticketing_llm.extract_tickets(conversations, model=llm_model, progress_callback=progress_callback)
    result.conversations = conversations
    result.tickets = tickets
    result.usage["ticketing"] = ticket_usage
    save_pipeline_result(result)
    return result


def run_step_embeddings(result: PipelineResult, embedding_model: str) -> PipelineResult:
    if result.tickets.empty:
        raise ValueError("No tickets available. Run ticketing first.")
    embedded, embed_usage = step_04_embeddings.generate_ticket_embeddings(result.tickets, model=embedding_model)
    grouped, sim_metrics = step_05_similarity_grouping.group_by_similarity(embedded) if not embedded.empty else (embedded, {})
    result.embedded_tickets = embedded
    result.grouped_tickets = grouped
    result.usage["embeddings"] = embed_usage
    result.usage["similarity_metrics"] = sim_metrics
    save_pipeline_result(result)
    return result


def run_step_labeling(result: PipelineResult, llm_model: str, progress_callback=None) -> PipelineResult:
    if result.embedded_tickets.empty:
        raise ValueError("No embedded tickets available. Run embeddings first.")
    taxonomy = load_json("taxonomy.json").get("intent_clusters", [])
    registry = load_json("flow_registry.json").get("flows", [])
    labeled, label_usage = step_06_labeling_llm.label_tickets(
        result.embedded_tickets, taxonomy, registry, model=llm_model, batch_size=10, progress_callback=progress_callback
    )
    result.labeled_tickets = labeled
    result.usage["labeling"] = label_usage
    save_pipeline_result(result)
    return result


def run_step_compression(result: PipelineResult, max_flows: int = 20, min_freq_ratio: float = 0.02) -> PipelineResult:
    if result.labeled_tickets.empty:
        raise ValueError("No labeled tickets available. Run labeling first.")
    compressed = step_07_compression.compress_flows(result.labeled_tickets, max_flows=max_flows, min_frequency_ratio=min_freq_ratio)
    result.compressed_tickets = compressed
    save_pipeline_result(result)
    return result


def run_step_viability(result: PipelineResult, llm_model: str) -> PipelineResult:
    if result.compressed_tickets.empty:
        raise ValueError("No compressed tickets available. Run compression first.")
    flow_rows = []
    for flow_name, df_flow in result.compressed_tickets.groupby("compressed_flow"):
        flow_rows.append(
            {
                "compressed_flow": flow_name,
                "frequency": len(df_flow),
                "avg_confidence": float(df_flow["confidence"].mean()) if len(df_flow) else 0.0,
                "example_tickets": df_flow["ticket_text"].head(3).tolist(),
                "category_counts": df_flow["category"].value_counts().to_dict(),
                "agent_resolutions": df_flow["agent_resolution"].dropna().head(3).tolist(),
            }
        )
    flow_df = pd.DataFrame(flow_rows)
    flows_with_viability, viab_usage = step_08_flow_viability.score_flow_viability(flow_df, model=llm_model)
    result.flows = flows_with_viability
    result.usage["viability"] = viab_usage
    save_pipeline_result(result)
    return result


def main() -> None:
    st.title("Intent & Flow Discovery Validator")
    st.write("Validates real intent extraction and flow mapping for chatbot build readiness.")

    taxonomy = load_json("taxonomy.json")["intent_clusters"]
    taxonomy_lookup = {item["intent_id"]: item["name"] for item in taxonomy}

    with st.sidebar:
        st.subheader("Run pipeline")
        st.caption("Uses real embeddings and chat completions. No mocks.")
        uploaded_file = st.file_uploader("Optional conversations JSON", type=["json"])
        uploaded_path = save_uploaded_conversations(uploaded_file) if uploaded_file else None
        sample_count = st.number_input(
            "Limit # conversations (0 = all)",
            min_value=0,
            step=1,
            value=0,
            help="Useful for large dumps. Applies to uploaded file or default data.",
        )
        api_key_input = st.text_input(
            "OpenAI API key",
            type="password",
            value=os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY or "",
            help="Overrides env var for this run.",
        )
        llm_model_input = st.text_input("LLM model", value=os.getenv("LLM_MODEL") or LLM_MODEL, help="Chat completion model")
        embedding_model_input = st.text_input(
            "Embedding model", value=os.getenv("EMBEDDING_MODEL") or EMBEDDING_MODEL, help="Embedding model for similarity"
        )
        step_option = st.selectbox(
            "Step to run",
            ["all", "ticketing", "embeddings", "labeling", "compression", "viability"],
            help="Run a single step or the full pipeline.",
        )
        if st.button("Clear data"):
            st.session_state["pending_clear"] = True
        if st.session_state.get("pending_clear"):
            st.warning("Confirm clearing cached results?")
            colc1, colc2 = st.columns(2)
            if colc1.button("Confirm clear", type="secondary"):
                st.session_state.pop("pipeline_result", None)
                clear_pipeline_cache()
                st.session_state["pending_clear"] = False
            if colc2.button("Cancel", type="secondary"):
                st.session_state["pending_clear"] = False
        run_clicked = st.button("Run selected", type="primary")
        st.write(f"Default LLM: `{LLM_MODEL}`")
        st.write(f"Default embedding: `{EMBEDDING_MODEL}`")

    progress_bar = st.empty()
    progress_text = st.empty()

    if run_clicked:
        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input
        if ensure_openai_key():
            def progress_callback(fraction: float, message: str) -> None:
                capped = min(max(fraction, 0.0), 1.0)
                progress_bar.progress(capped)
                progress_text.write(message)

            with st.spinner(f"Running step: {step_option}..."):
                try:
                    if step_option == "all":
                        result = run_pipeline(
                            conversation_path=str(uploaded_path) if uploaded_path else None,
                            llm_model=llm_model_input.strip() or None,
                            embedding_model=embedding_model_input.strip() or None,
                            sample_count=int(sample_count) if sample_count else None,
                            progress_callback=progress_callback,
                        )
                    else:
                        existing = st.session_state.get("pipeline_result") or load_pipeline_result() or PipelineResult(
                            conversations=pd.DataFrame(),
                            tickets=pd.DataFrame(),
                            embedded_tickets=pd.DataFrame(),
                            grouped_tickets=pd.DataFrame(),
                            labeled_tickets=pd.DataFrame(),
                            compressed_tickets=pd.DataFrame(),
                            flows=pd.DataFrame(),
                            usage={},
                            llm_model=llm_model_input,
                            embedding_model=embedding_model_input,
                        )
                        if step_option == "ticketing":
                            result = run_step_ticketing(existing, str(uploaded_path) if uploaded_path else None, llm_model_input, progress_callback)
                        elif step_option == "embeddings":
                            result = run_step_embeddings(existing, embedding_model_input)
                        elif step_option == "labeling":
                            result = run_step_labeling(existing, llm_model_input, progress_callback)
                        elif step_option == "compression":
                            result = run_step_compression(existing)
                        elif step_option == "viability":
                            result = run_step_viability(existing, llm_model_input)
                        else:
                            result = existing
                    st.session_state["pipeline_result"] = result
                    save_pipeline_result(result)
                except Exception as exc:  # pragma: no cover - surfaced in UI
                    st.error(f"Pipeline failed: {exc}")

    if "pipeline_result" not in st.session_state:
        cached = load_pipeline_result()
        if cached is not None:
            st.session_state["pipeline_result"] = cached

    result: Optional[PipelineResult] = st.session_state.get("pipeline_result")
    if result:
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Step-by-Step", "Tickets", "Intents & Flows"])
        with tab1:
            render_overview(result)
        with tab2:
            render_step_inspector(result)
        with tab3:
            render_tickets_tab(result)
        with tab4:
            render_intents_and_flows(result, taxonomy_lookup)
    else:
        st.info("Upload conversations (optional) and run the pipeline to inspect outputs.")


if __name__ == "__main__":
    main()
