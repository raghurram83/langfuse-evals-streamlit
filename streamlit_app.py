"""
Langfuse Evals Tool (V1)

This Streamlit app mirrors the PRD: manage golden datasets, sync to Langfuse,
run evals with a prompt/model, and inspect results. Langfuse + OpenAI SDK
integration is included; plug in your keys and host to make it live.
"""

from __future__ import annotations

import base64
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

# Optional imports; we guard them at runtime.
try:
    from langfuse import Langfuse
except Exception:  # pylint: disable=broad-except
    Langfuse = None  # type: ignore

try:
    from openai import OpenAI
except Exception:  # pylint: disable=broad-except
    OpenAI = None  # type: ignore

st.set_page_config(page_title="Langfuse Evals Tool", layout="wide")


# ---------- Session state helpers ----------
def get_state() -> Dict[str, Any]:
    if "datasets" not in st.session_state:
        st.session_state.datasets = {}  # id -> dataset dict
    if "runs" not in st.session_state:
        st.session_state.runs = []  # list of run dicts
    if "active_dataset" not in st.session_state:
        st.session_state.active_dataset = None
    if "active_run" not in st.session_state:
        st.session_state.active_run = None
    return st.session_state


def parse_json(text: str) -> Any:
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text  # keep as raw string so user sees what they entered


def dataset_summary(ds: Dict[str, Any]) -> str:
    status = "Uploaded" if ds.get("synced") else "Not Synced"
    return f"{ds['name']} • {status} • {len(ds['items'])} items"


# ---------- UI building blocks ----------
def sidebar_connection() -> Dict[str, str]:
    st.sidebar.header("Langfuse Connection")
    host = st.sidebar.text_input(
        "Base URL",
        value=st.secrets.get("langfuse_base_url", "https://cloud.langfuse.com"),
        help="Your Langfuse instance URL.",
    )
    project_id = st.sidebar.text_input(
        "Project ID (for trace links)",
        value=st.secrets.get("langfuse_project_id", ""),
        help="Used only to construct clickable trace URLs in the UI.",
    )
    public_key = st.sidebar.text_input(
        "Public Key",
        value=st.secrets.get("langfuse_public_key", ""),
    )
    secret_key = st.sidebar.text_input(
        "Secret Key",
        value=st.secrets.get("langfuse_secret_key", ""),
        type="password",
    )
    st.sidebar.caption("Keys are only used client-side in this app.")

    st.sidebar.divider()
    st.sidebar.header("Model Provider")
    provider = st.sidebar.selectbox("Provider", ["OpenAI"], index=0)
    default_openai = st.secrets.get("openai_api_key", "")
    openai_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=default_openai,
        type="password",
        help="Needed to fetch actual outputs for evals.",
    )
    return {
        "host": host,
        "project_id": project_id,
        "public_key": public_key,
        "secret_key": secret_key,
        "provider": provider,
        "openai_key": openai_key,
    }


def render_dataset_creator(state: Dict[str, Any]) -> None:
    st.subheader("Create Dataset")
    with st.form("create_dataset", clear_on_submit=True):
        name = st.text_input("Name", placeholder="Support flows - Nov regression")
        desc = st.text_area("Description", placeholder="Regression goldens for support chatbot")
        submitted = st.form_submit_button("Create dataset", type="primary")
    if submitted:
        if not name.strip():
            st.error("Dataset name is required.")
            return
        ds_id = str(uuid.uuid4())
        state.datasets[ds_id] = {
            "id": ds_id,
            "name": name.strip(),
            "description": desc.strip(),
            "items": [],
            "synced": False,
            "created_at": datetime.utcnow().isoformat(),
        }
        state.active_dataset = ds_id
        st.success(f"Dataset '{name}' created.")


def render_dataset_selector(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not state.datasets:
        st.info("No datasets yet. Create one to get started.")
        return None

    options = {dataset_summary(ds): ds_id for ds_id, ds in state.datasets.items()}
    selected_label = st.selectbox(
        "Select dataset",
        options=list(options.keys()),
        index=0 if state.active_dataset is None else list(options.values()).index(state.active_dataset),
    )
    ds_id = options[selected_label]
    state.active_dataset = ds_id
    return state.datasets[ds_id]


def render_items_table(ds: Dict[str, Any]) -> None:
    st.markdown("### Goldens")
    if not ds["items"]:
        st.info("Add rows below to build your golden dataset.")
        return

    df = pd.DataFrame(ds["items"])
    df = df[["input", "expected_output", "metadata", "id"]]
    df.rename(
        columns={"input": "Input", "expected_output": "Expected", "metadata": "Metadata", "id": "Item ID"},
        inplace=True,
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_add_row(ds: Dict[str, Any]) -> None:
    st.markdown("### Add Row")
    with st.form(f"add_row_{ds['id']}", clear_on_submit=True):
        input_text = st.text_area("Input (user message)", key=f"input_{ds['id']}")
        expected_text = st.text_area(
            "Expected Output (JSON)",
            placeholder='{"answer": "string"}',
            key=f"expected_{ds['id']}",
        )
        metadata_text = st.text_area(
            "Metadata (optional JSON)",
            placeholder='{"locale": "en-US"}',
            key=f"metadata_{ds['id']}",
        )
        add_btn = st.form_submit_button("Add row")
    if add_btn:
        if not input_text.strip() or not expected_text.strip():
            st.error("Input and Expected Output are required.")
            return
        item = {
            "id": str(uuid.uuid4()),
            "input": input_text.strip(),
            "expected_output": parse_json(expected_text),
            "metadata": parse_json(metadata_text) if metadata_text else None,
        }
        ds["items"].append(item)
        ds["synced"] = False
        st.success("Row added.")


def render_upload_actions(ds: Dict[str, Any], connection: Dict[str, str]) -> None:
    st.markdown("### Sync to Langfuse")
    col1, col2 = st.columns([1, 2])
    with col1:
        upload = st.button("Upload dataset", type="primary", key=f"upload_{ds['id']}")
    with col2:
        st.caption("Creates dataset + items in Langfuse via public API.")
    if upload:
        if not ds["items"]:
            st.error("Add at least one row before uploading.")
            return
        try:
            dataset_id, inserted, raw_body = upload_dataset_to_langfuse(ds, connection)
            ds["synced"] = True
            ds["langfuse_id"] = dataset_id
            st.success(
                f"Uploaded '{ds['name']}' to Langfuse at {connection['host']} "
                f"(dataset_id: {dataset_id}, items inserted: {inserted})."
            )
            if raw_body is not None:
                st.caption(f"Langfuse response: {raw_body}")
            verification = fetch_dataset_items(dataset_id, connection)
            if verification is not None:
                try:
                    items_found = len(verification.get("items", verification))
                    st.info(f"Verified items in Langfuse: {items_found}")
                except Exception:  # pylint: disable=broad-except
                    st.info(f"Verification response: {verification}")
            else:
                st.warning("Could not verify items via GET; check credentials/paths.")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Upload failed: {exc}")


def simulate_eval_run(ds: Dict[str, Any], prompt: str, model: str, temp: float, max_tokens: int) -> Dict[str, Any]:
    """Mock evaluator: echoes expected output as actual and scores exact string match."""
    run_id = str(uuid.uuid4())
    results = []
    for item in ds["items"]:
        expected = item["expected_output"]
        actual = expected  # naive echo for demo
        score = 1.0
        comment = "Matched expected output (simulated)"
        results.append(
            {
                "item_id": item["id"],
                "input": item["input"],
                "expected_output": expected,
                "actual_output": actual,
                "score": score,
                "comment": comment,
                "trace_url": f"{st.session_state.connection.get('host', '')}/trace/{item['id']}",
            }
        )
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    return {
        "id": run_id,
        "dataset_id": ds["id"],
        "dataset_name": ds["name"],
        "model": model,
        "prompt": prompt,
        "temperature": temp,
        "max_tokens": max_tokens,
        "status": "Completed",
        "created_at": datetime.utcnow().isoformat(),
        "avg_score": avg_score,
        "results": results,
    }


def run_eval_with_openai(ds: Dict[str, Any], prompt: str, model: str, temp: float, max_tokens: int, openai_key: str) -> Dict[str, Any]:
    """Call OpenAI for each row; no scoring yet."""
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    client = OpenAI(api_key=openai_key)
    run_id = str(uuid.uuid4())
    results = []
    for item in ds["items"]:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": item["input"]},
        ]
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temp,
            max_tokens=max_tokens,
        )
        actual = completion.choices[0].message.content
        results.append(
            {
                "item_id": item["id"],
                "input": item["input"],
                "expected_output": item["expected_output"],
                "actual_output": actual,
                "score": None,
                "comment": None,
                "trace_url": None,
            }
        )
    avg_score = 0.0
    return {
        "id": run_id,
        "dataset_id": ds["id"],
        "dataset_name": ds["name"],
        "model": model,
        "prompt": prompt,
        "temperature": temp,
        "max_tokens": max_tokens,
        "status": "Completed",
        "created_at": datetime.utcnow().isoformat(),
        "avg_score": avg_score,
        "results": results,
    }


def log_run_to_langfuse(run: Dict[str, Any], connection: Dict[str, str]) -> None:
    if Langfuse is None:
        st.warning("langfuse SDK not installed; skipping Langfuse logging. Run: pip install langfuse")
        return
    if not connection.get("public_key") or not connection.get("secret_key"):
        st.warning("Missing Langfuse keys; skipping Langfuse logging.")
        return
    client = Langfuse(
        public_key=connection["public_key"],
        secret_key=connection["secret_key"],
        host=connection["host"],
    )
    trace = client.trace(name="eval_run", input=run["prompt"], metadata={"model": run["model"], "dataset": run["dataset_name"]})
    for r in run["results"]:
        client.generation(
            trace_id=trace.id,
            name="eval_item",
            input=r["input"],
            output=r["actual_output"],
            expected_output=r["expected_output"],
            metadata={"dataset_item_id": r["item_id"]},
        )
    client.flush()
    # Populate trace URL for UI if project id is known.
    if connection.get("project_id"):
        base = connection["host"].rstrip("/")
        trace_url = f"{base}/project/{connection['project_id']}/traces?peek={trace.id}"
        for r in run["results"]:
            r["trace_url"] = trace_url


def build_basic_headers(connection: Dict[str, str]) -> Dict[str, str]:
    if not connection.get("public_key") or not connection.get("secret_key"):
        raise ValueError("Langfuse public/secret keys are required.")
    token = base64.b64encode(f"{connection['public_key']}:{connection['secret_key']}".encode()).decode()
    return {"Authorization": f"Basic {token}", "Content-Type": "application/json"}


def _post_with_fallback(urls: List[str], headers: Dict[str, str], payload: Dict[str, Any], timeout: int) -> requests.Response:
    last_err: Optional[Exception] = None
    for url in urls:
        if not url:
            continue
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as exc:  # pylint: disable=broad-except
            last_err = exc
            continue
    if last_err:
        raise last_err
    raise RuntimeError("No valid URL provided for Langfuse request.")


def upload_dataset_to_langfuse(ds: Dict[str, Any], connection: Dict[str, str]) -> tuple[str, int, Any]:
    """Create dataset and items via Langfuse public API (tries project-scoped then global)."""
    headers = build_basic_headers(connection)
    base_url = connection["host"].rstrip("/")
    project_id = connection.get("project_id")
    # Create dataset (try project-scoped, then default)
    payload = {"name": ds["name"], "description": ds.get("description", "")}
    create_urls = [
        f"{base_url}/api/public/projects/{project_id}/datasets" if project_id else None,
        f"{base_url}/api/public/datasets",
    ]
    resp = _post_with_fallback(create_urls, headers, payload, timeout=20)
    created = resp.json()
    dataset_id = created.get("id") or created.get("datasetId") or created.get("name")
    if not dataset_id:
        raise RuntimeError("Langfuse response missing dataset id.")
    # Upload items (try project-scoped, then default)
    items = []
    for item in ds["items"]:
        items.append(
            {
                "input": item["input"],
                "expectedOutput": item["expected_output"],
                "metadata": item.get("metadata"),
                "externalId": item["id"],
            }
        )
    items_payload = {"items": items}
    item_urls = [
        f"{base_url}/api/public/projects/{project_id}/datasets/{dataset_id}/items" if project_id else None,
        f"{base_url}/api/public/datasets/{dataset_id}/items",
    ]
    resp_items = _post_with_fallback(item_urls, headers, items_payload, timeout=30)
    # If still 404s, surface content for debugging.
    if resp_items.status_code >= 400:
        raise RuntimeError(f"Item upload failed: {resp_items.status_code} - {resp_items.text}")
    # Try to report inserted count if provided
    inserted = 0
    body: Any = None
    try:
        body = resp_items.json()
        inserted = body.get("count") or body.get("inserted") or body.get("itemsInserted") or 0
    except Exception:  # pylint: disable=broad-except
        inserted = 0
    if not inserted:
        inserted = len(ds["items"])
    return dataset_id, inserted, body


def fetch_dataset_items(dataset_id: str, connection: Dict[str, str]) -> Any:
    """Fetch items from Langfuse to verify insertion."""
    headers = build_basic_headers(connection)
    base_url = connection["host"].rstrip("/")
    project_id = connection.get("project_id")
    urls = [
        f"{base_url}/api/public/projects/{project_id}/datasets/{dataset_id}/items" if project_id else None,
        f"{base_url}/api/public/datasets/{dataset_id}/items",
    ]
    for url in urls:
        if not url:
            continue
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            continue
    return None


def render_run_eval(ds: Dict[str, Any], state: Dict[str, Any]) -> None:
    st.markdown("### Run Eval")
    with st.form(f"run_eval_{ds['id']}"):
        model = st.selectbox("Model", ["gpt-5.1", "gpt-4.1", "gpt-4o", "gpt-3.5-turbo"], index=0)
        prompt = st.text_area("System Prompt", height=160, placeholder="You are a helpful assistant...")
        col1, col2 = st.columns(2)
        with col1:
            temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        with col2:
            max_tokens = st.number_input("Max tokens", min_value=16, max_value=2048, value=512, step=16)
        run_btn = st.form_submit_button("Run Eval", type="primary")
    if run_btn:
        if not ds["items"]:
            st.error("Add at least one dataset row before running an eval.")
            return
        if not prompt.strip():
            st.error("Prompt is required.")
            return
        if state.connection.get("provider") == "OpenAI" and state.connection.get("openai_key"):
            try:
                run = run_eval_with_openai(ds, prompt, model, temp, max_tokens, state.connection["openai_key"])
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Eval failed: {exc}")
                return
        else:
            run = simulate_eval_run(ds, prompt, model, temp, max_tokens)
        state.runs.insert(0, run)
        state.active_run = run["id"]
        try:
            log_run_to_langfuse(run, state.connection)
        except Exception as exc:  # pylint: disable=broad-except
            st.warning(f"Logged locally but failed to log to Langfuse: {exc}")
        st.success(f"Eval run created (avg score {run['avg_score']:.2f}).")


def render_runs_list(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    st.subheader("Eval Runs")
    if not state.runs:
        st.info("No runs yet. Create a dataset and run an eval.")
        return None
    options = {
        f"{r['dataset_name']} • {r['model']} • {r['status']} • {r['avg_score']:.2f}": r["id"]
        for r in state.runs
    }
    selected_label = st.selectbox(
        "Select run",
        options=list(options.keys()),
        index=0 if state.active_run is None else list(options.values()).index(state.active_run),
    )
    run_id = options[selected_label]
    state.active_run = run_id
    run = next(r for r in state.runs if r["id"] == run_id)
    return run


def render_run_details(run: Dict[str, Any]) -> None:
    st.markdown("### Run Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", run["model"])
    col2.metric("Avg Score", f"{run['avg_score']:.2f}")
    col3.metric("Status", run["status"])
    col4.metric("Items", len(run["results"]))

    st.markdown("### Results")
    score_filter = st.selectbox("Filter by score", ["All", "Pass (1)", "Fail (0)"], index=0)
    search = st.text_input("Search (input or comment)")

    rows = run["results"]
    if score_filter == "Pass (1)":
        rows = [r for r in rows if r["score"] >= 1]
    elif score_filter == "Fail (0)":
        rows = [r for r in rows if r["score"] < 1]
    if search:
        rows = [
            r
            for r in rows
            if search.lower() in r["input"].lower()
            or search.lower() in str(r.get("comment", "")).lower()
        ]

    if not rows:
        st.info("No rows match the current filters.")
        return

    for r in rows:
        with st.expander(f"Input preview: {r['input'][:60]}"):
            st.markdown("**Input**")
            st.write(r["input"])
            st.markdown("**Expected Output**")
            st.json(r["expected_output"])
            st.markdown("**Actual Output**")
            st.json(r["actual_output"])
            st.markdown(f"**Score:** {r['score'] if r['score'] is not None else '—'}")
            st.markdown(f"**Comment:** {r.get('comment', '—')}")
            if r.get("trace_url"):
                st.markdown(f"[Langfuse Trace]({r['trace_url']})")


# ---------- App layout ----------
connection = sidebar_connection()
state = get_state()
state.connection = connection  # store for simulation

st.title("Langfuse Evals Tool (V1)")
st.caption("Create goldens, sync to Langfuse, run evals with custom prompts, and review results.")

tab_datasets, tab_runs = st.tabs(["Datasets", "Eval Runs"])

with tab_datasets:
    render_dataset_creator(state)
    ds = render_dataset_selector(state)
    if ds:
        st.markdown(f"**Description:** {ds.get('description') or '—'}")
        render_items_table(ds)
        render_add_row(ds)
        render_upload_actions(ds, connection)
        render_run_eval(ds, state)

with tab_runs:
    run = render_runs_list(state)
    if run:
        render_run_details(run)
