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
from pathlib import Path
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

try:
    from facebook_catalog_integration import CatalogItem, FacebookCatalogClient, FacebookCatalogError
except Exception:  # pylint: disable=broad-except
    CatalogItem = None  # type: ignore
    FacebookCatalogClient = None  # type: ignore
    FacebookCatalogError = None  # type: ignore


class InteraktError(RuntimeError):
    """Raised when Interakt API responds with an error payload or bad status."""


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
    if "interakt_products" not in st.session_state:
        st.session_state.interakt_products = []  # cached catalog products for Interakt messages
    if "persisted_sidebar" not in st.session_state:
        st.session_state.persisted_sidebar = load_persisted_sidebar()
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


def load_persisted_sidebar() -> Dict[str, str]:
    """Load persisted sidebar values from disk (plain text)."""
    path = Path(".streamlit/sidebar_state.json")
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def persist_sidebar(values: Dict[str, str]) -> None:
    """Persist sidebar values to disk (plain text)."""
    path = Path(".streamlit/sidebar_state.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    current = st.session_state.get("persisted_sidebar", {}) or {}
    current.update(values)
    st.session_state.persisted_sidebar = current
    path.write_text(json.dumps(current))


def send_interakt_product_message(
    access_token: str,
    waba_id: str,
    phone_number: str,
    phone_number_id: str,
    catalog_id: str,
    product_retailer_id: str,
    body_text: str = "",
    footer_text: str = "",
    timeout: int = 15,
) -> Dict[str, Any]:
    """
    Send a WhatsApp product message via Interakt's /phone_no_id/messages endpoint.
    """

    if not phone_number_id:
        raise InteraktError("Phone number ID is required to send messages.")

    url = f"https://amped-express.interakt.ai/api/v17.0/{phone_number_id}/messages"
    headers = {
        "x-access-token": access_token,
        "x-waba-id": waba_id,
        "Content-Type": "application/json",
    }
    interactive: Dict[str, Any] = {
        "type": "product",
        "action": {
            "catalog_id": catalog_id,
            "product_retailer_id": product_retailer_id,
        },
    }
    if body_text.strip():
        interactive["body"] = {"text": body_text.strip()}
    if footer_text.strip():
        interactive["footer"] = {"text": footer_text.strip()}

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": phone_number,
        "type": "interactive",
        "interactive": interactive,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        data = resp.json()
    except ValueError as exc:  # non-JSON response
        resp.raise_for_status()
        raise InteraktError("Interakt API returned a non-JSON response") from exc

    if not resp.ok:
        message = None
        if isinstance(data, dict):
            message = data.get("error") or data.get("message") or data.get("detail")
        if not message:
            message = f"Interakt API error: {resp.status_code}"
        raise InteraktError(message or f"Interakt API error: {resp.status_code}")
    if not isinstance(data, dict):
        raise InteraktError("Interakt API returned an unexpected payload shape")
    return data


def send_graph_message(
    access_token: str,
    phone_number_id: str,
    payload: Dict[str, Any],
    api_version: str = "v19.0",
    timeout: int = 15,
) -> Dict[str, Any]:
    if not phone_number_id:
        raise FacebookCatalogError("Phone number ID is required for Graph sends.")
    url = f"https://graph.facebook.com/{api_version}/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        data = resp.json()
    except ValueError as exc:
        resp.raise_for_status()
        raise FacebookCatalogError("Graph API returned a non-JSON response") from exc
    if not resp.ok or (isinstance(data, dict) and data.get("error")):
        message = None
        if isinstance(data, dict):
            err = data.get("error") or {}
            message = err.get("message") or data.get("message")
        raise FacebookCatalogError(message or f"Graph API error: {resp.status_code}")
    if not isinstance(data, dict):
        raise FacebookCatalogError("Graph API returned an unexpected payload shape")
    return data


def interakt_request(
    method: str,
    path: str,
    access_token: str,
    waba_id: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 15,
) -> Dict[str, Any]:
    """Generic Interakt wrapper."""
    base_url = "https://amped-express.interakt.ai/api/v17.0"
    url = f"{base_url}/{path.lstrip('/')}"
    headers = {
        "x-access-token": access_token,
        "x-waba-id": waba_id,
        "Content-Type": "application/json",
    }
    resp = requests.request(method, url, headers=headers, params=params or {}, timeout=timeout)
    try:
        data = resp.json()
    except ValueError as exc:  # non-JSON response
        resp.raise_for_status()
        raise InteraktError("Interakt API returned a non-JSON response") from exc
    if not resp.ok:
        message = None
        if isinstance(data, dict):
            message = data.get("error") or data.get("message") or data.get("detail")
        raise InteraktError(message or f"Interakt API error: {resp.status_code}")
    if not isinstance(data, dict):
        raise InteraktError("Interakt API returned an unexpected payload shape")
    return data


def interakt_list_catalogs(
    access_token: str,
    waba_id: str,
    phone_number_id: str,
    timeout: int = 15,
) -> Dict[str, Any]:
    """List catalogs linked to a phone number via Interakt/Cloud API."""
    if not phone_number_id:
        raise InteraktError("Phone number ID is required to list catalogs.")
    return interakt_request(
        "GET",
        f"{phone_number_id}/product_catalogs",
        access_token=access_token,
        waba_id=waba_id,
        timeout=timeout,
    )


def interakt_list_catalog_products(
    access_token: str,
    waba_id: str,
    catalog_id: str,
    limit: int = 25,
    timeout: int = 15,
) -> Dict[str, Any]:
    """List products for a catalog via Interakt/Cloud API."""
    if not catalog_id:
        raise InteraktError("Catalog ID is required to list products.")
    params = {"limit": limit, "fields": "id,name,retailer_id,price,availability,brand,condition"}
    return interakt_request(
        "GET",
        f"{catalog_id}/products",
        access_token=access_token,
        waba_id=waba_id,
        params=params,
        timeout=timeout,
    )


def send_interakt_product_list_message(
    access_token: str,
    waba_id: str,
    phone_number_id: str,
    phone_number: str,
    catalog_id: str,
    product_retailer_ids: List[str],
    header_text: str,
    body_text: str,
    footer_text: str = "",
    section_title: str = "Products",
    timeout: int = 15,
) -> Dict[str, Any]:
    """
    Send a WhatsApp multi-product message (product_list) via Interakt.
    """
    if not phone_number_id:
        raise InteraktError("Phone number ID is required to send messages.")
    if not product_retailer_ids:
        raise InteraktError("Select at least one product to send.")
    if not header_text.strip():
        raise InteraktError("Header text is required for product list messages.")
    if not body_text.strip():
        raise InteraktError("Body text is required for product list messages.")
    if not section_title.strip():
        raise InteraktError("Section title is required for product list messages.")

    url = f"https://amped-express.interakt.ai/api/v17.0/{phone_number_id}/messages"
    headers = {
        "x-access-token": access_token,
        "x-waba-id": waba_id,
        "Content-Type": "application/json",
    }

    interactive: Dict[str, Any] = {
        "type": "product_list",
        "header": {"type": "text", "text": header_text.strip()},
        "body": {"text": body_text.strip()},
        "action": {
            "catalog_id": catalog_id,
            "sections": [
                {
                    "title": section_title.strip(),
                    "product_items": [{"product_retailer_id": pid} for pid in product_retailer_ids],
                }
            ],
        },
    }
    if footer_text.strip():
        interactive["footer"] = {"text": footer_text.strip()}

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": phone_number,
        "type": "interactive",
        "interactive": interactive,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        data = resp.json()
    except ValueError as exc:  # non-JSON response
        resp.raise_for_status()
        raise InteraktError("Interakt API returned a non-JSON response") from exc

    if not resp.ok:
        message = None
        if isinstance(data, dict):
            message = data.get("error") or data.get("message") or data.get("detail")
        if not message:
            message = f"Interakt API error: {resp.status_code}"
        raise InteraktError(message)
    if not isinstance(data, dict):
        raise InteraktError("Interakt API returned an unexpected payload shape")
    return data


def send_interakt_catalog_message(
    access_token: str,
    waba_id: str,
    phone_number_id: str,
    phone_number: str,
    body_text: str,
    footer_text: str = "",
    thumbnail_product_retailer_id: Optional[str] = None,
    timeout: int = 15,
) -> Dict[str, Any]:
    """
    Send a WhatsApp catalog message (shows entire catalog) via Interakt.
    """
    if not phone_number_id:
        raise InteraktError("Phone number ID is required to send messages.")
    if not body_text.strip():
        raise InteraktError("Body text is required for catalog messages.")

    url = f"https://amped-express.interakt.ai/api/v17.0/{phone_number_id}/messages"
    headers = {
        "x-access-token": access_token,
        "x-waba-id": waba_id,
        "Content-Type": "application/json",
    }

    action: Dict[str, Any] = {"name": "catalog_message"}
    if thumbnail_product_retailer_id:
        action["parameters"] = {"thumbnail_product_retailer_id": thumbnail_product_retailer_id}

    interactive: Dict[str, Any] = {
        "type": "catalog_message",
        "body": {"text": body_text.strip()},
        "action": action,
    }
    if footer_text.strip():
        interactive["footer"] = {"text": footer_text.strip()}

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": phone_number,
        "type": "interactive",
        "interactive": interactive,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        data = resp.json()
    except ValueError as exc:  # non-JSON response
        resp.raise_for_status()
        raise InteraktError("Interakt API returned a non-JSON response") from exc

    if not resp.ok:
        message = None
        if isinstance(data, dict):
            message = data.get("error") or data.get("message") or data.get("detail")
        if not message:
            message = f"Interakt API error: {resp.status_code}"
        raise InteraktError(message)
    if not isinstance(data, dict):
        raise InteraktError("Interakt API returned an unexpected payload shape")
    return data


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


def sidebar_facebook_catalog() -> Dict[str, str]:
    st.sidebar.header("Facebook Catalog")
    st.sidebar.caption("Values persist locally in .streamlit/sidebar_state.json (plain text).")
    persisted = st.session_state.get("persisted_sidebar", {})
    fb_token = st.sidebar.text_input(
        "Access Token",
        value=persisted.get("facebook.access_token", st.secrets.get("facebook_access_token", "")),
        type="password",
        key="facebook_access_token_input",
    )
    fb_catalog_id = st.sidebar.text_input(
        "Catalog ID",
        value=persisted.get("facebook.catalog_id", st.secrets.get("facebook_catalog_id", "")),
        key="facebook_catalog_id_input",
    )
    api_version = st.sidebar.text_input(
        "Graph API version",
        value=persisted.get("facebook.api_version", "v19.0"),
        key="facebook_api_version_input",
    )
    fb_phone_number_id = st.sidebar.text_input(
        "WhatsApp Phone Number ID (Graph send)",
        value=persisted.get("facebook.phone_number_id", st.secrets.get("facebook_phone_number_id", "")),
        help="Used for sending via WhatsApp Cloud API.",
        key="facebook_phone_number_id_input",
    )
    persist_sidebar(
        {
            "facebook.access_token": fb_token,
            "facebook.catalog_id": fb_catalog_id,
            "facebook.api_version": api_version,
            "facebook.phone_number_id": fb_phone_number_id,
        }
    )
    return {"token": fb_token, "catalog_id": fb_catalog_id, "api_version": api_version, "phone_number_id": fb_phone_number_id}


def sidebar_interakt(fb_config: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    st.sidebar.header("WhatsApp Product Message")
    st.sidebar.caption("Values persist locally in .streamlit/sidebar_state.json (plain text).")
    persisted = st.session_state.get("persisted_sidebar", {})
    access_token = st.sidebar.text_input(
        "Interakt Access Token",
        value=persisted.get("interakt.access_token", st.secrets.get("interakt_access_token", "")),
        type="password",
        key="interakt_access_token_input",
    )
    waba_id = st.sidebar.text_input(
        "WABA ID",
        value=persisted.get("interakt.waba_id", st.secrets.get("interakt_waba_id", "")),
        key="interakt_waba_id_input",
    )
    phone_number_id = st.sidebar.text_input(
        "Phone number ID (sender)",
        value=persisted.get("interakt.phone_number_id", st.secrets.get("interakt_phone_number_id", "")),
        help="Sender phone number ID (same as Graph API phone_number_id).",
        key="interakt_phone_number_id_input",
    )
    phone_number = st.sidebar.text_input(
        "Recipient phone (+E.164)",
        value=persisted.get("interakt.phone_number", st.secrets.get("interakt_phone_number", "")),
        help="Destination WhatsApp number.",
        key="interakt_phone_number_input",
    )
    default_catalog = (
        persisted.get("interakt.catalog_id")
        or st.secrets.get("interakt_catalog_id", "")
        or (fb_config or {}).get("catalog_id", "")
    )
    catalog_id = st.sidebar.text_input(
        "Catalog ID for messages",
        value=default_catalog,
        help="Catalog ID used in the product message payload.",
        key="interakt_catalog_id_input",
    )
    body_text = st.sidebar.text_area(
        "Body text",
        value=persisted.get("interakt.body_text", st.secrets.get("interakt_body_text", "")),
        placeholder="optional body text",
        key="interakt_body_text_input",
    )
    footer_text = st.sidebar.text_input(
        "Footer text",
        value=persisted.get("interakt.footer_text", st.secrets.get("interakt_footer_text", "")),
        placeholder="optional footer text",
        key="interakt_footer_text_input",
    )
    persist_sidebar(
        {
            "interakt.access_token": access_token,
            "interakt.waba_id": waba_id,
            "interakt.phone_number_id": phone_number_id,
            "interakt.phone_number": phone_number,
            "interakt.catalog_id": catalog_id,
            "interakt.body_text": body_text,
            "interakt.footer_text": footer_text,
        }
    )
    return {
        "access_token": access_token,
        "waba_id": waba_id,
        "phone_number_id": phone_number_id,
        "phone_number": phone_number,
        "catalog_id": catalog_id,
        "body_text": body_text,
        "footer_text": footer_text,
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


def render_facebook_catalog_view(config: Dict[str, str], interakt_config: Dict[str, str]) -> None:
    st.subheader("Facebook Catalog")
    st.caption("Browse via Facebook Graph token or Interakt Cloud API (new option).")
    if FacebookCatalogClient is None or CatalogItem is None:
        st.error("facebook_catalog_integration module not available. Ensure the file exists.")
        return

    tab_browse, tab_add, tab_message = st.tabs(["Browse products", "Add product", "Send product message"])

    with tab_browse:
        col1, col2 = st.columns([1, 3])
        with col1:
            limit = st.number_input("Items to fetch", min_value=1, max_value=200, value=25, step=5)
        with col2:
            fields = st.text_input(
                "Fields (comma-separated)",
                value="id,name,retailer_id,price,availability,brand,inventory,condition",
                help="Fields returned per product item.",
            )
        source = st.radio(
            "Catalog source",
            ["Facebook Graph API token", "Interakt (Cloud API token)"],
            index=0,
            horizontal=True,
        )
        fetch_btn = st.button("Fetch products", type="primary", key="fetch_products_btn")
        if fetch_btn:
            try:
                if source == "Facebook Graph API token":
                    if not config.get("token") or not config.get("catalog_id"):
                        st.error("Enter Access Token and Catalog ID in the sidebar for Graph API browsing.")
                        return
                    client = FacebookCatalogClient(
                        access_token=config["token"],
                        catalog_id=config["catalog_id"],
                        api_version=config["api_version"],
                    )
                    with st.spinner("Calling Facebook Graph API..."):
                        resp = client.list_products(
                            limit=limit, fields=[f.strip() for f in fields.split(",") if f.strip()]
                        )
                    st.caption(f"Source: Graph API catalog {config['catalog_id']}")
                else:
                    if not interakt_config.get("access_token") or not interakt_config.get("waba_id"):
                        st.error("Fill Interakt Access Token and WABA ID in the sidebar for Cloud API browsing.")
                        return
                    if not interakt_config.get("phone_number_id"):
                        st.error("Fill Phone number ID (sender) in the sidebar for Cloud API browsing.")
                        return
                    catalog_id = interakt_config.get("catalog_id") or config.get("catalog_id") or ""
                    # If no catalog provided, try to fetch via phone number ID
                    if not catalog_id:
                        with st.spinner("Fetching catalogs via Interakt..."):
                            catalogs = interakt_list_catalogs(
                                access_token=interakt_config["access_token"],
                                waba_id=interakt_config["waba_id"],
                                phone_number_id=interakt_config["phone_number_id"],
                            )
                        catalog_data = catalogs.get("data") if isinstance(catalogs, dict) else None
                        first_catalog = catalog_data[0] if catalog_data else None
                        catalog_id = first_catalog.get("id") if isinstance(first_catalog, dict) else None
                        if catalog_id:
                            # cache locally so it persists
                            persist_sidebar({"interakt.catalog_id": catalog_id})
                        else:
                            st.error("No catalog_id provided and none returned for this phone number.")
                            return
                    with st.spinner("Fetching products via Interakt Cloud API..."):
                        resp = interakt_list_catalog_products(
                            access_token=interakt_config["access_token"],
                            waba_id=interakt_config["waba_id"],
                            catalog_id=catalog_id,
                            limit=limit,
                        )
                    st.caption(f"Source: Interakt catalog {catalog_id}")
            except (FacebookCatalogError, InteraktError, ValueError) as exc:
                st.error(f"Catalog request failed: {exc}")
                return
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Unexpected error: {exc}")
                return

            data = resp.get("data") if isinstance(resp, dict) else None
            if not data:
                st.info("No products returned for this catalog.")
                return

            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            paging = resp.get("paging", {}) if isinstance(resp, dict) else {}
            if paging.get("next"):
                st.caption("More products available. Increase limit or use the paging cursor.")
            if paging.get("cursors", {}).get("after"):
                st.code(f"Next cursor: {paging['cursors']['after']}", language="text")

    with tab_add:
        render_facebook_add_product(config)

    with tab_message:
        render_interakt_product_message(config, interakt_config)


def render_facebook_add_product(config: Dict[str, str]) -> None:
    st.markdown("### Create a product")
    st.caption("Minimal fields required: retailer ID, name, description, URL, image URL, currency, price.")
    with st.form("fb_add_product"):
        retailer_id = st.text_input("Retailer ID (SKU)")
        name = st.text_input("Name")
        description = st.text_area("Description", height=80)
        url = st.text_input("Product URL")
        image_url = st.text_input("Main image URL")
        currency = st.text_input("Currency (ISO 4217)", value="USD", max_chars=3)
        price = st.text_input("Price (e.g., 19.99)")
        col1, col2, col3 = st.columns(3)
        with col1:
            availability = st.selectbox("Availability", ["in stock", "out of stock", "preorder"], index=0)
        with col2:
            condition = st.selectbox("Condition", ["new", "used", "refurbished"], index=0)
        with col3:
            brand = st.text_input("Brand", value="")
        inventory = st.number_input("Inventory", min_value=0, value=0, step=1)
        category = st.text_input("Category (optional)")
        additional_images_text = st.text_area(
            "Additional image URLs (one per line)", value="", height=80, placeholder="https://..."
        )
        skip_validation = st.checkbox("Skip validation (Graph API)", value=False)
        submit = st.form_submit_button("Create product", type="primary")

    if not submit:
        return

    required_fields = [retailer_id, name, description, url, image_url, currency, price]
    if any(not f.strip() for f in required_fields):
        st.error("Please fill all required fields.")
        return

    additional_images = [line.strip() for line in additional_images_text.splitlines() if line.strip()]
    try:
        client = FacebookCatalogClient(
            access_token=config["token"],
            catalog_id=config["catalog_id"],
            api_version=config["api_version"],
        )
        item = CatalogItem(
            retailer_id=retailer_id.strip(),
            name=name.strip(),
            description=description.strip(),
            url=url.strip(),
            image_url=image_url.strip(),
            currency=currency.strip(),
            price=price.strip(),
            brand=brand.strip() or None,
            inventory=inventory,
            category=category.strip() or None,
            additional_image_urls=additional_images or None,
            availability=availability,
            condition=condition,
        )
        with st.spinner("Creating product..."):
            resp = client.create_product(item, skip_validation=skip_validation)
        st.success("Product created successfully.")
        st.json(resp)
    except (FacebookCatalogError, ValueError) as exc:
        st.error(f"Create product failed: {exc}")
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Unexpected error: {exc}")


def render_interakt_product_message(config: Dict[str, str], interakt_config: Dict[str, str]) -> None:
    st.markdown("### Send WhatsApp product message (Interakt)")
    st.caption("Pick a product from your Facebook catalog and send it as a WhatsApp product message.")

    missing = [
        label
        for label, val in [
            ("Interakt Access Token", interakt_config.get("access_token", "")),
            ("WABA ID", interakt_config.get("waba_id", "")),
            ("Phone number ID (sender)", interakt_config.get("phone_number_id", "")),
            ("Recipient phone", interakt_config.get("phone_number", "")),
        ]
        if not str(val or "").strip()
    ]
    if missing:
        st.info(f"Fill these in the left sidebar before sending: {', '.join(missing)}.")

    col1, col2 = st.columns([1, 2])
    with col1:
        limit = st.number_input("Items to fetch", min_value=1, max_value=200, value=25, step=5, key="interakt_fetch_limit")
    with col2:
        st.write(" ")  # spacer
        fetch_btn = st.button("Load catalog items", type="secondary")

    if fetch_btn:
        try:
            client = FacebookCatalogClient(
                access_token=config["token"],
                catalog_id=config["catalog_id"],
                api_version=config["api_version"],
            )
            with st.spinner("Loading catalog products..."):
                resp = client.list_products(
                    limit=limit, fields=["id", "name", "retailer_id", "price", "availability", "brand"]
                )
            data = resp.get("data") if isinstance(resp, dict) else None
            if not data:
                st.info("No products returned for this catalog.")
                st.session_state.interakt_products = []
                return
            st.session_state.interakt_products = data
        except (FacebookCatalogError, ValueError) as exc:
            st.error(f"Failed to load products: {exc}")
            return
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Unexpected error: {exc}")
            return

    products = st.session_state.interakt_products
    if not products:
        st.info("Load products from the catalog to pick one.")
        return

    options = {
        f"{p.get('name', '(no name)')} • {p.get('retailer_id', p.get('id', '')) or 'no retailer_id'}": p
        for p in products
    }
    product_list = []
    for label, p in options.items():
        if not p:
            continue
        raw_id = p.get("retailer_id") or p.get("id") or ""
        raw_id = str(raw_id).strip()
        if raw_id:
            product_list.append((label, raw_id))
    product_list = [item for item in product_list if item[1]]
    if not product_list:
        st.error("No products with retailer_id found in the catalog.")
        return

    provider = st.radio(
        "Send via",
        ["Interakt", "Graph Cloud API"],
        index=0,
        key="send_provider",
    )
    message_type = st.radio(
        "Message type",
        ["Single product", "Product list (variants)", "Catalog message"],
        index=0,
        key="interakt_message_type",
    )
    catalog_id = interakt_config.get("catalog_id") or config.get("catalog_id") or ""
    if not catalog_id:
        missing.append("Catalog ID (sidebar or Facebook)")
    body_text = interakt_config.get("body_text", "")
    footer_text = interakt_config.get("footer_text", "")

    if message_type == "Single product":
        labels = [lbl for lbl, _ in product_list]
        selected_label = st.selectbox(
            "Select product",
            options=labels,
            index=0,
            key="single_product_select",
        )
        product_retailer_id = dict(product_list)[selected_label]
        st.caption(f"Sending with Catalog ID: `{catalog_id}` and product_retailer_id: `{product_retailer_id}`")
        send_btn = st.button("Send WhatsApp product message", type="primary", key="send_single_product")
        if not send_btn:
            return
        if missing or not catalog_id:
            st.error(
                "Please complete the Interakt fields in the sidebar before sending "
                f"(Catalog ID used: {catalog_id or 'missing'})."
            )
            return
        try:
            if provider == "Interakt":
                with st.spinner("Sending product message via Interakt..."):
                    resp = send_interakt_product_message(
                        access_token=interakt_config["access_token"],
                        waba_id=interakt_config["waba_id"],
                        phone_number_id=interakt_config.get("phone_number_id") or "",
                        phone_number=interakt_config["phone_number"],
                        catalog_id=catalog_id,
                        product_retailer_id=product_retailer_id,
                        body_text=body_text,
                        footer_text=footer_text,
                    )
            else:
                if not config.get("token") or not config.get("phone_number_id"):
                    st.error("Fill Graph Access Token and Phone Number ID in the sidebar for Graph sends.")
                    return
                payload = {
                    "messaging_product": "whatsapp",
                    "recipient_type": "individual",
                    "to": interakt_config["phone_number"],
                    "type": "interactive",
                    "interactive": {
                        "type": "product",
                        "body": {"text": body_text or " "},
                        "footer": {"text": footer_text} if footer_text else None,
                        "action": {"catalog_id": catalog_id, "product_retailer_id": product_retailer_id},
                    },
                }
                # Clean None
                interactive = payload["interactive"]
                if not footer_text:
                    interactive.pop("footer", None)
                with st.spinner("Sending product message via Graph..."):
                    resp = send_graph_message(
                        access_token=config["token"],
                        phone_number_id=config.get("phone_number_id") or "",
                        payload=payload,
                        api_version=config.get("api_version", "v19.0"),
                    )
            st.success("Product message sent.")
            st.json(resp)
        except (InteraktError, FacebookCatalogError) as exc:
            st.error(f"Send failed: {exc}")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Unexpected error: {exc}")
        return

    # Multi-product / variants
    labels = [lbl for lbl, _ in product_list]
    default_selection = labels[:2] if len(labels) >= 2 else labels[:1]
    selected_labels = st.multiselect(
        "Select products (variants)",
        options=labels,
        default=default_selection,
        key="product_list_select",
    )
    selected_ids = [dict(product_list)[lbl] for lbl in selected_labels if lbl in dict(product_list)]
    if catalog_id:
        st.caption(f"Catalog ID for send: `{catalog_id}`")
    header_text = st.text_input("Header text (required)", value="Products", key="product_list_header")
    section_title = st.text_input("Section title", value="Items", key="product_list_section_title")
    body_input = st.text_area(
        "Body text (required)",
        value=body_text or "Browse products",
        key="product_list_body",
        height=80,
    )
    footer_input = st.text_input(
        "Footer text (optional)",
        value=footer_text,
        key="product_list_footer",
    )
    send_multi = st.button("Send WhatsApp product list message", type="primary", key="send_product_list")
    if not send_multi:
        return
    if missing or not catalog_id:
        st.error(
            "Please complete the Interakt fields in the sidebar before sending "
            f"(Catalog ID used: {catalog_id or 'missing'})."
        )
        return
    if not selected_ids:
        st.error("Pick at least one product to include in the list.")
        return

    try:
        if provider == "Interakt":
            with st.spinner("Sending product list message via Interakt..."):
                resp = send_interakt_product_list_message(
                    access_token=interakt_config["access_token"],
                    waba_id=interakt_config["waba_id"],
                    phone_number_id=interakt_config.get("phone_number_id") or "",
                    phone_number=interakt_config["phone_number"],
                    catalog_id=catalog_id,
                    product_retailer_ids=selected_ids,
                    header_text=header_text,
                    body_text=body_input,
                    footer_text=footer_input,
                    section_title=section_title,
                )
        else:
            if not config.get("token") or not config.get("phone_number_id"):
                st.error("Fill Graph Access Token and Phone Number ID in the sidebar for Graph sends.")
                return
            interactive: Dict[str, Any] = {
                "type": "product_list",
                "header": {"type": "text", "text": header_text.strip()},
                "body": {"text": body_input.strip()},
                "action": {
                    "catalog_id": catalog_id,
                    "sections": [
                        {
                            "title": section_title.strip(),
                            "product_items": [{"product_retailer_id": pid} for pid in selected_ids],
                        }
                    ],
                },
            }
            if footer_input.strip():
                interactive["footer"] = {"text": footer_input.strip()}
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": interakt_config["phone_number"],
                "type": "interactive",
                "interactive": interactive,
            }
            with st.spinner("Sending product list message via Graph..."):
                resp = send_graph_message(
                    access_token=config["token"],
                    phone_number_id=config.get("phone_number_id") or "",
                    payload=payload,
                    api_version=config.get("api_version", "v19.0"),
                )
        st.success("Product list message sent.")
        st.json(resp)
    except (InteraktError, FacebookCatalogError) as exc:
        st.error(f"Send failed: {exc}")
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Unexpected error: {exc}")
    return

    # Catalog message (full catalog)
    if message_type == "Catalog message":
        thumbnail_options = ["None"] + [lbl for lbl, _ in product_list]
        selected_thumb_label = st.selectbox(
            "Optional thumbnail product",
            options=thumbnail_options,
            index=0,
            key="catalog_message_thumb",
        )
        thumbnail_id = None
        if selected_thumb_label != "None":
            thumbnail_id = dict(product_list).get(selected_thumb_label)
        body_val = st.text_area(
            "Body text (required)",
            value=body_text or "Browse our catalog",
            height=80,
            key="catalog_message_body",
        )
        footer_val = st.text_input(
            "Footer text (optional)",
            value=footer_text,
            key="catalog_message_footer",
        )
        send_catalog = st.button("Send WhatsApp catalog message", type="primary", key="send_catalog_message")
        if not send_catalog:
            return
        if missing:
            st.error("Please complete the Interakt fields in the sidebar before sending.")
            return
        try:
            if provider == "Interakt":
                with st.spinner("Sending catalog message via Interakt..."):
                    resp = send_interakt_catalog_message(
                        access_token=interakt_config["access_token"],
                        waba_id=interakt_config["waba_id"],
                        phone_number_id=interakt_config.get("phone_number_id") or "",
                        phone_number=interakt_config["phone_number"],
                        body_text=body_val,
                        footer_text=footer_val,
                        thumbnail_product_retailer_id=thumbnail_id,
                    )
            else:
                if not config.get("token") or not config.get("phone_number_id"):
                    st.error("Fill Graph Access Token and Phone Number ID in the sidebar for Graph sends.")
                    return
                interactive: Dict[str, Any] = {
                    "type": "catalog_message",
                    "body": {"text": body_val.strip()},
                    "action": {"name": "catalog_message"},
                }
                if footer_val.strip():
                    interactive["footer"] = {"text": footer_val.strip()}
                if thumbnail_id:
                    interactive["action"]["parameters"] = {"thumbnail_product_retailer_id": thumbnail_id}
                payload = {
                    "messaging_product": "whatsapp",
                    "recipient_type": "individual",
                    "to": interakt_config["phone_number"],
                    "type": "interactive",
                    "interactive": interactive,
                }
                with st.spinner("Sending catalog message via Graph..."):
                    resp = send_graph_message(
                        access_token=config["token"],
                        phone_number_id=config.get("phone_number_id") or "",
                        payload=payload,
                        api_version=config.get("api_version", "v19.0"),
                    )
            st.success("Catalog message sent.")
            st.json(resp)
        except (InteraktError, FacebookCatalogError) as exc:
            st.error(f"Send failed: {exc}")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Unexpected error: {exc}")


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


def render_langfuse_section(state: Dict[str, Any], connection: Dict[str, str]) -> None:
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


def render_home_cards() -> None:
    st.subheader("Choose a workspace")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Langfuse Evals")
        st.caption("Create goldens, sync to Langfuse, and run evals.")
        if st.button("Open Langfuse tools", key="open_langfuse"):
            st.session_state.main_view = "Langfuse"
            st.experimental_rerun()
    with col2:
        st.markdown("### Facebook Catalog")
        st.caption("Connect with an access token and view catalog products.")
        if st.button("Open Catalog viewer", key="open_catalog"):
            st.session_state.main_view = "Facebook Catalog"
            st.experimental_rerun()


# ---------- App layout ----------
state = get_state()
connection = sidebar_connection()
facebook_config = sidebar_facebook_catalog()
interakt_config = sidebar_interakt(facebook_config)
state.connection = connection  # store for simulation
if "main_view" not in st.session_state:
    st.session_state.main_view = "Home"

st.title("Workspace")
st.caption("Pick a workspace below: Langfuse eval tools or Facebook Catalog viewer.")

nav_options = ["Home", "Langfuse", "Facebook Catalog"]
current_view = st.session_state.get("main_view", "Home")
try:
    default_index = nav_options.index(current_view)
except ValueError:
    default_index = 0
selected_view = st.radio("Navigation", nav_options, horizontal=True, index=default_index)
if selected_view != current_view:
    st.session_state.main_view = selected_view

if st.session_state.main_view == "Home":
    render_home_cards()
elif st.session_state.main_view == "Langfuse":
    render_langfuse_section(state, connection)
elif st.session_state.main_view == "Facebook Catalog":
    render_facebook_catalog_view(facebook_config, interakt_config)
