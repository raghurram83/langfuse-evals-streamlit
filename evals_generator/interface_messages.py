"""
Hybrid pipeline for generating WhatsApp interface messages.

Entry point: generate_interface_messages(raw_message: str) -> dict

Flow:
1) Preprocess + feature extraction
2) Rule-based type decision
3) Fallback LLM type classifier (small model) if needed
4) Deterministic JSON builders per type
5) Validator + safe text fallback
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from openai import OpenAI

# Allowed message types
MESSAGE_TYPES = {"text", "interactive_buttons", "interactive_list", "media"}

# Small model for classification; can be overridden in tests.
CLASSIFIER_MODEL = "gpt-4.1-mini"


@dataclass
class Features:
    media_urls: List[str]
    choice_phrases: bool
    asked_inputs: bool
    bullet_lines: List[str]
    action_lines: List[str]


def normalize_message(raw_message: str) -> str:
    """Normalize whitespace while preserving emojis and line breaks."""
    if not isinstance(raw_message, str):
        return ""
    return "\n".join(line.strip() for line in raw_message.strip().splitlines())


def extract_media_urls(text: str) -> List[str]:
    """Extract media URLs with supported extensions."""
    media_exts = (
        "jpg",
        "jpeg",
        "png",
        "gif",
        "webp",
        "mp4",
        "mov",
        "avi",
        "mpeg",
        "pdf",
    )
    pattern = rf"https?://[^\s]+?\.(?:{'|'.join(media_exts)})(?:\b|$)"
    urls = re.findall(pattern, text, flags=re.IGNORECASE)
    return urls


def detect_choice_phrases(text: str) -> bool:
    phrases = [
        "please choose",
        "select an option",
        "pick one",
        "menu:",
        "choose from",
        "tell me which",
        "choose an option",
    ]
    lower = text.lower()
    return any(p in lower for p in phrases)


def detect_inputs_requested(text: str) -> bool:
    keywords = [
        "name",
        "phone",
        "email",
        "e-mail",
        "address",
        "order id",
        "order number",
        "model number",
        "tracking",
    ]
    lower = text.lower()
    return any(k in lower for k in keywords)


def extract_bullet_lines(text: str) -> List[str]:
    """Return lines that look like bullets or menu items."""
    lines = []
    for line in text.splitlines():
        if re.match(r"^\s*[-*••●▪️▶️➤➜➔➡️►]", line):
            lines.append(line.strip("-*• \t"))
    return [ln for ln in lines if ln.strip()]


def extract_action_lines(text: str) -> List[str]:
    """Heuristic: short lines (<=40 chars) that look like actions."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    action_keywords = ["buy", "track", "order", "support", "contact", "talk", "upgrade", "pay", "cancel", "retry"]
    actions: List[str] = []
    for ln in lines:
        if len(ln) <= 40 and any(kw in ln.lower() for kw in action_keywords):
            actions.append(ln)
    return actions


def extract_features(raw_message: str) -> Features:
    normalized = normalize_message(raw_message)
    return Features(
        media_urls=extract_media_urls(normalized),
        choice_phrases=detect_choice_phrases(normalized),
        asked_inputs=detect_inputs_requested(normalized),
        bullet_lines=extract_bullet_lines(normalized),
        action_lines=extract_action_lines(normalized),
    )


def rule_based_message_type(raw_message: str, features: Features) -> Optional[str]:
    """Return a type based on deterministic rules, or None if undecided."""
    if features.media_urls:
        return "media"
    if features.asked_inputs and not features.choice_phrases:
        return "text"
    if features.choice_phrases and len(features.bullet_lines) >= 2:
        return "interactive_list"
    if 1 <= len(features.action_lines) <= 3 and not features.choice_phrases:
        return "interactive_buttons"
    return None


def llm_message_type_classifier(raw_message: str, client: Optional[OpenAI] = None) -> str:
    """
    Call a small model to classify the message type.
    Returns one of: text, interactive_buttons, interactive_list, media.
    """
    client = client or OpenAI()
    system_prompt = (
        "You are a classifier. Given a WhatsApp message, return ONLY the best message type as one of: "
        '"text", "interactive_buttons", "interactive_list", "media".'
    )
    user_prompt = f"""
Message:
{raw_message}

Return JSON:
{{"type": "<one_of_the_four>"}}
"""
    completion = client.chat.completions.create(
        model=CLASSIFIER_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0,
        max_tokens=10,
    )
    content = completion.choices[0].message.content or "{}"
    try:
        parsed = json.loads(content)
        tpe = parsed.get("type")
        if tpe in MESSAGE_TYPES:
            return tpe
    except Exception:
        pass
    # Fallback default
    return "text"


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text or "option"


def build_text_message(raw_message: str) -> dict:
    return {"messages": [{"type": "text", "content": {"body": raw_message}}]}


def build_media_messages(raw_message: str) -> dict:
    urls = extract_media_urls(raw_message)
    messages = []
    text_without_urls = raw_message
    for url in urls:
        text_without_urls = text_without_urls.replace(url, "").strip()
        messages.append({"type": "media", "content": {"media_link": url}})
    if text_without_urls:
        messages.insert(0, {"type": "text", "content": {"body": text_without_urls}})
    return {"messages": messages}


def parse_option_lines(lines: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in lines:
        parts = re.split(r"\s*[-–:]\s*", line, maxsplit=1)
        title = parts[0].strip()
        desc = parts[1].strip() if len(parts) > 1 else ""
        if not title:
            continue
        rows.append({"id": slugify(title)[:32], "title": title[:64], "description": desc[:72]})
    return rows


def build_interactive_list_messages(raw_message: str) -> dict:
    lines = [ln.strip() for ln in raw_message.splitlines() if ln.strip()]
    bullet_lines = extract_bullet_lines(raw_message)
    intro_lines = [ln for ln in lines if ln not in bullet_lines]
    intro_text = "\n".join(intro_lines).strip()
    rows = parse_option_lines(bullet_lines if bullet_lines else lines[1:])
    content_body = "Please choose an option:"
    sections = [
        {
            "title": "Options",
            "rows": rows if rows else [{"id": "option_1", "title": raw_message[:64] or "Option", "description": ""}],
        }
    ]
    messages = [
        {
            "type": "interactive_list",
            "content": {
                "header": {"title": "Options"},
                "body": content_body,
                "sections": sections,
            },
        }
    ]
    if intro_text and intro_text != content_body:
        messages.insert(0, {"type": "text", "content": {"body": intro_text}})
    return {"messages": messages}


def build_interactive_buttons_messages(raw_message: str) -> dict:
    lines = [ln.strip() for ln in raw_message.splitlines() if ln.strip()]
    action_lines = extract_action_lines(raw_message)
    buttons = [{"id": slugify(btn)[:32], "title": btn[:32]} for btn in action_lines[:3]]
    intro_candidates = [ln for ln in lines if ln not in action_lines]
    intro_text = "\n".join(intro_candidates).strip() if intro_candidates else "Choose an option:"
    content = {"body": intro_text or "Choose an option:", "buttons": buttons or [{"id": "more_info", "title": "More Info"}]}
    return {"messages": [{"type": "interactive_buttons", "content": content}]}


def build_messages_for_type(raw_message: str, msg_type: str) -> dict:
    if msg_type == "media":
        return build_media_messages(raw_message)
    if msg_type == "interactive_list":
        return build_interactive_list_messages(raw_message)
    if msg_type == "interactive_buttons":
        return build_interactive_buttons_messages(raw_message)
    return build_text_message(raw_message)


def validate_messages_payload(payload: dict) -> bool:
    """Validate core structural rules."""
    if not isinstance(payload, dict) or "messages" not in payload:
        return False
    messages = payload.get("messages")
    if not isinstance(messages, list) or not (1 <= len(messages) <= 6):
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            return False
        if msg.get("type") not in MESSAGE_TYPES:
            return False
        content = msg.get("content")
        if not isinstance(content, dict):
            return False
        if msg["type"] == "text" and not content.get("body"):
            return False
        if msg["type"] == "media" and not content.get("media_link"):
            return False
        if msg["type"] == "interactive_buttons":
            if not content.get("body") or not content.get("buttons"):
                return False
            for btn in content["buttons"]:
                if not isinstance(btn, dict) or not btn.get("id") or not btn.get("title"):
                    return False
        if msg["type"] == "interactive_list":
            header = content.get("header", {})
            sections = content.get("sections", [])
            if not content.get("body") or not header.get("title") or not sections:
                return False
            for section in sections:
                rows = section.get("rows", [])
                if not rows:
                    return False
                for row in rows:
                    if not isinstance(row, dict) or not row.get("id") or not row.get("title"):
                        return False
    return True


def generate_interface_messages(
    raw_message: str,
    llm_classifier: Callable[[str], str] = None,
) -> dict:
    """
    Takes raw natural-language input and returns a JSON dict:
    { "messages": [ ... ] } following the existing schema.
    """
    llm_classifier = llm_classifier or llm_message_type_classifier
    features = extract_features(raw_message)
    msg_type = rule_based_message_type(raw_message, features)
    if msg_type is None:
        msg_type = llm_classifier(raw_message)
    if msg_type not in MESSAGE_TYPES:
        msg_type = "text"
    payload = build_messages_for_type(raw_message, msg_type)
    if not validate_messages_payload(payload):
        # Fallback to plain text
        payload = build_text_message(raw_message)
    return payload

