import pytest

from evals_generator.interface_messages import generate_interface_messages, validate_messages_payload


def test_text_greeting():
    msg = "Hello there!"
    result = generate_interface_messages(msg)
    assert result["messages"][0]["type"] == "text"
    assert msg in result["messages"][0]["content"]["body"]


def test_menu_to_list():
    msg = "Hi, please choose:\n- Pricing\n- Support\n- Contact"
    result = generate_interface_messages(msg)
    types = [m["type"] for m in result["messages"]]
    assert "interactive_list" in types
    list_msg = next(m for m in result["messages"] if m["type"] == "interactive_list")
    rows = list_msg["content"]["sections"][0]["rows"]
    assert len(rows) >= 3


def test_action_buttons():
    msg = "What do you need?\nBuy now\nTrack order\nContact support"
    result = generate_interface_messages(msg)
    assert result["messages"][0]["type"] == "interactive_buttons"
    buttons = result["messages"][0]["content"]["buttons"]
    assert len(buttons) == 3


def test_media_message():
    msg = "Here is the brochure: https://example.com/file.pdf and image https://example.com/pic.jpg"
    result = generate_interface_messages(msg)
    types = [m["type"] for m in result["messages"]]
    assert "media" in types
    assert sum(1 for t in types if t == "media") == 2


def test_inputs_requested_stays_text():
    msg = "Please share your name and phone number so we can assist."
    result = generate_interface_messages(msg)
    assert result["messages"][0]["type"] == "text"


def test_validation_fallback(monkeypatch):
    # Force an invalid payload to trigger fallback.
    def bad_builder(_):
        return {"messages": [{"type": "interactive_buttons", "content": {"buttons": []}}]}

    monkeypatch.setattr("interface_messages.build_messages_for_type", lambda raw, t: bad_builder(raw))
    result = generate_interface_messages("fallback please", llm_classifier=lambda _: "interactive_buttons")
    assert validate_messages_payload(result)
    assert result["messages"][0]["type"] == "text"
