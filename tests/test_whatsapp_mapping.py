import json

from expected_output_generator.app import build_whatsapp_sequence, InterfaceUISpec


def test_buttons_three_options():
    spec: InterfaceUISpec = {
        "body": "Pick one",
        "options": [
            {"id": "a", "title": "Buy now", "description": None},
            {"id": "b", "title": "Track order", "description": None},
            {"id": "c", "title": "Support", "description": None},
        ],
        "media": None,
    }
    seq = build_whatsapp_sequence(spec)
    assert seq["messages"][0]["type"] == "interactive_button"
    buttons = seq["messages"][0]["content"]["buttons"]
    assert len(buttons) == 3
    assert buttons[0]["title"] == "Buy now"


def test_media_pdf_no_options():
    spec: InterfaceUISpec = {
        "body": "See brochure",
        "options": [],
        "media": {"kind": "document", "url": "https://example.com/file.pdf", "caption": None},
    }
    seq = build_whatsapp_sequence(spec)
    msg = seq["messages"][0]
    assert msg["type"] == "media"
    assert msg["content"]["media_link"].endswith(".pdf")
