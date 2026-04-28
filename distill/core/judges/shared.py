import json
from typing import Any, Dict, List, Optional


def assistant_text(messages: List[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def normalize_message_list(raw_messages: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_messages, str):
        text = raw_messages.strip()
        if not text or text[0] not in "[{":
            return []
        try:
            raw_messages = json.loads(text)
        except Exception:
            return []

    if hasattr(raw_messages, "tolist"):
        try:
            converted = raw_messages.tolist()
            if converted is not raw_messages:
                raw_messages = converted
        except Exception:
            return []

    if isinstance(raw_messages, tuple):
        raw_messages = list(raw_messages)

    if not isinstance(raw_messages, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for message in raw_messages:
        if hasattr(message, "as_py"):
            try:
                message = message.as_py()
            except Exception:
                return []
        if not isinstance(message, dict):
            return []
        normalized.append(message)
    return normalized


def last_message_text(messages: List[Dict[str, Any]], role: str) -> Optional[str]:
    for message in reversed(messages):
        if message.get("role") == role:
            content = message.get("content")
            if isinstance(content, str):
                return content
    return None

