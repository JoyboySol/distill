import json
from typing import Any, Dict


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def ensure_message_shape(message: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "role": message.get("role"),
        "content": message.get("content"),
        "reasoning_content": message.get("reasoning_content"),
        "tool_calls": message.get("tool_calls"),
        "tool_call_id": message.get("tool_call_id"),
        "name": message.get("name"),
    }
