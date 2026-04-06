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


def usage_to_dict(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {}

    if isinstance(usage, dict):
        data = usage
    else:
        data = {}
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_tokens_details",
            "completion_tokens_details",
        ):
            value = getattr(usage, key, None)
            if value is not None:
                data[key] = value

    return json.loads(safe_json_dumps(data))
