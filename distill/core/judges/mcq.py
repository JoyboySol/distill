import re
from typing import Any, Dict, Optional

from .math import MathJudge
from .shared import last_message_text, normalize_message_list


def _looks_like_choice_prompt(text: Optional[str]) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    return len(re.findall(r"(?:^|\n)\s*[A-J]\s*:", text)) >= 3


def _extract_boxed_choice_letter(text: Optional[str]) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return None
    answer = MathJudge.extract_boxed_answer(
        text, strip_double_curly_brace=True)
    if answer is None:
        return None
    cleaned = re.sub(r"[^A-Za-z]", "", answer).upper()
    if len(cleaned) == 1 and cleaned in "ABCDEFGHIJ":
        return cleaned
    return None


def _extract_freeform_choice_letter(text: Optional[str]) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return None

    normalized = text.replace("**", " ").replace("__", " ")
    pattern_groups = [
        [
            re.compile(
                r"\b(?:final|correct)\s+answer\s*(?:is|:)?\s*"
                r"(?:option\s+)?\(?\s*([A-J])\s*\)?(?=\s*[:).\-]|\b)",
                re.IGNORECASE,
            ),
            re.compile(
                r"\banswer\s*(?:is|:)?\s*(?:option\s+)?\(?\s*([A-J])\s*\)?"
                r"(?=\s*[:).\-]|\b)",
                re.IGNORECASE,
            ),
            re.compile(
                r"\bthe\s+best\s+(?:answer|choice)\s*(?:is|would\s+be|:)?\s*"
                r"(?:option\s+)?\(?\s*([A-J])\s*\)?(?=\s*[:).\-]|\b)",
                re.IGNORECASE,
            ),
        ],
        [
            re.compile(
                r"\b(?:therefore|thus|hence|so|overall|in\s+conclusion)\b"
                r"[^\n.]{0,120}?\b(?:option\s+)?\(?\s*([A-J])\s*\)?\s*"
                r"(?:is|would\s+be)?\s*(?:the\s+)?(?:correct|best)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\boption\s+([A-J])\s*(?:[:).\-][^\n]{0,120})?\s+is\s+"
                r"(?:the\s+)?(?:correct|best)\b",
                re.IGNORECASE,
            ),
        ],
    ]
    for group in pattern_groups:
        matches = []
        for pattern in group:
            matches.extend(pattern.finditer(normalized))
        if matches:
            return matches[-1].group(1).upper()
    return None


def extract_mcq_choice_letter(text: Optional[str]) -> Optional[str]:
    boxed = _extract_boxed_choice_letter(text)
    if boxed is not None:
        return boxed
    return _extract_freeform_choice_letter(text)


def try_mcq_boxed_reference(row_data: Dict[str, Any]) -> Optional[str]:
    messages = normalize_message_list(row_data.get("messages"))
    if not messages:
        return None
    user_text = last_message_text(messages, "user")
    if not _looks_like_choice_prompt(user_text):
        return None
    assistant_reference = last_message_text(messages, "assistant")
    return _extract_boxed_choice_letter(assistant_reference)


def judge_mcq(row_data: Dict[str, Any],
              content: str) -> Optional[Dict[str, Any]]:
    mcq_reference = try_mcq_boxed_reference(row_data)
    if mcq_reference is None:
        return None
    prediction = extract_mcq_choice_letter(content)
    is_correct = prediction == mcq_reference
    return {
        "judge_type": "mcq_boxed",
        "judge_backend": "boxed_choice_v1",
        "is_correct": is_correct,
        "judge_status": "pass" if is_correct else "wrong_answer",
        "judge_detail": {
            "prediction": prediction,
            "reference": mcq_reference,
        },
    }

