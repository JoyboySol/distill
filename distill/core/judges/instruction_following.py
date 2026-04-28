import ast
import re
from typing import Any, Dict, List, Optional, Tuple

from .shared import assistant_text


SUPPORTED_BACKEND = "instruction_following_v1"


def _split_words(text: str) -> List[str]:
    return re.findall(r"[^\W_]+", text, flags=re.UNICODE)


def _normalized_last_word(text: str) -> Optional[str]:
    words = _split_words(text)
    if not words:
        return None
    return words[-1].lower()


def _normalized_first_word(text: str) -> Optional[str]:
    words = _split_words(text)
    if not words:
        return None
    return words[0].lower()


def _paragraphs(text: str) -> List[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def _count_placeholders(text: str) -> int:
    return len(re.findall(r"\[[^\[\]\n]+\]", text))


def _count_uppercase_words(text: str) -> int:
    return sum(1 for word in _split_words(text) if len(word) > 1 and word.isupper())


def _extract_quoted_phrases(text: str) -> List[str]:
    return [match[0] or match[1] for match in re.findall(r'"([^"]+)"|\'([^\']+)\'', text)]


def _extract_excluded_words(prompt: str) -> List[str]:
    match = re.search(
        r'excluding (?:the )?words? (.+?)(?:[.](?:\s|$)|$)',
        prompt,
        re.IGNORECASE,
    )
    if not match:
        return []
    return _extract_quoted_phrases(match.group(1))


def _normalize_prompt_keywords(values: List[str]) -> List[str]:
    normalized = []
    for value in values:
        cleaned = value.strip().strip(".,!?;:")
        if cleaned:
            normalized.append(cleaned)
    return normalized


def _sentence_blocks(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _count_bullet_lists(text: str) -> int:
    count = 0
    in_list = False
    for line in text.splitlines():
        is_bullet = bool(re.match(r"^\s*(?:[-*]|\d+\.)\s+", line))
        if is_bullet and not in_list:
            count += 1
            in_list = True
        elif not is_bullet:
            in_list = False
    return count


def _has_title_like_line(text: str) -> bool:
    for idx, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return True
        if stripped.startswith("[") and stripped.endswith("]"):
            return True
        if len(_split_words(stripped)) <= 12 and not stripped.endswith((".", "?", "!")):
            next_lines = text.splitlines()[idx + 1:]
            if next_lines and not next_lines[0].strip():
                return True
        return False
    return False


def _contains_prompt_text(answer: str, prompt_to_repeat: str, count: int = 1) -> bool:
    return answer.count(prompt_to_repeat) >= count


def _compare_count(actual: int, expected: int, relation: str) -> bool:
    if relation == "at least":
        return actual >= expected
    if relation == "less than":
        return actual < expected
    return actual == expected


def _check_constraint(spec: Dict[str, Any], answer: str) -> Tuple[bool, Dict[str, Any]]:
    kind = spec["kind"]
    if kind == "last_word":
        actual = _normalized_last_word(answer)
        expected = str(spec["value"]).lower()
        return actual == expected, {"expected": expected, "actual": actual}

    if kind == "first_word":
        actual = _normalized_first_word(answer)
        expected = str(spec["value"]).lower()
        return actual == expected, {"expected": expected, "actual": actual}

    if kind == "exclude_keywords":
        lowered = answer.lower()
        missing = [word for word in spec["values"] if word.lower() in lowered]
        return not missing, {"forbidden_present": missing}

    if kind == "include_keywords":
        lowered = answer.lower()
        missing = [word for word in spec["values"] if word.lower() not in lowered]
        return not missing, {"missing_keywords": missing}

    if kind == "punctuation_forbidden":
        symbol = str(spec["value"])
        present = symbol in answer
        return not present, {"symbol": symbol, "present": present}

    if kind == "sentence_hyphens":
        compact = answer.strip()
        ok = bool(compact) and " " not in compact and "-" in compact
        return ok, {"value": compact[:120]}

    if kind == "num_paragraphs":
        actual = len(_paragraphs(answer))
        expected = int(spec["value"])
        return actual == expected, {"expected": expected, "actual": actual}

    if kind == "specific_ending":
        expected = str(spec["value"]).strip()
        actual = answer.rstrip()
        return actual.endswith(expected), {"expected": expected}

    if kind == "repeat_prompt":
        expected = str(spec["value"])
        return _contains_prompt_text(answer, expected), {"expected": expected[:120]}

    if kind == "repeat_phrase_count":
        phrase = str(spec["value"])
        count = int(spec["count"])
        actual = answer.count(phrase)
        return actual == count, {"phrase": phrase, "expected": count, "actual": actual}

    if kind == "copy_span":
        span = str(spec["value"])
        return span in answer, {"expected": span}

    if kind == "placeholder_count":
        actual = _count_placeholders(answer)
        expected = int(spec["value"])
        return actual >= expected, {"expected_at_least": expected, "actual": actual}

    if kind == "title":
        ok = _has_title_like_line(answer)
        return ok, {"first_nonempty_line": next((line.strip() for line in answer.splitlines() if line.strip()), None)}

    if kind == "bullet_list_count":
        actual = _count_bullet_lists(answer)
        expected = int(spec["value"])
        return actual == expected, {"expected": expected, "actual": actual}

    if kind == "quotation":
        stripped = answer.strip()
        ok = (stripped.startswith('"') and stripped.endswith('"')) or (
            stripped.startswith("'") and stripped.endswith("'"))
        return ok, {"value": stripped[:120]}

    if kind == "english_lowercase":
        letters = [ch for ch in answer if ch.isalpha()]
        ok = bool(letters) and all(ch == ch.lower() for ch in letters if ch.isascii())
        return ok, {"checked_letters": len(letters)}

    if kind == "capital_word_frequency":
        actual = _count_uppercase_words(answer)
        expected = int(spec["value"])
        relation = str(spec.get("relation") or "exactly")
        return _compare_count(actual, expected, relation), {
            "expected": expected,
            "relation": relation,
            "actual": actual,
        }

    if kind == "two_responses":
        normalized = answer.lower()
        ok = (
            "summary 1" in normalized and "summary 2" in normalized
        ) or (
            "response 1" in normalized and "response 2" in normalized
        ) or (
            len(re.findall(r"\n\s*\*{4,}\s*\n", answer)) >= 1
        )
        return ok, {"value": answer[:160]}

    raise KeyError(kind)


def _structured_constraint_specs(row_data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    raw = row_data.get("ground_truth")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        payload = ast.literal_eval(raw)
    except Exception:
        return None
    if not isinstance(payload, list) or not payload:
        return None

    specs: List[Dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            return None
        instruction_ids = entry.get("instruction_id") or []
        kwargs_list = entry.get("kwargs") or []
        for idx, instruction_id in enumerate(instruction_ids):
            kwargs = kwargs_list[idx] if idx < len(kwargs_list) else None
            spec = _spec_from_instruction_id(str(instruction_id), kwargs)
            if spec is None:
                return [{
                    "kind": "__unsupported__",
                    "source": str(instruction_id),
                }]
            specs.append(spec)
    return specs


def _spec_from_instruction_id(instruction_id: str,
                              kwargs: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    kwargs = kwargs or {}
    if instruction_id == "detectable_format:sentence_hyphens":
        return {"kind": "sentence_hyphens", "source": instruction_id}
    if instruction_id == "last_word:last_word_answer":
        return {"kind": "last_word", "value": kwargs.get("last_word"), "source": instruction_id}
    if instruction_id in {"first_word:first_word_answer", "first_word:first_word_sent"}:
        return {"kind": "first_word", "value": kwargs.get("first_word"), "source": instruction_id}
    if instruction_id == "keywords:word_once":
        return {"kind": "include_keywords", "values": [kwargs.get("keyword")], "source": instruction_id}
    if instruction_id == "keywords:exclude_word_harder":
        return {"kind": "exclude_keywords", "values": [kwargs.get("keyword")], "source": instruction_id}
    if instruction_id == "length_constraints:number_paragraphs":
        return {"kind": "num_paragraphs", "value": kwargs.get("num_paragraphs"), "source": instruction_id}
    if instruction_id == "detectable_content:number_placeholders":
        return {"kind": "placeholder_count", "value": kwargs.get("num_placeholders"), "source": instruction_id}
    if instruction_id == "startend:end_checker":
        return {"kind": "specific_ending", "value": kwargs.get("end_phrase"), "source": instruction_id}
    if instruction_id == "startend:quotation":
        return {"kind": "quotation", "source": instruction_id}
    if instruction_id == "change_case:english_lowercase":
        return {"kind": "english_lowercase", "source": instruction_id}
    if instruction_id == "change_case:capital_word_frequency":
        return {
            "kind": "capital_word_frequency",
            "value": kwargs.get("capital_frequency"),
            "relation": kwargs.get("capital_relation") or "exactly",
            "source": instruction_id,
        }
    if instruction_id == "copy:copying_simple":
        return {"kind": "repeat_prompt", "value": kwargs.get("prompt_to_repeat"), "source": instruction_id}
    if instruction_id == "copy:copying_multiple":
        return {
            "kind": "repeat_phrase_count",
            "value": kwargs.get("prompt_to_repeat"),
            "count": kwargs.get("N"),
            "source": instruction_id,
        }
    if instruction_id == "copy:repeat_phrase":
        return {
            "kind": "repeat_phrase_count",
            "value": kwargs.get("phrase"),
            "count": kwargs.get("small_n"),
            "source": instruction_id,
        }
    if instruction_id == "new:copy_span_idx":
        prompt = str(kwargs.get("prompt_to_repeat") or "")
        start = kwargs.get("n_start")
        end = kwargs.get("n_end")
        if isinstance(start, int) and isinstance(end, int) and prompt:
            return {
                "kind": "copy_span",
                "value": prompt[start:end + 1],
                "source": instruction_id,
            }
    if instruction_id == "punctuation:punctuation_dot":
        return {"kind": "punctuation_forbidden", "value": ".", "source": instruction_id}
    if instruction_id == "punctuation:punctuation_exclamation":
        return {"kind": "punctuation_forbidden", "value": "!", "source": instruction_id}
    if instruction_id == "detectable_format:title":
        return {"kind": "title", "source": instruction_id}
    if instruction_id == "combination:repeat_prompt":
        return {"kind": "repeat_prompt", "value": kwargs.get("prompt_to_repeat"), "source": instruction_id}
    if instruction_id == "combination:two_responses":
        return {"kind": "two_responses", "source": instruction_id}
    return None


def _prompt_constraint_specs(row_data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    prompt = row_data.get("prompt")
    constraints = row_data.get("constraints")
    if not isinstance(prompt, str) or not prompt.strip() or not isinstance(constraints, list):
        return None

    specs: List[Dict[str, Any]] = []
    for raw_constraint in constraints:
        constraint = str(raw_constraint).strip().lower()
        if constraint == "keywords:exclude words":
            values = _normalize_prompt_keywords(_extract_excluded_words(prompt))
            if values:
                specs.append({"kind": "exclude_keywords", "values": values, "source": raw_constraint})
                continue
        if constraint == "specific ending":
            match = re.search(r'end (?:your )?response with "([^"]+)"', prompt, re.IGNORECASE)
            if match:
                specs.append({"kind": "specific_ending", "value": match.group(1), "source": raw_constraint})
                continue
        if constraint == "include keywords":
            values = _normalize_prompt_keywords(_extract_quoted_phrases(prompt))
            if values:
                specs.append({"kind": "include_keywords", "values": values, "source": raw_constraint})
                continue
        if constraint == "punctuation:use no comma":
            specs.append({"kind": "punctuation_forbidden", "value": ",", "source": raw_constraint})
            continue
        if constraint == "repeat the prompt":
            specs.append({"kind": "repeat_prompt", "value": prompt, "source": raw_constraint})
            continue
        if constraint == "use quotation":
            specs.append({"kind": "quotation", "source": raw_constraint})
            continue
        if constraint == "format:title":
            specs.append({"kind": "title", "source": raw_constraint})
            continue
        if constraint == "give two responses":
            specs.append({"kind": "two_responses", "source": raw_constraint})
            continue
        if constraint == "keywords:exclude words":
            continue
        return [{
            "kind": "__unsupported__",
            "source": raw_constraint,
        }]
    return specs


def _evaluate_specs(specs: List[Dict[str, Any]], answer: str) -> Dict[str, Any]:
    unsupported = [spec["source"] for spec in specs if spec.get("kind") == "__unsupported__"]
    if unsupported or not specs:
        return {
            "judge_type": "instruction_following",
            "judge_backend": SUPPORTED_BACKEND,
            "is_correct": None,
            "judge_status": "not_applicable",
            "judge_detail": {
                "unsupported_constraints": unsupported or ["no_supported_constraints"],
            },
        }

    results = []
    all_pass = True
    for spec in specs:
        passed, detail = _check_constraint(spec, answer)
        results.append({
            "source": spec.get("source"),
            "kind": spec["kind"],
            "passed": passed,
            "detail": detail,
        })
        all_pass = all_pass and passed

    return {
        "judge_type": "instruction_following",
        "judge_backend": SUPPORTED_BACKEND,
        "is_correct": all_pass,
        "judge_status": "pass" if all_pass else "wrong_answer",
        "judge_detail": {
            "constraint_results": results,
        },
    }


def instruction_following_hint(row_data: Dict[str, Any]) -> Optional[str]:
    if _structured_constraint_specs(row_data) is not None:
        return "instruction_following"
    if _prompt_constraint_specs(row_data) is not None:
        return "instruction_following"
    return None


def judge_instruction_following(row_data: Dict[str, Any],
                                messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    specs = _structured_constraint_specs(row_data)
    if specs is None:
        specs = _prompt_constraint_specs(row_data)
    if specs is None:
        return None
    return _evaluate_specs(specs, assistant_text(messages))
