import re
from typing import Any, Dict, List, Optional


EXAMPLE_LINE_RE = re.compile(r"^\s*(?:examples?|samples?)\s*:?\s*$",
                             re.IGNORECASE)
HEADER_LINE_RE = re.compile(
    r"^\s*(?:(?:sample|example)\s+)?(input|output)\s*:?\s*(.*?)\s*$",
    re.IGNORECASE,
)
END_LINE_RE = re.compile(
    r"^\s*(?:"
    r"explanation|comment|comments|note|notes|"
    r"constraint|constraints|subtask|subtasks|"
    r"warning|warnings|limit|limits"
    r")\s*:?\s*$",
    re.IGNORECASE,
)


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")


def _clean_block(lines: List[str]) -> str:
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def _find_example_start(lines: List[str]) -> int:
    last_idx = -1
    for idx, line in enumerate(lines):
        if EXAMPLE_LINE_RE.match(line):
            last_idx = idx
    return last_idx + 1 if last_idx >= 0 else -1


def _detect_header(line: str) -> Optional[Dict[str, str]]:
    match = HEADER_LINE_RE.match(line)
    if not match:
        return None
    return {
        "kind": match.group(1).lower(),
        "inline": match.group(2) or "",
    }


def parse_prompt_examples(prompt_text: str) -> List[Dict[str, str]]:
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        return []

    lines = _normalize_text(prompt_text).split("\n")
    example_start = _find_example_start(lines)
    if example_start < 0:
        return []
    scan_lines = lines[example_start:]

    pairs: List[Dict[str, str]] = []
    current_input: Optional[List[str]] = None
    current_output: Optional[List[str]] = None
    current_kind: Optional[str] = None

    def flush():
        nonlocal current_input, current_output, current_kind
        if current_input is None or current_output is None:
            current_input = None
            current_output = None
            current_kind = None
            return
        cleaned_input = _clean_block(current_input[:])
        cleaned_output = _clean_block(current_output[:])
        if cleaned_input and cleaned_output:
            pairs.append({
                "input": cleaned_input,
                "output": cleaned_output,
            })
        current_input = None
        current_output = None
        current_kind = None

    for line in scan_lines:
        header = _detect_header(line)
        if header:
            kind = header["kind"]
            inline = header["inline"]
            if kind == "input":
                if current_input is not None and current_output is not None:
                    flush()
                elif current_input is not None and current_output is None:
                    current_input = None
                current_input = [inline] if inline else []
                current_output = None
                current_kind = "input"
                continue

            if kind == "output" and current_input is not None:
                current_output = [inline] if inline else []
                current_kind = "output"
                continue

        if END_LINE_RE.match(line):
            flush()
            break

        if current_kind == "input" and current_input is not None:
            current_input.append(line)
        elif current_kind == "output" and current_output is not None:
            current_output.append(line)

    flush()
    return pairs


def build_prompt_example_sample(row_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prompt_candidates: List[str] = []
    for key in ("input", "question", "prompt", "problem", "question_content"):
        value = row_data.get(key)
        if isinstance(value, str) and value.strip():
            prompt_candidates.append(value)

    for prompt_text in prompt_candidates:
        examples = parse_prompt_examples(prompt_text)
        if not examples:
            continue
        return {
            "inputs": [example["input"] for example in examples],
            "outputs": [example["output"] for example in examples],
            "fn_name": None,
            "source": "prompt_examples",
        }
    return None


def reference_solution_text(row_data: Dict[str, Any]) -> Optional[str]:
    for key in ("solution", "reference_solution", "canonical_solution",
                "ground_truth_solution"):
        value = row_data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
