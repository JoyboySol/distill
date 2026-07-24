import contextlib
import io
import json
import multiprocessing
import re
import signal
from typing import Any, Dict, List, Optional

from .shared import assistant_text

try:
    from ..livecodebench import (build_livecodebench_sample,
                                 evaluate_livecodebench_generation)
    from ..prompt_examples import (build_prompt_example_sample,
                                   reference_solution_text)
except ImportError:
    from distill.core.livecodebench import (  # type: ignore
        build_livecodebench_sample,
        evaluate_livecodebench_generation,
    )
    from distill.core.prompt_examples import (  # type: ignore
        build_prompt_example_sample,
        reference_solution_text,
    )


class TimeOutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeOutException("Time out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _execution(program: str, timeout: int, result_holder):
    try:
        exec_globals: Dict[str, Any] = {}
        with swallow_io():
            with time_limit(timeout):
                exec(program, exec_globals)
        result_holder.append("pass")
    except TimeOutException:
        result_holder.append("timeout")
    except AssertionError:
        result_holder.append("wrong_answer")
    except BaseException:
        result_holder.append("failed")


def run_python_program(program: str, timeout: int = 10) -> str:
    manager = multiprocessing.Manager()
    result_holder = manager.list()
    process = multiprocessing.Process(
        target=_execution,
        args=(program, max(1, timeout - 1), result_holder),
    )
    process.start()
    process.join(timeout=timeout)
    if process.is_alive():
        process.kill()
        process.join()
        return "timeout"
    if len(result_holder) == 0:
        return "failed"
    return str(result_holder[0])


def _fenced_code_blocks(text: str) -> List[Dict[str, str]]:
    blocks = re.findall(r"```([A-Za-z0-9_+#-]*)\s*\n(.*?)```", text,
                        re.DOTALL)
    return [{
        "lang": (lang or "").strip(),
        "code": code.lstrip(),
    } for lang, code in blocks]


def _python_block_score(code: str) -> int:
    text = code.strip()
    lower = text.lower()
    score = 0

    positive_patterns = (
        "def ",
        "print(",
        "input(",
        "sys.stdin",
        "sys.stdout",
        "map(int",
        "for _ in range",
        "elif ",
        "__name__ ==",
        "import sys",
        "from collections",
    )
    score += sum(1 for pattern in positive_patterns if pattern in lower)

    if "class solution" in lower:
        score += 2
    if text.count("\n") >= 2:
        score += 1

    negative_patterns = (
        "#include <",
        "using namespace std",
        "int main(",
        "std::",
        "public static void main",
        "system.out.",
        "scanner ",
        "bufferedreader",
        "package main",
        "func main()",
        "fmt.",
        "fn main()",
        "println!",
    )
    score -= sum(3 for pattern in negative_patterns if pattern in lower)

    if re.fullmatch(r"[\d\s\.\-]+", text):
        score -= 5

    return score


def _best_python_like_block(text: str, prefer_last: bool) -> Optional[str]:
    blocks = _fenced_code_blocks(text)
    if not blocks:
        return None

    candidates = []
    for idx, block in enumerate(blocks):
        lang = block["lang"].lower()
        if lang in {"python", "py"}:
            candidates.append((1000 + idx, idx, block["code"]))
            continue
        score = _python_block_score(block["code"])
        if score > 0:
            candidates.append((score, idx, block["code"]))

    if not candidates:
        return None

    if prefer_last:
        _, _, code = max(candidates, key=lambda item: (item[0], item[1]))
    else:
        _, _, code = max(candidates, key=lambda item: (item[0], -item[1]))
    return code


def extract_code_text(text: str) -> str:
    best_block = _best_python_like_block(text, prefer_last=False)
    if best_block is not None:
        return best_block

    blocks = re.findall(r"```\w*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[0].lstrip()

    patterns = [
        r"\[BEGIN\]\s*'(.*)'\s*\[DONE\]",
        r"BEGIN\s*'(.*)'\s*\[DONE\]",
        r"\[BEGIN\]\s*'(.*)\s*\[DONE\]",
        r"BEGIN\s*'(.*)\s*\[DONE\]",
        r"\[BEGIN\]\s*(.*)\s*\[DONE\]",
        r"BEGIN\s*(.*)\s*\[DONE\]",
        r"\[BEGIN\](.*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            extracted = match.group(1)
            extracted = extracted.split("```")[0]
            extracted = re.split(r"'?\s*\[?DONE\]?", extracted)[0]
            return extracted.replace("\\_", "_").strip()
    return text.strip()


def extract_code_text_last_block(text: str) -> str:
    best_block = _best_python_like_block(text, prefer_last=True)
    if best_block is not None:
        return best_block

    blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].lstrip()
    return extract_code_text(text)


def _unit_tests_from_tests_field(row_data: Dict[str, Any]) -> Optional[str]:
    raw = row_data.get("tests")
    if isinstance(raw, str):
        if not raw.strip():
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
    elif isinstance(raw, dict):
        parsed = raw
    else:
        return None

    unit_tests = parsed.get("unit_tests")
    if not isinstance(unit_tests, list):
        return None

    statuses = parsed.get("tests_execution_status")
    selected: List[str] = []
    if isinstance(statuses, list) and len(statuses) == len(unit_tests):
        for test, status in zip(unit_tests, statuses):
            if str(status).strip().lower() == "pass" and isinstance(test, str):
                selected.append(test)
    else:
        selected = [test for test in unit_tests if isinstance(test, str)]

    selected = [test.strip() for test in selected if test.strip()]
    if not selected:
        return None
    return "\n".join(selected)


def _looks_like_code_text(text: Optional[str]) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    lower = text.lower()
    indicators = (
        "def ",
        "print(",
        "input(",
        "sys.stdin",
        "sys.stdout",
        "map(int",
        "for _ in range",
        "class solution",
        "__name__ ==",
        "#include <",
        "using namespace std",
        "public static void main",
        "system.out.",
        "func main()",
        "package main",
    )
    return any(indicator in lower for indicator in indicators)


def _looks_like_code_task(row_data: Dict[str, Any]) -> bool:
    direct_code_fields = (
        "input_output",
        "evaluation_sample",
        "public_test_cases",
        "private_test_cases",
        "tests",
        "test",
        "test_list",
        "test_list_2",
        "entry_point",
    )
    if any(row_data.get(field) for field in direct_code_fields):
        return True

    metadata_text = " ".join(
        str(row_data.get(field) or "")
        for field in ("dataset", "dataset_name", "source_dataset", "source")
    ).lower()
    code_dataset_markers = (
        "code_contests",
        "codeforces",
        "codechef",
        "atcoder",
        "hackerrank",
        "hackerearth",
        "aizu",
        "kattis",
        "leetcode",
        "geeksforgeeks",
        "codewars",
        "humaneval",
        "mbpp",
        "livecodebench",
        "opencode",
    )
    if any(marker in metadata_text for marker in code_dataset_markers):
        return True

    for field in ("solution", "reference_solution", "canonical_solution",
                  "ground_truth_solution", "output"):
        if _looks_like_code_text(row_data.get(field)):
            return True

    prompt_text = row_data.get("input") or row_data.get("prompt") or row_data.get(
        "question")
    if isinstance(prompt_text, str):
        prompt_lower = prompt_text.lower()
        prompt_markers = (
            "write a program",
            "standard input",
            "standard output",
            "input:",
            "output:",
            "constraints",
        )
        if any(marker in prompt_lower for marker in prompt_markers):
            return True

    return False


def build_humaneval_program(row_data: Dict[str, Any],
                            completion: str) -> Optional[str]:
    prompt = row_data.get("prompt")
    tests = row_data.get("test")
    entry_point = row_data.get("entry_point")
    if not isinstance(prompt, str) or not isinstance(tests, str):
        return None
    program = prompt + completion.rstrip() + "\n" + tests
    if isinstance(entry_point, str) and entry_point.strip():
        program += f"\ncheck({entry_point.strip()})"
    return program


def judge_code(row_data: Dict[str, Any],
               messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    content = assistant_text(messages)

    code_test = row_data.get("test_list_2") or row_data.get("test_list")
    if isinstance(code_test, list):
        code_test = "\n".join(str(x) for x in code_test)

    if isinstance(code_test, str) and code_test.strip():
        status = run_python_program(extract_code_text(content).rstrip() + "\n" +
                                    code_test,
                                    timeout=10)
        return {
            "judge_type": "code_mbpp",
            "is_correct": status == "pass",
            "judge_status": status,
            "judge_detail": {
                "timeout_seconds": 10
            },
        }

    unit_tests = _unit_tests_from_tests_field(row_data)
    if unit_tests:
        status = run_python_program(extract_code_text(content).rstrip() + "\n" +
                                    unit_tests,
                                    timeout=10)
        return {
            "judge_type": "code_unit_tests",
            "is_correct": status == "pass",
            "judge_status": status,
            "judge_detail": {
                "timeout_seconds": 10,
                "test_source": "tests_unit_tests",
            },
        }

    humaneval_program = build_humaneval_program(row_data,
                                                extract_code_text(content))
    if humaneval_program:
        status = run_python_program(humaneval_program, timeout=10)
        return {
            "judge_type": "code_humaneval",
            "is_correct": status == "pass",
            "judge_status": status,
            "judge_detail": {
                "timeout_seconds": 10
            },
        }

    livecodebench_sample = build_livecodebench_sample(row_data)
    if livecodebench_sample:
        lcb_result = evaluate_livecodebench_generation(
            livecodebench_sample,
            extract_code_text_last_block(content),
            timeout=6,
        )
        return {
            "judge_type": "code_livecodebench_generation",
            "judge_backend": "livecodebench_v6",
            "is_correct": lcb_result["is_correct"],
            "judge_status": lcb_result["status"],
            "judge_detail": lcb_result["detail"],
        }

    prompt_example_sample = build_prompt_example_sample(row_data)
    if prompt_example_sample:
        detail: Dict[str, Any] = {
            "example_count": len(prompt_example_sample["inputs"]),
            "test_source": prompt_example_sample.get("source"),
        }
        reference_solution = reference_solution_text(row_data)
        if reference_solution:
            oracle_result = evaluate_livecodebench_generation(
                prompt_example_sample,
                reference_solution,
                timeout=6,
            )
            detail["oracle_status"] = oracle_result["status"]
            detail["oracle_detail"] = oracle_result["detail"]
            if not oracle_result["is_correct"]:
                return {
                    "judge_type": "code_prompt_examples",
                    "judge_backend": "prompt_examples_v1",
                    "is_correct": None,
                    "judge_status": "not_applicable",
                    "judge_detail": detail,
                }

        prompt_result = evaluate_livecodebench_generation(
            prompt_example_sample,
            extract_code_text_last_block(content),
            timeout=6,
        )
        detail.update(prompt_result["detail"])
        return {
            "judge_type": "code_prompt_examples",
            "judge_backend": "prompt_examples_v1",
            "is_correct": prompt_result["is_correct"],
            "judge_status": prompt_result["status"],
            "judge_detail": detail,
        }

    if _looks_like_code_task(row_data):
        return {
            "judge_type": "code_unverified",
            "judge_backend": "code_safeguard_v1",
            "is_correct": None,
            "judge_status": "not_applicable",
            "judge_detail": {
                "reason": "code_task_without_reliable_tests",
            },
        }

    return None


def code_judge_type_hint(row_data: Dict[str, Any],
                         messages: List[Dict[str, Any]]) -> Optional[str]:
    code_test = row_data.get("test_list_2") or row_data.get("test_list")
    if isinstance(code_test, list):
        code_test = "\n".join(str(x) for x in code_test)
    if isinstance(code_test, str) and code_test.strip():
        return "code_mbpp"
    if _unit_tests_from_tests_field(row_data):
        return "code_unit_tests"

    humaneval_program = build_humaneval_program(row_data,
                                                extract_code_text(
                                                    assistant_text(messages)))
    if humaneval_program:
        return "code_humaneval"

    if build_livecodebench_sample(row_data):
        return "code_livecodebench_generation"

    if build_prompt_example_sample(row_data):
        return "code_prompt_examples"

    if _looks_like_code_task(row_data):
        return "code_unverified"

    return None
