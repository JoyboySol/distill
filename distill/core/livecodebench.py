import ast
import base64
import faulthandler
import json
import math
import multiprocessing
import os
import pickle
import platform
import queue
import shutil
import signal
import subprocess
import sys
import types
import zlib
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import mock_open, patch


LCB_SUPPORT_IMPORTS = """from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
sys.setrecursionlimit(6 * 10 ** 5)
"""


class LiveCodeBenchTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise LiveCodeBenchTimeout("time_limit_exceeded")


def _is_digit_like(value: Any) -> bool:
    return isinstance(value, int) or (isinstance(value, str)
                                      and value.lstrip("-").isdigit())


def _allclose(left: List[float], right: List[float]) -> bool:
    if len(left) != len(right):
        return False
    return all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
               for a, b in zip(left, right))


def _truncate(text: str, length: int = 300) -> str:
    if len(text) <= length:
        return text
    half = max(1, length // 2)
    return text[:half] + "...(truncated)..." + text[-half:]


def _strip_compare(left: str, right: str) -> bool:
    return left.strip() == right.strip()


def _custom_compare_output(output_lines: List[str], expected: Any) -> bool:
    if isinstance(expected, list):
        expected_text = "\n".join(str(item) for item in expected)
    else:
        expected_text = str(expected)

    output_text = "\n".join(output_lines)
    if _strip_compare(output_text, expected_text):
        return True

    stripped_output = "\n".join(line.strip() for line in output_lines)
    return _strip_compare(stripped_output, expected_text)


def _normalize_jsonish(value: Any) -> Any:
    if isinstance(value, dict):
        normalized: Dict[Any, Any] = {}
        for key, item in value.items():
            normalized_key = key
            if isinstance(key, str) and key.lstrip("-").isdigit():
                normalized_key = int(key)
            normalized[normalized_key] = _normalize_jsonish(item)
        return normalized
    if isinstance(value, tuple):
        return [_normalize_jsonish(item) for item in value]
    if isinstance(value, list):
        return [_normalize_jsonish(item) for item in value]
    return value


def _module_from_source(module_name: str, source: str) -> types.SimpleNamespace:
    namespace: Dict[str, Any] = {"__name__": module_name}
    exec(compile(source, f"<{module_name}>", "exec"), namespace, namespace)
    return types.SimpleNamespace(**namespace)


def _call_with_mocked_stdin(method, inputs: str):
    line_iter = iter(inputs.split("\n"))

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", StringIO(inputs))
    @patch("sys.stdin.readline", lambda *args: next(line_iter))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def _wrapped(target):
        try:
            return target()
        except SystemExit:
            return None

    return _wrapped(method)


def _reliability_guard(maximum_memory_bytes: Optional[int] = None):
    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS,
                           (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA,
                           (maximum_memory_bytes, maximum_memory_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK,
                               (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    def _guarded_exit(*args, **kwargs):
        raise SystemExit

    builtins.exit = _guarded_exit
    builtins.quit = _guarded_exit

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    subprocess.Popen = None  # type: ignore

    builtins.help = None

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None


def _ensure_list_of_strings(value: Any) -> List[str]:
    if isinstance(value, list):
        return ["" if item is None else str(item) for item in value]
    return []


def _load_json_string(raw: Any) -> Optional[Any]:
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str) and raw.strip():
        return json.loads(raw)
    return None


def _decode_private_tests(raw: Any) -> Optional[Any]:
    if not isinstance(raw, str) or not raw.strip():
        return None

    try:
        return json.loads(raw)
    except Exception:
        pass

    try:
        decoded = pickle.loads(
            zlib.decompress(base64.b64decode(raw.encode("utf-8"))))
        if isinstance(decoded, (bytes, bytearray)):
            decoded = decoded.decode("utf-8")
        if isinstance(decoded, str):
            return json.loads(decoded)
        return decoded
    except Exception:
        return None


def _parse_test_cases(raw: Any, allow_encoded: bool = False) -> List[Dict[str, Any]]:
    parsed = _load_json_string(raw)
    if parsed is None and allow_encoded:
        parsed = _decode_private_tests(raw)
    if not isinstance(parsed, list):
        return []
    test_cases: List[Dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, dict):
            test_cases.append(item)
    return test_cases


def _extract_fn_name(metadata: Any) -> Optional[str]:
    parsed = _load_json_string(metadata)
    if not isinstance(parsed, dict):
        return None
    for key in ("func_name", "fn_name", "function_name", "entry_point"):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def build_livecodebench_sample(row_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for field in ("input_output", "evaluation_sample"):
        parsed = _load_json_string(row_data.get(field))
        if isinstance(parsed, dict):
            inputs = _ensure_list_of_strings(parsed.get("inputs"))
            outputs = _ensure_list_of_strings(parsed.get("outputs"))
            if inputs and len(inputs) == len(outputs):
                fn_name = parsed.get("fn_name")
                return {
                    "inputs": inputs,
                    "outputs": outputs,
                    "fn_name": fn_name if isinstance(fn_name, str) and fn_name.strip() else None,
                    "source": field,
                }

    public_tests = _parse_test_cases(row_data.get("public_test_cases"))
    private_tests = _parse_test_cases(row_data.get("private_test_cases"),
                                      allow_encoded=True)
    all_tests = public_tests + private_tests
    if not all_tests:
        return None

    inputs: List[str] = []
    outputs: List[str] = []
    for test_case in all_tests:
        if "input" not in test_case or "output" not in test_case:
            continue
        inputs.append("" if test_case["input"] is None else str(test_case["input"]))
        outputs.append("" if test_case["output"] is None else str(test_case["output"]))

    if not inputs or len(inputs) != len(outputs):
        return None

    return {
        "inputs": inputs,
        "outputs": outputs,
        "fn_name": _extract_fn_name(row_data.get("metadata")),
        "source": "public_private_test_cases",
    }


def _parse_call_based_inputs(raw_inputs: str) -> List[Any]:
    values: List[Any] = []
    for line in raw_inputs.split("\n"):
        if not line.strip():
            continue
        values.append(_normalize_jsonish(json.loads(line)))
    return values


def _parse_call_based_output(raw_output: str) -> Any:
    return _normalize_jsonish(json.loads(raw_output))


def _prepare_standard_input_code(generation: str) -> str:
    test_code = generation
    try:
        tree = ast.parse(test_code)
        if tree.body and isinstance(tree.body[-1], ast.If):
            condition = tree.body[-1].test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                prefix = ""
                if tree.body[:-1]:
                    prefix = ast.unparse(
                        ast.Module(body=tree.body[:-1], type_ignores=[]))
                suffix = ast.unparse(
                    ast.Module(body=tree.body[-1].body, type_ignores=[]))
                test_code = (prefix + "\n" + suffix).strip()
    except Exception:
        pass

    lines = test_code.split("\n")
    transformed_lines: List[str] = []
    for line in lines:
        if line.startswith("from ") or line.startswith("import "):
            transformed_lines.append(line + "\n")
        else:
            transformed_lines.append("\t" + line + "\n")

    wrapped = ""
    started = False
    for line in transformed_lines:
        if line.startswith("\t") and not started:
            wrapped += "stdin = sys.stdin\nstdout = sys.stdout\n"
            wrapped += "def code():\n"
            wrapped += line
            started = True
        elif started and (line.startswith("from ") or line.startswith("import ")):
            wrapped += "\t" + line
        else:
            wrapped += line
    return wrapped


def _compare_call_based(actual: Any, expected: Any) -> bool:
    actual = _normalize_jsonish(actual)
    expected = _normalize_jsonish(expected)

    if isinstance(actual, tuple):
        actual = list(actual)
    if actual == expected:
        return True

    if isinstance(expected, list) and expected:
        return actual == expected[0]

    if isinstance(actual, list) and actual and isinstance(actual[0], tuple):
        return [list(item) for item in actual] == expected

    return False


def _compare_standard_output(raw_output: str, expected: Any) -> bool:
    output_lines = raw_output.splitlines()
    if _custom_compare_output(output_lines, expected):
        return True

    if isinstance(expected, list):
        expected_lines = [str(item) for item in expected]
    else:
        expected_lines = str(expected).split("\n")

    normalized_output = [line.strip() for line in output_lines if line.strip()]
    normalized_expected = [line.strip() for line in expected_lines if line.strip()]
    if normalized_output == normalized_expected:
        return True

    if normalized_output and normalized_expected:
        if all(_is_digit_like(left) and _is_digit_like(right)
               for left, right in zip(normalized_output, normalized_expected)):
            return normalized_output == normalized_expected
        try:
            output_float = [float(item) for item in normalized_output]
            expected_float = [float(item) for item in normalized_expected]
            if _allclose(output_float, expected_float):
                return True
        except Exception:
            pass

    output_tokens = [set(line.split()) for line in normalized_output if line.split()]
    expected_tokens = [set(line.split()) for line in normalized_expected
                       if line.split()]
    return output_tokens == expected_tokens and bool(output_tokens)


def _run_call_based_tests(sample: Dict[str, Any], generation: str,
                          timeout: int) -> Tuple[List[Any], Dict[str, Any]]:
    source = LCB_SUPPORT_IMPORTS + "\n" + generation
    signal.alarm(timeout)
    try:
        module = _module_from_source("distill_lcb_solution", source)
        signal.alarm(0)
    except Exception as exc:
        signal.alarm(0)
        return ([-2], {
            "error": repr(exc),
            "error_code": -1,
            "error_message": "Compilation Error",
        })

    fn_name = sample.get("fn_name")
    try:
        target = getattr(module, "Solution")() if hasattr(module, "Solution") else module
        method = getattr(target, str(fn_name))
    except Exception as exc:
        return ([-2], {
            "error": repr(exc),
            "error_code": -1,
            "error_message": "Unable to extract code",
        })

    results: List[Any] = []
    for index, raw_inputs in enumerate(sample["inputs"]):
        raw_expected = sample["outputs"][index]
        try:
            inputs = _parse_call_based_inputs(raw_inputs)
            expected = _parse_call_based_output(raw_expected)
        except Exception as exc:
            return ([-2], {
                "error": repr(exc),
                "error_code": -1,
                "error_message": "Malformed Test Case",
            })

        faulthandler.enable()
        signal.alarm(timeout)
        try:
            output = method(*inputs)
            signal.alarm(0)
        except Exception as exc:
            signal.alarm(0)
            faulthandler.disable()
            results.append(-1)
            return (results, {
                "error": repr(exc),
                "error_code": -3 if "timeout" in repr(exc).lower() else -4,
                "error_message": "Time Limit Exceeded"
                if "timeout" in repr(exc).lower() else "Runtime Error",
                "inputs": _truncate(raw_inputs),
                "expected": _truncate(raw_expected, 200),
            })
        faulthandler.disable()

        if _compare_call_based(output, expected):
            results.append(True)
            continue

        results.append(False)
        return (results, {
            "output": _truncate(json.dumps(_normalize_jsonish(output),
                                           ensure_ascii=False,
                                           default=str), 200),
            "expected": _truncate(raw_expected, 200),
            "inputs": _truncate(raw_inputs),
            "error_code": -2,
            "error_message": "Wrong Answer",
        })

    return results, {}


def _run_standard_input_tests(sample: Dict[str, Any], generation: str,
                              timeout: int) -> Tuple[List[Any], Dict[str, Any]]:
    wrapped_code = _prepare_standard_input_code(generation)
    source = LCB_SUPPORT_IMPORTS + "\n" + wrapped_code
    signal.alarm(timeout)
    try:
        module = _module_from_source("distill_lcb_stdio", source)
        method = getattr(module, "code")
        signal.alarm(0)
    except Exception as exc:
        signal.alarm(0)
        return ([-2], {
            "error": repr(exc),
            "error_code": -1,
            "error_message": "Compilation Error",
        })

    results: List[Any] = []
    for index, raw_inputs in enumerate(sample["inputs"]):
        raw_expected = sample["outputs"][index]
        expected: Any = raw_expected
        if isinstance(expected, list):
            expected = "\n".join(str(item) for item in expected)

        faulthandler.enable()
        output_buffer = StringIO()
        signal.alarm(timeout)
        try:
            with patch("sys.stdout", output_buffer):
                _call_with_mocked_stdin(method, raw_inputs)
            signal.alarm(0)
        except Exception as exc:
            signal.alarm(0)
            faulthandler.disable()
            results.append(-1)
            return (results, {
                "error": repr(exc),
                "error_code": -3 if "timeout" in repr(exc).lower() else -4,
                "error_message": "Time Limit Exceeded"
                if "timeout" in repr(exc).lower() else "Runtime Error",
                "inputs": _truncate(raw_inputs),
                "expected": _truncate(str(raw_expected), 200),
            })
        faulthandler.disable()

        output_text = output_buffer.getvalue()
        if _compare_standard_output(output_text, expected):
            results.append(True)
            continue

        results.append(False)
        return (results, {
            "output": _truncate(output_text, 200),
            "expected": _truncate(str(raw_expected), 200),
            "inputs": _truncate(raw_inputs),
            "error_code": -2,
            "error_message": "Wrong Answer",
        })

    return results, {}


def _run_livecodebench_tests(sample: Dict[str, Any], generation: str,
                             timeout: int) -> Tuple[List[Any], Dict[str, Any]]:
    _reliability_guard()
    signal.signal(signal.SIGALRM, _timeout_handler)
    if sample.get("fn_name"):
        return _run_call_based_tests(sample, generation, timeout)
    return _run_standard_input_tests(sample, generation, timeout)


def _run_livecodebench_tests_subprocess(sample: Dict[str, Any],
                                        generation: str,
                                        timeout: int,
                                        queue: multiprocessing.Queue):
    try:
        queue.put(_run_livecodebench_tests(sample, generation, timeout))
    except BaseException as exc:
        queue.put(([-1] * max(1, len(sample.get("inputs", []))), {
            "error": repr(exc),
            "error_code": -5,
            "error_message": "Judge Internal Error",
        }))


def evaluate_livecodebench_generation(sample: Dict[str, Any],
                                      generation: str,
                                      timeout: int = 6) -> Dict[str, Any]:
    case_count = max(1, len(sample.get("inputs", [])))
    ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_run_livecodebench_tests_subprocess,
        args=(sample, generation, timeout, result_queue),
    )
    process.start()
    process.join((timeout + 1) * case_count + 5)

    if process.is_alive():
        process.kill()
        process.join()
        return {
            "status": "timeout",
            "is_correct": False,
            "detail": {
                "error_code": -3,
                "error_message": "Global Time Limit Exceeded",
                "timeout_seconds": timeout,
                "total_tests": case_count,
                "passed_tests": 0,
            },
        }

    try:
        results, detail = result_queue.get_nowait()
    except queue.Empty:
        return {
            "status": "failed",
            "is_correct": False,
            "detail": {
                "error_code": -5,
                "error_message": "No Judge Result",
                "timeout_seconds": timeout,
                "total_tests": case_count,
                "passed_tests": 0,
            },
        }
    passed_tests = sum(1 for item in results if item is True)
    status = "pass"
    if not all(item is True for item in results):
        error_code = detail.get("error_code")
        if error_code == -2:
            status = "wrong_answer"
        elif error_code == -3:
            status = "timeout"
        else:
            status = "failed"

    detail = dict(detail or {})
    detail.update({
        "timeout_seconds": timeout,
        "total_tests": case_count,
        "passed_tests": passed_tests,
        "raw_results": results,
        "mode": "call_based" if sample.get("fn_name") else "standard_input",
        "test_source": sample.get("source"),
    })
    return {
        "status": status,
        "is_correct": status == "pass",
        "detail": detail,
    }
