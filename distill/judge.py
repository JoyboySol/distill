import contextlib
import io
import multiprocessing
import re
import signal
from typing import Any, Dict, List, Optional


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


class MathJudge:
    @staticmethod
    def last_boxed_only_string(string: str):
        idx = string.rfind("\\boxed")
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is None:
            return None
        return string[idx:right_brace_idx + 1]

    @staticmethod
    def remove_boxed(s: str):
        left = "\\boxed{"
        try:
            assert s[:len(left)] == left
            assert s[-1] == "}"
            return s[len(left):-1]
        except Exception:
            return None

    @classmethod
    def extract_boxed_answer(cls,
                             pred_str: str,
                             strip_double_curly_brace: bool = False):
        boxed_str = cls.last_boxed_only_string(pred_str)
        if boxed_str is None:
            return None
        answer = cls.remove_boxed(boxed_str)
        if answer is None:
            return None
        if strip_double_curly_brace:
            match = re.match(r"^\{(.*)\}$", answer)
            if match:
                answer = match.group(1)
        return answer

    @staticmethod
    def normalize_final_answer(final_answer: str) -> str:
        substitutions = [
            ("an ", ""),
            ("a ", ""),
            (".$", "$"),
            ("\\$", ""),
            (r"\ ", ""),
            (" ", ""),
            ("mbox", "text"),
            (",\\text{and}", ","),
            ("\\text{and}", ","),
            ("\\text{m}", "\\text{}"),
            ("\\le", "<"),
        ]
        removed_expressions = [
            "square",
            "ways",
            "integers",
            "dollars",
            "mph",
            "inches",
            "ft",
            "hours",
            "km",
            "units",
            "\\ldots",
            "sue",
            "points",
            "feet",
            "minutes",
            "digits",
            "cents",
            "degrees",
            "cm",
            "gm",
            "pounds",
            "meters",
            "meals",
            "edges",
            "students",
            "childrentickets",
            "multiples",
            "\\text{s}",
            "\\text{.}",
            "\\text{\ns}",
            "\\text{}^2",
            "\\text{}^3",
            "\\text{\n}",
            "\\text{}",
            r"\mathrm{th}",
            r"^\circ",
            r"^{\circ}",
            r"\;",
            r",\!",
            "{,}",
            '"',
            "\\dots",
            "\n",
            "\r",
            "\f",
        ]
        for before, after in substitutions:
            final_answer = final_answer.replace(before, after)
        for expr in removed_expressions:
            final_answer = final_answer.replace(expr, "")

        final_answer = re.sub(r"(\\text\{)\((.*?)\)(\})", r"\2",
                              final_answer)
        final_answer = re.sub(r"(\\text\{)(.*?)(\})", r"\2", final_answer)
        final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", r"\2", final_answer)
        final_answer = re.sub(r"(\\overline\{)(.*?)(\})", r"\2",
                              final_answer)
        final_answer = re.sub(r"(\\boxed\{)(.*)(\})", r"\2", final_answer)
        if len(re.findall(r"finalansweris(.*)", final_answer)) > 0:
            final_answer = re.findall(r"finalansweris(.*)",
                                      final_answer)[-1]
        if len(re.findall(r"answer?is:?(.*)", final_answer)) > 0:
            final_answer = re.findall(r"answer?is:?(.*)",
                                      final_answer)[-1]
        if len(re.findall(r"oxed\{(.*?)\}", final_answer)) > 0:
            final_answer = re.findall(r"oxed\{(.*?)\}", final_answer)[-1]
        if len(re.findall(r"\$(.*?)\$", final_answer)) > 0:
            final_answer = re.findall(r"\$(.*?)\$", final_answer)[-1]

        final_answer = final_answer.strip()
        if "rac" in final_answer and "\\frac" not in final_answer:
            final_answer = final_answer.replace("rac", "\\frac")
        final_answer = re.sub(r"(frac)([^{])(.)", r"frac{\2}{\3}",
                              final_answer)
        final_answer = re.sub(r"(sqrt)([^{])", r"sqrt{\2}", final_answer)
        final_answer = final_answer.replace("$", "")
        if final_answer.replace(",", "").isdigit():
            final_answer = final_answer.replace(",", "")
        return final_answer

    @classmethod
    def math_postprocess_v2(cls, text: str) -> str:
        cand_ans = cls.extract_boxed_answer(
            text, strip_double_curly_brace=True)
        if cand_ans:
            return cand_ans
        for maybe_ans in text.split("."):
            if re.search(r"final answer|answer is", maybe_ans.lower()):
                return cls.normalize_final_answer(maybe_ans)
        return cls.normalize_final_answer(text.split(".")[0])

    @staticmethod
    def _fix_fracs(string: str):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if len(substr) > 0 and substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except AssertionError:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        post_substr = substr[2:] if len(substr) > 2 else ""
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        post_substr = substr[2:] if len(substr) > 2 else ""
                        new_str += "{" + a + "}" + b + post_substr
        return new_str

    @staticmethod
    def _fix_a_slash_b(string: str):
        if len(string.split("/")) != 2:
            return string
        a, b = string.split("/")
        try:
            a = int(a)
            b = int(b)
            assert string == f"{a}/{b}"
            return "\\frac{" + str(a) + "}{" + str(b) + "}"
        except Exception:
            return string

    @staticmethod
    def _fix_sqrt_v2(string: str):
        return re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)

    @classmethod
    def _strip_string_v2(cls, string: str):
        string = str(string).strip()
        string = string.replace("\n", "")
        string = string.rstrip(".")
        string = string.replace("\\!", "")
        string = string.replace("\\ ", "")
        string = string.replace("\\\\", "\\")
        string = string.replace("tfrac", "frac")
        string = string.replace("dfrac", "frac")
        string = string.replace("\\left", "")
        string = string.replace("\\right", "")

        stripped = re.sub(r"\\text{.*?}$", "", string).strip()
        if stripped not in ("", string):
            string = stripped

        string = string.replace("^{\\circ}", "")
        string = string.replace("^\\circ", "")
        string = string.replace("\\$", "")
        string = string.replace("$", "")
        string = string.replace("\\text", "")
        string = string.replace("x\\in", "")
        string = string.replace("\\%", "")
        string = string.replace("%", "")
        string = string.replace(" .", " 0.")
        string = string.replace("{.", "{0.")
        string = string.replace("\\cdot", "")
        string = string.replace("infinity", "\\infty")
        if "\\infty" not in string:
            string = string.replace("inf", "\\infty")
        string = string.replace("+\\inity", "\\infty")
        string = string.replace("and", "")
        string = string.replace("\\mathbf", "")
        string = re.sub(r"\\mbox{.*?}", "", string)
        string = string.replace("'", "")
        string = string.replace('"', "")
        if "j" in string and "i" not in string:
            string = string.replace("j", "i")
        string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
        string = re.sub(r"(\d+)\.0+$", r"\1", string)
        if len(string) == 0:
            return string
        if string[0] == ".":
            string = "0" + string
        if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
        string = cls._fix_sqrt_v2(string)
        string = string.replace(" ", "")
        string = cls._fix_fracs(string)
        string = cls._fix_a_slash_b(string)
        return string

    @classmethod
    def is_equiv(cls, str1: Optional[str], str2: Optional[str]) -> bool:
        if str1 is None and str2 is None:
            return True
        if str1 is None or str2 is None:
            return False

        try:
            ss1 = cls._strip_string_v2(str1)
            ss2 = cls._strip_string_v2(str2)
            if ss1 == ss2:
                return True
            ss1 = cls.normalize_final_answer(ss1)
            ss2 = cls.normalize_final_answer(ss2)
            if ss1 == ss2:
                return True
        except Exception:
            pass

        try:
            ss1 = cls.normalize_final_answer(str1)
            ss2 = cls.normalize_final_answer(str2)
            if ss1 == ss2:
                return True
        except Exception:
            pass

        return str1 == str2


class MathVerifyJudge:
    @staticmethod
    def is_available() -> bool:
        try:
            from latex2sympy2_extended import NormalizationConfig  # noqa: F401
            from math_verify import (ExprExtractionConfig,  # noqa: F401
                                     LatexExtractionConfig, parse,
                                     verify)  # noqa: F401
            return True
        except Exception:
            return False

    @staticmethod
    def verify(prediction: str, reference: str) -> Dict[str, Any]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import (ExprExtractionConfig, LatexExtractionConfig,
                                 parse, verify)

        gold_with_env = f"${reference}$"
        gold_parsed = parse(
            gold_with_env,
            extraction_mode="first_match",
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=False,
                        boxed="all",
                        units=False,
                    )),
                ExprExtractionConfig(),
            ],
        )

        if len(gold_parsed) == 0:
            return {
                "usable": False,
                "is_correct": None,
                "detail": {
                    "reason": "gold_parse_failed",
                    "prediction": prediction,
                    "reference": reference,
                },
            }

        answer_parsed = parse(
            prediction,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=False,
                        boxed="all",
                        units=False,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        if len(answer_parsed) == 0:
            return {
                "usable": False,
                "is_correct": None,
                "detail": {
                    "reason": "prediction_parse_failed",
                    "prediction": prediction,
                    "reference": reference,
                },
            }

        is_correct = bool(verify(answer_parsed, gold_parsed))
        return {
            "usable": True,
            "is_correct": is_correct,
            "detail": {
                "prediction": str(answer_parsed),
                "reference": str(gold_parsed),
            },
        }


def assistant_text(messages: List[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def extract_code_text(text: str) -> str:
    blocks = re.findall(r"```\w*\n(.*?)```", text, re.DOTALL)
    if len(blocks) >= 1:
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


def try_math_reference(row_data: Dict[str, Any],
                       preferred_field: Optional[str] = None) -> Optional[str]:
    candidates = []
    if preferred_field:
        candidates.append(preferred_field)
    candidates.extend([
        "solution",
        "answer",
        "expected_answer",
        "final_answer",
        "gold",
        "gold_answer",
        "target",
        "reference",
        "label",
    ])
    for key in candidates:
        value = row_data.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


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


def judge_output(row_data: Dict[str, Any],
                 messages: List[Dict[str, Any]],
                 label_field: Optional[str] = None) -> Dict[str, Any]:
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

    reference = try_math_reference(row_data, preferred_field=label_field)
    if reference is not None:
        raw_reference = reference
        pred = MathJudge.math_postprocess_v2(content)
        ref = MathJudge.math_postprocess_v2(reference)

        if MathVerifyJudge.is_available():
            try:
                verify_result = MathVerifyJudge.verify(content, raw_reference)
                if verify_result["usable"]:
                    return {
                        "judge_type": "math",
                        "judge_backend": "math_verify",
                        "is_correct": verify_result["is_correct"],
                        "judge_status":
                        "pass" if verify_result["is_correct"] else "wrong_answer",
                        "judge_detail": {
                            "label_field": label_field,
                            **verify_result["detail"],
                        },
                    }
            except Exception as exc:
                verify_failure = {
                    "verify_error": f"{type(exc).__name__}:{exc}",
                }
            else:
                verify_failure = {
                    "verify_fallback_reason":
                    verify_result["detail"].get("reason", "not_usable"),
                }
        else:
            verify_failure = {
                "verify_fallback_reason": "math_verify_unavailable",
            }

        is_correct = MathJudge.is_equiv(pred, ref)
        detail = {
            "prediction": pred,
            "reference": ref,
            "label_field": label_field,
        }
        detail.update(verify_failure)
        return {
            "judge_type": "math",
            "judge_backend": "math_rule",
            "is_correct": is_correct,
            "judge_status": "pass" if is_correct else "wrong_answer",
            "judge_detail": detail,
        }

    return {
        "judge_type": None,
        "judge_backend": None,
        "is_correct": None,
        "judge_status": "not_applicable",
        "judge_detail": None,
    }
