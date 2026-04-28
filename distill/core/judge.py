import multiprocessing
import queue
from typing import Any, Dict, List, Optional

from .judges.code import extract_code_text, extract_code_text_last_block, \
    run_python_program
from .judges.dispatcher import judge_output_impl, judge_type_hint
from .judges.math import MathJudge, MathVerifyJudge


class JudgeSubprocessTimeoutError(TimeoutError):
    pass


def judge_output(row_data: Dict[str, Any],
                 messages: List[Dict[str, Any]],
                 label_field: Optional[str] = None,
                 judge_mode: Optional[str] = None) -> Dict[str, Any]:
    return judge_output_impl(row_data, messages, label_field, judge_mode)


def _guarded_judge_result(row_data: Dict[str, Any],
                          messages: List[Dict[str, Any]],
                          label_field: Optional[str],
                          judge_mode: Optional[str],
                          judge_status: str,
                          detail: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "judge_type": judge_type_hint(row_data, messages, label_field,
                                      judge_mode),
        "judge_backend": "judge_subprocess_guard_v1",
        "is_correct": None,
        "judge_status": judge_status,
        "judge_detail": detail,
    }


def _judge_output_process_entry(result_queue,
                                row_data: Dict[str, Any],
                                messages: List[Dict[str, Any]],
                                label_field: Optional[str],
                                judge_mode: Optional[str]):
    try:
        result_queue.put({
            "ok": True,
            "result": judge_output_impl(row_data, messages, label_field,
                                        judge_mode),
        })
    except BaseException as exc:
        result_queue.put({
            "ok": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        })


def judge_output_with_timeout(
    row_data: Dict[str, Any],
    messages: List[Dict[str, Any]],
    label_field: Optional[str] = None,
    judge_mode: Optional[str] = None,
    timeout: Optional[float] = 20.0,
) -> Dict[str, Any]:
    if timeout is None or timeout <= 0:
        return judge_output_impl(row_data, messages, label_field, judge_mode)

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_judge_output_process_entry,
        args=(result_queue, row_data, messages, label_field, judge_mode),
    )
    process.start()

    try:
        process.join(timeout=timeout)
        if process.is_alive():
            process.kill()
            process.join()
            return _guarded_judge_result(
                row_data,
                messages,
                label_field,
                judge_mode,
                "timeout",
                {
                    "reason": "judge_process_timeout",
                    "timeout_seconds": timeout,
                },
            )

        try:
            payload = result_queue.get_nowait()
        except queue.Empty:
            return _guarded_judge_result(
                row_data,
                messages,
                label_field,
                judge_mode,
                "failed",
                {
                    "reason": "judge_process_no_result",
                    "exitcode": process.exitcode,
                },
            )

        if payload.get("ok"):
            return payload["result"]

        return _guarded_judge_result(
            row_data,
            messages,
            label_field,
            judge_mode,
            "failed",
            {
                "reason": "judge_process_error",
                "error_type": payload.get("error_type"),
                "error_message": payload.get("error_message"),
            },
        )
    finally:
        result_queue.close()
        result_queue.join_thread()
