import argparse
import glob
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, Iterable, Optional


TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _iter_segment_records(output_dir: str,
                          stream: str = "all") -> Iterable[Dict[str, Any]]:
    pattern = os.path.join(output_dir, stream, "segments", "segment_*.jsonl")
    for file_path in sorted(glob.glob(pattern)):
        with open(file_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                yield json.loads(line)


def _normalize_token_value(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(100.0 * numerator / denominator, 4)


def _empty_token_stats() -> Dict[str, Dict[str, Any]]:
    return {
        "sums": {field: 0 for field in TOKEN_FIELDS},
        "counts": {field: 0 for field in TOKEN_FIELDS},
        "averages": {field: None for field in TOKEN_FIELDS},
    }


def _finalize_token_stats(stats: Dict[str, Dict[str, Any]]) -> None:
    for field in TOKEN_FIELDS:
        count = int(stats["counts"].get(field, 0) or 0)
        total = int(stats["sums"].get(field, 0) or 0)
        stats["averages"][field] = round(total / count, 4) if count else None


def _load_resume_state(output_dir: str) -> Dict[str, Any]:
    path = os.path.join(output_dir, ".resume", "resume_state.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return state if isinstance(state, dict) else {}


def summarize_current_stats(output_dir: str) -> Dict[str, Any]:
    total = 0
    correct = 0
    false = 0
    none = 0
    finish_reason_counts: Counter = Counter()
    judge_backend_counts: Counter = Counter()
    all_tokens = _empty_token_stats()
    correct_tokens = _empty_token_stats()

    for record in _iter_segment_records(output_dir, "all"):
        total += 1
        is_correct = record.get("is_correct")
        if is_correct is True:
            correct += 1
        elif is_correct is False:
            false += 1
        else:
            none += 1

        finish_reason_counts[record.get("generation_finish_reason")
                             or "none"] += 1
        judge_backend_counts[record.get("judge_backend") or "none"] += 1

        for field in TOKEN_FIELDS:
            value = _normalize_token_value(record.get(field))
            if value is None:
                continue
            all_tokens["sums"][field] += value
            all_tokens["counts"][field] += 1
            if is_correct is True:
                correct_tokens["sums"][field] += value
                correct_tokens["counts"][field] += 1

    _finalize_token_stats(all_tokens)
    _finalize_token_stats(correct_tokens)

    evaluated = correct + false
    resume_state = _load_resume_state(output_dir)

    return {
        "output_dir": output_dir,
        "records": {
            "total": total,
            "correct": correct,
            "false": false,
            "none": none,
            "evaluated": evaluated,
        },
        "accuracy": {
            "correct_over_total": _pct(correct, total),
            "correct_over_evaluated": _pct(correct, evaluated),
        },
        "tokens": {
            "all": all_tokens,
            "correct": correct_tokens,
        },
        "generation_finish_reason_counts": dict(finish_reason_counts),
        "judge_backend_counts": dict(judge_backend_counts),
        "resume_state": resume_state,
    }


def _print_pretty(summary: Dict[str, Any]) -> None:
    records = summary["records"]
    accuracy = summary["accuracy"]
    all_tokens = summary["tokens"]["all"]
    correct_tokens = summary["tokens"]["correct"]
    resume_progress = (summary.get("resume_state") or {}).get("progress") or {}

    print(f"output_dir: {summary['output_dir']}")
    print(
        "records: "
        f"total={records['total']} correct={records['correct']} "
        f"false={records['false']} none={records['none']} "
        f"evaluated={records['evaluated']}")
    print(
        "accuracy: "
        f"correct/total={accuracy['correct_over_total']}% "
        f"correct/evaluated={accuracy['correct_over_evaluated']}%")
    print("tokens_all:")
    for field in TOKEN_FIELDS:
        print(
            f"  {field}: sum={all_tokens['sums'][field]} "
            f"count={all_tokens['counts'][field]} "
            f"avg={all_tokens['averages'][field]}")
    print("tokens_correct:")
    for field in TOKEN_FIELDS:
        print(
            f"  {field}: sum={correct_tokens['sums'][field]} "
            f"count={correct_tokens['counts'][field]} "
            f"avg={correct_tokens['averages'][field]}")
    if resume_progress:
        print(
            "resume_progress: "
            f"written={resume_progress.get('written')} "
            f"correct={resume_progress.get('correct')} "
            f"overlong={resume_progress.get('overlong')}")


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Summarize current distillation progress from JSONL "
                     "segment files, including correctness and token totals."),
        add_help=add_help,
    )
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--pretty",
                        action="store_true",
                        help="Print a compact human-readable summary.")
    parser.add_argument("--save-json", type=str, default=None)
    return parser


def run_namespace(args: argparse.Namespace) -> Dict[str, Any]:
    summary = summarize_current_stats(args.output_dir)
    if args.pretty:
        _print_pretty(summary)
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return summary


def main(argv: Optional[list[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    run_namespace(args)


if __name__ == "__main__":
    main()
