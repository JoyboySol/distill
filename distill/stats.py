import argparse
import glob
import json
import os
from collections import Counter
from typing import Any, Dict, Iterable

import pyarrow.parquet as pq


def iter_records(output_dir: str, stream: str = "all") -> Iterable[Dict[str,
                                                                       Any]]:
    pattern = os.path.join(output_dir, stream, "shards", "shard_*.parquet")
    for file_path in sorted(glob.glob(pattern)):
        table = pq.read_table(file_path)
        for row in table.to_pylist():
            yield row


def pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def summarize(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    judge_type_counter: Counter = Counter()
    judge_backend_counter: Counter = Counter()
    judge_status_counter: Counter = Counter()
    correct_counter: Counter = Counter()
    backend_correct_counter: Counter = Counter()
    backend_total_counter: Counter = Counter()
    fallback_reasons: Counter = Counter()

    for row in records:
        total += 1
        judge_type = row.get("judge_type") or "none"
        judge_backend = row.get("judge_backend") or "none"
        judge_status = row.get("judge_status") or "none"
        is_correct = row.get("is_correct")
        judge_detail = row.get("judge_detail") or {}

        judge_type_counter[judge_type] += 1
        judge_backend_counter[judge_backend] += 1
        judge_status_counter[judge_status] += 1
        backend_total_counter[judge_backend] += 1

        if is_correct is True:
            correct_counter["true"] += 1
            backend_correct_counter[judge_backend] += 1
        elif is_correct is False:
            correct_counter["false"] += 1
        else:
            correct_counter["none"] += 1

        fallback_reason = judge_detail.get("verify_fallback_reason")
        if fallback_reason:
            fallback_reasons[fallback_reason] += 1

    backend_accuracy = {}
    for backend, backend_total in backend_total_counter.items():
        backend_accuracy[backend] = {
            "correct": backend_correct_counter[backend],
            "total": backend_total,
            "accuracy": round(
                pct(backend_correct_counter[backend], backend_total), 4),
        }

    math_total = sum(
        count for backend, count in judge_backend_counter.items()
        if backend in {"math_verify", "math_rule"})
    math_verify_total = judge_backend_counter["math_verify"]
    math_rule_total = judge_backend_counter["math_rule"]

    return {
        "total_records": total,
        "judge_type_counts": dict(judge_type_counter),
        "judge_backend_counts": dict(judge_backend_counter),
        "judge_status_counts": dict(judge_status_counter),
        "correct_counts": dict(correct_counter),
        "overall_accuracy":
        round(pct(correct_counter["true"], correct_counter["true"] +
                  correct_counter["false"]), 4)
        if (correct_counter["true"] + correct_counter["false"]) > 0 else 0.0,
        "backend_accuracy": backend_accuracy,
        "math_backend_summary": {
            "math_total": math_total,
            "math_verify_total": math_verify_total,
            "math_rule_total": math_rule_total,
            "math_verify_ratio":
            round(pct(math_verify_total, math_total), 4) if math_total else 0.0,
            "math_rule_ratio":
            round(pct(math_rule_total, math_total), 4) if math_total else 0.0,
        },
        "math_verify_fallback_reasons": dict(fallback_reasons),
    }


def print_summary(summary: Dict[str, Any]):
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--stream",
                        type=str,
                        default="all",
                        choices=["all", "correct"])
    parser.add_argument("--save-json", type=str, default=None)
    args = parser.parse_args()

    summary = summarize(iter_records(args.output_dir, args.stream))
    print_summary(summary)

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
