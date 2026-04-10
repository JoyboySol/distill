import argparse
import glob
import json
import os
from typing import Any, Dict, Iterable, Optional

import pyarrow.parquet as pq


TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def iter_records(output_dir: str, stream: str = "correct") -> Iterable[Dict[str, Any]]:
    pattern = os.path.join(output_dir, stream, "shards", "shard_*.parquet")
    for file_path in sorted(glob.glob(pattern)):
        table = pq.read_table(file_path)
        for row in table.to_pylist():
            yield row


def _normalize_token_value(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def summarize_average_tokens(output_dir: str,
                             stream: str = "correct",
                             only_correct_flag: bool = False) -> Dict[str, Any]:
    record_count = 0
    counted_records = 0
    sums = {field: 0 for field in TOKEN_FIELDS}
    counts = {field: 0 for field in TOKEN_FIELDS}

    for row in iter_records(output_dir, stream):
        record_count += 1
        if only_correct_flag and row.get("is_correct") is not True:
            continue

        counted_records += 1
        for field in TOKEN_FIELDS:
            value = _normalize_token_value(row.get(field))
            if value is None:
                continue
            sums[field] += value
            counts[field] += 1

    averages = {}
    for field in TOKEN_FIELDS:
        if counts[field] == 0:
            averages[field] = None
        else:
            averages[field] = round(sums[field] / counts[field], 4)

    return {
        "output_dir": output_dir,
        "stream": stream,
        "only_correct_flag": only_correct_flag,
        "record_count": record_count,
        "counted_records": counted_records,
        "token_field_counts": counts,
        "token_field_sums": sums,
        "average_tokens": averages,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute average token counts for saved distill outputs.")
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--stream",
                        default="correct",
                        choices=["all", "correct"],
                        help="Which output stream to read.")
    parser.add_argument(
        "--only-correct-flag",
        action="store_true",
        help=("When reading the 'all' stream, keep only rows with "
              "'is_correct == True'."))
    parser.add_argument("--save-json", type=str, default=None)
    args = parser.parse_args()

    summary = summarize_average_tokens(
        output_dir=args.output_dir,
        stream=args.stream,
        only_correct_flag=args.only_correct_flag,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
