import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pyarrow.parquet as pq

from distill.core.judge import judge_output


def iter_rows(input_dir: Path, limit: int, batch_size: int = 128):
    seen = 0
    for parquet_path in sorted(input_dir.glob("*.parquet")):
        parquet_file = pq.ParquetFile(parquet_path)
        available_columns = set(parquet_file.schema.names)
        preferred_columns = [
            "id",
            "input",
            "output",
            "solution",
            "input_output",
            "evaluation_sample",
            "public_test_cases",
            "private_test_cases",
            "metadata",
            "prompt",
            "test",
            "entry_point",
            "test_list",
            "test_list_2",
        ]
        columns = [name for name in preferred_columns if name in available_columns]
        row_offset = 0
        for batch in parquet_file.iter_batches(batch_size=batch_size,
                                               columns=columns):
            rows = batch.to_pylist()
            for inner_idx, row in enumerate(rows):
                yield parquet_path, row_offset + inner_idx, row
                seen += 1
                if limit > 0 and seen >= limit:
                    return
            row_offset += len(rows)


def evaluate_rows(input_dir: Path,
                  limit: int,
                  progress_every: int = 0) -> Dict[str, object]:
    status_counter: Counter = Counter()
    judge_type_counter: Counter = Counter()
    backend_counter: Counter = Counter()
    code_fence_counter: Counter = Counter()
    sample_records: List[Dict[str, object]] = []

    total = 0
    for parquet_path, row_idx, row in iter_rows(input_dir, limit):
        assistant_output = row.get("output") or ""
        messages = [{
            "role": "assistant",
            "content": str(assistant_output),
        }]
        result = judge_output(row, messages)

        total += 1
        status_counter[result.get("judge_status")] += 1
        judge_type_counter[result.get("judge_type")] += 1
        backend_counter[result.get("judge_backend")] += 1
        code_fence_counter["has_code_fence" if "```" in str(assistant_output)
                           else "no_code_fence"] += 1

        if len(sample_records) < 20:
            sample_records.append({
                "source_file": parquet_path.name,
                "source_row": row_idx,
                "id": row.get("id"),
                "judge_type": result.get("judge_type"),
                "judge_backend": result.get("judge_backend"),
                "judge_status": result.get("judge_status"),
                "is_correct": result.get("is_correct"),
                "judge_detail": result.get("judge_detail"),
            })

        if progress_every > 0 and total % progress_every == 0:
            print(f"evaluated={total}", flush=True)

    return {
        "total": total,
        "status_counter": dict(status_counter),
        "judge_type_counter": dict(judge_type_counter),
        "backend_counter": dict(backend_counter),
        "code_fence_counter": dict(code_fence_counter),
        "sample_records": sample_records,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate existing dataset outputs using distill judges.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = evaluate_rows(args.input_dir,
                            args.limit,
                            progress_every=args.progress_every)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return

    print(f"Total rows: {summary['total']}")
    print("Judge types:", summary["judge_type_counter"])
    print("Judge backends:", summary["backend_counter"])
    print("Judge status:", summary["status_counter"])
    print("Code fences:", summary["code_fence_counter"])
    print("Sample records:")
    for record in summary["sample_records"][:5]:
        print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
