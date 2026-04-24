import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distill.runtime.manifest import load_manifest_tasks, select_manifest_tasks
from scripts.avg_correct_tokens import summarize_average_tokens


CSV_COLUMNS = (
    "task_name",
    "field",
    "total_records",
    "token_count_total",
    "token_sum_total",
    "avg_total_tokens",
)
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _build_task_rows(task_name: str,
                     output_dir: str,
                     stream: str = "correct") -> List[Dict[str, Any]]:
    summary = summarize_average_tokens(output_dir=output_dir, stream=stream)
    token_counts = summary.get("token_field_counts") or {}
    token_sums = summary.get("token_field_sums") or {}
    avg_total_tokens = (summary.get("average_tokens") or {}).get("total_tokens")
    total_records = int(summary.get("record_count", 0) or 0)

    rows: List[Dict[str, Any]] = []
    for field in TOKEN_FIELDS:
        rows.append({
            "task_name": task_name,
            "field": field,
            "total_records": total_records,
            "token_count_total": int(token_counts.get(field, 0) or 0),
            "token_sum_total": int(token_sums.get(field, 0) or 0),
            "avg_total_tokens": avg_total_tokens,
        })
    return rows


def summarize_manifest_token_stats(config_path: str,
                                   task_name: Optional[str] = None,
                                   stream: str = "correct") -> List[Dict[str, Any]]:
    manifest_path = Path(config_path).expanduser().resolve()
    task_values_list = select_manifest_tasks(load_manifest_tasks(manifest_path),
                                             task_name=task_name)

    rows: List[Dict[str, Any]] = []
    for task_values in task_values_list:
        output_dir = task_values.get("output_dir")
        if not output_dir:
            raise ValueError(
                f"Task {task_values.get('task_name')!r} is missing output_dir in {manifest_path}"
            )
        rows.extend(
            _build_task_rows(
                task_name=str(task_values.get("task_name") or ""),
                output_dir=str(output_dir),
                stream=stream,
            ))
    return rows


def write_summary_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in CSV_COLUMNS})


def main():
    parser = argparse.ArgumentParser(
        description=("Summarize token counts for each task in a manifest and "
                     "write a strict CSV report."))
    parser.add_argument("--config",
                        required=True,
                        type=str,
                        help="Manifest/task YAML path.")
    parser.add_argument("--task-name",
                        type=str,
                        default=None,
                        help="Optional single task_name filter inside manifest.")
    parser.add_argument("--stream",
                        type=str,
                        default="correct",
                        choices=["all", "correct"],
                        help="Which output stream to read. Defaults to correct.")
    parser.add_argument("--summary-csv-path",
                        type=str,
                        default="manifest_token_stats.csv",
                        help="Where to write the CSV summary.")
    parser.add_argument("--summary-json-path",
                        type=str,
                        default=None,
                        help="Optional path to also save the row payload as JSON.")
    args = parser.parse_args()

    rows = summarize_manifest_token_stats(
        config_path=args.config,
        task_name=args.task_name,
        stream=args.stream,
    )

    csv_path = Path(args.summary_csv_path)
    write_summary_csv(rows, csv_path)

    if args.summary_json_path:
        json_path = Path(args.summary_json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )

    print(json.dumps({
        "config_path": str(Path(args.config).expanduser().resolve()),
        "task_name": args.task_name,
        "stream": args.stream,
        "row_count": len(rows),
        "summary_csv_path": str(csv_path.resolve()),
        "summary_json_path": (str(Path(args.summary_json_path).expanduser().resolve())
                               if args.summary_json_path else None),
    },
                     ensure_ascii=False,
                     indent=2,
                     sort_keys=True))


if __name__ == "__main__":
    main()
