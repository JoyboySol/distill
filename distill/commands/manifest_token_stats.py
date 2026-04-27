import argparse
import concurrent.futures
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from distill.runtime.manifest import load_manifest_tasks, select_manifest_tasks
from .avg_correct_tokens import summarize_average_tokens

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


CSV_COLUMNS = (
    "task_name",
    "field",
    "total_records",
    "token_count_total",
    "token_sum_total",
    "avg_total_tokens",
)


def _build_task_row(task_name: str,
                    field: str,
                    output_dir: str,
                     stream: str = "correct") -> Dict[str, Any]:
    summary = summarize_average_tokens(output_dir=output_dir, stream=stream)
    token_counts = summary.get("token_field_counts") or {}
    token_sums = summary.get("token_field_sums") or {}
    avg_total_tokens = (summary.get("average_tokens") or {}).get("total_tokens")
    total_records = int(summary.get("record_count", 0) or 0)

    return {
        "task_name": task_name,
        "field": field,
        "total_records": total_records,
        "token_count_total": int(token_counts.get("total_tokens", 0) or 0),
        "token_sum_total": int(token_sums.get("total_tokens", 0) or 0),
        "avg_total_tokens": avg_total_tokens,
    }


def _build_task_row_from_manifest_entry(task_index: int,
                                        task_values: Dict[str, Any],
                                        stream: str = "correct") -> Dict[str, Any]:
    output_dir = task_values.get("output_dir")
    if not output_dir:
        raise ValueError(
            f"Task {task_values.get('task_name')!r} is missing output_dir")

    row = _build_task_row(
        task_name=str(task_values.get("task_name") or ""),
        field=str(task_values.get("field") or ""),
        output_dir=str(output_dir),
        stream=stream,
    )
    row["_task_index"] = task_index
    return row


def summarize_manifest_token_stats(config_path: str,
                                   task_name: Optional[str] = None,
                                   stream: str = "correct",
                                   max_workers: Optional[int] = None,
                                   show_progress: bool = False) -> List[Dict[str, Any]]:
    manifest_path = Path(config_path).expanduser().resolve()
    task_values_list = select_manifest_tasks(load_manifest_tasks(manifest_path),
                                             task_name=task_name)

    field_order: Dict[str, int] = {}
    indexed_tasks = []
    for task_index, task_values in enumerate(task_values_list):
        field = str(task_values.get("field") or "")
        if field not in field_order:
            field_order[field] = len(field_order)
        indexed_tasks.append((task_index, task_values))

    if max_workers is None:
        cpu_count = os.cpu_count() or 1
        max_workers = min(max(1, cpu_count), max(1, len(indexed_tasks)))
    else:
        max_workers = max(1, int(max_workers))

    rows: List[Dict[str, Any]] = []
    progress = None
    if show_progress and tqdm is not None:
        progress = tqdm(total=len(indexed_tasks),
                        desc="Summarizing tasks",
                        unit="task",
                        dynamic_ncols=True)

    try:
        if max_workers == 1:
            for task_index, task_values in indexed_tasks:
                row = _build_task_row_from_manifest_entry(task_index,
                                                          task_values,
                                                          stream=stream)
                row["_field_order"] = field_order[str(row.get("field") or "")]
                rows.append(row)
                if progress is not None:
                    progress.update(1)
        else:
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_build_task_row_from_manifest_entry,
                                    task_index,
                                    task_values,
                                    stream)
                    for task_index, task_values in indexed_tasks
                ]
                for future in concurrent.futures.as_completed(futures):
                    row = future.result()
                    row["_field_order"] = field_order[str(row.get("field") or "")]
                    rows.append(row)
                    if progress is not None:
                        progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    rows.sort(key=lambda row: (int(row["_field_order"]), int(row["_task_index"])))
    for row in rows:
        row.pop("_task_index", None)
        row.pop("_field_order", None)
    return rows


def write_summary_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in CSV_COLUMNS})


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Summarize token counts for each task in a manifest and "
                     "write a strict CSV report."),
        add_help=add_help,
    )
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
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=("How many tasks to summarize in parallel. Defaults to a bounded "
              "CPU-based value."),
    )
    return parser


def run_namespace(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = summarize_manifest_token_stats(
        config_path=args.config,
        task_name=args.task_name,
        stream=args.stream,
        max_workers=args.max_workers,
        show_progress=True,
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
        "max_workers": args.max_workers,
        "summary_csv_path": str(csv_path.resolve()),
        "summary_json_path": (str(Path(args.summary_json_path).expanduser().resolve())
                               if args.summary_json_path else None),
    },
                     ensure_ascii=False,
                     indent=2,
                     sort_keys=True))
    return rows


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    run_namespace(args)


if __name__ == "__main__":
    main()
