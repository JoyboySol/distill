import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pyarrow.parquet as pq

from distill.core.judges.instruction_following import (
    _prompt_constraint_specs,
    _structured_constraint_specs,
)
from distill.runtime.manifest import load_manifest_tasks, select_manifest_tasks


PROFILE_FIELDS: Sequence[str] = ("prompt", "constraints", "ground_truth")


def _is_nonempty_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _iter_task_files(input_dir: str, file_pattern: Optional[str]) -> List[Path]:
    pattern = file_pattern or "*.parquet"
    return sorted(Path(input_dir).expanduser().glob(pattern))


def _parquet_available_columns(file_path: Path) -> List[str]:
    return list(pq.ParquetFile(file_path).schema_arrow.names)


def _iter_parquet_rows(file_path: Path) -> Iterable[Dict[str, Any]]:
    columns = [name for name in PROFILE_FIELDS if name in _parquet_available_columns(file_path)]
    table = pq.read_table(file_path, columns=columns or None)
    for row in table.to_pylist():
        yield row


def _iter_jsonl_rows(file_path: Path) -> Iterable[Dict[str, Any]]:
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_rows(file_path: Path) -> Iterable[Dict[str, Any]]:
    if file_path.suffix.lower() == ".parquet":
        yield from _iter_parquet_rows(file_path)
        return
    if file_path.suffix.lower() == ".jsonl":
        yield from _iter_jsonl_rows(file_path)
        return
    raise ValueError(f"Unsupported input file type: {file_path}")


def _estimate_file_rows(file_path: Path) -> int:
    if file_path.suffix.lower() == ".parquet":
        return int(pq.ParquetFile(file_path).metadata.num_rows)
    if file_path.suffix.lower() == ".jsonl":
        with file_path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    raise ValueError(f"Unsupported input file type: {file_path}")


def _raw_constraint_values(row: Dict[str, Any]) -> List[str]:
    constraints = row.get("constraints")
    if isinstance(constraints, list):
        return [str(item) for item in constraints if _is_nonempty_value(item)]

    specs = _structured_constraint_specs(row)
    if specs is None:
        return []
    values = []
    for spec in specs:
        source = spec.get("source")
        if _is_nonempty_value(source):
            values.append(str(source))
    return values


def _supportability_bucket(row: Dict[str, Any]) -> str:
    if _is_nonempty_value(row.get("ground_truth")):
        specs = _structured_constraint_specs(row)
    else:
        constraints = row.get("constraints")
        if not isinstance(constraints, list) or not constraints:
            return "no_constraint_rows"
        specs = _prompt_constraint_specs(row)

    if not specs:
        return "unsupported_rows"
    if any(spec.get("kind") == "__unsupported__" for spec in specs):
        return "unsupported_rows"
    return "supported_rows"


def _profile_task(task_values: Dict[str, Any]) -> Dict[str, Any]:
    input_dir = str(task_values.get("input_dir") or "")
    file_pattern = str(task_values.get("file_pattern") or "*.parquet")
    file_paths = _iter_task_files(input_dir, file_pattern)

    field_presence: Counter = Counter()
    constraint_counts: Counter = Counter()
    constraint_combo_counts: Counter = Counter()
    supportability: Counter = Counter({
        "supported_rows": 0,
        "unsupported_rows": 0,
        "no_constraint_rows": 0,
    })
    estimated_row_count = 0

    for file_path in file_paths:
        estimated_row_count += _estimate_file_rows(file_path)
        for row in _iter_rows(file_path):
            for field in PROFILE_FIELDS:
                if _is_nonempty_value(row.get(field)):
                    field_presence[field] += 1

            constraints = _raw_constraint_values(row)
            for constraint in constraints:
                constraint_counts[constraint] += 1
            if constraints:
                constraint_combo_counts[" | ".join(constraints)] += 1

            supportability[_supportability_bucket(row)] += 1

    return {
        "task_name": str(task_values.get("task_name") or ""),
        "input_dir": input_dir,
        "file_pattern": file_pattern,
        "file_count": len(file_paths),
        "estimated_row_count": estimated_row_count,
        "field_presence": {
            field: int(field_presence.get(field, 0))
            for field in PROFILE_FIELDS
        },
        "constraint_counts": dict(constraint_counts),
        "constraint_combo_counts": dict(constraint_combo_counts),
        "supportability": {
            "supported_rows": int(supportability["supported_rows"]),
            "unsupported_rows": int(supportability["unsupported_rows"]),
            "no_constraint_rows": int(supportability["no_constraint_rows"]),
        },
    }


def summarize_manifest_input_profiles(config_path: str,
                                      task_name: Optional[str] = None
                                      ) -> List[Dict[str, Any]]:
    manifest_path = Path(config_path).expanduser().resolve()
    task_values_list = select_manifest_tasks(load_manifest_tasks(manifest_path),
                                             task_name=task_name)
    return [_profile_task(task_values) for task_values in task_values_list]


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Profile raw manifest-backed input datasets before "
                     "distillation."),
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
    parser.add_argument("--summary-json-path",
                        type=str,
                        default=None,
                        help="Optional path to save the profile rows as JSON.")
    return parser


def run_namespace(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = summarize_manifest_input_profiles(
        config_path=args.config,
        task_name=args.task_name,
    )
    if args.summary_json_path:
        output_path = Path(args.summary_json_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) +
                               "\n",
                               encoding="utf-8")

    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return rows


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    run_namespace(args)


if __name__ == "__main__":
    main()
