import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pyarrow as pa
import pyarrow.parquet as pq


DATASET_SPECS = {
    "apps": {
        "hf_path": "codeparrot/apps",
        "question_field": "question",
        "tests_field": "input_output",
    },
    "taco": {
        "hf_path": "BAAI/TACO",
        "question_field": "question",
        "tests_field": "input_output",
    },
}


def _load_remote_split(dataset_name: str, split_name: str):
    from datasets import load_dataset

    spec = DATASET_SPECS[dataset_name]
    return load_dataset(spec["hf_path"], split=split_name)


def _normalize_tests(raw_value) -> str:
    if raw_value is None:
        return ""
    if isinstance(raw_value, str):
        return raw_value
    return json.dumps(raw_value, ensure_ascii=False)


def hydrate_row(row: Dict[str, object], source_row: Dict[str, object]) -> Dict[str, object]:
    dataset_name = str(row.get("dataset") or "").lower()
    spec = DATASET_SPECS[dataset_name]

    question = source_row.get(spec["question_field"])
    if isinstance(question, str) and question.strip():
        row["input"] = question

    row["input_output"] = _normalize_tests(source_row.get(spec["tests_field"]))
    row["generated_test_source"] = f"{dataset_name}_source_dataset"
    row["generated_test_parser"] = "source_dataset_v1"
    row["generated_test_count"] = 1 if row["input_output"] else 0
    row["generated_test_status"] = "hydrated" if row["input_output"] else "missing"
    return row


def hydrate_file(input_path: Path,
                 output_path: Path,
                 batch_size: int = 128) -> Dict[str, int]:
    parquet_file = pq.ParquetFile(input_path)
    datasets_cache: Dict[str, object] = {}
    writer: Optional[pq.ParquetWriter] = None
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    hydrated = 0
    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            rows_batch = []
            for row in batch.to_pylist():
                row = dict(row)
                dataset_name = str(row.get("dataset") or "").lower()
                split_name = str(row.get("split") or "train")
                index_text = str(row.get("index") or "").strip()

                if dataset_name in DATASET_SPECS and index_text.isdigit():
                    if dataset_name not in datasets_cache:
                        datasets_cache[dataset_name] = _load_remote_split(
                            dataset_name, split_name)
                    source_ds = datasets_cache[dataset_name]
                    source_row = source_ds[int(index_text)]
                    row = hydrate_row(row, source_row)
                    if row.get("generated_test_status") == "hydrated":
                        hydrated += 1
                else:
                    row["generated_test_source"] = ""
                    row["generated_test_parser"] = "source_dataset_v1"
                    row["generated_test_count"] = 0
                    row["generated_test_status"] = "missing"
                    row["input_output"] = row.get("input_output") or ""

                rows_batch.append(row)
                rows += 1

            table = pa.Table.from_pylist(rows_batch)
            if writer is None:
                writer = pq.ParquetWriter(temp_path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    temp_path.replace(output_path)
    return {
        "rows": rows,
        "hydrated": hydrated,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Hydrate OpenCodeReasoning split_1 with source prompts/tests.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*.parquet"))
    total_rows = 0
    total_hydrated = 0

    for idx, input_path in enumerate(files, 1):
        output_path = args.output_dir / input_path.name
        if output_path.exists() and not args.overwrite:
            continue
        summary = hydrate_file(input_path,
                               output_path,
                               batch_size=args.batch_size)
        total_rows += summary["rows"]
        total_hydrated += summary["hydrated"]
        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(
                f"processed_files={idx}/{len(files)} hydrated={total_hydrated} rows={total_rows}",
                flush=True,
            )

    print(json.dumps({
        "files_processed": len(files),
        "rows": total_rows,
        "hydrated": total_hydrated,
        "hydrate_rate": total_hydrated / total_rows if total_rows else 0.0,
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
