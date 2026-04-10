import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from distill.core.prompt_examples import build_prompt_example_sample


def iter_parquet_files(root: Path, splits: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            continue
        files.extend(sorted(split_dir.glob("*.parquet")))
    return files


def augment_row(row: Dict[str, object]) -> Dict[str, object]:
    sample = build_prompt_example_sample(row)
    if sample:
        row["input_output"] = json.dumps({
            "inputs": sample["inputs"],
            "outputs": sample["outputs"],
            "fn_name": None,
        }, ensure_ascii=False)
        row["generated_test_source"] = sample.get("source")
        row["generated_test_parser"] = "prompt_examples_v1"
        row["generated_test_count"] = len(sample["inputs"])
        row["generated_test_status"] = "parsed"
    else:
        row["input_output"] = ""
        row["generated_test_source"] = ""
        row["generated_test_parser"] = "prompt_examples_v1"
        row["generated_test_count"] = 0
        row["generated_test_status"] = "missing"
    return row


def process_file(input_path: Path,
                 output_path: Path,
                 batch_size: int = 256) -> Dict[str, object]:
    parquet_file = pq.ParquetFile(input_path)
    writer: Optional[pq.ParquetWriter] = None

    rows = 0
    parsed_rows = 0
    total_examples = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()

    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            rows_batch = []
            for row in batch.to_pylist():
                augmented = augment_row(dict(row))
                rows_batch.append(augmented)
                rows += 1
                if augmented["generated_test_status"] == "parsed":
                    parsed_rows += 1
                    total_examples += int(augmented["generated_test_count"])

            table = pa.Table.from_pylist(rows_batch)
            if writer is None:
                writer = pq.ParquetWriter(temp_path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    temp_path.replace(output_path)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows": rows,
        "parsed_rows": parsed_rows,
        "parse_rate": (parsed_rows / rows) if rows else 0.0,
        "total_examples": total_examples,
        "avg_examples_per_parsed_row":
        (total_examples / parsed_rows) if parsed_rows else 0.0,
    }


def prepare_dataset(input_root: Path,
                    output_root: Path,
                    splits: List[str],
                    batch_size: int,
                    overwrite: bool,
                    progress_every: int) -> Dict[str, object]:
    summaries: List[Dict[str, object]] = []
    files = iter_parquet_files(input_root, splits)

    total_rows = 0
    total_parsed_rows = 0
    total_examples = 0

    for file_idx, input_path in enumerate(files, 1):
        relative = input_path.relative_to(input_root)
        output_path = output_root / relative
        if output_path.exists() and not overwrite:
            continue

        summary = process_file(input_path, output_path, batch_size=batch_size)
        summaries.append(summary)
        total_rows += int(summary["rows"])
        total_parsed_rows += int(summary["parsed_rows"])
        total_examples += int(summary["total_examples"])

        if progress_every > 0 and file_idx % progress_every == 0:
            print(
                f"processed_files={file_idx}/{len(files)} parsed_rows={total_parsed_rows} total_rows={total_rows}",
                flush=True,
            )

    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "splits": splits,
        "files_processed": len(summaries),
        "rows": total_rows,
        "parsed_rows": total_parsed_rows,
        "parse_rate": (total_parsed_rows / total_rows) if total_rows else 0.0,
        "total_examples": total_examples,
        "avg_examples_per_parsed_row":
        (total_examples / total_parsed_rows) if total_parsed_rows else 0.0,
        "files": summaries,
    }

    summary_path = output_root / "prepare_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(manifest,
                                       ensure_ascii=False,
                                       indent=2,
                                       sort_keys=True),
                            encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Materialize prompt-example tests for OpenCodeReasoning.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["split_0", "split_1"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = prepare_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        splits=args.splits,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
    )
    print(json.dumps({
        "files_processed": summary["files_processed"],
        "rows": summary["rows"],
        "parsed_rows": summary["parsed_rows"],
        "parse_rate": summary["parse_rate"],
        "output_root": summary["output_root"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
