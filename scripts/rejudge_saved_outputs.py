import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pyarrow.parquet as pq

from distill.core.judge import judge_output


logger = logging.getLogger("rejudge_saved_outputs")


def iter_segment_paths(segment_dir: Path) -> Iterable[Path]:
    return sorted(segment_dir.glob("segment_*.jsonl"))


class RowCache:

    def __init__(self):
        self._cache: Dict[str, list] = {}

    def get_row(self, source_file: str, source_row: int) -> Dict[str, object]:
        if source_file not in self._cache:
            self._cache[source_file] = pq.read_table(source_file).to_pylist()
        return self._cache[source_file][source_row]


def rejudge_record(record: Dict[str, object], row_cache: RowCache) -> Tuple[bool, Dict[str, object]]:
    source_file = str(record["source_file"])
    source_row = int(record["source_row"])
    row_data = row_cache.get_row(source_file, source_row)
    new_result = judge_output(row_data, record["messages"])

    old_tuple = (
        record.get("judge_type"),
        record.get("judge_backend"),
        record.get("judge_status"),
        record.get("is_correct"),
    )
    new_tuple = (
        new_result.get("judge_type"),
        new_result.get("judge_backend"),
        new_result.get("judge_status"),
        new_result.get("is_correct"),
    )

    record["judge_type"] = new_result.get("judge_type")
    record["judge_backend"] = new_result.get("judge_backend")
    record["judge_status"] = new_result.get("judge_status")
    record["judge_detail"] = new_result.get("judge_detail")
    record["is_correct"] = new_result.get("is_correct")
    return old_tuple != new_tuple, record


def is_suspicious_record(record: Dict[str, object]) -> bool:
    dataset_name = str(record.get("dataset_name") or "")
    judge_type = str(record.get("judge_type") or "")
    judge_status = str(record.get("judge_status") or "")
    judge_detail = record.get("judge_detail") or {}
    expected = str(judge_detail.get("expected") or "")
    output = str(judge_detail.get("output") or "")
    messages = record.get("messages") or []
    assistant = next((m.get("content") or "" for m in messages
                      if isinstance(m, dict) and m.get("role") == "assistant"),
                     "")

    if dataset_name == "code_contests" and judge_type == "math":
        return True
    if "Constraints" in expected or "Warning" in expected:
        return True
    if "Input" in expected and "Output" in expected:
        return True
    if judge_status == "wrong_answer" and output == "" and "```python" in assistant.lower(
    ) and assistant.count("```") >= 4:
        return True
    if judge_status == "failed" and "```python" in assistant.lower(
    ) and assistant.count("```") >= 4:
        return True
    return False


def segment_has_suspicious_records(segment_path: Path) -> bool:
    with segment_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if is_suspicious_record(json.loads(line)):
                return True
    return False


def rejudge_segments(segment_dir: Path,
                     limit: int = 0,
                     suspicious_only: bool = False,
                     progress_every: int = 50) -> Dict[str, object]:
    row_cache = RowCache()
    counters: Counter = Counter()
    changed_examples = []
    processed = 0
    total_segments = sum(1 for _ in iter_segment_paths(segment_dir))

    logger.info(
        "Starting rejudge: segment_dir=%s suspicious_only=%s limit=%s total_segments=%s",
        segment_dir,
        suspicious_only,
        limit,
        total_segments,
    )

    for segment_idx, segment_path in enumerate(iter_segment_paths(segment_dir), start=1):
        if suspicious_only and not segment_has_suspicious_records(segment_path):
            counters["processed_segments"] += 1
            if progress_every > 0 and segment_idx % progress_every == 0:
                logger.info(
                    "Progress: segments=%s/%s processed_records=%s changed_records=%s",
                    segment_idx,
                    total_segments,
                    processed,
                    counters["changed_records"],
                )
            continue

        temp_path = segment_path.with_suffix(segment_path.suffix + ".tmp")
        changed_in_file = 0
        file_rejudged = 0
        with segment_path.open("r", encoding="utf-8") as src, temp_path.open(
                "w", encoding="utf-8") as dst:
            for line in src:
                if not line.strip():
                    continue
                record = json.loads(line)
                if suspicious_only and not is_suspicious_record(record):
                    dst.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue
                processed += 1
                file_rejudged += 1
                changed, updated = rejudge_record(record, row_cache)
                if changed:
                    changed_in_file += 1
                    counters["changed_records"] += 1
                    key = (str(record.get("judge_status")), str(record.get("is_correct")))
                    counters[f"changed_to::{key}"] += 1
                    if len(changed_examples) < 20:
                        changed_examples.append({
                            "segment": segment_path.name,
                            "source_file": updated["source_file"],
                            "source_row": updated["source_row"],
                            "judge_type": updated.get("judge_type"),
                            "judge_backend": updated.get("judge_backend"),
                            "judge_status": updated.get("judge_status"),
                            "is_correct": updated.get("is_correct"),
                            "judge_detail": updated.get("judge_detail"),
                        })
                dst.write(json.dumps(updated, ensure_ascii=False) + "\n")
                if limit > 0 and processed >= limit:
                    break

        if changed_in_file > 0:
            segment_path.replace(segment_path.with_suffix(segment_path.suffix + ".bak"))
            temp_path.replace(segment_path)
            logger.info(
                "Updated %s: rejudged_records=%s changed_records=%s",
                segment_path.name,
                file_rejudged,
                changed_in_file,
            )
        else:
            temp_path.unlink(missing_ok=True)
            if file_rejudged > 0:
                logger.info(
                    "Checked %s: rejudged_records=%s changed_records=0",
                    segment_path.name,
                    file_rejudged,
                )

        counters["processed_segments"] += 1
        counters["processed_records"] = processed
        if progress_every > 0 and segment_idx % progress_every == 0:
            logger.info(
                "Progress: segments=%s/%s processed_records=%s changed_records=%s",
                segment_idx,
                total_segments,
                processed,
                counters["changed_records"],
            )
        if limit > 0 and processed >= limit:
            break

    summary = {
        "processed_records": processed,
        "processed_segments": counters["processed_segments"],
        "changed_records": counters["changed_records"],
        "total_segments": total_segments,
        "change_counters": dict(counters),
        "changed_examples": changed_examples,
    }
    logger.info(
        "Finished rejudge: processed_segments=%s/%s processed_records=%s changed_records=%s",
        summary["processed_segments"],
        total_segments,
        summary["processed_records"],
        summary["changed_records"],
    )
    return summary


def configure_logging(log_path: Path | None) -> None:
    handlers = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Rejudge saved segment outputs in-place using the current judge logic.")
    parser.add_argument("--segment-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--suspicious-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--summary-path", type=Path)
    args = parser.parse_args()

    configure_logging(args.log_path)

    summary = rejudge_segments(
        args.segment_dir,
        limit=args.limit,
        suspicious_only=args.suspicious_only,
        progress_every=args.progress_every,
    )
    if args.summary_path is not None:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote summary to %s", args.summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
