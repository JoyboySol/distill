import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq


TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def iter_segment_paths(segment_dir: Path) -> List[Path]:
    return sorted(segment_dir.glob("segment_*.jsonl"))


def load_state(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        return {
            "version": 1,
            "merged_segments": [],
            "next_shard_idx": 0,
            "stats": {
                "total_records": 0,
                "token_sums": {
                    field: 0
                    for field in TOKEN_FIELDS
                },
                "token_counts": {
                    field: 0
                    for field in TOKEN_FIELDS
                },
            },
        }
    with state_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state_path: Path, state: Dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(state_path)


def _normalize_token_value(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _empty_stats() -> Dict[str, Any]:
    return {
        "total_records": 0,
        "token_sums": {
            field: 0
            for field in TOKEN_FIELDS
        },
        "token_counts": {
            field: 0
            for field in TOKEN_FIELDS
        },
    }


def _accumulate_record_stats(stats: Dict[str, Any], record: Dict[str, Any]) -> None:
    stats["total_records"] += 1
    for field in TOKEN_FIELDS:
        value = _normalize_token_value(record.get(field))
        if value is None:
            continue
        stats["token_sums"][field] += value
        stats["token_counts"][field] += 1


def _merge_stats_into_state(state: Dict[str, Any], stats: Dict[str, Any]) -> None:
    state["stats"]["total_records"] += stats["total_records"]
    for field in TOKEN_FIELDS:
        state["stats"]["token_sums"][field] += stats["token_sums"][field]
        state["stats"]["token_counts"][field] += stats["token_counts"][field]


def _average_tokens_from_state(state: Dict[str, Any]) -> Dict[str, float | None]:
    averages: Dict[str, float | None] = {}
    for field in TOKEN_FIELDS:
        count = int(state["stats"]["token_counts"].get(field, 0) or 0)
        total = int(state["stats"]["token_sums"].get(field, 0) or 0)
        averages[field] = round(total / count, 4) if count else None
    return averages


def _write_parquet_shard(shard_path: Path, records: List[Dict[str, Any]]) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
    table = pa.Table.from_pylist(records)
    pq.write_table(table, tmp_path)
    tmp_path.replace(shard_path)


def merge_correct_segments(output_dir: str,
                           stream: str = "correct",
                           shard_size_mb: int = 200,
                           state_path: str | None = None) -> Dict[str, Any]:
    output_root = Path(output_dir)
    segment_dir = output_root / stream / "segments"
    shard_dir = output_root / stream / "shards"
    if not segment_dir.exists():
        raise FileNotFoundError(f"segment dir not found: {segment_dir}")

    resolved_state_path = Path(state_path) if state_path else (
        output_root / stream / "external_merge_state.json")
    state = load_state(resolved_state_path)
    merged_segments = set(state.get("merged_segments", []))
    segment_paths = iter_segment_paths(segment_dir)
    pending_paths = [
        path for path in segment_paths if path.name not in merged_segments
    ]

    shard_target_bytes = max(1, shard_size_mb) * 1024 * 1024
    records_buffer: List[Dict[str, Any]] = []
    buffer_size_bytes = 0
    current_segment_names: List[str] = []
    pending_stats = _empty_stats()

    shards_written_this_run = 0
    merged_segments_this_run = 0
    merged_records_this_run = 0

    def flush_pending() -> None:
        nonlocal records_buffer
        nonlocal buffer_size_bytes
        nonlocal current_segment_names
        nonlocal pending_stats
        nonlocal shards_written_this_run
        nonlocal merged_segments_this_run
        nonlocal merged_records_this_run

        if not current_segment_names:
            return

        if records_buffer:
            shard_idx = int(state.get("next_shard_idx", 0) or 0)
            shard_path = shard_dir / f"shard_{shard_idx:05d}.parquet"
            _write_parquet_shard(shard_path, records_buffer)
            state["next_shard_idx"] = shard_idx + 1
            shards_written_this_run += 1

        merged_segments.update(current_segment_names)
        state["merged_segments"] = sorted(merged_segments)
        _merge_stats_into_state(state, pending_stats)
        save_state(resolved_state_path, state)

        merged_segments_this_run += len(current_segment_names)
        merged_records_this_run += pending_stats["total_records"]
        records_buffer = []
        buffer_size_bytes = 0
        current_segment_names = []
        pending_stats = _empty_stats()

    for segment_path in pending_paths:
        segment_records: List[Dict[str, Any]] = []
        segment_size_bytes = 0
        segment_stats = _empty_stats()

        with segment_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                segment_records.append(record)
                segment_size_bytes += len(line.encode("utf-8"))
                _accumulate_record_stats(segment_stats, record)

        if (records_buffer and segment_records
                and buffer_size_bytes + segment_size_bytes > shard_target_bytes):
            flush_pending()

        records_buffer.extend(segment_records)
        buffer_size_bytes += segment_size_bytes
        current_segment_names.append(segment_path.name)
        pending_stats["total_records"] += segment_stats["total_records"]
        for field in TOKEN_FIELDS:
            pending_stats["token_sums"][field] += segment_stats["token_sums"][field]
            pending_stats["token_counts"][field] += segment_stats["token_counts"][field]

    flush_pending()

    summary = {
        "output_dir": str(output_root),
        "stream": stream,
        "state_path": str(resolved_state_path),
        "segment_dir": str(segment_dir),
        "shard_dir": str(shard_dir),
        "total_segment_count": len(segment_paths),
        "pending_segment_count": len(pending_paths),
        "merged_segments_total": len(state.get("merged_segments", [])),
        "merged_segments_this_run": merged_segments_this_run,
        "shards_written_this_run": shards_written_this_run,
        "next_shard_idx": int(state.get("next_shard_idx", 0) or 0),
        "total_records": int(state["stats"].get("total_records", 0) or 0),
        "merged_records_this_run": merged_records_this_run,
        "token_sums": state["stats"]["token_sums"],
        "token_counts": state["stats"]["token_counts"],
        "average_tokens": _average_tokens_from_state(state),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Incrementally merge correct segments into parquet shards "
                     "and maintain token statistics."))
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--stream",
                        default="correct",
                        choices=["all", "correct"],
                        help="Which output stream to merge.")
    parser.add_argument("--shard-size-mb",
                        default=200,
                        type=int,
                        help="Approximate target parquet shard size.")
    parser.add_argument("--state-path",
                        type=str,
                        default=None,
                        help=("Optional state file path. Defaults to "
                              "<output-dir>/<stream>/external_merge_state.json"))
    parser.add_argument("--summary-path", type=str, default=None)
    args = parser.parse_args()

    summary = merge_correct_segments(
        output_dir=args.output_dir,
        stream=args.stream,
        shard_size_mb=args.shard_size_mb,
        state_path=args.state_path,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))

    if args.summary_path:
        summary_path = Path(args.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
