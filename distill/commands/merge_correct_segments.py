import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq

from distill.runtime.manifest import load_manifest_tasks, select_manifest_tasks


TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def iter_segment_paths(segment_dir: Path) -> List[Path]:
    return sorted(segment_dir.glob("segment_*.jsonl"))


def iter_shard_paths(shard_dir: Path) -> List[Path]:
    return sorted(shard_dir.glob("shard_*.parquet"))


def _default_state() -> Dict[str, Any]:
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


def _normalize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _default_state()
    normalized["version"] = int(state.get("version", 1) or 1)
    merged_segments = state.get("merged_segments", [])
    normalized["merged_segments"] = (
        list(merged_segments) if isinstance(merged_segments, list) else [])
    normalized["next_shard_idx"] = int(state.get("next_shard_idx", 0) or 0)

    raw_stats = state.get("stats") or {}
    if not isinstance(raw_stats, dict):
        raw_stats = {}
    normalized["stats"]["total_records"] = int(raw_stats.get("total_records", 0)
                                                or 0)

    raw_token_sums = raw_stats.get("token_sums") or {}
    if not isinstance(raw_token_sums, dict):
        raw_token_sums = {}
    raw_token_counts = raw_stats.get("token_counts") or {}
    if not isinstance(raw_token_counts, dict):
        raw_token_counts = {}

    for field in TOKEN_FIELDS:
        normalized["stats"]["token_sums"][field] = int(
            raw_token_sums.get(field, 0) or 0)
        normalized["stats"]["token_counts"][field] = int(
            raw_token_counts.get(field, 0) or 0)
    return normalized


def _state_has_nonzero_stats(state: Dict[str, Any]) -> bool:
    stats = state.get("stats") or {}
    if int(stats.get("total_records", 0) or 0) > 0:
        return True
    for field in TOKEN_FIELDS:
        if int((stats.get("token_counts") or {}).get(field, 0) or 0) > 0:
            return True
        if int((stats.get("token_sums") or {}).get(field, 0) or 0) > 0:
            return True
    return False


def _stats_from_existing_shards(shard_paths: List[Path]) -> Dict[str, Any]:
    stats = _empty_stats()
    for shard_path in shard_paths:
        table = pq.read_table(shard_path)
        for record in table.to_pylist():
            _accumulate_record_stats(stats, record)
    return stats


def _next_shard_idx_from_paths(shard_paths: List[Path]) -> int:
    next_idx = 0
    for shard_path in shard_paths:
        name = shard_path.stem
        if not name.startswith("shard_"):
            continue
        try:
            shard_idx = int(name.split("_", 1)[1])
        except ValueError:
            continue
        next_idx = max(next_idx, shard_idx + 1)
    return next_idx


def _repair_state_if_needed(state: Dict[str, Any], segment_paths: List[Path],
                            shard_paths: List[Path]) -> Dict[str, Any]:
    repaired = _normalize_state(state)
    has_stats = _state_has_nonzero_stats(repaired)
    if shard_paths:
        repaired["next_shard_idx"] = max(
            int(repaired.get("next_shard_idx", 0) or 0),
            _next_shard_idx_from_paths(shard_paths),
        )
        if not has_stats:
            repaired["stats"] = _stats_from_existing_shards(shard_paths)
        return repaired

    if has_stats:
        return repaired

    segment_names = {path.name for path in segment_paths}
    merged_segments = set(repaired.get("merged_segments", []))
    if segment_names and merged_segments.issuperset(segment_names):
        repaired["merged_segments"] = []
    return repaired


def load_state(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        return _default_state()
    with state_path.open("r", encoding="utf-8") as f:
        return _normalize_state(json.load(f))


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
    normalized_records = [_normalize_record_for_parquet(record) for record in records]
    table = pa.Table.from_pylist(normalized_records)
    pq.write_table(table, tmp_path)
    tmp_path.replace(shard_path)


def _normalize_record_for_parquet(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(record)
    judge_detail = normalized.get("judge_detail")
    if isinstance(judge_detail, (dict, list)):
        normalized["judge_detail"] = json.dumps(
            judge_detail,
            ensure_ascii=False,
            sort_keys=True,
        )
    return normalized


def _remove_existing_shards(shard_dir: Path) -> None:
    for shard_path in iter_shard_paths(shard_dir):
        shard_path.unlink()


def _flatten_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    hf_upload = summary.get("hf_upload") or {}
    token_counts = summary.get("token_counts") or {}
    token_sums = summary.get("token_sums") or {}
    average_tokens = summary.get("average_tokens") or {}
    return {
        "task_name": summary.get("task_name", ""),
        "output_dir": summary.get("output_dir", ""),
        "stream": summary.get("stream", ""),
        "total_segment_count": summary.get("total_segment_count", 0),
        "pending_segment_count_before_run":
        summary.get("pending_segment_count_before_run", 0),
        "pending_segment_count": summary.get("pending_segment_count", 0),
        "merged_segments_total": summary.get("merged_segments_total", 0),
        "merged_segments_this_run": summary.get("merged_segments_this_run", 0),
        "shards_written_this_run": summary.get("shards_written_this_run", 0),
        "next_shard_idx": summary.get("next_shard_idx", 0),
        "total_records": summary.get("total_records", 0),
        "merged_records_this_run": summary.get("merged_records_this_run", 0),
        "token_count_total": token_counts.get("total_tokens", 0),
        "token_sum_total": token_sums.get("total_tokens", 0),
        "avg_total_tokens": average_tokens.get("total_tokens", ""),
        "hf_uploaded": hf_upload.get("uploaded", 0),
        "hf_skipped": hf_upload.get("skipped", 0),
        "hf_reason": hf_upload.get("reason", ""),
    }


def write_summary_csv(summaries: List[Dict[str, Any]], csv_path: Path) -> None:
    rows = [_flatten_summary(summary) for summary in summaries]
    fieldnames = [
        "task_name",
        "output_dir",
        "stream",
        "total_segment_count",
        "pending_segment_count_before_run",
        "pending_segment_count",
        "merged_segments_total",
        "merged_segments_this_run",
        "shards_written_this_run",
        "next_shard_idx",
        "total_records",
        "merged_records_this_run",
        "token_count_total",
        "token_sum_total",
        "avg_total_tokens",
        "hf_uploaded",
        "hf_skipped",
        "hf_reason",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_upload_state(upload_state_path: Path) -> Dict[str, Any]:
    if not upload_state_path.exists():
        return {"version": 1, "uploaded_paths": []}
    with upload_state_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_upload_state(upload_state_path: Path, state: Dict[str, Any]) -> None:
    upload_state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = upload_state_path.with_suffix(upload_state_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(upload_state_path)


def _upload_correct_shards_to_hf(
        output_root: Path,
        shard_dir: Path,
        hf_repo_id: str,
        hf_token: str | None = None,
        hf_repo_type: str = "dataset",
        hf_remote_prefix: str | None = None,
        task_name: str | None = None,
        upload_state_path: str | None = None,
        force_reupload: bool = False) -> Dict[str, Any]:
    if not hf_repo_id:
        return {"uploaded": 0, "skipped": 0, "reason": "missing_repo_id"}

    token = hf_token or os.getenv("HF_TOKEN")
    if not token:
        return {"uploaded": 0, "skipped": 0, "reason": "missing_token"}

    try:
        from huggingface_hub import HfApi
    except ImportError:
        return {"uploaded": 0, "skipped": 0, "reason": "missing_dependency"}

    shard_paths = sorted(shard_dir.glob("shard_*.parquet"))
    if not shard_paths:
        return {"uploaded": 0, "skipped": 0, "reason": "no_shards"}

    remote_prefix = (hf_remote_prefix or task_name or "").strip().strip("/")
    remote_prefix = remote_prefix.replace("\\", "/")
    resolved_upload_state_path = (
        Path(upload_state_path)
        if upload_state_path else output_root / ".hf_upload_state.json")
    state = _load_upload_state(resolved_upload_state_path)
    uploaded_paths = set()
    if not force_reupload:
        uploaded_paths = set(str(path) for path in state.get("uploaded_paths", []))

    api = HfApi(token=token)
    api.create_repo(
        repo_id=hf_repo_id,
        repo_type=hf_repo_type,
        exist_ok=True,
    )

    uploaded = 0
    skipped = 0
    for shard_path in shard_paths:
        relative_path = os.path.relpath(shard_path, output_root)
        if relative_path in uploaded_paths:
            skipped += 1
            continue

        remote_name = shard_path.name
        path_in_repo = f"{remote_prefix}/{remote_name}" if remote_prefix else remote_name
        api.upload_file(
            path_or_fileobj=str(shard_path),
            path_in_repo=path_in_repo,
            repo_id=hf_repo_id,
            repo_type=hf_repo_type,
        )
        uploaded_paths.add(relative_path)
        state["uploaded_paths"] = sorted(uploaded_paths)
        _save_upload_state(resolved_upload_state_path, state)
        uploaded += 1

    return {
        "uploaded": uploaded,
        "skipped": skipped,
        "repo_id": hf_repo_id,
        "repo_type": hf_repo_type,
        "remote_prefix": remote_prefix,
        "upload_state_path": str(resolved_upload_state_path),
        "force_reupload": force_reupload,
    }


def merge_correct_segments(output_dir: str,
                           stream: str = "correct",
                           shard_size_mb: int = 200,
                           state_path: str | None = None,
                           upload_to_hf: bool = False,
                           hf_repo_id: str = "JoyboyGo/hf_data",
                           hf_token: str | None = None,
                           hf_repo_type: str = "dataset",
                           hf_remote_prefix: str | None = None,
                           task_name: str | None = None,
                           hf_upload_state_path: str | None = None,
                           force_reupload: bool = False,
                           force_merge: bool = False) -> Dict[str, Any]:
    output_root = Path(output_dir)
    segment_dir = output_root / stream / "segments"
    shard_dir = output_root / stream / "shards"
    if not segment_dir.exists():
        raise FileNotFoundError(f"segment dir not found: {segment_dir}")

    resolved_state_path = Path(state_path) if state_path else (
        output_root / stream / "merge_state.json")
    segment_paths = iter_segment_paths(segment_dir)
    if force_merge:
        _remove_existing_shards(shard_dir)
        state = _default_state()
    else:
        state = load_state(resolved_state_path)
        shard_paths = iter_shard_paths(shard_dir)
        state = _repair_state_if_needed(state, segment_paths, shard_paths)
    merged_segments = set(state.get("merged_segments", []))
    pending_paths = [
        path for path in segment_paths if path.name not in merged_segments
    ]
    pending_segment_count_before_run = len(pending_paths)

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
    final_pending_paths = [
        path for path in segment_paths if path.name not in set(state.get("merged_segments", []))
    ]

    summary = {
        "output_dir": str(output_root),
        "stream": stream,
        "state_path": str(resolved_state_path),
        "segment_dir": str(segment_dir),
        "shard_dir": str(shard_dir),
        "total_segment_count": len(segment_paths),
        "pending_segment_count_before_run": pending_segment_count_before_run,
        "pending_segment_count": len(final_pending_paths),
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

    if upload_to_hf:
        if stream != "correct":
            summary["hf_upload"] = {
                "uploaded": 0,
                "skipped": 0,
                "reason": "unsupported_stream",
            }
        else:
            summary["hf_upload"] = _upload_correct_shards_to_hf(
                output_root=output_root,
                shard_dir=shard_dir,
                hf_repo_id=hf_repo_id,
                hf_token=hf_token,
                hf_repo_type=hf_repo_type,
                hf_remote_prefix=hf_remote_prefix,
                task_name=task_name,
                upload_state_path=hf_upload_state_path,
                force_reupload=force_reupload,
            )
    return summary


def merge_correct_segments_from_manifest(
        config_path: str,
        task_name: str | None = None,
        stream: str = "correct",
        state_path: str | None = None,
        upload_to_hf: bool | None = None,
        hf_repo_id: str | None = None,
        hf_token: str | None = None,
        hf_repo_type: str | None = None,
        hf_remote_prefix: str | None = None,
        hf_upload_state_path: str | None = None,
        shard_size_mb: int | None = None,
        force_reupload: bool = False,
        force_merge: bool = False) -> List[Dict[str, Any]]:
    manifest_path = Path(config_path).expanduser().resolve()
    task_values_list = select_manifest_tasks(load_manifest_tasks(manifest_path),
                                             task_name)
    summaries: List[Dict[str, Any]] = []

    for task_values in task_values_list:
        resolved_output_dir = task_values.get("output_dir")
        if not resolved_output_dir:
            raise ValueError(
                f"Task {task_values.get('task_name')!r} is missing output_dir in {manifest_path}"
            )

        summary = merge_correct_segments(
            output_dir=str(resolved_output_dir),
            stream=stream,
            shard_size_mb=int(shard_size_mb if shard_size_mb is not None else
                              (task_values.get("shard_size_mb", 200) or 200)),
            state_path=state_path,
            upload_to_hf=bool(upload_to_hf if upload_to_hf is not None else
                              task_values.get("upload_merged_shards", False)),
            hf_repo_id=str(hf_repo_id if hf_repo_id is not None else
                           (task_values.get("hf_repo_id")
                            or "JoyboyGo/hf_data")),
            hf_token=hf_token if hf_token is not None else task_values.get("hf_token"),
            hf_repo_type=str(hf_repo_type if hf_repo_type is not None else
                             (task_values.get("hf_repo_type", "dataset")
                              or "dataset")),
            hf_remote_prefix=hf_remote_prefix if hf_remote_prefix is not None else
            task_values.get("hf_remote_prefix"),
            task_name=str(task_values.get("task_name") or ""),
            hf_upload_state_path=hf_upload_state_path,
            force_reupload=force_reupload,
            force_merge=force_merge,
        )
        summary["task_name"] = str(task_values.get("task_name") or "")
        summary["config_path"] = str(manifest_path)
        summaries.append(summary)

    return summaries


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Incrementally merge correct segments into parquet shards "
                     "and maintain token statistics."),
        add_help=add_help,
    )
    parser.add_argument("--config",
                        type=str,
                        default=None,
                        help=("Manifest/task YAML path. When provided, merge "
                              "all matching tasks from the config."))
    parser.add_argument("--task",
                        type=str,
                        default=None,
                        help="Optional task_name filter when --config is used.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--stream",
                        default="correct",
                        choices=["all", "correct"],
                        help="Which output stream to merge.")
    parser.add_argument("--shard-size-mb",
                        default=None,
                        type=int,
                        help="Approximate target parquet shard size.")
    parser.add_argument("--state-path",
                        type=str,
                        default=None,
                        help=("Optional state file path. Defaults to "
                              "<output-dir>/<stream>/merge_state.json"))
    parser.add_argument("--upload-to-hf",
                        action="store_true",
                        help="Upload correct shards to a Hugging Face dataset repo.")
    parser.add_argument("--hf-repo-id",
                        type=str,
                        default=None,
                        help="Target Hugging Face dataset repo.")
    parser.add_argument("--hf-token",
                        type=str,
                        default="",
                        help="Hugging Face token. Defaults to HF_TOKEN env var.")
    parser.add_argument("--hf-repo-type",
                        type=str,
                        default=None,
                        help="Hugging Face repo type.")
    parser.add_argument("--hf-remote-prefix",
                        type=str,
                        default=None,
                        help=("Remote single-level directory name. Defaults to the "
                              "pipeline task_name when provided."))
    parser.add_argument("--task-name",
                        type=str,
                        default=None,
                        help=("Pipeline task_name. Used as the default HF remote "
                              "directory when hf-remote-prefix is not set."))
    parser.add_argument("--hf-upload-state-path",
                        type=str,
                        default=None,
                        help=("Optional HF upload state path. Defaults to "
                              "<output-dir>/.hf_upload_state.json"))
    parser.add_argument("--force-reupload",
                        action="store_true",
                        help=("Ignore local HF upload state and re-upload all "
                              "currently existing shard files."))
    parser.add_argument("--force-merge",
                        action="store_true",
                        help=("Ignore existing merge state, remove current shard "
                              "files, and rebuild shards from all segments."))
    parser.add_argument("--summary-path", type=str, default=None)
    parser.add_argument("--summary-csv-path",
                        type=str,
                        default="merge_correct_segments_summary.csv",
                        help=("CSV snapshot path. Defaults to a file in the "
                              "current working directory."))
    return parser


def run_namespace(args: argparse.Namespace) -> Dict[str, Any] | List[Dict[str, Any]]:

    if args.config:
        summary: Dict[str, Any] | List[Dict[str, Any]] = (
            merge_correct_segments_from_manifest(
                config_path=args.config,
                task_name=args.task,
                stream=args.stream,
                state_path=args.state_path,
                upload_to_hf=args.upload_to_hf if args.upload_to_hf else None,
                hf_repo_id=args.hf_repo_id if args.hf_repo_id else None,
                hf_token=args.hf_token,
                hf_repo_type=args.hf_repo_type if args.hf_repo_type else None,
                hf_remote_prefix=args.hf_remote_prefix,
                hf_upload_state_path=args.hf_upload_state_path,
                shard_size_mb=args.shard_size_mb,
                force_reupload=args.force_reupload,
                force_merge=args.force_merge,
            ))
    else:
        if not args.output_dir:
            parser.error("--output-dir is required unless --config is provided.")
        summary = merge_correct_segments(
            output_dir=args.output_dir,
            stream=args.stream,
            shard_size_mb=args.shard_size_mb if args.shard_size_mb is not None else 200,
            state_path=args.state_path,
            upload_to_hf=args.upload_to_hf,
            hf_repo_id=args.hf_repo_id or "JoyboyGo/hf_data",
            hf_token=args.hf_token,
            hf_repo_type=args.hf_repo_type or "dataset",
            hf_remote_prefix=args.hf_remote_prefix,
            task_name=args.task,
            hf_upload_state_path=args.hf_upload_state_path,
            force_reupload=args.force_reupload,
            force_merge=args.force_merge,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))

    summary_rows = summary if isinstance(summary, list) else [summary]
    write_summary_csv(summary_rows, Path(args.summary_csv_path))

    if args.summary_path:
        summary_path = Path(args.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    run_namespace(args)


if __name__ == "__main__":
    main()
