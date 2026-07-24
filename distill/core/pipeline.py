import asyncio
import glob
import hashlib
import json
import os
import re
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

import pyarrow as pa
import pyarrow.parquet as pq
from openai import APITimeoutError, BadRequestError
from tqdm.asyncio import tqdm

try:
    from ..common.judge_mode import judge_mode_disables_all, normalize_judge_mode
    from ..runtime.settings import PipelineConfig, logger
    from ..common.utils import ensure_message_shape, safe_json_dumps
    from .failure import FailureRecorder
    from .judge import judge_output, judge_output_with_timeout
    from .llm import (AsyncLLMManager, GenerationResponse,
                      NoHealthyBackendsError)
except ImportError:
    from common.judge_mode import judge_mode_disables_all, normalize_judge_mode
    from runtime.settings import PipelineConfig, logger
    from common.utils import ensure_message_shape, safe_json_dumps
    from core.failure import FailureRecorder
    from core.judge import judge_output, judge_output_with_timeout
    from core.llm import (AsyncLLMManager, GenerationResponse,
                          NoHealthyBackendsError)


@dataclass
class TaskItem:
    source_file: str
    source_row: int
    rollout_index: int
    row_data: Dict[str, Any]
    prompt: Optional[str] = None
    input_messages: Optional[List[Dict[str, Any]]] = None
    task_mode: str = "single_turn"


@dataclass
class ResultItem:
    source_file: str
    source_row: int
    record: Dict[str, Any]
    estimated_size: int


@dataclass
class GenerationResultItem:
    task: TaskItem
    messages: List[Dict[str, Any]]
    finish_reason: Optional[str]
    usage: Dict[str, Any]
    distill_status: str = "success"
    distill_error: Optional[str] = None
    assistant_turns_completed: int = 1
    assistant_turns_total: int = 1


class InterruptFinalizeRequested(Exception):
    pass


class DistillPipeline:
    STREAM_ALL = "all"
    STREAM_CORRECT = "correct"
    WRITER_POLL_INTERVAL_SEC = 0.2
    OPEN_CODE_REASONING_PYTHON_WRAPPER = (
        "Solve this problem using Python.\n"
        "Return a Python solution that can be executed directly.\n"
        "Use standard input and standard output.\n"
        "Put the final answer inside a ```python``` or ```Python``` code block.\n"
        "A brief explanation is okay, but make sure the final answer includes Python code.\n\n"
        "Problem:\n"
    )

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.llm_manager = AsyncLLMManager(config)
        self.failure_recorder = FailureRecorder(config.failure_log)
        self.task_queue: asyncio.Queue = asyncio.Queue(
            maxsize=config.queue_max_size)
        self.judge_queue: asyncio.Queue = asyncio.Queue()
        self.result_queue: asyncio.Queue = asyncio.Queue()
        self.stop_requested = False
        self.interrupt_finalize_requested = False
        self.input_exhausted = False

        self.completed_records_loaded = False
        self.completed_records: Set[str] = set()
        self.resume_progress = {
            "written": 0,
            "correct": 0,
            "overlong": 0,
        }

        self.segment_counters = {
            self.STREAM_ALL: 0,
            self.STREAM_CORRECT: 0,
        }
        self.shard_counters = {
            self.STREAM_ALL: 0,
            self.STREAM_CORRECT: 0,
        }
        self.counter_locks = {
            self.STREAM_ALL: asyncio.Lock(),
            self.STREAM_CORRECT: asyncio.Lock(),
        }
        self.last_interrupt_summary: Dict[str, Any] = {}

    def _hf_upload_state_path(self) -> str:
        return os.path.join(self.config.output_dir, ".hf_upload_state.json")

    def _load_hf_upload_state(self) -> Dict[str, Any]:
        path = self._hf_upload_state_path()
        if not os.path.exists(path):
            return {"uploaded_paths": []}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            uploaded_paths = data.get("uploaded_paths", [])
            if not isinstance(uploaded_paths, list):
                uploaded_paths = []
            return {"uploaded_paths": uploaded_paths}
        except Exception as e:
            logger.warning("Failed to load HF upload state: %s", e)
            return {"uploaded_paths": []}

    def _save_hf_upload_state(self, state: Dict[str, Any]):
        target = self._hf_upload_state_path()
        temp = target + ".tmp"
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(temp, target)

    def _hf_remote_dir(self) -> str:
        remote_dir = (self.config.hf_remote_prefix
                      or self.config.task_name or "").strip().strip("/")
        return remote_dir.replace("\\", "/")

    def _upload_pending_correct_shards_sync(self) -> Dict[str, Any]:
        if not self.config.upload_merged_shards:
            return {"uploaded": 0, "skipped": 0, "reason": "disabled"}
        if not self.config.hf_repo_id:
            logger.warning("HF upload skipped: hf_repo_id is missing")
            return {"uploaded": 0, "skipped": 0, "reason": "missing_repo_id"}
        token = self.config.hf_token or os.getenv("HF_TOKEN")
        if not token:
            logger.warning("HF upload skipped: hf_token is missing")
            return {"uploaded": 0, "skipped": 0, "reason": "missing_token"}

        try:
            from huggingface_hub import HfApi
        except ImportError:
            logger.warning("HF upload skipped: huggingface_hub is not installed")
            return {"uploaded": 0, "skipped": 0, "reason": "missing_dependency"}

        shard_dir = self._stream_dir(self.STREAM_CORRECT, "shards")
        shard_paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.parquet")))
        if not shard_paths:
            return {"uploaded": 0, "skipped": 0, "reason": "no_shards"}

        state = self._load_hf_upload_state()
        uploaded_paths = set(str(path) for path in state.get("uploaded_paths", []))
        api = HfApi(token=token)
        api.create_repo(
            repo_id=self.config.hf_repo_id,
            repo_type=self.config.hf_repo_type,
            exist_ok=True,
        )

        uploaded = 0
        skipped = 0
        remote_dir = self._hf_remote_dir()
        for shard_path in shard_paths:
            relative_path = os.path.relpath(shard_path, self.config.output_dir)
            if relative_path in uploaded_paths:
                skipped += 1
                continue
            remote_name = os.path.basename(shard_path)
            path_in_repo = (f"{remote_dir}/{remote_name}"
                            if remote_dir else remote_name)
            api.upload_file(
                path_or_fileobj=shard_path,
                path_in_repo=path_in_repo,
                repo_id=self.config.hf_repo_id,
                repo_type=self.config.hf_repo_type,
            )
            uploaded_paths.add(relative_path)
            state["uploaded_paths"] = sorted(uploaded_paths)
            self._save_hf_upload_state(state)
            uploaded += 1

        return {"uploaded": uploaded, "skipped": skipped}

    async def _upload_pending_correct_shards(self) -> Dict[str, Any]:
        return await asyncio.to_thread(self._upload_pending_correct_shards_sync)

    @staticmethod
    def _refresh_progress_postfix(pbar: tqdm):
        resumed = int(getattr(pbar, "_resumed_tasks", 0))
        discovered = int(getattr(pbar, "_discovered_tasks", 0))
        written = int(getattr(pbar, "_written_tasks", 0))
        correct = int(getattr(pbar, "_correct_tasks", 0))
        overlong = int(getattr(pbar, "_overlong_tasks", 0))
        discovered_new = max(0, discovered - resumed)
        written_new = max(0, written - resumed)
        pbar.set_postfix_str(
            " ".join([
                f"resumed={resumed}",
                f"discovered_new={discovered_new}",
                f"written_total={written}",
                f"written_new={written_new}",
                f"correct={correct}",
                f"overlong={overlong}",
            ]))

    @staticmethod
    def _completed_key(source_file: str,
                       source_row: int,
                       rollout_index: int = 0) -> str:
        return f"{source_file}::{source_row}::{rollout_index}"

    def _stream_root(self, stream_name: str) -> str:
        path = os.path.join(self.config.output_dir, stream_name)
        os.makedirs(path, exist_ok=True)
        return path

    def _stream_dir(self, stream_name: str, kind: str) -> str:
        path = os.path.join(self._stream_root(stream_name), kind)
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _count_jsonl_rows(file_path: str) -> int:
        count = 0
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                count += chunk.count(b"\n")
        return count

    def _estimate_input_rows(self, input_files: List[str]) -> Optional[int]:
        total_rows = 0
        try:
            for file_path in input_files:
                ext = os.path.splitext(file_path)[1].lower()
                if ext == ".parquet":
                    total_rows += pq.ParquetFile(file_path).metadata.num_rows
                    continue
                if ext == ".jsonl":
                    total_rows += self._count_jsonl_rows(file_path)
                    continue
                logger.warning("Skipping row estimate for unsupported file: %s",
                               file_path)
            return total_rows
        except Exception as e:
            logger.warning("Failed to estimate input row count: %s", e)
            return None

    def _effective_input_row_limit(self,
                                   estimated_rows: Optional[int]) -> Optional[int]:
        if estimated_rows is None:
            return None
        if self.config.sample_limit is None:
            return estimated_rows
        return min(estimated_rows, max(0, int(self.config.sample_limit)))

    def _request_stop(self, signame: str):
        if self.stop_requested:
            return
        self.stop_requested = True
        self.interrupt_finalize_requested = True
        self.llm_manager.request_stop(signame)
        logger.warning(
            "Received %s, cancelling in-flight work and finalizing persisted outputs...",
            signame,
        )

    def _stats_summary_path(self, stream_name: str) -> str:
        return os.path.join(self._stream_root(stream_name), "judge_stats.json")

    def _write_stats_summary(self, stream_name: str) -> Dict[str, Any]:
        try:
            from ..tools.stats import iter_records, summarize
        except ImportError:
            from tools.stats import iter_records, summarize

        summary = summarize(iter_records(self.config.output_dir, stream_name))
        target = self._stats_summary_path(stream_name)
        temp = target + ".tmp"
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(temp, target)
        return summary

    @staticmethod
    def _percent(numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return round(100.0 * numerator / denominator, 4)

    def _build_interrupt_quick_stats_summary(
            self, stream_name: str) -> Dict[str, Any]:
        written = int(self.resume_progress["written"])
        correct = int(self.resume_progress["correct"])
        overlong = int(self.resume_progress["overlong"])

        if stream_name == self.STREAM_CORRECT:
            total_records = correct
            correct_counts = {
                "true": correct,
            }
            finish_reason_counts = {}
        else:
            total_records = written
            unknown_count = max(0, written - correct)
            correct_counts = {
                "true": correct,
                "unknown": unknown_count,
            }
            other_finish_reasons = max(0, written - overlong)
            finish_reason_counts = {
                "length": overlong,
            }
            if other_finish_reasons > 0:
                finish_reason_counts["other_or_unknown"] = other_finish_reasons

        if stream_name == self.STREAM_CORRECT:
            overlong_total = 0
            overlong_correct = 0
        else:
            overlong_total = overlong
            overlong_correct = 0

        return {
            "summary_type": "interrupt_quick",
            "stream": stream_name,
            "total_records": total_records,
            "judge_type_counts": {},
            "judge_backend_counts": {},
            "judge_status_counts": {},
            "generation_finish_reason_counts": finish_reason_counts,
            "correct_counts": correct_counts,
            "overall_accuracy": None,
            "overlong_summary": {
                "overlong_total": overlong_total,
                "overlong_ratio": self._percent(overlong_total, total_records),
                "overlong_correct": overlong_correct,
                "overlong_accuracy": None,
            },
            "backend_accuracy": {},
            "math_backend_summary": {
                "math_total": 0,
                "math_verify_total": 0,
                "math_rule_total": 0,
                "math_verify_ratio": 0.0,
                "math_rule_ratio": 0.0,
            },
            "math_verify_fallback_reasons": {},
            "note": (
                "Interrupt quick summary derived from persisted progress "
                "counters. Detailed judge breakdowns were skipped to keep "
                "SIGINT finalization fast. Run distill.tools.stats for a "
                "full scan if needed."
            ),
        }

    def _write_interrupt_quick_stats_summary(
            self, stream_name: str) -> Dict[str, Any]:
        summary = self._build_interrupt_quick_stats_summary(stream_name)
        target = self._stats_summary_path(stream_name)
        temp = target + ".tmp"
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(temp, target)
        return summary

    def _merge_state_path(self, stream_name: str) -> str:
        return os.path.join(self._stream_root(stream_name), "merge_state.json")

    def _resume_dir(self) -> str:
        path = os.path.join(self.config.output_dir, ".resume")
        os.makedirs(path, exist_ok=True)
        return path

    def _completed_index_path(self) -> str:
        return os.path.join(self._resume_dir(), "completed_index.jsonl")

    def _resume_state_path(self) -> str:
        return os.path.join(self._resume_dir(), "resume_state.json")

    @staticmethod
    def _max_index_from_names(names: List[str], pattern: str) -> int:
        max_idx = -1
        compiled = re.compile(pattern)
        for name in names:
            match = compiled.search(name)
            if not match:
                continue
            max_idx = max(max_idx, int(match.group(1)))
        return max_idx

    def _default_resume_state(self) -> Dict[str, Any]:
        return {
            "version": 2,
            "progress": {
                "written": 0,
                "correct": 0,
                "overlong": 0,
            },
            "streams": {
                self.STREAM_ALL: {
                    "next_segment_idx": 0,
                    "next_shard_idx": 0,
                },
                self.STREAM_CORRECT: {
                    "next_segment_idx": 0,
                    "next_shard_idx": 0,
                },
            },
        }

    def _load_resume_state(self) -> Optional[Dict[str, Any]]:
        path = self._resume_state_path()
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            if not isinstance(state, dict):
                return None
            return state
        except Exception as e:
            logger.warning("Failed to load resume state: %s", e)
            return None

    def _save_resume_state(self):
        state = self._default_resume_state()
        state["progress"] = {
            "written": int(self.resume_progress["written"]),
            "correct": int(self.resume_progress["correct"]),
            "overlong": int(self.resume_progress["overlong"]),
        }
        state["streams"][self.STREAM_ALL] = {
            "next_segment_idx": int(self.segment_counters[self.STREAM_ALL]),
            "next_shard_idx": int(self.shard_counters[self.STREAM_ALL]),
        }
        state["streams"][self.STREAM_CORRECT] = {
            "next_segment_idx": int(self.segment_counters[self.STREAM_CORRECT]),
            "next_shard_idx": int(self.shard_counters[self.STREAM_CORRECT]),
        }

        target = self._resume_state_path()
        temp = target + ".tmp"
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(temp, target)

    def _append_completed_index(self, records: List[Dict[str, Any]]):
        if not records:
            return
        path = self._completed_index_path()
        with open(path, "a", encoding="utf-8") as f:
            for record in records:
                source_file = str(record["source_file"]).replace("\t", " ")
                source_row = int(record["source_row"])
                rollout_index = int(record.get("rollout_index", 0) or 0)
                f.write(f"{source_file}\t{source_row}\t{rollout_index}\n")

    def _parse_completed_index_line(self, line: str) -> Optional[str]:
        text = line.strip()
        if not text:
            return None
        if text.startswith("{"):
            row = json.loads(text)
            return self._completed_key(
                str(row["source_file"]),
                int(row["source_row"]),
                int(row.get("rollout_index", 0) or 0),
            )

        parts = text.split("\t")
        if len(parts) != 3:
            raise ValueError(f"Invalid completed index line: {text[:200]}")
        return self._completed_key(
            str(parts[0]),
            int(parts[1]),
            int(parts[2]),
        )

    def _completed_key_source_file(self, completed_key: str) -> str:
        return completed_key.rsplit("::", 2)[0]

    def _load_completed_index(
            self,
            allowed_source_files: Optional[Set[str]] = None) -> Optional[Set[str]]:
        path = self._completed_index_path()
        if not os.path.exists(path):
            return None
        completed_sources: Set[str] = set()
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    completed_key = self._parse_completed_index_line(line)
                    if completed_key is None:
                        continue
                    if (allowed_source_files is not None
                            and self._completed_key_source_file(completed_key)
                            not in allowed_source_files):
                        continue
                    completed_sources.add(completed_key)
            return completed_sources
        except Exception as e:
            logger.warning("Failed to load completed index: %s", e)
            return None

    def _load_merge_state(self, stream_name: str) -> Dict[str, Any]:
        path = self._merge_state_path(stream_name)
        if not os.path.exists(path):
            return {"merged_segments": []}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            merged_segments = data.get("merged_segments", [])
            if not isinstance(merged_segments, list):
                merged_segments = []
            return {"merged_segments": merged_segments}
        except Exception as e:
            logger.warning("Failed to load merge state for %s: %s", stream_name,
                           e)
            return {"merged_segments": []}

    def _discover_next_stream_index(self, stream_name: str, kind: str) -> int:
        if kind == "segment":
            dir_path = self._stream_dir(stream_name, "segments")
            names = os.listdir(dir_path) if os.path.isdir(dir_path) else []
            return self._max_index_from_names(names, r"segment_(\d+)\.jsonl$") + 1

        dir_path = self._stream_dir(stream_name, "shards")
        names = os.listdir(dir_path) if os.path.isdir(dir_path) else []
        return self._max_index_from_names(names, r"shard_(\d+)\.parquet$") + 1

    def _load_progress_from_resume_state(self,
                                         resume_state: Optional[Dict[str, Any]]):
        progress = (resume_state or {}).get("progress") or {}
        self.resume_progress = {
            "written": int(progress.get("written", 0) or 0),
            "correct": int(progress.get("correct", 0) or 0),
            "overlong": int(progress.get("overlong", 0) or 0),
        }

    def _load_stream_counters_from_resume_state(
            self, resume_state: Optional[Dict[str, Any]]) -> bool:
        streams = (resume_state or {}).get("streams") or {}
        all_stream = streams.get(self.STREAM_ALL) or {}
        correct_stream = streams.get(self.STREAM_CORRECT) or {}
        has_any = bool(all_stream or correct_stream)
        if not has_any:
            return False

        self.segment_counters[self.STREAM_ALL] = int(
            all_stream.get("next_segment_idx", 0) or 0)
        self.segment_counters[self.STREAM_CORRECT] = int(
            correct_stream.get("next_segment_idx", 0) or 0)
        self.shard_counters[self.STREAM_ALL] = int(
            all_stream.get("next_shard_idx", 0) or 0)
        self.shard_counters[self.STREAM_CORRECT] = int(
            correct_stream.get("next_shard_idx", 0) or 0)
        return True

    def _load_stream_counters_lightweight(self):
        self.segment_counters[self.STREAM_ALL] = self._discover_next_stream_index(
            self.STREAM_ALL, "segment")
        self.segment_counters[
            self.STREAM_CORRECT] = self._discover_next_stream_index(
                self.STREAM_CORRECT, "segment")
        self.shard_counters[self.STREAM_ALL] = self._discover_next_stream_index(
            self.STREAM_ALL, "shard")
        self.shard_counters[
            self.STREAM_CORRECT] = self._discover_next_stream_index(
                self.STREAM_CORRECT, "shard")

    def _save_merge_state(self, stream_name: str, state: Dict[str, Any]):
        target = self._merge_state_path(stream_name)
        temp = target + ".tmp"
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(temp, target)

    def _iter_completed_keys_from_file(self, file_path: str) -> Iterator[str]:
        if file_path.endswith(".parquet"):
            table = pq.read_table(
                file_path,
                columns=["source_file", "source_row", "rollout_index"],
            )
            for row in table.to_pylist():
                yield self._completed_key(
                    str(row["source_file"]),
                    int(row["source_row"]),
                    int(row.get("rollout_index", 0) or 0),
                )
            return

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                yield self._completed_key(
                    str(row["source_file"]),
                    int(row["source_row"]),
                    int(row.get("rollout_index", 0) or 0),
                )

    def _iter_records_from_file(self, file_path: str) -> Iterator[Dict[str, Any]]:
        if file_path.endswith(".parquet"):
            table = pq.read_table(
                file_path,
                columns=[
                    "source_file",
                    "source_row",
                    "rollout_index",
                    "generation_finish_reason",
                    "is_correct",
                ],
            )
            for row in table.to_pylist():
                yield row
            return

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                yield json.loads(line)

    def _rebuild_overlong_progress(self) -> int:
        overlong_completed_keys: Set[str] = set()
        shard_dir = self._stream_dir(self.STREAM_ALL, "shards")
        shard_files = sorted(glob.glob(os.path.join(shard_dir, "shard_*.parquet")))
        segment_dir = self._stream_dir(self.STREAM_ALL, "segments")
        segment_files = sorted(
            glob.glob(os.path.join(segment_dir, "segment_*.jsonl")))
        existing_files = [("shard", file_path) for file_path in shard_files]
        existing_files.extend(("segment", file_path) for file_path in segment_files)

        progress = tqdm(
            total=len(existing_files),
            desc="Resume repair [overlong]",
            unit="file",
            leave=False,
            dynamic_ncols=True,
        )

        for _, file_path in existing_files:
            try:
                for row in self._iter_records_from_file(file_path):
                    if row.get("generation_finish_reason") != "length":
                        continue
                    overlong_completed_keys.add(
                        self._completed_key(
                            str(row["source_file"]),
                            int(row["source_row"]),
                            int(row.get("rollout_index", 0) or 0),
                        ))
                progress.set_postfix_str(
                    f"overlong={len(overlong_completed_keys)}")
            except Exception as e:
                logger.warning("Failed to rebuild overlong from %s: %s",
                               file_path, e)
            finally:
                progress.update(1)

        progress.close()
        return len(overlong_completed_keys)

    def _resume_state_needs_progress_repair(
            self,
            resume_state: Optional[Dict[str, Any]],
            completed_index: Optional[Set[str]],
    ) -> bool:
        if resume_state is None or completed_index is None:
            return False
        version = int(resume_state.get("version", 0) or 0)
        if version >= 2:
            return False
        return len(completed_index) > 0

    def _scan_stream_state(self, stream_name: str) -> Dict[str, Any]:
        completed_sources: Set[str] = set()
        max_segment_idx = -1
        max_shard_idx = -1

        shard_dir = self._stream_dir(stream_name, "shards")
        shard_files = sorted(glob.glob(os.path.join(shard_dir, "shard_*.parquet")))

        segment_dir = self._stream_dir(stream_name, "segments")
        segment_files = sorted(
            glob.glob(os.path.join(segment_dir, "segment_*.jsonl")))

        existing_files = [("shard", file_path) for file_path in shard_files]
        existing_files.extend(("segment", file_path) for file_path in segment_files)

        progress = tqdm(
            total=len(existing_files),
            desc=f"Resume scan [{stream_name}]",
            unit="file",
            leave=False,
            dynamic_ncols=True,
        )

        for file_kind, file_path in existing_files:
            try:
                if file_kind == "shard":
                    match = re.search(r"shard_(\d+)\.parquet$",
                                      os.path.basename(file_path))
                    if match:
                        max_shard_idx = max(max_shard_idx, int(match.group(1)))
                else:
                    match = re.search(r"segment_(\d+)\.jsonl$",
                                      os.path.basename(file_path))
                    if match:
                        max_segment_idx = max(max_segment_idx, int(match.group(1)))

                before_count = len(completed_sources)
                for completed_key in self._iter_completed_keys_from_file(file_path):
                    completed_sources.add(completed_key)
                loaded_rows = len(completed_sources) - before_count
                progress.set_postfix_str(
                    f"rows={len(completed_sources)} last={loaded_rows}")
            except Exception as e:
                logger.warning("Failed to read existing %s %s: %s", file_kind,
                               file_path, e)
            finally:
                progress.update(1)

        progress.close()

        return {
            "completed_sources": completed_sources,
            "next_segment_idx": max_segment_idx + 1,
            "next_shard_idx": max_shard_idx + 1,
        }

    def _load_completed_records(self,
                                input_files: Optional[List[str]] = None):
        if self.completed_records_loaded:
            return

        self.completed_records_loaded = True
        os.makedirs(self.config.output_dir, exist_ok=True)
        allowed_source_files = set(input_files) if input_files is not None else None

        logger.info("Loading completed-record index for resume support...")
        resume_state = self._load_resume_state()
        completed_index = self._load_completed_index(allowed_source_files)
        if resume_state is not None and completed_index is not None:
            self.completed_records = completed_index
            self._load_progress_from_resume_state(resume_state)
            self._load_stream_counters_from_resume_state(resume_state)
            if self._resume_state_needs_progress_repair(resume_state,
                                                       completed_index):
                logger.info(
                    "Repairing legacy resume progress counters from persisted outputs..."
                )
                self.resume_progress["written"] = len(self.completed_records)
                self.resume_progress["overlong"] = self._rebuild_overlong_progress(
                )
                self._save_resume_state()
            logger.info(
                "Loaded resume state without full scan: completed=%s written=%s correct=%s overlong=%s filter=%s",
                len(self.completed_records),
                self.resume_progress["written"],
                self.resume_progress["correct"],
                self.resume_progress["overlong"],
                "all" if allowed_source_files is None else len(
                    allowed_source_files),
            )
            return

        if completed_index is not None:
            self.completed_records = completed_index
            self._load_progress_from_resume_state(resume_state)
            if self.resume_progress["written"] <= 0:
                self.resume_progress["written"] = len(self.completed_records)
            if not self._load_stream_counters_from_resume_state(resume_state):
                self._load_stream_counters_lightweight()
            self._save_resume_state()
            logger.info(
                "Recovered from completed index without full content scan: completed=%s written=%s correct=%s overlong=%s filter=%s",
                len(self.completed_records),
                self.resume_progress["written"],
                self.resume_progress["correct"],
                self.resume_progress["overlong"],
                "all" if allowed_source_files is None else len(
                    allowed_source_files),
            )
            return

        if resume_state is not None:
            self._load_progress_from_resume_state(resume_state)
            self._load_stream_counters_from_resume_state(resume_state)
            logger.warning(
                "Resume state exists but completed index is missing; falling back to one-time stream scan to rebuild skip index."
            )

        all_state = self._scan_stream_state(self.STREAM_ALL)
        correct_state = self._scan_stream_state(self.STREAM_CORRECT)

        self.completed_records = all_state["completed_sources"]
        self.segment_counters[self.STREAM_ALL] = all_state["next_segment_idx"]
        self.segment_counters[self.STREAM_CORRECT] = correct_state[
            "next_segment_idx"]
        self.shard_counters[self.STREAM_ALL] = all_state["next_shard_idx"]
        self.shard_counters[self.STREAM_CORRECT] = correct_state[
            "next_shard_idx"]
        self.resume_progress = {
            "written": len(all_state["completed_sources"]),
            "correct": len(correct_state["completed_sources"]),
            "overlong": 0,
        }
        self._append_completed_index([
            {
                "source_file": source_file,
                "source_row": source_row,
                "rollout_index": rollout_index,
            } for source_file, source_row, rollout_index in (
                key.rsplit("::", 2) for key in self.completed_records)
        ])
        self._save_resume_state()

        logger.info(
            "Loaded %s completed rows from persisted outputs and bootstrapped lightweight resume state",
            len(self.completed_records),
        )

    def _is_completed(self,
                      source_file: str,
                      source_row: int,
                      rollout_index: int = 0) -> bool:
        return self._completed_key(source_file, source_row,
                                   rollout_index) in self.completed_records

    def _normalize_prompt(self, raw_prompt) -> Optional[str]:
        if raw_prompt is None:
            return None

        if isinstance(raw_prompt, str):
            prompt = raw_prompt.strip()
            if not prompt:
                return None
            if prompt[0] in "[{":
                try:
                    return self._normalize_prompt(json.loads(prompt))
                except Exception:
                    pass
            return prompt

        if isinstance(raw_prompt, dict):
            content = raw_prompt.get("content")
            if isinstance(content, str):
                text = content.strip()
                return text or None
            if content is not None:
                return self._normalize_prompt(content)
            return None

        if hasattr(raw_prompt, "tolist"):
            try:
                converted = raw_prompt.tolist()
                if converted is not raw_prompt:
                    return self._normalize_prompt(converted)
            except Exception:
                pass

        if isinstance(raw_prompt, tuple):
            raw_prompt = list(raw_prompt)

        if isinstance(raw_prompt, list):
            user_parts = []
            for message in raw_prompt:
                if isinstance(message, str):
                    text = message.strip()
                    if text:
                        user_parts.append(text)
                    continue

                if hasattr(message, "as_py"):
                    try:
                        message = message.as_py()
                    except Exception:
                        pass

                if not isinstance(message, dict):
                    continue
                if message.get("role") != "user":
                    continue

                content = message.get("content")
                normalized_content = self._normalize_prompt(content)
                if normalized_content:
                    user_parts.append(normalized_content)

            if user_parts:
                return "\n\n".join(user_parts)

        return None

    @staticmethod
    def _normalize_message(raw_message: Any) -> Optional[Dict[str, Any]]:
        if hasattr(raw_message, "as_py"):
            try:
                raw_message = raw_message.as_py()
            except Exception:
                pass
        if not isinstance(raw_message, dict):
            return None
        role = raw_message.get("role")
        if not isinstance(role, str) or not role.strip():
            return None
        normalized = ensure_message_shape(raw_message)
        normalized["role"] = role.strip()
        return normalized

    def _normalize_multi_turn_messages(
            self, raw_input: Any) -> Optional[List[Dict[str, Any]]]:
        if raw_input is None:
            return None

        if isinstance(raw_input, str):
            text = raw_input.strip()
            if not text or text[0] not in "[{":
                return None
            try:
                return self._normalize_multi_turn_messages(json.loads(text))
            except Exception:
                return None

        if isinstance(raw_input, dict):
            messages = raw_input.get("messages")
            if messages is not None:
                return self._normalize_multi_turn_messages(messages)
            return None

        if hasattr(raw_input, "tolist"):
            try:
                converted = raw_input.tolist()
                if converted is not raw_input:
                    return self._normalize_multi_turn_messages(converted)
            except Exception:
                pass

        if isinstance(raw_input, tuple):
            raw_input = list(raw_input)

        if not isinstance(raw_input, list):
            return None

        normalized_messages = []
        has_assistant = False
        for message in raw_input:
            normalized = self._normalize_message(message)
            if normalized is None:
                return None
            normalized_messages.append(normalized)
            has_assistant = has_assistant or normalized["role"] == "assistant"

        if not has_assistant:
            return None
        return normalized_messages

    def _prepare_task_input(self, raw_input: Any) -> Optional[TaskItem]:
        input_messages = self._normalize_multi_turn_messages(raw_input)
        if input_messages is not None:
            return TaskItem(
                source_file="",
                source_row=0,
                rollout_index=0,
                row_data={},
                input_messages=input_messages,
                task_mode="multi_turn",
            )

        normalized_prompt = self._normalize_prompt(raw_input)
        if normalized_prompt is None:
            return None
        return TaskItem(
            source_file="",
            source_row=0,
            rollout_index=0,
            row_data={},
            prompt=self._format_task_prompt(normalized_prompt),
            task_mode="single_turn",
        )

    def _should_wrap_open_code_reasoning_prompt(self) -> bool:
        task_name = (self.config.task_name or "").lower()
        config_path = (self.config.config_path or "").lower()
        input_dir = (self.config.input_dir or "").lower()
        output_dir = (self.config.output_dir or "").lower()
        joined = " ".join((task_name, config_path, input_dir, output_dir))
        return "opencode_reasoning" in joined or "opencodereasoning" in joined

    def _format_task_prompt(self, normalized_prompt: str) -> str:
        if self._should_wrap_open_code_reasoning_prompt():
            return self.OPEN_CODE_REASONING_PYTHON_WRAPPER + normalized_prompt
        return normalized_prompt

    @staticmethod
    def _extract_row_dict(row: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {}
        for key, value in row.items():
            if hasattr(value, "as_py"):
                try:
                    value = value.as_py()
                except Exception:
                    pass
            normalized[key] = value
        return normalized

    @staticmethod
    def _row_judge_suite(row_data: Dict[str, Any]) -> Optional[str]:
        raw_spec = row_data.get("judge_spec")
        if isinstance(raw_spec, str):
            if not raw_spec.strip():
                return None
            try:
                raw_spec = json.loads(raw_spec)
            except Exception:
                return None
        if not isinstance(raw_spec, dict):
            return None
        suite = raw_spec.get("suite")
        if suite is None:
            return None
        return str(suite)

    def _row_matches_judge_suite_filter(self, row_data: Dict[str, Any]) -> bool:
        allowed = self.config.judge_suites or []
        if not allowed:
            return True
        suite = self._row_judge_suite(row_data)
        return suite in set(str(item) for item in allowed)

    def _jsonl_batch_iterator(self,
                              file_path: str,
                              batch_size: int) -> Iterator[pa.Table]:
        with open(file_path, "r", encoding="utf-8") as f:
            batch_data = []
            for line in f:
                if not line.strip():
                    continue
                try:
                    batch_data.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON line in %s", file_path)
                    continue
                if len(batch_data) >= batch_size:
                    yield pa.Table.from_pylist(batch_data)
                    batch_data = []
            if batch_data:
                yield pa.Table.from_pylist(batch_data)

    async def producer(self, input_files: List[str], pbar: tqdm):
        self.input_exhausted = False
        self._load_completed_records(input_files)
        logger.info("Producer started. Total files to process in this node: %s",
                    len(input_files))
        if not input_files:
            logger.warning("No files assigned. Exiting producer.")
            return

        row_budget = None
        limit_reached = False
        if self.config.sample_limit is not None:
            row_budget = max(0, int(self.config.sample_limit))
            if row_budget == 0:
                logger.info("Sample limit is 0. Producer will not queue any tasks.")
                return

        pending_files = list(input_files)
        active_iterators = []

        def open_next_file():
            if not pending_files:
                return None
            file_path = pending_files.pop(0)
            ext = os.path.splitext(file_path)[1].lower()
            try:
                if ext == ".jsonl":
                    return (file_path, self._jsonl_batch_iterator(
                        file_path, self.config.batch_size))
                if ext == ".parquet":
                    parquet_file = pq.ParquetFile(file_path)
                    return (file_path,
                            parquet_file.iter_batches(
                                batch_size=self.config.batch_size))
                logger.warning("Unsupported file format: %s for %s", ext,
                               file_path)
                return None
            except Exception as e:
                logger.error("Failed to open %s: %s", file_path, e)
                return None

        while len(active_iterators) < self.config.active_file_window and pending_files:
            opened = open_next_file()
            if opened:
                active_iterators.append(opened)

        global_indices_tracker = {
            file_path: 0
            for file_path, _ in active_iterators
        }

        while active_iterators:
            if self.stop_requested:
                logger.info("Producer stop requested. No more new rows will be queued.")
                break
            for file_path, iterator in list(active_iterators):
                if self.stop_requested:
                    break
                try:
                    batch = next(iterator)
                    rows = [self._extract_row_dict(row) for row in batch.to_pylist()]
                    current_idx = global_indices_tracker.get(file_path, 0)

                    for row_data in rows:
                        if self.stop_requested:
                            break
                        if row_budget is not None and row_budget <= 0:
                            limit_reached = True
                            logger.info(
                                "Reached sample limit (%s). Producer will stop queueing new rows.",
                                self.config.sample_limit,
                            )
                            break
                        source_file = os.path.abspath(file_path)
                        source_row = current_idx

                        if not self._row_matches_judge_suite_filter(row_data):
                            current_idx += 1
                            continue

                        if row_budget is not None:
                            row_budget -= 1

                        prepared_task = self._prepare_task_input(
                            row_data.get(self.config.input_content_field))
                        if prepared_task:
                            for rollout_index in range(self.config.rollout_count):
                                if self._is_completed(source_file, source_row,
                                                      rollout_index):
                                    continue

                                if self.failure_recorder.should_skip(
                                        source_file, source_row,
                                        rollout_index):
                                    continue

                                await self.task_queue.put(
                                    TaskItem(
                                        source_file=source_file,
                                        source_row=source_row,
                                        rollout_index=rollout_index,
                                        row_data=row_data,
                                        prompt=prepared_task.prompt,
                                        input_messages=prepared_task.input_messages,
                                        task_mode=prepared_task.task_mode,
                                    ))
                                discovered = int(getattr(pbar, "_discovered_tasks",
                                                         0)) + 1
                                pbar._discovered_tasks = discovered
                                self._refresh_progress_postfix(pbar)
                        else:
                            logger.warning(
                                "Skipping %s row %s: unsupported or empty input in field '%s'",
                                source_file,
                                source_row,
                                self.config.input_content_field,
                            )
                            await self.failure_recorder.record_failure(
                                source_file,
                                source_row,
                                f"empty_or_unsupported_input:{self.config.input_content_field}",
                                rollout_index=0,
                            )
                        current_idx += 1

                    global_indices_tracker[file_path] = current_idx
                    await asyncio.sleep(0)
                    if limit_reached:
                        break

                except StopIteration:
                    logger.info("File finished reading: %s", file_path)
                    active_iterators = [(f, it) for f, it in active_iterators
                                        if f != file_path]
                    global_indices_tracker.pop(file_path, None)
                    new_it = open_next_file()
                    if new_it:
                        active_iterators.append(new_it)
                        global_indices_tracker[new_it[0]] = 0
                except Exception as e:
                    logger.error("Error reading batch from %s: %s", file_path,
                                 e)
                    active_iterators = [(f, it) for f, it in active_iterators
                                        if f != file_path]
                    global_indices_tracker.pop(file_path, None)
            if limit_reached:
                break

        self.input_exhausted = (not limit_reached and not self.stop_requested
                                and not pending_files and not active_iterators)

        logger.info("All assigned files read completely.")

    @staticmethod
    def _extract_metadata(row_data: Dict[str, Any]) -> str:
        metadata = row_data.get("metadata")
        if isinstance(metadata, str):
            return metadata
        if metadata is not None:
            return safe_json_dumps(metadata)
        return safe_json_dumps({
            "avatarUrl": None,
            "category": None,
            "custom_instruction": None,
            "hash": None,
            "id": None,
            "idx": None,
            "language": None,
            "model": None,
            "model_name": None,
            "skip_prompt_formatting": None,
            "source": None,
            "system_prompt": None,
            "title": None,
            "topic": None,
            "views": None,
        })

    @staticmethod
    def _dataset_name(row_data: Dict[str, Any], source_file: str) -> str:
        for key in ("dataset_name", "dataset", "source_dataset"):
            value = row_data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return Path(source_file).stem

    @staticmethod
    def _message_char_count(messages: List[Dict[str, Any]]) -> int:
        total = 0
        for message in messages:
            for key in ("content", "reasoning_content", "name", "tool_call_id"):
                value = message.get(key)
                if isinstance(value, str):
                    total += len(value)
            if message.get("tool_calls") is not None:
                total += len(safe_json_dumps(message["tool_calls"]))
        return total

    @staticmethod
    def _assistant_turn_count(messages: List[Dict[str, Any]]) -> int:
        return sum(1 for message in messages if message.get("role") == "assistant")

    @staticmethod
    def _ends_with_user(messages: List[Dict[str, Any]]) -> bool:
        return bool(messages) and messages[-1].get("role") == "user"

    @staticmethod
    def _merge_usage_totals(current: Dict[str, Any],
                            usage: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(current)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                merged[key] = int(merged.get(key, 0) or 0) + int(value)
        return merged

    @staticmethod
    def _summarize_finish_reason(finish_reasons: List[Optional[str]]
                                 ) -> Optional[str]:
        if any(reason == "length" for reason in finish_reasons):
            return "length"
        non_empty = [reason for reason in finish_reasons if reason]
        return non_empty[-1] if non_empty else None

    @staticmethod
    def _partial_distill_error(exc: Exception) -> str:
        if isinstance(exc, APITimeoutError):
            return "timeout_1h"
        if isinstance(exc, BadRequestError):
            return f"bad_request:{exc}"
        return f"worker_error:{type(exc).__name__}:{exc}"

    async def _generate_single_turn_result(
            self, item: TaskItem) -> GenerationResultItem:
        if not item.prompt:
            raise RuntimeError("empty_single_turn_prompt")
        response = await self.llm_manager.generate(item.prompt)
        if not response:
            raise RuntimeError("empty_response")
        return GenerationResultItem(
            task=item,
            messages=response["messages"],
            finish_reason=response.get("finish_reason"),
            usage=response.get("usage") or {},
        )

    async def _generate_multi_turn_result(
            self, item: TaskItem) -> GenerationResultItem:
        input_messages = item.input_messages or []
        distilled_messages: List[Dict[str, Any]] = []
        aggregated_usage: Dict[str, Any] = {}
        finish_reasons: List[Optional[str]] = []
        assistant_turns_total = self._assistant_turn_count(input_messages)
        if (self.config.complete_trailing_user_turn
                and self._ends_with_user(input_messages)):
            assistant_turns_total += 1
        assistant_turns_completed = 0
        distill_status = "success"
        distill_error = None

        for message in input_messages:
            normalized_message = ensure_message_shape(message)
            if normalized_message.get("role") != "assistant":
                distilled_messages.append(normalized_message)
                continue

            try:
                response = await self.llm_manager.generate_messages(
                    distilled_messages)
                if not response or response.get("assistant_message") is None:
                    raise RuntimeError("empty_response")
            except NoHealthyBackendsError:
                raise
            except Exception as exc:
                distill_status = "partial"
                distill_error = self._partial_distill_error(exc)
                logger.warning(
                    "Stopping multi-turn distillation early for %s row %s after %s/%s assistant turns: %s",
                    item.source_file,
                    item.source_row,
                    assistant_turns_completed,
                    assistant_turns_total,
                    distill_error,
                )
                break

            distilled_messages.append(
                ensure_message_shape(response["assistant_message"]))
            assistant_turns_completed += 1
            finish_reasons.append(response.get("finish_reason"))
            aggregated_usage = self._merge_usage_totals(
                aggregated_usage, response.get("usage") or {})

        if (distill_status == "success" and self.config.complete_trailing_user_turn
                and self._ends_with_user(distilled_messages)):
            try:
                response = await self.llm_manager.generate_messages(
                    distilled_messages)
                if not response or response.get("assistant_message") is None:
                    raise RuntimeError("empty_response")
            except NoHealthyBackendsError:
                raise
            except Exception as exc:
                distill_status = "partial"
                distill_error = self._partial_distill_error(exc)
                logger.warning(
                    "Failed to complete trailing user turn for %s row %s after %s/%s assistant turns: %s",
                    item.source_file,
                    item.source_row,
                    assistant_turns_completed,
                    assistant_turns_total,
                    distill_error,
                )
            else:
                distilled_messages.append(
                    ensure_message_shape(response["assistant_message"]))
                assistant_turns_completed += 1
                finish_reasons.append(response.get("finish_reason"))
                aggregated_usage = self._merge_usage_totals(
                    aggregated_usage, response.get("usage") or {})

        return GenerationResultItem(
            task=item,
            messages=distilled_messages,
            finish_reason=self._summarize_finish_reason(finish_reasons),
            usage=aggregated_usage,
            distill_status=distill_status,
            distill_error=distill_error,
            assistant_turns_completed=assistant_turns_completed,
            assistant_turns_total=assistant_turns_total,
        )

    async def _generate_task_result(self, item: TaskItem) -> GenerationResultItem:
        if item.task_mode == "multi_turn":
            return await self._generate_multi_turn_result(item)
        return await self._generate_single_turn_result(item)

    def _build_output_record(
            self,
            task: TaskItem,
            messages: List[Dict[str, Any]],
        finish_reason: Optional[str],
        usage: Dict[str, Any],
        distill_status: str = "success",
        distill_error: Optional[str] = None,
        assistant_turns_completed: int = 1,
        assistant_turns_total: int = 1,
    ) -> Dict[str, Any]:
        messages = [ensure_message_shape(message) for message in messages]
        dataset_name = self._dataset_name(task.row_data, task.source_file)
        judge_mode = normalize_judge_mode(getattr(self.config, "judge_mode",
                                                  "auto"))
        if judge_mode_disables_all(judge_mode):
            judge_result = {
                "judge_type": None,
                "judge_backend": None,
                "is_correct": None,
                "judge_status": "not_applicable",
                "judge_detail": {},
            }
        elif task.task_mode == "multi_turn":
            judge_result = judge_output(
                task.row_data,
                messages,
                label_field=self.config.label_field,
                judge_mode=judge_mode,
            )
        else:
            judge_result = judge_output_with_timeout(
                task.row_data,
                messages,
                label_field=self.config.label_field,
                judge_mode=judge_mode,
                timeout=self.config.judge_timeout_sec,
            )
        if judge_result.get("judge_type") is None:
            assume_no_judge_correct = (
                self.config.treat_no_judge_as_correct
                and finish_reason != "length")
            judge_result = {
                "judge_type": "none",
                "judge_backend": "none",
                "is_correct": True if assume_no_judge_correct else None,
                "judge_status": (
                    "assumed_correct"
                    if assume_no_judge_correct else
                    "not_applicable_overlong"
                    if (self.config.treat_no_judge_as_correct
                        and finish_reason == "length") else "not_applicable"),
                "judge_detail": {
                    "assumed_correct": True
                } if assume_no_judge_correct else {},
            }

        return {
            "dataset_name": dataset_name,
            "sample_id":
            f"{dataset_name}:{task.source_file}:{task.source_row}:rollout_{task.rollout_index}",
            "dedup_hash":
            hashlib.sha256(safe_json_dumps(messages).encode("utf-8")).hexdigest(),
            "content_chars": self._message_char_count(messages),
            "turn_count": len(messages),
            "has_reasoning":
            any(isinstance(msg.get("reasoning_content"), str)
                and msg.get("reasoning_content").strip() for msg in messages),
            "text": None,
            "messages": messages,
            "chosen": None,
            "rejected": None,
            "tools": None,
            "system": None,
            "metadata": self._extract_metadata(task.row_data),
            "adapter_status": "success",
            "adapter_error": None,
            "adapter_name": "sharegpt",
            "record_mode": "sft",
            "generation_finish_reason": finish_reason,
            "generation_usage": usage or None,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "distill_status": distill_status,
            "distill_error": distill_error,
            "assistant_turns_completed": assistant_turns_completed,
            "assistant_turns_total": assistant_turns_total,
            "source_file": task.source_file,
            "source_row": task.source_row,
            "rollout_index": task.rollout_index,
            "judge_type": judge_result["judge_type"],
            "judge_backend": judge_result.get("judge_backend"),
            "is_correct": judge_result["is_correct"],
            "judge_status": judge_result["judge_status"],
            "judge_detail": judge_result["judge_detail"],
        }

    async def worker(self, worker_id: int, pbar: tqdm):
        while True:
            item = await self.task_queue.get()
            if item is None:
                self.task_queue.task_done()
                break

            try:
                result = await self._generate_task_result(item)
                await self.judge_queue.put(result)

            except NoHealthyBackendsError as e:
                logger.error(
                    "Fatal backend outage in worker %s for %s row %s: %s",
                    worker_id,
                    item.source_file,
                    item.source_row,
                    e,
                )
                self._request_stop("all_vllm_backends_terminated")
                raise
            except APITimeoutError:
                logger.warning("TIMEOUT: %s row %s", item.source_file,
                               item.source_row)
                await self.failure_recorder.record_failure(
                    item.source_file,
                    item.source_row,
                    "timeout_1h",
                    rollout_index=item.rollout_index,
                )
            except BadRequestError as e:
                logger.error("BAD REQUEST: %s row %s - %s", item.source_file,
                             item.source_row, e)
                await self.failure_recorder.record_failure(
                    item.source_file,
                    item.source_row,
                    f"bad_request:{e}",
                    rollout_index=item.rollout_index,
                )
            except Exception as e:
                logger.error("Worker Error: %s row %s - %s: %s",
                             item.source_file, item.source_row,
                             type(e).__name__, e)
                await self.failure_recorder.record_failure(
                    item.source_file,
                    item.source_row,
                    f"worker_error:{type(e).__name__}:{e}",
                    rollout_index=item.rollout_index,
                )
            finally:
                self.task_queue.task_done()
                pbar.update(1)

    async def _wait_tasks(self, tasks: List[asyncio.Task], stage_name: str):
        results = await asyncio.gather(*tasks, return_exceptions=True)
        unexpected_errors = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Unhandled %s task error [%s]: %r", stage_name, idx,
                             result)
                unexpected_errors.append(result)
        if unexpected_errors:
            raise RuntimeError(
                f"{stage_name}_tasks_failed:{len(unexpected_errors)}")

    async def _wait_task(self, task: asyncio.Task, stage_name: str):
        result = await asyncio.gather(task, return_exceptions=True)
        if result and isinstance(result[0], Exception):
            logger.error("Unhandled %s task error: %r", stage_name, result[0])
            raise RuntimeError(f"{stage_name}_task_failed")

    async def judge_worker(self, worker_id: int):
        while True:
            item = await self.judge_queue.get()
            if item is None:
                self.judge_queue.task_done()
                break

            try:
                record = await asyncio.to_thread(
                    self._build_output_record,
                    item.task,
                    item.messages,
                    item.finish_reason,
                    item.usage,
                    item.distill_status,
                    item.distill_error,
                    item.assistant_turns_completed,
                    item.assistant_turns_total,
                )
                await self.result_queue.put(
                    ResultItem(
                        source_file=item.task.source_file,
                        source_row=item.task.source_row,
                        record=record,
                        estimated_size=len(
                            safe_json_dumps(record).encode("utf-8")),
                    ))
            except Exception as e:
                logger.error("Judge Error: %s row %s - %s",
                             item.task.source_file, item.task.source_row, e)
                await self.failure_recorder.record_failure(
                    item.task.source_file,
                    item.task.source_row,
                    f"judge_error:{type(e).__name__}:{e}",
                    rollout_index=item.task.rollout_index,
                )
            finally:
                self.judge_queue.task_done()

    async def _next_index(self, stream_name: str, kind: str) -> int:
        async with self.counter_locks[stream_name]:
            if kind == "segment":
                current = self.segment_counters[stream_name]
                self.segment_counters[stream_name] += 1
                return current
            current = self.shard_counters[stream_name]
            self.shard_counters[stream_name] += 1
            return current

    def _segment_target_bytes(self) -> int:
        return self.config.segment_target_size_mb * 1024 * 1024

    async def _write_segment_with_retry(self, records: List[Dict[str, Any]],
                                        stream_name: str):
        if not records:
            return

        segment_idx = await self._next_index(stream_name, "segment")
        target = os.path.join(
            self._stream_dir(stream_name, "segments"),
            f"segment_{segment_idx:06d}.jsonl",
        )
        temp = target + ".tmp"
        last_error = None

        for attempt in range(1, self.config.write_retries + 1):
            try:
                with open(temp, "w", encoding="utf-8") as f:
                    for record in records:
                        f.write(safe_json_dumps(record) + "\n")
                os.replace(temp, target)
                if stream_name == self.STREAM_ALL:
                    for record in records:
                        self.completed_records.add(
                            self._completed_key(record["source_file"],
                                                int(record["source_row"]),
                                                int(record.get("rollout_index", 0)
                                                    or 0)))
                    self._append_completed_index(records)
                    self.resume_progress["written"] += len(records)
                    self.resume_progress["correct"] += sum(
                        1 for record in records
                        if record.get("is_correct") is True)
                    self.resume_progress["overlong"] += sum(
                        1 for record in records
                        if record.get("generation_finish_reason") == "length")
                self._save_resume_state()
                logger.info("Wrote %s segment_%06d.jsonl with %s rows",
                            stream_name, segment_idx, len(records))
                return
            except Exception as e:
                last_error = e
                logger.error(
                    "Segment write failed for %s segment_%06d attempt %s/%s: %s",
                    stream_name,
                    segment_idx,
                    attempt,
                    self.config.write_retries,
                    e,
                )
                await asyncio.sleep(min(5 * attempt, 15))

        raise RuntimeError(
            f"failed_to_write_{stream_name}_segment_{segment_idx:06d}: {last_error}")

    def _flush_records(self, records: List[Dict[str, Any]], stream_name: str,
                       shard_idx: int):
        target = os.path.join(
            self._stream_dir(stream_name, "shards"),
            f"shard_{shard_idx:05d}.parquet",
        )
        temp = target + ".tmp"
        table = pa.Table.from_pylist(
            [self._normalize_for_parquet(record) for record in records])
        pq.write_table(table, temp)
        os.replace(temp, target)

    def _normalize_for_parquet(self, value: Any):
        if isinstance(value, dict):
            if not value:
                return None
            return {
                key: self._normalize_for_parquet(sub_value)
                for key, sub_value in value.items()
            }
        if isinstance(value, list):
            return [self._normalize_for_parquet(item) for item in value]
        return value

    async def _flush_with_retry(self, records: List[Dict[str, Any]],
                                stream_name: str):
        if not records:
            return None

        shard_idx = await self._next_index(stream_name, "shard")
        last_error = None
        for attempt in range(1, self.config.write_retries + 1):
            try:
                self._flush_records(records, stream_name, shard_idx)
                self._save_resume_state()
                logger.info("Flushed %s shard_%05d.parquet with %s rows",
                            stream_name, shard_idx, len(records))
                return shard_idx
            except Exception as e:
                last_error = e
                logger.error(
                    "Write failed for %s shard_%05d attempt %s/%s: %s",
                    stream_name,
                    shard_idx,
                    attempt,
                    self.config.write_retries,
                    e,
                )
                await asyncio.sleep(min(5 * attempt, 15))

        raise RuntimeError(
            f"failed_to_flush_{stream_name}_shard_{shard_idx:05d}: {last_error}")

    async def _flush_segment_buffer(self, buffers: Dict[str, List[Dict[str,
                                                                       Any]]],
                                    buffer_sizes: Dict[str, int],
                                    stream_name: str):
        records = buffers[stream_name]
        if not records:
            return
        await self._write_segment_with_retry(records, stream_name)
        buffers[stream_name] = []
        buffer_sizes[stream_name] = 0

    async def _merge_stream_segments(self, stream_name: str):
        segment_dir = self._stream_dir(stream_name, "segments")
        segment_paths = sorted(
            glob.glob(os.path.join(segment_dir, "segment_*.jsonl")))
        if not segment_paths:
            return {
                "merged_segments_this_run": 0,
                "shards_written_this_run": 0,
            }

        state = self._load_merge_state(stream_name)
        merged_segments = set(state.get("merged_segments", []))
        pending_segments = [
            path for path in segment_paths if os.path.basename(path)
            not in merged_segments
        ]
        if not pending_segments:
            return {
                "merged_segments_this_run": 0,
                "shards_written_this_run": 0,
            }

        shard_target_bytes = self.config.shard_target_size_mb * 1024 * 1024
        records_buffer: List[Dict[str, Any]] = []
        records_size = 0
        current_segment_names: List[str] = []
        merged_segments_this_run = 0
        shards_written_this_run = 0

        async def flush_pending():
            nonlocal records_buffer, records_size, current_segment_names
            nonlocal merged_segments_this_run, shards_written_this_run
            if not records_buffer:
                return
            shard_idx = await self._flush_with_retry(records_buffer, stream_name)
            merged_segments.update(current_segment_names)
            state["merged_segments"] = sorted(merged_segments)
            self._save_merge_state(stream_name, state)
            merged_segments_this_run += len(current_segment_names)
            if shard_idx is not None:
                shards_written_this_run += 1
            records_buffer = []
            records_size = 0
            current_segment_names = []

        for segment_path in pending_segments:
            segment_name = os.path.basename(segment_path)
            segment_records = []
            segment_size = 0
            with open(segment_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    segment_records.append(record)
                    segment_size += len(line.encode("utf-8"))

            if records_buffer and records_size + segment_size > shard_target_bytes:
                await flush_pending()

            records_buffer.extend(segment_records)
            records_size += segment_size
            current_segment_names.append(segment_name)

        await flush_pending()
        return {
            "merged_segments_this_run": merged_segments_this_run,
            "shards_written_this_run": shards_written_this_run,
        }

    async def _run_merge_cycle(self, buffers: Dict[str, List[Dict[str, Any]]],
                               buffer_sizes: Dict[str, int]) -> Dict[str, Any]:
        await self._flush_segment_buffer(buffers, buffer_sizes, self.STREAM_ALL)
        await self._flush_segment_buffer(buffers, buffer_sizes,
                                         self.STREAM_CORRECT)
        all_summary = await self._merge_stream_segments(self.STREAM_ALL)
        correct_summary = await self._merge_stream_segments(self.STREAM_CORRECT)
        return {
            "all": all_summary,
            "correct": correct_summary,
        }

    async def _finalize_interrupt_outputs(
            self,
            buffers: Dict[str, List[Dict[str, Any]]],
            buffer_sizes: Dict[str, int]) -> Dict[str, Any]:
        merge_summary = await self._run_merge_cycle(buffers, buffer_sizes)
        upload_summary: Dict[str, Any] = {
            "uploaded": 0,
            "skipped": 0,
            "reason": "disabled",
        }
        if self.config.upload_merged_shards:
            try:
                upload_summary = await self._upload_pending_correct_shards()
            except Exception as e:
                logger.error("Failed to upload correct shards during interrupt finalization: %s",
                             e)
                upload_summary = {
                    "uploaded": 0,
                    "skipped": 0,
                    "reason": f"error:{type(e).__name__}",
                }

        stats_summary = {
            self.STREAM_ALL:
            self._write_interrupt_quick_stats_summary(self.STREAM_ALL),
            self.STREAM_CORRECT:
            self._write_interrupt_quick_stats_summary(self.STREAM_CORRECT),
        }
        summary = {
            "merge": merge_summary,
            "upload": upload_summary,
            "stats": stats_summary,
        }
        self.last_interrupt_summary = summary
        return summary

    async def writer_daemon(self, pbar: tqdm):
        buffers = {
            self.STREAM_ALL: [],
            self.STREAM_CORRECT: [],
        }
        buffer_sizes = {
            self.STREAM_ALL: 0,
            self.STREAM_CORRECT: 0,
        }
        segment_target_bytes = self._segment_target_bytes()
        flush_interval = max(0.0, self.config.segment_flush_interval_sec)
        writes_since_merge = 0
        merges_since_upload = 0

        while True:
            if self.interrupt_finalize_requested:
                await self._finalize_interrupt_outputs(buffers, buffer_sizes)
                break
            try:
                poll_timeout = (flush_interval if flush_interval > 0 else
                                self.WRITER_POLL_INTERVAL_SEC)
                item = await asyncio.wait_for(self.result_queue.get(),
                                              timeout=poll_timeout)
            except asyncio.TimeoutError:
                if self.interrupt_finalize_requested:
                    await self._finalize_interrupt_outputs(buffers, buffer_sizes)
                    break
                await self._flush_segment_buffer(buffers, buffer_sizes,
                                                 self.STREAM_ALL)
                await self._flush_segment_buffer(buffers, buffer_sizes,
                                                 self.STREAM_CORRECT)
                continue

            if item is None:
                try:
                    await self._run_merge_cycle(buffers, buffer_sizes)
                    if self.config.upload_merged_shards:
                        try:
                            await self._upload_pending_correct_shards()
                        except Exception as e:
                            logger.error("Failed to upload correct shards: %s",
                                         e)
                finally:
                    self.result_queue.task_done()
                break

            buffers[self.STREAM_ALL].append(item.record)
            buffer_sizes[self.STREAM_ALL] += item.estimated_size
            pbar._written_tasks = int(getattr(pbar, "_written_tasks", 0)) + 1

            if item.record.get("is_correct") is True:
                buffers[self.STREAM_CORRECT].append(item.record)
                buffer_sizes[self.STREAM_CORRECT] += item.estimated_size
                pbar._correct_tasks = int(
                    getattr(pbar, "_correct_tasks", 0)) + 1
            if item.record.get("generation_finish_reason") == "length":
                pbar._overlong_tasks = int(
                    getattr(pbar, "_overlong_tasks", 0)) + 1

            try:
                self._refresh_progress_postfix(pbar)
                writes_since_merge += 1
                if buffer_sizes[self.STREAM_ALL] >= segment_target_bytes:
                    await self._flush_segment_buffer(buffers, buffer_sizes,
                                                     self.STREAM_ALL)
                if buffer_sizes[self.STREAM_CORRECT] >= segment_target_bytes:
                    await self._flush_segment_buffer(buffers, buffer_sizes,
                                                     self.STREAM_CORRECT)
                if (self.config.merge_every_n_writes > 0
                        and writes_since_merge >= self.config.merge_every_n_writes):
                    periodic_summary = await self._run_merge_cycle(buffers,
                                                                   buffer_sizes)
                    writes_since_merge = 0
                    if periodic_summary["correct"].get(
                            "merged_segments_this_run", 0) > 0:
                        merges_since_upload += 1
                    if (self.config.upload_merged_shards
                            and merges_since_upload >= max(
                                1, int(self.config.upload_every_n_merges or 1))):
                        try:
                            await self._upload_pending_correct_shards()
                        except Exception as e:
                            logger.error("Failed to upload correct shards: %s",
                                         e)
                        merges_since_upload = 0
            finally:
                self.result_queue.task_done()

    async def run(self):
        all_files = sorted(
            glob.glob(os.path.join(self.config.input_dir,
                                   self.config.file_pattern)))

        start_idx = max(0, self.config.range_start)
        end_idx = self.config.range_end if self.config.range_end is not None else len(
            all_files)
        end_idx = min(end_idx, len(all_files))
        input_files = all_files[start_idx:end_idx]

        print("=" * 50)
        print("Tasks Assigned to this Node")
        print(f"Pattern : {self.config.file_pattern}")
        print(f"Range   : [{start_idx} : {end_idx}]")
        print(f"Files ({len(input_files)}):")
        for file_path in input_files:
            print(f"   - {os.path.basename(file_path)}")
        estimated_rows = self._estimate_input_rows(input_files)
        effective_rows = self._effective_input_row_limit(estimated_rows)
        if effective_rows is not None:
            estimated_tasks = effective_rows * self.config.rollout_count
            if self.config.sample_limit is not None and estimated_rows is not None:
                print(
                    f"Estimated input rows : {effective_rows} (capped from {estimated_rows} by sample_limit)"
                )
            else:
                print(f"Estimated input rows : {effective_rows}")
            print(f"Estimated max tasks  : {estimated_tasks}")
        print("=" * 50)

        if not input_files:
            logger.error("No files matched the range criteria. Exiting.")
            return

        self._load_completed_records(input_files)

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._request_stop, sig.name)
            except NotImplementedError:
                pass

        estimated_rows = self._estimate_input_rows(input_files)
        estimated_tasks = None
        effective_rows = self._effective_input_row_limit(estimated_rows)
        if effective_rows is not None:
            estimated_tasks = effective_rows * self.config.rollout_count
        start_progress = dict(self.resume_progress)

        pbar = tqdm(
            total=estimated_tasks,
            desc=f"Processing [{start_idx}:{end_idx}]",
            unit="task",
            dynamic_ncols=True,
        )
        pbar._resumed_tasks = int(self.resume_progress["written"])
        pbar._discovered_tasks = int(self.resume_progress["written"])
        pbar._written_tasks = int(self.resume_progress["written"])
        pbar._correct_tasks = int(self.resume_progress["correct"])
        pbar._overlong_tasks = int(self.resume_progress["overlong"])
        pbar.n = int(self.resume_progress["written"])
        self._refresh_progress_postfix(pbar)
        writer_task = asyncio.create_task(self.writer_daemon(pbar))
        producer_task = asyncio.create_task(self.producer(input_files, pbar))
        generation_workers = [
            asyncio.create_task(self.worker(i, pbar))
            for i in range(self.config.max_concurrency)
        ]
        judge_workers = [
            asyncio.create_task(self.judge_worker(i))
            for i in range(self.config.judge_concurrency)
        ]
        interrupted = False

        try:
            producer_and_generation_tasks = {
                producer_task,
                *generation_workers,
            }
            while not producer_task.done():
                if self.interrupt_finalize_requested:
                    raise InterruptFinalizeRequested()
                done, _ = await asyncio.wait(
                    producer_and_generation_tasks,
                    timeout=self.WRITER_POLL_INTERVAL_SEC,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if self.interrupt_finalize_requested:
                    raise InterruptFinalizeRequested()
                if not done:
                    continue
                task_error = next(
                    (task.exception() for task in done if task.exception() is not None),
                    None,
                )
                if task_error is not None:
                    raise task_error
            if producer_task.exception() is not None:
                raise producer_task.exception()
            if self.interrupt_finalize_requested:
                raise InterruptFinalizeRequested()
            for _ in generation_workers:
                await self.task_queue.put(None)
            await self._wait_tasks(generation_workers, "generation")
            pbar.close()
            if self.interrupt_finalize_requested:
                raise InterruptFinalizeRequested()
            for _ in judge_workers:
                await self.judge_queue.put(None)
            await self._wait_tasks(judge_workers, "judge")
            if self.interrupt_finalize_requested:
                raise InterruptFinalizeRequested()
            await self.result_queue.put(None)
            await self._wait_task(writer_task, "writer")
        except InterruptFinalizeRequested:
            interrupted = True
            tasks_to_cancel = []
            for task in [producer_task, *generation_workers, *judge_workers]:
                if not task.done():
                    task.cancel()
                    tasks_to_cancel.append(task)
            if tasks_to_cancel:
                await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            await self._wait_task(writer_task, "writer")
        except Exception:
            self.stop_requested = True
            tasks_to_cancel = []
            for task in [producer_task, *generation_workers, *judge_workers,
                         writer_task]:
                if not task.done():
                    task.cancel()
                    tasks_to_cancel.append(task)
            if tasks_to_cancel:
                await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            raise
        finally:
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.remove_signal_handler(sig)
                except NotImplementedError:
                    pass
            if not pbar.disable:
                pbar.close()

        logger.info("Pipeline Complete.")
        return {
            "written_delta": int(self.resume_progress["written"]) - int(
                start_progress["written"]),
            "correct_delta": int(self.resume_progress["correct"]) - int(
                start_progress["correct"]),
            "overlong_delta": int(self.resume_progress["overlong"]) - int(
                start_progress["overlong"]),
            "input_exhausted": bool(self.input_exhausted),
            "interrupted": interrupted,
        }


RoundRobinPipeline = DistillPipeline
