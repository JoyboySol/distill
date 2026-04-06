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
    from ..runtime.settings import PipelineConfig, logger
    from ..common.utils import ensure_message_shape, safe_json_dumps
    from .failure import FailureRecorder
    from .judge import judge_output
    from .llm import AsyncLLMManager, GenerationResponse
except ImportError:
    from runtime.settings import PipelineConfig, logger
    from common.utils import ensure_message_shape, safe_json_dumps
    from core.failure import FailureRecorder
    from core.judge import judge_output
    from core.llm import AsyncLLMManager, GenerationResponse


@dataclass
class TaskItem:
    source_file: str
    source_row: int
    rollout_index: int
    prompt: str
    row_data: Dict[str, Any]


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


class DistillPipeline:
    STREAM_ALL = "all"
    STREAM_CORRECT = "correct"

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.llm_manager = AsyncLLMManager(config)
        self.failure_recorder = FailureRecorder(config.failure_log)
        self.task_queue: asyncio.Queue = asyncio.Queue(
            maxsize=config.queue_max_size)
        self.judge_queue: asyncio.Queue = asyncio.Queue()
        self.result_queue: asyncio.Queue = asyncio.Queue()
        self.stop_requested = False

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

    @staticmethod
    def _refresh_progress_postfix(pbar: tqdm):
        discovered = int(getattr(pbar, "_discovered_tasks", 0))
        written = int(getattr(pbar, "_written_tasks", 0))
        correct = int(getattr(pbar, "_correct_tasks", 0))
        overlong = int(getattr(pbar, "_overlong_tasks", 0))
        pbar.set_postfix_str(
            f"discovered={discovered} written={written} correct={correct} overlong={overlong}"
        )

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

    def _request_stop(self, signame: str):
        if self.stop_requested:
            return
        self.stop_requested = True
        logger.warning(
            "Received %s, stopping producer and draining queued work before exit...",
            signame,
        )

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
                f.write(
                    safe_json_dumps({
                        "source_file": record["source_file"],
                        "source_row": int(record["source_row"]),
                        "rollout_index": int(record.get("rollout_index", 0)
                                             or 0),
                    }) + "\n")

    def _load_completed_index(self) -> Optional[Set[str]]:
        path = self._completed_index_path()
        if not os.path.exists(path):
            return None
        completed_sources: Set[str] = set()
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    completed_sources.add(
                        self._completed_key(
                            str(row["source_file"]),
                            int(row["source_row"]),
                            int(row.get("rollout_index", 0) or 0),
                        ))
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

    def _load_completed_records(self):
        if self.completed_records_loaded:
            return

        self.completed_records_loaded = True
        os.makedirs(self.config.output_dir, exist_ok=True)

        logger.info("Loading completed-record index for resume support...")
        resume_state = self._load_resume_state()
        completed_index = self._load_completed_index()
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
                "Loaded resume state without full scan: completed=%s written=%s correct=%s overlong=%s",
                len(self.completed_records),
                self.resume_progress["written"],
                self.resume_progress["correct"],
                self.resume_progress["overlong"],
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
                "Recovered from completed index without full content scan: completed=%s written=%s correct=%s overlong=%s",
                len(self.completed_records),
                self.resume_progress["written"],
                self.resume_progress["correct"],
                self.resume_progress["overlong"],
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
        self._load_completed_records()
        logger.info("Producer started. Total files to process in this node: %s",
                    len(input_files))
        if not input_files:
            logger.warning("No files assigned. Exiting producer.")
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
                        source_file = os.path.abspath(file_path)
                        source_row = current_idx

                        normalized_prompt = self._normalize_prompt(
                            row_data.get(self.config.input_content_field))
                        if normalized_prompt:
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
                                        prompt=normalized_prompt,
                                        row_data=row_data,
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

    def _build_output_record(
        self,
        task: TaskItem,
        messages: List[Dict[str, Any]],
        finish_reason: Optional[str],
        usage: Dict[str, Any],
    ) -> Dict[str, Any]:
        messages = [ensure_message_shape(message) for message in messages]
        dataset_name = self._dataset_name(task.row_data, task.source_file)
        judge_result = judge_output(
            task.row_data,
            messages,
            label_field=self.config.label_field,
        )

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
                response = await self.llm_manager.generate(item.prompt)
                if not response:
                    raise RuntimeError("empty_response")
                await self.judge_queue.put(
                    GenerationResultItem(
                        task=item,
                        messages=response["messages"],
                        finish_reason=response.get("finish_reason"),
                        usage=response.get("usage") or {},
                    ))

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
                record = self._build_output_record(
                    item.task,
                    item.messages,
                    item.finish_reason,
                    item.usage,
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
            return

        shard_idx = await self._next_index(stream_name, "shard")
        last_error = None
        for attempt in range(1, self.config.write_retries + 1):
            try:
                self._flush_records(records, stream_name, shard_idx)
                self._save_resume_state()
                logger.info("Flushed %s shard_%05d.parquet with %s rows",
                            stream_name, shard_idx, len(records))
                return
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
            return

        state = self._load_merge_state(stream_name)
        merged_segments = set(state.get("merged_segments", []))
        pending_segments = [
            path for path in segment_paths if os.path.basename(path)
            not in merged_segments
        ]
        if not pending_segments:
            return

        shard_target_bytes = self.config.shard_target_size_mb * 1024 * 1024
        records_buffer: List[Dict[str, Any]] = []
        records_size = 0
        current_segment_names: List[str] = []

        async def flush_pending():
            nonlocal records_buffer, records_size, current_segment_names
            if not records_buffer:
                return
            await self._flush_with_retry(records_buffer, stream_name)
            merged_segments.update(current_segment_names)
            state["merged_segments"] = sorted(merged_segments)
            self._save_merge_state(stream_name, state)
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

        while True:
            try:
                if flush_interval > 0:
                    item = await asyncio.wait_for(self.result_queue.get(),
                                                  timeout=flush_interval)
                else:
                    item = await self.result_queue.get()
            except asyncio.TimeoutError:
                await self._flush_segment_buffer(buffers, buffer_sizes,
                                                 self.STREAM_ALL)
                await self._flush_segment_buffer(buffers, buffer_sizes,
                                                 self.STREAM_CORRECT)
                continue

            if item is None:
                try:
                    await self._flush_segment_buffer(buffers, buffer_sizes,
                                                     self.STREAM_ALL)
                    await self._flush_segment_buffer(buffers, buffer_sizes,
                                                     self.STREAM_CORRECT)
                    await self._merge_stream_segments(self.STREAM_ALL)
                    await self._merge_stream_segments(self.STREAM_CORRECT)
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
                if buffer_sizes[self.STREAM_ALL] >= segment_target_bytes:
                    await self._flush_segment_buffer(buffers, buffer_sizes,
                                                     self.STREAM_ALL)
                if buffer_sizes[self.STREAM_CORRECT] >= segment_target_bytes:
                    await self._flush_segment_buffer(buffers, buffer_sizes,
                                                     self.STREAM_CORRECT)
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
        if estimated_rows is not None:
            estimated_tasks = estimated_rows * self.config.rollout_count
            print(f"Estimated input rows : {estimated_rows}")
            print(f"Estimated max tasks  : {estimated_tasks}")
        print("=" * 50)

        if not input_files:
            logger.error("No files matched the range criteria. Exiting.")
            return

        self._load_completed_records()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._request_stop, sig.name)
            except NotImplementedError:
                pass

        estimated_rows = self._estimate_input_rows(input_files)
        estimated_tasks = None
        if estimated_rows is not None:
            estimated_tasks = estimated_rows * self.config.rollout_count

        pbar = tqdm(
            total=estimated_tasks,
            desc=f"Processing [{start_idx}:{end_idx}]",
            unit="task",
            dynamic_ncols=True,
        )
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

        try:
            await producer_task
            for _ in generation_workers:
                await self.task_queue.put(None)
            await self._wait_tasks(generation_workers, "generation")
            pbar.close()
            for _ in judge_workers:
                await self.judge_queue.put(None)
            await self._wait_tasks(judge_workers, "judge")
            await self.result_queue.put(None)
            await self._wait_task(writer_task, "writer")
        finally:
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.remove_signal_handler(sig)
                except NotImplementedError:
                    pass
            if not pbar.disable:
                pbar.close()

        logger.info("Pipeline Complete.")


RoundRobinPipeline = DistillPipeline
