import asyncio
import json
import os
import time
from typing import Set

from config import logger
from utils import safe_json_dumps


class FailureRecorder:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.failed_set: Set[str] = set()
        self.lock = asyncio.Lock()
        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self._load()

    @staticmethod
    def _build_key(source_file: str, source_row: int) -> str:
        return f"{source_file}::{source_row}"

    def _load(self):
        if not os.path.exists(self.log_path):
            return
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        source_file = data["source_file"]
                        source_row = int(data["source_row"])
                        self.failed_set.add(
                            self._build_key(source_file, source_row))
                    except Exception:
                        pass
            logger.info(
                "Loaded %s previously failed tasks from %s",
                len(self.failed_set),
                self.log_path,
            )
        except Exception as e:
            logger.error("Failed to load failure log: %s", e)

    async def record_failure(self, source_file: str, source_row: int,
                             reason: str):
        self.failed_set.add(self._build_key(source_file, source_row))
        entry = safe_json_dumps({
            "source_file": source_file,
            "source_row": source_row,
            "reason": str(reason),
            "time": time.time(),
        })
        async with self.lock:
            log_dir = os.path.dirname(self.log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(entry + "\n")

    def should_skip(self, source_file: str, source_row: int) -> bool:
        return self._build_key(source_file, source_row) in self.failed_set
