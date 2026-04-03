import logging
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PipelineConfig:
    input_dir: str
    output_dir: str
    failure_log: str
    model_name: str
    api_key: Optional[str]
    base_urls: List[str]

    file_pattern: str = "*.parquet"
    range_start: int = 0
    range_end: Optional[int] = None

    input_content_field: str = "question"
    label_field: Optional[str] = None
    output_content_field: str = "messages"

    max_concurrency: int = 64
    judge_concurrency: int = 16
    queue_max_size: int = 2000
    active_file_window: int = 6
    llm_timeout: float = 3600

    write_retries: int = 3
    shard_target_size_mb: int = 200
    batch_size: int = 1000
    segment_target_size_mb: int = 4


def setup_logger(log_file: str):
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    return logging.getLogger(__name__)


logger = setup_logger("pipeline_runner.log")
