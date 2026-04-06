import logging
import os
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional


DEFAULT_URL_TEMPLATE = "http://localhost:{port}/v1"
DEFAULT_PORT_RANGE = "6758-6765"


def split_text_items(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        items: List[str] = []
        for item in raw:
            items.extend(split_text_items(item))
        return items
    text = str(raw)
    return [item.strip() for item in text.replace(",", "\n").splitlines() if item.strip()]


def parse_port_range(port_range: str) -> List[int]:
    text = port_range.strip()
    if not text:
        return []

    if "-" in text:
        start_text, end_text = text.split("-", 1)
        start = int(start_text)
        end = int(end_text)
        if end < start:
            raise ValueError(
                f"Invalid port range '{port_range}': end must be >= start.")
        return list(range(start, end + 1))

    return [int(text)]


def parse_ports_text(ports_text: Any) -> List[int]:
    ports: List[int] = []
    for item in split_text_items(ports_text):
        ports.extend(parse_port_range(item))
    return ports


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        deduped.append(item)
        seen.add(item)
    return deduped


def resolve_base_urls(
    direct_urls: Optional[Any] = None,
    ports_text: Optional[Any] = None,
    url_template: str = DEFAULT_URL_TEMPLATE,
    default_port_range: str = DEFAULT_PORT_RANGE,
) -> List[str]:
    urls: List[str] = []

    if direct_urls:
        urls.extend(split_text_items(direct_urls))

    env_urls = os.getenv("DISTILL_BASE_URLS")
    if env_urls:
        urls.extend(split_text_items(env_urls))

    ports: List[int] = []
    if ports_text:
        ports.extend(parse_ports_text(ports_text))

    env_ports = os.getenv("DISTILL_PORTS")
    if env_ports:
        ports.extend(parse_ports_text(env_ports))

    if not urls and not ports:
        ports.extend(parse_port_range(default_port_range))

    urls.extend(url_template.format(port=port) for port in ports)
    deduped_urls = dedupe_preserve_order(urls)
    if not deduped_urls:
        raise ValueError("No base URLs resolved for the LLM client.")
    return deduped_urls


@dataclass
class PipelineConfig:
    input_dir: str
    output_dir: str
    failure_log: str
    model_name: str
    api_key: Optional[str]
    base_urls: List[str]
    task_name: Optional[str] = None
    config_path: Optional[str] = None
    manifest_dir: Optional[str] = None

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
    rollout_count: int = 1
    llm_timeout: float = 3600
    llm_max_tokens: int = 7000
    segment_flush_interval_sec: float = 0.0

    write_retries: int = 3
    shard_target_size_mb: int = 200
    batch_size: int = 1000
    segment_target_size_mb: int = 4

    @property
    def primary_base_url(self) -> Optional[str]:
        return self.base_urls[0] if self.base_urls else None


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
