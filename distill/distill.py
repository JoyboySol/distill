import argparse
import asyncio
import os
from pathlib import Path
from typing import List

from config import PipelineConfig, logger
from pipeline import RoundRobinPipeline


DEFAULT_URL_TEMPLATE = "http://localhost:{port}/v1"
DEFAULT_PORT_RANGE = "6758-6765"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _split_url_text(raw: str) -> List[str]:
    return [
        item.strip() for item in raw.replace(",", "\n").splitlines()
        if item.strip()
    ]


def _parse_port_range(port_range: str) -> List[int]:
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


def _parse_ports_text(ports_text: str) -> List[int]:
    ports: List[int] = []
    for item in _split_url_text(ports_text):
        ports.extend(_parse_port_range(item))
    return ports


def resolve_base_urls(args: argparse.Namespace) -> List[str]:
    ports: List[int] = []
    if args.ports:
        ports.extend(_parse_ports_text(args.ports))

    if not ports:
        env_ports = os.getenv("DISTILL_PORTS")
        if env_ports:
            ports.extend(_parse_ports_text(env_ports))

    if not ports:
        ports.extend(_parse_port_range(DEFAULT_PORT_RANGE))

    urls = [DEFAULT_URL_TEMPLATE.format(port=port) for port in ports]
    deduped_urls: List[str] = []
    seen = set()
    for url in urls:
        if url not in seen:
            deduped_urls.append(url)
            seen.add(url)

    if not deduped_urls:
        raise ValueError("No base URLs resolved for the LLM client.")

    return deduped_urls


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/mnt/hdd/lvzhihao/data/KodCode-V1-SFT-4o/data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/hdd/lvzhihao/output/KodCode-V1-SFT-4o",
    )
    parser.add_argument("--failure-log", type=str, default="failed_tasks.jsonl")

    parser.add_argument(
        "--file-pattern",
        type=str,
        default="*.parquet",
        help="Glob pattern, e.g., '*.parquet' or '*.jsonl'",
    )
    parser.add_argument("--range-start",
                        type=int,
                        default=0,
                        help="Start index of file list (inclusive)")
    parser.add_argument("--range-end",
                        type=int,
                        default=None,
                        help="End index of file list (exclusive)")

    parser.add_argument("--model",
                        type=str,
                        default="Qwen3-30B-A3B-Thinking-2507")
    parser.add_argument(
        "--ports",
        type=str,
        default=None,
        help="Local port list, e.g. '6758,6759,6761-6765'.",
    )
    parser.add_argument("--concurrency", type=int, default=2048)
    parser.add_argument("--judge-concurrency", type=int, default=32)
    parser.add_argument("--active-files", type=int, default=6)
    parser.add_argument("--input-field", type=str, default="question")
    parser.add_argument(
        "--label-field",
        type=str,
        default=None,
        help="Reference answer field for judge, e.g. 'label'.",
    )
    parser.add_argument("--shard-size-mb", type=int, default=200)
    parser.add_argument("--segment-size-mb", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--write-retries", type=int, default=3)
    return parser


def build_config(args: argparse.Namespace) -> PipelineConfig:
    range_suffix = f"_{args.range_start}_{args.range_end if args.range_end is not None else 'END'}"
    failure_log_path = Path(args.failure_log)
    log_name_stem = failure_log_path.stem
    log_name_ext = failure_log_path.suffix
    unique_failure_log = f"{log_name_stem}{range_suffix}{log_name_ext}"
    if failure_log_path.is_absolute():
        resolved_failure_log = str(failure_log_path.with_name(unique_failure_log))
    else:
        resolved_failure_log = str(PROJECT_ROOT / "failures" / unique_failure_log)
    base_urls = resolve_base_urls(args)

    return PipelineConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        failure_log=resolved_failure_log,
        model_name=args.model,
        api_key="EMPTY",
        base_urls=base_urls,
        max_concurrency=args.concurrency,
        judge_concurrency=args.judge_concurrency,
        active_file_window=args.active_files,
        file_pattern=args.file_pattern,
        range_start=args.range_start,
        range_end=args.range_end,
        input_content_field=args.input_field,
        label_field=args.label_field,
        shard_target_size_mb=args.shard_size_mb,
        segment_target_size_mb=args.segment_size_mb,
        batch_size=args.batch_size,
        write_retries=args.write_retries,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = build_config(args)
    logger.info("Failure log path: %s", config.failure_log)
    pipeline = RoundRobinPipeline(config)
    try:
        asyncio.run(pipeline.run())
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
