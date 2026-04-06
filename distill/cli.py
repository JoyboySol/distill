import argparse
import asyncio
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from .runtime.manifest import (DEFAULT_MANIFEST_DIRNAME,
                                   DEFAULT_RULE_EXAMPLES_DIRNAME,
                                   list_task_configs, load_manifest_tasks,
                                   resolve_manifest_dir,
                                   resolve_task_config_path,
                                   select_manifest_tasks)
    from .runtime.settings import PipelineConfig, logger, resolve_base_urls
except ImportError:
    from runtime.manifest import (DEFAULT_MANIFEST_DIRNAME,
                                  DEFAULT_RULE_EXAMPLES_DIRNAME,
                                  list_task_configs, load_manifest_tasks,
                                  resolve_manifest_dir,
                                  resolve_task_config_path,
                                  select_manifest_tasks)
    from runtime.settings import PipelineConfig, logger, resolve_base_urls

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PIPELINE_VALUES: Dict[str, Any] = {
    "input_dir": "/mnt/hdd/lvzhihao/data/KodCode-V1-SFT-4o/data",
    "output_dir": "/mnt/hdd/lvzhihao/output/KodCode-V1-SFT-4o",
    "failure_log": "failed_tasks.jsonl",
    "file_pattern": "*.parquet",
    "range_start": 0,
    "range_end": None,
    "model": "Qwen3-30B-A3B-Thinking-2507",
    "api_key": os.getenv("OPENAI_API_KEY", "EMPTY"),
    "base_urls": None,
    "ports": None,
    "concurrency": 2048,
    "judge_concurrency": 32,
    "active_files": 6,
    "rollout_count": 1,
    "input_field": "question",
    "label_field": None,
    "max_tokens": 7000,
    "shard_size_mb": 200,
    "segment_size_mb": 4,
    "segment_flush_interval_sec": 0.0,
    "batch_size": 1000,
    "write_retries": 3,
}

CONFIG_KEY_ALIASES = {
    "base_url": "base_urls",
    "input-field": "input_field",
    "label-field": "label_field",
    "file-pattern": "file_pattern",
    "range-start": "range_start",
    "range-end": "range_end",
    "api-key": "api_key",
    "judge-concurrency": "judge_concurrency",
    "active-files": "active_files",
    "rollout-count": "rollout_count",
    "max-tokens": "max_tokens",
    "shard-size-mb": "shard_size_mb",
    "segment-size-mb": "segment_size_mb",
    "segment-flush-interval-sec": "segment_flush_interval_sec",
    "batch-size": "batch_size",
    "write-retries": "write_retries",
}


def _normalize_config_keys(mapping: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in mapping.items():
        normalized_key = CONFIG_KEY_ALIASES.get(key, key.replace("-", "_"))
        normalized[normalized_key] = value
    return normalized


def _add_argument(parser: argparse.ArgumentParser, *names: str, **kwargs):
    kwargs.setdefault("default", argparse.SUPPRESS)
    parser.add_argument(*names, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Asynchronous distillation pipeline for SFT-style data "
                     "generation with optional YAML task manifests."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_argument(
        parser,
        "--config",
        type=str,
        help="Path to a single YAML task config file.",
    )
    _add_argument(
        parser,
        "--config-name",
        type=str,
        help=("Task config name under the manifest directory. Supports bare "
              "name, *.yaml, or relative path."),
    )
    _add_argument(
        parser,
        "--task",
        type=str,
        help=("When the loaded YAML contains multiple tasks, only run the task "
              "with this task_name."),
    )
    _add_argument(
        parser,
        "--manifest-dir",
        type=str,
        help=("Manifest root directory. Defaults to "
              f"'{DEFAULT_MANIFEST_DIRNAME}/'."),
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help=("List task YAML files found under the manifest directory and "
              "exit."),
    )
    _add_argument(
        parser,
        "--input-dir",
        type=str,
        help="Directory containing input files.",
    )
    _add_argument(
        parser,
        "--output-dir",
        type=str,
        help="Directory where all/correct outputs will be written.",
    )
    _add_argument(
        parser,
        "--failure-log",
        type=str,
        help=("Failure log filename or absolute path. Relative paths are stored "
              "under <project>/failures/ and suffixed by file range."),
    )

    _add_argument(
        parser,
        "--file-pattern",
        type=str,
        help="Glob pattern, e.g., '*.parquet' or '*.jsonl'",
    )
    _add_argument(parser,
                  "--range-start",
                  type=int,
                  help="Start index of file list (inclusive)")
    _add_argument(parser,
                  "--range-end",
                  type=int,
                  help="End index of file list (exclusive)")

    _add_argument(parser, "--model", type=str)
    _add_argument(
        parser,
        "--api-key",
        type=str,
        help="API key for the OpenAI-compatible backend.",
    )
    _add_argument(
        parser,
        "--base-url",
        "--base-urls",
        dest="base_urls",
        nargs="+",
        help=("Explicit backend base URLs, comma/newline separated. "
              "Example: 'http://host1:8000/v1,http://host2:8000/v1'."),
    )
    _add_argument(
        parser,
        "--ports",
        nargs="+",
        help=("Backward-compatible shorthand for local ports. "
              "Example: '6758,6759,6761-6765'."),
    )
    _add_argument(parser, "--concurrency", type=int)
    _add_argument(parser, "--judge-concurrency", type=int)
    _add_argument(parser, "--active-files", type=int)
    _add_argument(
        parser,
        "--rollout-count",
        type=int,
        help="Number of independent rollouts to generate for each input sample.",
    )
    _add_argument(
        parser,
        "--max-tokens",
        type=int,
        help="Maximum number of tokens to generate for each assistant response.",
    )
    _add_argument(parser, "--input-field", type=str)
    _add_argument(
        parser,
        "--label-field",
        type=str,
        help="Reference answer field for judge, e.g. 'label'.",
    )
    _add_argument(parser, "--shard-size-mb", type=int)
    _add_argument(parser, "--segment-size-mb", type=int)
    _add_argument(
        parser,
        "--segment-flush-interval-sec",
        type=float,
        help=("Flush in-memory segment buffers every N seconds even if size "
              "threshold is not reached. 0 disables time-based flushing."),
    )
    _add_argument(parser, "--batch-size", type=int)
    _add_argument(parser, "--write-retries", type=int)
    return parser


def _manifest_task_values(args: argparse.Namespace) -> List[Dict[str, Any]]:
    cli_values = vars(args).copy()
    manifest_dir = resolve_manifest_dir(
        cli_values.get("manifest_dir"),
        project_root=PROJECT_ROOT,
    )

    config_path = resolve_task_config_path(
        config_path=cli_values.get("config"),
        manifest_dir=manifest_dir,
        config_name=cli_values.get("config_name"),
    )
    task_values_list = load_manifest_tasks(config_path) if config_path else [{}]
    task_values_list = [_normalize_config_keys(task) for task in task_values_list]
    task_values_list = select_manifest_tasks(task_values_list, cli_values.get("task"))

    effective_values: List[Dict[str, Any]] = []
    for task_values in task_values_list:
        effective = dict(DEFAULT_PIPELINE_VALUES)
        effective.update(task_values)
        for key, value in cli_values.items():
            if key in {
                    "config",
                    "config_name",
                    "manifest_dir",
                    "list_configs",
                    "task",
            }:
                continue
            effective[key] = value

        effective["manifest_dir"] = str(manifest_dir)
        effective["config_path"] = str(config_path) if config_path else None
        if config_path and not effective.get("task_name"):
            effective["task_name"] = config_path.stem
        effective_values.append(effective)
    return effective_values


def _build_config_from_values(values: Dict[str, Any]) -> PipelineConfig:
    range_suffix = (
        f"_{values['range_start']}_"
        f"{values['range_end'] if values['range_end'] is not None else 'END'}"
    )
    failure_log_path = Path(values["failure_log"])
    log_name_stem = failure_log_path.stem
    log_name_ext = failure_log_path.suffix
    unique_failure_log = f"{log_name_stem}{range_suffix}{log_name_ext}"
    if failure_log_path.is_absolute():
        resolved_failure_log = str(failure_log_path.with_name(unique_failure_log))
    else:
        resolved_failure_log = str(PROJECT_ROOT / "failures" / unique_failure_log)
    base_urls = resolve_base_urls(
        direct_urls=values["base_urls"],
        ports_text=values["ports"],
    )

    return PipelineConfig(
        input_dir=values["input_dir"],
        output_dir=values["output_dir"],
        failure_log=resolved_failure_log,
        model_name=values["model"],
        api_key=values["api_key"],
        base_urls=base_urls,
        task_name=values.get("task_name"),
        config_path=values.get("config_path"),
        manifest_dir=values.get("manifest_dir"),
        max_concurrency=values["concurrency"],
        judge_concurrency=values["judge_concurrency"],
        active_file_window=values["active_files"],
        rollout_count=values["rollout_count"],
        llm_max_tokens=values["max_tokens"],
        file_pattern=values["file_pattern"],
        range_start=values["range_start"],
        range_end=values["range_end"],
        input_content_field=values["input_field"],
        label_field=values["label_field"],
        shard_target_size_mb=values["shard_size_mb"],
        segment_target_size_mb=values["segment_size_mb"],
        segment_flush_interval_sec=values["segment_flush_interval_sec"],
        batch_size=values["batch_size"],
        write_retries=values["write_retries"],
    )


def build_configs(args: argparse.Namespace) -> List[PipelineConfig]:
    return [_build_config_from_values(values) for values in _manifest_task_values(args)]


def build_config(args: argparse.Namespace) -> PipelineConfig:
    configs = build_configs(args)
    if len(configs) != 1:
        raise ValueError(
            f"Expected exactly one task config, but resolved {len(configs)} tasks. "
            "Use build_configs() or pass --task to select one task.")
    return configs[0]


def _print_available_configs(manifest_dir: Path):
    print(f"Manifest dir: {manifest_dir}")
    print(f"Rule examples: {manifest_dir / DEFAULT_RULE_EXAMPLES_DIRNAME}")
    config_paths = list_task_configs(manifest_dir)
    if not config_paths:
        print("No YAML task configs found.")
        return
    for config_path in config_paths:
        rel_path = config_path.relative_to(manifest_dir)
        try:
            tasks = load_manifest_tasks(config_path)
        except Exception:
            print(rel_path)
            continue
        if len(tasks) <= 1:
            print(rel_path)
            continue
        task_names = ", ".join(str(task.get("task_name")) for task in tasks)
        print(f"{rel_path}  [tasks: {task_names}]")


def main():
    parser = build_parser()
    args = parser.parse_args()
    manifest_dir = resolve_manifest_dir(
        getattr(args, "manifest_dir", None),
        project_root=PROJECT_ROOT,
    )
    if getattr(args, "list_configs", False):
        _print_available_configs(manifest_dir)
        return
    configs = build_configs(args)
    try:
        from .core.pipeline import DistillPipeline
    except ImportError:
        from core.pipeline import DistillPipeline
    logger.info("Resolved %s task(s) for this run", len(configs))
    try:
        for index, config in enumerate(configs, 1):
            logger.info("Starting task %s/%s", index, len(configs))
            if config.config_path:
                logger.info("Loaded task config: %s", config.config_path)
            if config.task_name:
                logger.info("Task name: %s", config.task_name)
            logger.info("Failure log path: %s", config.failure_log)
            logger.info("Resolved %s backend(s): %s", len(config.base_urls),
                        ", ".join(config.base_urls))
            pipeline = DistillPipeline(config)
            asyncio.run(pipeline.run())
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
