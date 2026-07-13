import argparse
import asyncio
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..common.judge_mode import normalize_judge_mode
from ..runtime.manifest import (DEFAULT_MANIFEST_DIRNAME,
                                DEFAULT_RULE_EXAMPLES_DIRNAME,
                                list_task_configs, load_manifest_tasks,
                                resolve_manifest_dir,
                                resolve_task_config_path,
                                select_manifest_tasks)
from ..runtime.settings import (DEFAULT_LLM_TIMEOUT,
                                DEFAULT_VLLM_LS_COMMAND, PipelineConfig,
                                logger, resolve_base_urls, split_text_items)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PIPELINE_VALUES: Dict[str, Any] = {
    "input_dir": "/mnt/hdd/lvzhihao/data/KodCode-V1-SFT-4o/data",
    "output_dir": "/mnt/hdd/lvzhihao/output/KodCode-V1-SFT-4o",
    "failure_log": "failed_tasks.jsonl",
    "file_pattern": "*.parquet",
    "range_start": 0,
    "range_end": None,
    "sample_limit": None,
    "judge_mode": "auto",
    "complete_trailing_user_turn": False,
    "model": "Qwen3-30B-A3B-Thinking-2507",
    "api_key": os.getenv("OPENAI_API_KEY", "EMPTY"),
    "api_keys": None,
    "api_key_concurrency": 0,
    "vllm_ls_command": DEFAULT_VLLM_LS_COMMAND,
    "base_urls": None,
    "ports": None,
    "concurrency": 2048,
    "judge_concurrency": 32,
    "judge_timeout_sec": 20.0,
    "active_files": 6,
    "rollout_count": 1,
    "task_schedule": "serial",
    "round_robin_chunk_size": 1000,
    "input_field": "question",
    "label_field": None,
    "llm_timeout": DEFAULT_LLM_TIMEOUT,
    "max_tokens": 7000,
    "shard_size_mb": 200,
    "segment_size_mb": 4,
    "segment_flush_interval_sec": 0.0,
    "batch_size": 1000,
    "write_retries": 3,
    "merge_every_n_writes": 0,
    "upload_every_n_merges": 1,
    "upload_merged_shards": False,
    "treat_no_judge_as_correct": False,
    "hf_repo_id": None,
    "hf_repo_type": "dataset",
    "hf_remote_prefix": None,
    "hf_token": None,
}

CONFIG_KEY_ALIASES = {
    "base_url": "base_urls",
    "input-field": "input_field",
    "label-field": "label_field",
    "file-pattern": "file_pattern",
    "range-start": "range_start",
    "range-end": "range_end",
    "sample-limit": "sample_limit",
    "judge-mode": "judge_mode",
    "complete-trailing-user-turn": "complete_trailing_user_turn",
    "api-key": "api_key",
    "api-keys": "api_keys",
    "api-key-concurrency": "api_key_concurrency",
    "judge-concurrency": "judge_concurrency",
    "judge-timeout-sec": "judge_timeout_sec",
    "active-files": "active_files",
    "rollout-count": "rollout_count",
    "task-schedule": "task_schedule",
    "round-robin-chunk-size": "round_robin_chunk_size",
    "llm-timeout": "llm_timeout",
    "max-tokens": "max_tokens",
    "shard-size-mb": "shard_size_mb",
    "segment-size-mb": "segment_size_mb",
    "segment-flush-interval-sec": "segment_flush_interval_sec",
    "batch-size": "batch_size",
    "write-retries": "write_retries",
    "merge-every-n-writes": "merge_every_n_writes",
    "upload-every-n-merges": "upload_every_n_merges",
    "upload-merged-shards": "upload_merged_shards",
    "treat-no-judge-as-correct": "treat_no_judge_as_correct",
    "hf-repo-id": "hf_repo_id",
    "hf-repo-type": "hf_repo_type",
    "hf-remote-prefix": "hf_remote_prefix",
    "hf-token": "hf_token",
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


def _resolve_optional_env_text(raw: Any) -> Any:
    if raw is None:
        return None
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if not text:
        return None
    if text.startswith("${") and text.endswith("}") and len(text) > 3:
        resolved = os.getenv(text[2:-1], "")
        return resolved or None
    if text.startswith("$") and len(text) > 1:
        resolved = os.getenv(text[1:], "")
        return resolved or None
    return raw


def _resolve_optional_env_items(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        items: List[str] = []
        for item in raw:
            resolved = _resolve_optional_env_text(item)
            if resolved is None:
                continue
            items.extend(_resolve_optional_env_items(resolved))
        return items
    resolved = _resolve_optional_env_text(raw)
    if resolved is None:
        return []
    return [item for item in split_text_items(resolved) if item]


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Asynchronous distillation pipeline for SFT-style data "
                      "generation with optional YAML task manifests."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=add_help,
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
    _add_argument(
        parser,
        "--sample-limit",
        type=int,
        help=("Maximum number of input samples to distill after file range "
              "selection. Applied before rollout expansion."),
    )
    _add_argument(
        parser,
        "--judge-mode",
        type=str,
        help=("Judge mode: auto, none, a single family "
              "(code|instruction_following|mcq|math), or a comma-separated "
              "ordered list such as 'instruction_following,mcq'."),
    )
    _add_argument(
        parser,
        "--complete-trailing-user-turn",
        action="store_true",
        help=("For multi-turn inputs ending in a user turn, generate one extra "
              "assistant reply to complete the conversation."),
    )

    _add_argument(parser, "--model", type=str)
    _add_argument(
        parser,
        "--api-key",
        type=str,
        help="API key for the OpenAI-compatible backend.",
    )
    _add_argument(
        parser,
        "--api-keys",
        nargs="+",
        help=("Multiple API keys for the OpenAI-compatible backend. Values may "
              "also be comma/newline separated."),
    )
    _add_argument(
        parser,
        "--api-key-concurrency",
        type=int,
        help=("Maximum in-flight requests per API key. 0 means unlimited except "
              "for the global concurrency limit."),
    )
    _add_argument(
        parser,
        "--vllm-ls-command",
        type=str,
        help=("Command used to verify whether a local vLLM process is still "
              "alive for a given port. Example: "
              "'/mnt/ssd/yulan/bin/vllm_ls'."),
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
    _add_argument(parser, "--judge-timeout-sec", type=float)
    _add_argument(parser, "--active-files", type=int)
    _add_argument(
        parser,
        "--rollout-count",
        type=int,
        help="Number of independent rollouts to generate for each input sample.",
    )
    _add_argument(
        parser,
        "--task-schedule",
        type=str,
        help="Task scheduling mode across manifest tasks: serial or round_robin.",
    )
    _add_argument(
        parser,
        "--round-robin-chunk-size",
        type=int,
        help="Per-task sample_limit increment for each round-robin pass.",
    )
    _add_argument(
        parser,
        "--llm-timeout",
        type=float,
        help="Per-request timeout in seconds for chat completions.",
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
    _add_argument(
        parser,
        "--merge-every-n-writes",
        type=int,
        help="Trigger a periodic segment merge every N written records. 0 disables it.",
    )
    _add_argument(
        parser,
        "--upload-every-n-merges",
        type=int,
        help="Upload newly merged correct shards every N merge events.",
    )
    _add_argument(
        parser,
        "--upload-merged-shards",
        action="store_true",
        help="Upload merged correct shards to a Hugging Face dataset repo.",
    )
    _add_argument(
        parser,
        "--treat-no-judge-as-correct",
        action="store_true",
        help=("Treat judge_type=none samples as assumed-correct so they also "
              "enter the correct stream."),
    )
    _add_argument(parser, "--hf-repo-id", type=str)
    _add_argument(parser, "--hf-repo-type", type=str)
    _add_argument(parser, "--hf-remote-prefix", type=str)
    _add_argument(parser, "--hf-token", type=str)
    return parser


def build_list_configs_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List task YAML files found under the manifest directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=add_help,
    )
    _add_argument(
        parser,
        "--manifest-dir",
        type=str,
        help=("Manifest root directory. Defaults to "
              f"'{DEFAULT_MANIFEST_DIRNAME}/'."),
    )
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
        direct_urls=_resolve_optional_env_text(values["base_urls"]),
        ports_text=_resolve_optional_env_text(values["ports"]),
    )
    normalized_judge_mode = normalize_judge_mode(values.get("judge_mode",
                                                            "auto"))

    return PipelineConfig(
        input_dir=values["input_dir"],
        output_dir=values["output_dir"],
        failure_log=resolved_failure_log,
        model_name=values["model"],
        api_key=values["api_key"],
        base_urls=base_urls,
        api_keys=_resolve_optional_env_items(values.get("api_keys")),
        api_key_concurrency=int(values.get("api_key_concurrency", 0) or 0),
        vllm_ls_command=values.get("vllm_ls_command"),
        task_name=values.get("task_name"),
        config_path=values.get("config_path"),
        manifest_dir=values.get("manifest_dir"),
        task_schedule=str(values.get("task_schedule", "serial") or "serial"),
        round_robin_chunk_size=int(values.get("round_robin_chunk_size", 1000)
                                   or 1000),
        max_concurrency=values["concurrency"],
        judge_concurrency=values["judge_concurrency"],
        judge_timeout_sec=values["judge_timeout_sec"],
        active_file_window=values["active_files"],
        rollout_count=values["rollout_count"],
        llm_timeout=values["llm_timeout"],
        llm_max_tokens=values["max_tokens"],
        file_pattern=values["file_pattern"],
        range_start=values["range_start"],
        range_end=values["range_end"],
        sample_limit=values["sample_limit"],
        judge_mode=normalized_judge_mode,
        complete_trailing_user_turn=bool(values.get(
            "complete_trailing_user_turn", False)),
        input_content_field=values["input_field"],
        label_field=values["label_field"],
        shard_target_size_mb=values["shard_size_mb"],
        segment_target_size_mb=values["segment_size_mb"],
        segment_flush_interval_sec=values["segment_flush_interval_sec"],
        batch_size=values["batch_size"],
        write_retries=values["write_retries"],
        merge_every_n_writes=int(values.get("merge_every_n_writes", 0) or 0),
        upload_every_n_merges=int(values.get("upload_every_n_merges", 1)
                                  or 1),
        upload_merged_shards=bool(values.get("upload_merged_shards", False)),
        treat_no_judge_as_correct=bool(values.get(
            "treat_no_judge_as_correct", False)),
        hf_repo_id=values.get("hf_repo_id"),
        hf_repo_type=str(values.get("hf_repo_type", "dataset") or "dataset"),
        hf_remote_prefix=values.get("hf_remote_prefix"),
        hf_token=_resolve_optional_env_text(values.get("hf_token")),
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


async def _run_pipeline_once(config: PipelineConfig, pipeline_cls):
    pipeline = pipeline_cls(config)
    summary = await pipeline.run()
    return summary or {}


def _run_round_robin_configs(configs: List[PipelineConfig], pipeline_cls):
    if not configs:
        return
    chunk_size = int(configs[0].round_robin_chunk_size or 0)
    if chunk_size <= 0:
        raise ValueError("round_robin_chunk_size must be > 0")

    task_states = [{
        "base_config": config,
        "current_limit": 0,
        "done": False,
    } for config in configs]
    round_index = 0

    while any(not state["done"] for state in task_states):
        round_index += 1
        logger.info("Starting round-robin pass %s", round_index)
        for state in task_states:
            if state["done"]:
                continue

            base_config = state["base_config"]
            total_limit = base_config.sample_limit
            next_limit = state["current_limit"] + chunk_size
            if total_limit is not None:
                next_limit = min(next_limit, int(total_limit))
                if next_limit <= state["current_limit"]:
                    state["done"] = True
                    continue

            run_config = replace(base_config, sample_limit=next_limit)
            logger.info("Round-robin task %s -> sample_limit=%s",
                        base_config.task_name or "<unnamed>", next_limit)
            summary = asyncio.run(_run_pipeline_once(run_config, pipeline_cls))
            if summary.get("interrupted"):
                logger.info(
                    "Stopping round-robin scheduling after interrupt in task %s",
                    base_config.task_name or "<unnamed>",
                )
                return
            state["current_limit"] = next_limit

            if total_limit is not None and next_limit >= int(total_limit):
                state["done"] = True
                continue
            if summary.get("input_exhausted"):
                state["done"] = True


def run_resolved_configs(configs: List[PipelineConfig], pipeline_cls=None):
    if pipeline_cls is None:
        from ..core.pipeline import DistillPipeline
        pipeline_cls = DistillPipeline

    schedule = str(configs[0].task_schedule or "serial").lower() if configs else "serial"
    if configs and schedule == "round_robin":
        _run_round_robin_configs(configs, pipeline_cls)
        return

    for index, config in enumerate(configs, 1):
        logger.info("Starting task %s/%s", index, len(configs))
        if config.config_path:
            logger.info("Loaded task config: %s", config.config_path)
        if config.task_name:
            logger.info("Task name: %s", config.task_name)
        logger.info("Failure log path: %s", config.failure_log)
        logger.info("Resolved %s backend(s): %s", len(config.base_urls),
                    ", ".join(config.base_urls))
        summary = asyncio.run(_run_pipeline_once(config, pipeline_cls))
        if summary.get("interrupted"):
            logger.info("Stopping remaining task scheduling after interrupt.")
            return


def run_list_configs_namespace(args: argparse.Namespace):
    manifest_dir = resolve_manifest_dir(
        getattr(args, "manifest_dir", None),
        project_root=PROJECT_ROOT,
    )
    _print_available_configs(manifest_dir)


def run_namespace(args: argparse.Namespace):
    configs = build_configs(args)
    logger.info("Resolved %s task(s) for this run", len(configs))
    try:
        run_resolved_configs(configs)
    except KeyboardInterrupt:
        print("\nStopped by user.")


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if getattr(args, "list_configs", False):
        run_list_configs_namespace(args)
        return
    run_namespace(args)


if __name__ == "__main__":
    main()
