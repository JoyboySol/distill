import argparse
import sys
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from .commands import (avg_correct_tokens, download, eval_existing_outputs,
                       hydrate_opencode_reasoning_split1,
                       manifest_token_stats, merge_correct_segments,
                       prepare_opencode_reasoning,
                       rebuild_correct_segments, rejudge_saved_outputs,
                       rewrite_completed_index_paths, run, stats)

DEFAULT_PIPELINE_VALUES = run.DEFAULT_PIPELINE_VALUES
_build_config_from_values = run._build_config_from_values
_normalize_config_keys = run._normalize_config_keys
build_config = run.build_config
build_configs = run.build_configs
build_parser = run.build_parser
run_resolved_configs = run.run_resolved_configs


@dataclass(frozen=True)
class CommandSpec:
    name: str
    help_text: str
    build_parser: Callable[..., argparse.ArgumentParser]
    handler: Callable[[argparse.Namespace], object]


COMMAND_SPECS = (
    CommandSpec(
        name="run",
        help_text="Run the asynchronous distillation pipeline.",
        build_parser=run.build_parser,
        handler=run.run_namespace,
    ),
    CommandSpec(
        name="list-configs",
        help_text="List manifest task config files.",
        build_parser=run.build_list_configs_parser,
        handler=run.run_list_configs_namespace,
    ),
    CommandSpec(
        name="stats",
        help_text="Summarize saved distill output shards.",
        build_parser=stats.build_parser,
        handler=stats.run_namespace,
    ),
    CommandSpec(
        name="download",
        help_text="Download Hugging Face repos with the local hfd tool.",
        build_parser=download.build_parser,
        handler=download.run_namespace,
    ),
    CommandSpec(
        name="avg-correct-tokens",
        help_text="Compute average token counts for saved outputs.",
        build_parser=avg_correct_tokens.build_parser,
        handler=avg_correct_tokens.run_namespace,
    ),
    CommandSpec(
        name="manifest-token-stats",
        help_text="Summarize per-task token stats from a manifest.",
        build_parser=manifest_token_stats.build_parser,
        handler=manifest_token_stats.run_namespace,
    ),
    CommandSpec(
        name="merge-correct-segments",
        help_text="Merge segment JSONL files into parquet shards.",
        build_parser=merge_correct_segments.build_parser,
        handler=merge_correct_segments.run_namespace,
    ),
    CommandSpec(
        name="rebuild-correct-segments",
        help_text="Rebuild the correct segment stream from all segments.",
        build_parser=rebuild_correct_segments.build_parser,
        handler=rebuild_correct_segments.run_namespace,
    ),
    CommandSpec(
        name="rejudge-saved-outputs",
        help_text="Rejudge saved segment outputs in place.",
        build_parser=rejudge_saved_outputs.build_parser,
        handler=rejudge_saved_outputs.run_namespace,
    ),
    CommandSpec(
        name="eval-existing-outputs",
        help_text="Evaluate existing parquet outputs with current judges.",
        build_parser=eval_existing_outputs.build_parser,
        handler=eval_existing_outputs.run_namespace,
    ),
    CommandSpec(
        name="prepare-opencode-reasoning",
        help_text="Materialize prompt-example tests for OpenCodeReasoning.",
        build_parser=prepare_opencode_reasoning.build_parser,
        handler=prepare_opencode_reasoning.run_namespace,
    ),
    CommandSpec(
        name="hydrate-opencode-reasoning-split1",
        help_text="Hydrate OpenCodeReasoning split_1 prompts/tests.",
        build_parser=hydrate_opencode_reasoning_split1.build_parser,
        handler=hydrate_opencode_reasoning_split1.run_namespace,
    ),
    CommandSpec(
        name="rewrite-completed-index-paths",
        help_text="Rewrite source path prefixes in completed indexes.",
        build_parser=rewrite_completed_index_paths.build_parser,
        handler=rewrite_completed_index_paths.run_namespace,
    ),
)

COMMAND_NAMES = {spec.name for spec in COMMAND_SPECS}


def _register_subcommand(subparsers, spec: CommandSpec):
    parent_parser = spec.build_parser(add_help=False)
    description = getattr(parent_parser, "description", None) or spec.help_text
    parser = subparsers.add_parser(
        spec.name,
        parents=[parent_parser],
        help=spec.help_text,
        description=description,
    )
    parser.set_defaults(command=spec.name, command_handler=spec.handler)
    return parser


def build_root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Unified CLI for the distillation pipeline and common "
                     "maintenance tools."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True
    for spec in COMMAND_SPECS:
        _register_subcommand(subparsers, spec)
    return parser


def _run_command_from_namespace(args: argparse.Namespace):
    handler = getattr(args, "command_handler", None)
    if handler is None:
        raise ValueError(f"No command handler registered for {args!r}")
    return handler(args)


def _dispatch_legacy_run(argv: Sequence[str]):
    return run.main(list(argv))


def main(argv: Optional[Sequence[str]] = None):
    tokens = list(argv if argv is not None else sys.argv[1:])
    if not tokens or tokens[0] in {"-h", "--help"}:
        build_root_parser().print_help()
        return 0
    if tokens[0] in COMMAND_NAMES:
        args = build_root_parser().parse_args(tokens)
        return _run_command_from_namespace(args)
    return _dispatch_legacy_run(tokens)


__all__ = [
    "DEFAULT_PIPELINE_VALUES",
    "_build_config_from_values",
    "_normalize_config_keys",
    "build_config",
    "build_configs",
    "build_parser",
    "build_root_parser",
    "main",
    "run_resolved_configs",
]
