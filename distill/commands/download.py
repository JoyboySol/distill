import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..runtime.manifest import (DEFAULT_MANIFEST_DIRNAME, load_manifest_tasks,
                                resolve_manifest_dir,
                                resolve_task_config_path,
                                select_manifest_tasks)


def _resolve_optional_env_text(raw: Any) -> Any:
    if raw is None:
        return None
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if not text:
        return None
    if text.startswith("${") and text.endswith("}") and len(text) > 3:
        return os.getenv(text[2:-1], "")
    if text.startswith("$") and len(text) > 1:
        return os.getenv(text[1:], "")
    return raw

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DOWNLOAD_VALUES: Dict[str, Any] = {
    "repo_id": None,
    "local_dir": None,
    "output_dir": None,
    "include": None,
    "exclude": None,
    "hf_username": None,
    "hf_token": None,
    "tool": None,
    "threads": None,
    "jobs": None,
    "dataset": False,
    "revision": None,
    "hfd_command": "hfd",
}

CONFIG_KEY_ALIASES = {
    "repo-id": "repo_id",
    "local-dir": "local_dir",
    "hf-username": "hf_username",
    "hf-token": "hf_token",
    "hfd-command": "hfd_command",
}


def _normalize_config_keys(mapping: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in mapping.items():
        normalized_key = CONFIG_KEY_ALIASES.get(key, key.replace("-", "_"))
        normalized[normalized_key] = value
    return normalized


def _normalize_patterns(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if isinstance(raw, (list, tuple)):
        patterns = []
        for item in raw:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                patterns.append(text)
        return patterns
    text = str(raw).strip()
    return [text] if text else []


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Download a Hugging Face model or dataset by wrapping the "
                     "local hfd utility. Supports direct arguments or "
                     "manifest-driven task bundles."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=add_help,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--config",
                             type=str,
                             help="Path to a manifest/task YAML file.")
    input_group.add_argument(
        "--config-name",
        type=str,
        help=("Task config name under the manifest directory. Supports bare "
              "name, *.yaml, or relative path."))
    input_group.add_argument("--repo-id", type=str, help="Hugging Face repo ID.")
    parser.add_argument("--task",
                        type=str,
                        help="Optional task_name filter when using a manifest.")
    parser.add_argument("--manifest-dir",
                        type=str,
                        help=("Manifest root directory. Defaults to "
                              f"'{DEFAULT_MANIFEST_DIRNAME}/'."))
    parser.add_argument("--local-dir",
                        type=str,
                        help=("Target download directory. Preferred over "
                              "output_dir in manifests."))
    parser.add_argument("--include",
                        nargs="+",
                        default=None,
                        help="Optional include patterns.")
    parser.add_argument("--exclude",
                        nargs="+",
                        default=None,
                        help="Optional exclude patterns.")
    parser.add_argument("--hf-username", type=str)
    parser.add_argument("--hf-token", type=str)
    parser.add_argument("--tool",
                        choices=["aria2c", "wget"],
                        default=None,
                        help="Download backend passed to hfd.")
    parser.add_argument("-x",
                        "--threads",
                        type=int,
                        default=None,
                        help="Per-file download threads for aria2c.")
    parser.add_argument("-j",
                        "--jobs",
                        type=int,
                        default=None,
                        help="Concurrent downloads for aria2c.")
    parser.add_argument("--dataset",
                        action="store_true",
                        help="Download as a dataset repo.")
    parser.add_argument("--revision",
                        type=str,
                        default=None,
                        help="Model/dataset revision.")
    parser.add_argument("--hfd-command",
                        type=str,
                        default=None,
                        help="Executable path for the hfd wrapper.")
    return parser


def _manifest_task_values(args: argparse.Namespace) -> List[Dict[str, Any]]:
    cli_values = vars(args).copy()
    manifest_dir = resolve_manifest_dir(cli_values.get("manifest_dir"),
                                        project_root=PROJECT_ROOT)
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
        effective = dict(DEFAULT_DOWNLOAD_VALUES)
        effective.update(task_values)
        for key, value in cli_values.items():
            if key in {"config", "config_name", "manifest_dir", "task"}:
                continue
            if value is None:
                continue
            if key == "dataset" and value is False:
                continue
            effective[key] = value

        effective["manifest_dir"] = str(manifest_dir)
        effective["config_path"] = str(config_path) if config_path else None
        effective_values.append(effective)
    return effective_values


def _build_hfd_command(values: Dict[str, Any]) -> List[str]:
    repo_id = str(values.get("repo_id") or "").strip()
    if not repo_id:
        raise ValueError("repo_id is required for download tasks")

    command = [str(values.get("hfd_command") or "hfd"), repo_id]
    include_patterns = _normalize_patterns(values.get("include"))
    exclude_patterns = _normalize_patterns(values.get("exclude"))
    if include_patterns:
        command.extend(["--include", *include_patterns])
    if exclude_patterns:
        command.extend(["--exclude", *exclude_patterns])

    hf_username = _resolve_optional_env_text(values.get("hf_username"))
    hf_token = _resolve_optional_env_text(values.get("hf_token"))
    if hf_username:
        command.extend(["--hf_username", str(hf_username)])
    if hf_token:
        command.extend(["--hf_token", str(hf_token)])

    tool = values.get("tool")
    if tool:
        command.extend(["--tool", str(tool)])

    threads = values.get("threads")
    if threads is not None:
        command.extend(["-x", str(int(threads))])

    jobs = values.get("jobs")
    if jobs is not None:
        command.extend(["-j", str(int(jobs))])

    if bool(values.get("dataset")):
        command.append("--dataset")

    local_dir = values.get("local_dir") or values.get("output_dir")
    if local_dir:
        command.extend(["--local-dir", str(local_dir)])

    revision = values.get("revision")
    if revision:
        command.extend(["--revision", str(revision)])
    return command


def run_download_task(task_values: Dict[str, Any]) -> Dict[str, Any]:
    command = _build_hfd_command(task_values)
    subprocess.run(command, check=True)
    return {
        "task_name": str(task_values.get("task_name") or ""),
        "repo_id": str(task_values.get("repo_id") or ""),
        "local_dir": str(task_values.get("local_dir") or
                         task_values.get("output_dir") or ""),
        "dataset": bool(task_values.get("dataset")),
        "revision": str(task_values.get("revision") or ""),
        "command": command,
    }


def download_from_manifest(config_path: str,
                           task_name: Optional[str] = None,
                           manifest_dir: Optional[str] = None,
                           cli_overrides: Optional[Dict[str, Any]] = None
                           ) -> List[Dict[str, Any]]:
    args = argparse.Namespace(
        config=config_path,
        config_name=None,
        task=task_name,
        manifest_dir=manifest_dir,
        repo_id=None,
        local_dir=None,
        include=None,
        exclude=None,
        hf_username=None,
        hf_token=None,
        tool=None,
        threads=None,
        jobs=None,
        dataset=False,
        revision=None,
        hfd_command=None,
    )
    if cli_overrides:
        for key, value in cli_overrides.items():
            setattr(args, key, value)
    return [run_download_task(values) for values in _manifest_task_values(args)]


def run_namespace(args: argparse.Namespace) -> Dict[str, Any] | List[Dict[str, Any]]:
    if args.config or args.config_name:
        summaries = [run_download_task(values) for values in _manifest_task_values(args)]
        print(json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True))
        return summaries

    direct_values = dict(DEFAULT_DOWNLOAD_VALUES)
    direct_values.update({
        "repo_id": args.repo_id,
        "local_dir": args.local_dir,
        "include": args.include,
        "exclude": args.exclude,
        "hf_username": args.hf_username,
        "hf_token": args.hf_token,
        "tool": args.tool,
        "threads": args.threads,
        "jobs": args.jobs,
        "dataset": args.dataset,
        "revision": args.revision,
        "hfd_command": args.hfd_command,
        "task_name": args.task,
    })
    summary = run_download_task(direct_values)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    run_namespace(args)


if __name__ == "__main__":
    main()
