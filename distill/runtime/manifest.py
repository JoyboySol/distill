from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


DEFAULT_MANIFEST_DIRNAME = "manifest"
DEFAULT_RULE_EXAMPLES_DIRNAME = "rule_examples"
YAML_SUFFIXES = (".yaml", ".yml")


def resolve_manifest_dir(raw_manifest_dir: Optional[str],
                         project_root: Path) -> Path:
    if raw_manifest_dir:
        return Path(raw_manifest_dir).expanduser().resolve()
    return (project_root / DEFAULT_MANIFEST_DIRNAME).resolve()


def load_task_file(task_path: Path) -> Dict[str, Any]:
    with open(task_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Task config must be a mapping: {task_path}")
    return payload


def expand_manifest_payload(payload: Dict[str, Any],
                            config_path: Path) -> List[Dict[str, Any]]:
    tasks = payload.get("tasks")
    if tasks is None:
        task = dict(payload)
        if not task.get("task_name"):
            task["task_name"] = config_path.stem
        return [task]

    if not isinstance(tasks, list):
        raise ValueError(f"'tasks' must be a list in manifest: {config_path}")

    defaults = payload.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError(f"'defaults' must be a mapping in manifest: {config_path}")

    top_level_common = {
        key: value
        for key, value in payload.items() if key not in {"tasks", "defaults"}
    }
    common = dict(defaults)
    common.update(top_level_common)

    expanded_tasks: List[Dict[str, Any]] = []
    for idx, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(
                f"Each task must be a mapping in manifest {config_path}: index={idx}"
            )
        merged = dict(common)
        merged.update(task)
        if not merged.get("task_name"):
            merged["task_name"] = f"{config_path.stem}_{idx:03d}"
        expanded_tasks.append(merged)
    return expanded_tasks


def load_manifest_tasks(task_path: Path) -> List[Dict[str, Any]]:
    return expand_manifest_payload(load_task_file(task_path), task_path)


def list_task_configs(manifest_dir: Path) -> List[Path]:
    if not manifest_dir.exists():
        return []
    paths = [
        path for path in manifest_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in YAML_SUFFIXES
    ]
    return sorted(paths)


def select_manifest_tasks(tasks: List[Dict[str, Any]],
                          task_name: Optional[str] = None) -> List[Dict[str, Any]]:
    if not task_name:
        return tasks

    selected = [task for task in tasks if task.get("task_name") == task_name]
    if selected:
        return selected

    raise FileNotFoundError(
        f"Unable to find task '{task_name}' in manifest bundle. "
        f"Available tasks: {[task.get('task_name') for task in tasks]}")


def resolve_task_config_path(
    config_path: Optional[str],
    manifest_dir: Path,
    config_name: Optional[str] = None,
) -> Optional[Path]:
    if config_path:
        path = Path(config_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Task config not found: {path}")
        return path

    if not config_name:
        return None

    candidates = []
    raw = Path(config_name)
    if raw.suffix.lower() in YAML_SUFFIXES:
        candidates.extend([
            manifest_dir / raw,
            manifest_dir / DEFAULT_RULE_EXAMPLES_DIRNAME / raw,
        ])
    else:
        candidates.extend([
            manifest_dir / f"{config_name}.yaml",
            manifest_dir / f"{config_name}.yml",
            manifest_dir / DEFAULT_RULE_EXAMPLES_DIRNAME /
            f"{config_name}.yaml",
            manifest_dir / DEFAULT_RULE_EXAMPLES_DIRNAME /
            f"{config_name}.yml",
            manifest_dir / raw,
            manifest_dir / DEFAULT_RULE_EXAMPLES_DIRNAME / raw,
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Unable to resolve config '{config_name}' under {manifest_dir}")
