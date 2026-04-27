import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _normalize_prefix(prefix: str) -> str:
    normalized = str(prefix).strip()
    if not normalized:
        raise ValueError("prefix must not be empty")
    if normalized != "/":
        normalized = normalized.rstrip("/")
    return normalized


def _rewrite_source_path(source_path: str, old_prefix: str,
                         new_prefix: str) -> tuple[str, bool]:
    if source_path == old_prefix:
        return new_prefix, True
    if source_path.startswith(old_prefix + "/"):
        suffix = source_path[len(old_prefix):]
        return new_prefix + suffix, True
    return source_path, False


def _rewrite_tsv_line(line: str, old_prefix: str,
                      new_prefix: str) -> tuple[str, bool]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 3:
        raise ValueError(f"Invalid completed index TSV line: {line[:200]!r}")
    source_file, source_row, rollout_index = parts
    rewritten_source, changed = _rewrite_source_path(source_file, old_prefix,
                                                     new_prefix)
    return f"{rewritten_source}\t{source_row}\t{rollout_index}\n", changed


def _rewrite_json_line(line: str, old_prefix: str,
                       new_prefix: str) -> tuple[str, bool]:
    payload = json.loads(line)
    if not isinstance(payload, dict) or "source_file" not in payload:
        raise ValueError(f"Invalid completed index JSON line: {line[:200]!r}")
    rewritten_source, changed = _rewrite_source_path(str(payload["source_file"]),
                                                     old_prefix, new_prefix)
    payload["source_file"] = rewritten_source
    return json.dumps(payload, ensure_ascii=False) + "\n", changed


def rewrite_completed_index_paths(
    input_path: Path,
    old_prefix: str,
    new_prefix: str,
    output_path: Optional[Path] = None,
    dry_run: bool = False,
    in_place: bool = False,
) -> Dict[str, Any]:
    if dry_run and in_place:
        raise ValueError("dry_run and in_place cannot both be true")
    if output_path is not None and in_place:
        raise ValueError("output_path and in_place cannot both be set")

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"completed index not found: {input_path}")

    old_prefix = _normalize_prefix(old_prefix)
    new_prefix = _normalize_prefix(new_prefix)

    total_lines = 0
    matched_lines = 0
    rewritten_lines = 0
    rewritten_content: list[str] = []

    with input_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            if not raw_line.strip():
                continue
            total_lines += 1
            stripped = raw_line.lstrip()
            if stripped.startswith("{"):
                rewritten_line, changed = _rewrite_json_line(
                    raw_line, old_prefix, new_prefix)
            else:
                rewritten_line, changed = _rewrite_tsv_line(
                    raw_line, old_prefix, new_prefix)
            if changed:
                matched_lines += 1
                rewritten_lines += 1
            rewritten_content.append(rewritten_line)

    target_path = output_path
    if in_place:
        target_path = input_path
    elif output_path is None and not dry_run:
        target_path = input_path.with_name(
            input_path.stem + ".rewritten" + input_path.suffix)

    wrote_output = False
    if not dry_run and target_path is not None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            for line in rewritten_content:
                f.write(line)
        temp_path.replace(target_path)
        wrote_output = True

    return {
        "input_path": str(input_path),
        "output_path": str(target_path) if target_path is not None else None,
        "old_prefix": old_prefix,
        "new_prefix": new_prefix,
        "total_lines": total_lines,
        "matched_lines": matched_lines,
        "rewritten_lines": rewritten_lines,
        "wrote_output": wrote_output,
        "dry_run": dry_run,
        "in_place": in_place,
    }


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rewrite source_file prefixes inside completed_index.jsonl.",
        add_help=add_help,
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--old-prefix", required=True)
    parser.add_argument("--new-prefix", required=True)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--in-place", action="store_true")
    return parser


def run_namespace(args: argparse.Namespace) -> Dict[str, Any]:
    summary = rewrite_completed_index_paths(
        input_path=args.input_path,
        old_prefix=args.old_prefix,
        new_prefix=args.new_prefix,
        output_path=args.output_path,
        dry_run=args.dry_run,
        in_place=args.in_place,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    run_namespace(args)


if __name__ == "__main__":
    main()
