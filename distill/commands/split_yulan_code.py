import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from distill.core.judges.code import code_judge_type_hint, extract_code_text_last_block


def _load_json_object(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _canonical_language(raw: Any) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    text = raw.strip().lower().replace("_", "-")
    if not text:
        return None
    mapping = {
        "py": "python",
        "python": "python",
        "python2": "python",
        "python3": "python",
        "pypy": "python",
        "pypy2": "python",
        "pypy3": "python",
        "cpp": "cpp",
        "c++": "cpp",
        "g++": "cpp",
        "gnu-c++": "cpp",
        "gnu-c++17": "cpp",
        "cxx": "cpp",
        "c": "c",
        "java": "java",
        "javascript": "javascript",
        "js": "javascript",
        "node": "javascript",
        "typescript": "typescript",
        "ts": "typescript",
        "go": "go",
        "golang": "go",
        "rust": "rust",
        "rs": "rust",
        "c#": "csharp",
        "csharp": "csharp",
        "cs": "csharp",
        "kotlin": "kotlin",
        "kt": "kotlin",
        "swift": "swift",
        "ruby": "ruby",
        "rb": "ruby",
        "php": "php",
    }
    if text in mapping:
        return mapping[text]
    for needle, language in (
        ("python", "python"),
        ("pypy", "python"),
        ("c++", "cpp"),
        ("cpp", "cpp"),
        ("java", "java"),
        ("javascript", "javascript"),
        ("typescript", "typescript"),
        ("golang", "go"),
        ("rust", "rust"),
        ("kotlin", "kotlin"),
    ):
        if needle in text:
            return language
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-") or "unknown"


def _normalize_code_for_match(code: Any) -> str:
    if not isinstance(code, str):
        return ""
    return code.replace("\r\n", "\n").replace("\r", "\n").strip()


def _language_from_metadata_match(response: str,
                                  metadata: Dict[str, Any]) -> Optional[str]:
    normalized_response = _normalize_code_for_match(response)
    if not normalized_response:
        return None

    for key in ("language", "lang", "programming_language"):
        language = _canonical_language(metadata.get(key))
        if language:
            return language

    solutions = metadata.get("all_accepted_solutions")
    if not isinstance(solutions, list):
        return None

    for solution in solutions:
        if not isinstance(solution, dict):
            continue
        if _normalize_code_for_match(solution.get("code")) == normalized_response:
            language = _canonical_language(solution.get("language"))
            if language:
                return language
    return None


def infer_response_language(row: Dict[str, Any]) -> str:
    response = row.get("response") or ""
    metadata = _load_json_object(row.get("metadata"))
    metadata_language = _language_from_metadata_match(str(response), metadata)
    if metadata_language:
        return metadata_language

    raw = str(response)
    code = extract_code_text_last_block(raw)
    text = (code or raw).strip()
    lower = text.lower()
    raw_lower = raw.lower()

    fence_match = re.search(r"```([A-Za-z0-9_+#.-]+)", raw)
    if fence_match:
        language = _canonical_language(fence_match.group(1))
        if language:
            return language

    if any(marker in lower for marker in (
        "#include <",
        "using namespace std",
        "std::",
        "int main(",
        "int32_t main(",
        "long long",
        "ios::sync_with_stdio",
    )):
        return "cpp"
    if any(marker in lower for marker in (
        "import java.",
        "public class ",
        "public static void main",
        "system.out.",
        "bufferedreader",
        "stringtokenizer",
    )):
        return "java"
    if any(marker in lower for marker in (
        "package main",
        "func main()",
        "fmt.",
        "bufio.",
    )):
        return "go"
    if any(marker in lower for marker in (
        "fn main()",
        "use std::",
        "println!",
    )):
        return "rust"
    if any(marker in lower for marker in (
        "console.log",
        "function ",
        "const ",
        "let ",
        "require(",
    )):
        return "javascript"
    if any(marker in lower for marker in (
        "def ",
        "import ",
        "from ",
        "input()",
        "print(",
        "sys.stdin",
        "class solution",
    )) or "```python" in raw_lower:
        return "python"
    return "unknown"


def _judge_hint(row: Dict[str, Any]) -> Optional[str]:
    try:
        return code_judge_type_hint(
            row,
            [{
                "role": "assistant",
                "content": str(row.get("response") or ""),
            }],
        )
    except Exception:
        return None


def classify_row(row: Dict[str, Any]) -> Tuple[str, str, str]:
    judge_source = _judge_hint(row)
    judge_bucket = "judge" if judge_source and judge_source != "code_unverified" else "no_judge"
    language = infer_response_language(row)
    return judge_bucket, language, judge_source or ""


def _safe_bucket_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip().lower())
    return value.strip("-") or "unknown"


class BucketWriter:

    def __init__(self, output_dir: Path, shard_rows: int,
                 merge_judge: bool, merge_no_judge: bool):
        self.output_dir = output_dir
        self.shard_rows = shard_rows
        self.merge_judge = merge_judge
        self.merge_no_judge = merge_no_judge
        self.buffers: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        self.shard_indices: Counter = Counter()
        self.written_rows: Counter = Counter()
        self.parquet_writers: Dict[Tuple[str, str], pq.ParquetWriter] = {}

    def add(self, bucket: str, language: str, row: Dict[str, Any]) -> None:
        safe_bucket = _safe_bucket_name(bucket)
        safe_language = _safe_bucket_name(language)
        should_merge = ((safe_bucket == "judge" and self.merge_judge) or
                        (safe_bucket == "no_judge" and self.merge_no_judge))
        key = (safe_bucket, "" if should_merge else safe_language)
        buffer = self.buffers[key]
        buffer.append(row)
        if len(buffer) >= self.shard_rows:
            self.flush_key(key)

    def _target_path(self, key: Tuple[str, str]) -> Path:
        bucket, language = key
        if language == "":
            target_dir = self.output_dir / bucket
            target_dir.mkdir(parents=True, exist_ok=True)
            return target_dir / "code.parquet"
        target_dir = self.output_dir / bucket / language
        target_dir.mkdir(parents=True, exist_ok=True)
        shard_idx = self.shard_indices[key]
        return target_dir / f"code-{shard_idx:05d}.parquet"

    def flush_key(self, key: Tuple[str, str]) -> None:
        rows = self.buffers.get(key) or []
        if not rows:
            return
        target_path = self._target_path(key)
        table = pa.Table.from_pylist(rows)
        if key[1] == "":
            writer = self.parquet_writers.get(key)
            if writer is None:
                writer = pq.ParquetWriter(target_path, table.schema)
                self.parquet_writers[key] = writer
            writer.write_table(table)
        else:
            pq.write_table(table, target_path)
            self.shard_indices[key] += 1
        self.written_rows[key] += len(rows)
        self.buffers[key] = []

    def close(self) -> None:
        for key in list(self.buffers):
            self.flush_key(key)
        for writer in self.parquet_writers.values():
            writer.close()
        self.parquet_writers.clear()


def split_yulan_code(input_path: Path,
                     output_dir: Path,
                     batch_size: int,
                     shard_rows: int,
                     overwrite: bool,
                     limit: Optional[int],
                     progress_every: int,
                     merge_judge: bool = True,
                     merge_no_judge: bool = True) -> Dict[str, Any]:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = pq.ParquetFile(input_path)
    writer = BucketWriter(output_dir,
                          shard_rows=shard_rows,
                          merge_judge=merge_judge,
                          merge_no_judge=merge_no_judge)
    bucket_counts: Counter = Counter()
    language_counts: Counter = Counter()
    joint_counts: Counter = Counter()
    judge_source_counts: Counter = Counter()

    total_rows = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            if limit is not None and total_rows >= limit:
                writer.close()
                return _write_summary(input_path, output_dir, total_rows,
                                      bucket_counts, language_counts,
                                      joint_counts, judge_source_counts,
                                      merge_judge,
                                      merge_no_judge)

            row = dict(row)
            bucket, language, judge_source = classify_row(row)
            row["distill_has_judge"] = bucket == "judge"
            row["distill_judge_source"] = judge_source
            row["distill_response_language"] = language
            writer.add(bucket, language, row)

            bucket_counts[bucket] += 1
            language_counts[language] += 1
            joint_counts[f"{bucket}/{language}"] += 1
            judge_source_counts[judge_source or "none"] += 1
            total_rows += 1

            if progress_every > 0 and total_rows % progress_every == 0:
                print(
                    f"processed_rows={total_rows} judge={bucket_counts['judge']} no_judge={bucket_counts['no_judge']}",
                    flush=True,
                )

    writer.close()
    return _write_summary(input_path, output_dir, total_rows, bucket_counts,
                          language_counts, joint_counts, judge_source_counts,
                          merge_judge,
                          merge_no_judge)


def _write_summary(input_path: Path, output_dir: Path, total_rows: int,
                   bucket_counts: Counter, language_counts: Counter,
                   joint_counts: Counter,
                   judge_source_counts: Counter,
                   merge_judge: bool,
                   merge_no_judge: bool) -> Dict[str, Any]:
    summary = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "rows": total_rows,
        "bucket_counts": dict(bucket_counts),
        "language_counts": dict(language_counts),
        "joint_counts": dict(joint_counts),
        "judge_source_counts": dict(judge_source_counts),
        "merge_judge": merge_judge,
        "merge_no_judge": merge_no_judge,
    }
    summary_path = output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary,
                                       ensure_ascii=False,
                                       indent=2,
                                       sort_keys=True),
                            encoding="utf-8")
    return summary


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Split YuLan-Code parquet rows by judge availability and "
                     "response language."),
        add_help=add_help,
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--shard-rows", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--split-judge-by-language",
        action="store_true",
        help="Keep the older layout judge/<language>/code-*.parquet.",
    )
    parser.add_argument(
        "--split-no-judge-by-language",
        action="store_true",
        help="Keep the older layout no_judge/<language>/code-*.parquet.",
    )
    return parser


def run_namespace(args: argparse.Namespace) -> Dict[str, Any]:
    summary = split_yulan_code(
        input_path=args.input_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        shard_rows=args.shard_rows,
        overwrite=args.overwrite,
        limit=args.limit,
        progress_every=args.progress_every,
        merge_judge=not args.split_judge_by_language,
        merge_no_judge=not args.split_no_judge_by_language,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    run_namespace(args)


if __name__ == "__main__":
    main()
