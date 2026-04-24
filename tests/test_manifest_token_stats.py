import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.manifest_token_stats import (CSV_COLUMNS,
                                          summarize_manifest_token_stats,
                                          write_summary_csv)


def _write_shard(output_dir: Path, stream: str, shard_index: int,
                 rows: list[dict]) -> None:
    shard_dir = output_dir / stream / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, shard_dir / f"shard_{shard_index:05d}.parquet")


class ManifestTokenStatsTests(unittest.TestCase):

    def test_summarize_manifest_token_stats_outputs_expected_rows_grouped_by_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_a = root / "task_a_out"
            output_b = root / "task_b_out"
            output_c = root / "task_c_out"

            _write_shard(
                output_a,
                "correct",
                0,
                [
                    {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30,
                    },
                    {
                        "prompt_tokens": 14,
                        "completion_tokens": 16,
                        "total_tokens": 30,
                    },
                ],
            )
            _write_shard(
                output_b,
                "correct",
                0,
                [
                    {
                        "prompt_tokens": 7,
                        "completion_tokens": 5,
                        "total_tokens": 12,
                    },
                ],
            )
            _write_shard(
                output_c,
                "correct",
                0,
                [
                    {
                        "prompt_tokens": 11,
                        "completion_tokens": 9,
                        "total_tokens": 20,
                    },
                ],
            )

            manifest_path = root / "batch.yaml"
            manifest_path.write_text(
                "\n".join([
                    "tasks:",
                    "  - task_name: task_a",
                    "    field: math",
                    f"    output_dir: {output_a}",
                    "  - task_name: task_b",
                    "    field: code",
                    f"    output_dir: {output_b}",
                    "  - task_name: task_c",
                    "    field: math",
                    f"    output_dir: {output_c}",
                    "",
                ]),
                encoding="utf-8",
            )

            rows = summarize_manifest_token_stats(str(manifest_path))

            self.assertEqual(
                rows,
                [
                    {
                        "task_name": "task_a",
                        "field": "math",
                        "total_records": 2,
                        "token_count_total": 2,
                        "token_sum_total": 60,
                        "avg_total_tokens": 30.0,
                    },
                    {
                        "task_name": "task_c",
                        "field": "math",
                        "total_records": 1,
                        "token_count_total": 1,
                        "token_sum_total": 20,
                        "avg_total_tokens": 20.0,
                    },
                    {
                        "task_name": "task_b",
                        "field": "code",
                        "total_records": 1,
                        "token_count_total": 1,
                        "token_sum_total": 12,
                        "avg_total_tokens": 12.0,
                    },
                ],
            )

    def test_write_summary_csv_uses_strict_column_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "summary.csv"
            write_summary_csv(
                [
                    {
                        "task_name": "task_a",
                        "field": "math",
                        "total_records": 2,
                        "token_count_total": 2,
                        "token_sum_total": 60,
                        "avg_total_tokens": 30.0,
                    }
                ],
                csv_path,
            )

            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                row = next(reader)

            self.assertEqual(header, list(CSV_COLUMNS))
            self.assertEqual(row, ["task_a", "math", "2", "2", "60", "30.0"])

    def test_missing_field_defaults_to_empty_string(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "task_out"
            _write_shard(
                output_dir,
                "correct",
                0,
                [
                    {
                        "total_tokens": 42,
                    },
                ],
            )

            manifest_path = root / "single.yaml"
            manifest_path.write_text(
                "\n".join([
                    "task_name: task_a",
                    f"output_dir: {output_dir}",
                    "",
                ]),
                encoding="utf-8",
            )

            rows = summarize_manifest_token_stats(str(manifest_path))

            self.assertEqual(
                rows,
                [
                    {
                        "task_name": "task_a",
                        "field": "",
                        "total_records": 1,
                        "token_count_total": 1,
                        "token_sum_total": 42,
                        "avg_total_tokens": 42.0,
                    }
                ],
            )

    def test_parallel_collection_preserves_field_grouping_and_task_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "batch.yaml"
            manifest_path.write_text(
                "\n".join([
                    "tasks:",
                    "  - task_name: task_a",
                    "    field: math",
                    "    output_dir: /tmp/a",
                    "  - task_name: task_b",
                    "    field: code",
                    "    output_dir: /tmp/b",
                    "  - task_name: task_c",
                    "    field: math",
                    "    output_dir: /tmp/c",
                    "",
                ]),
                encoding="utf-8",
            )

            fake_rows = {
                "/tmp/a": {
                    "task_name": "task_a",
                    "field": "math",
                    "total_records": 2,
                    "token_count_total": 2,
                    "token_sum_total": 60,
                    "avg_total_tokens": 30.0,
                },
                "/tmp/b": {
                    "task_name": "task_b",
                    "field": "code",
                    "total_records": 1,
                    "token_count_total": 1,
                    "token_sum_total": 12,
                    "avg_total_tokens": 12.0,
                },
                "/tmp/c": {
                    "task_name": "task_c",
                    "field": "math",
                    "total_records": 1,
                    "token_count_total": 1,
                    "token_sum_total": 20,
                    "avg_total_tokens": 20.0,
                },
            }

            def fake_build_task_row(task_name: str, field: str, output_dir: str,
                                    stream: str = "correct"):
                return dict(fake_rows[output_dir])

            with patch("scripts.manifest_token_stats._build_task_row",
                       side_effect=fake_build_task_row):
                rows = summarize_manifest_token_stats(
                    str(manifest_path),
                    max_workers=3,
                    show_progress=False,
                )

            self.assertEqual(
                rows,
                [
                    fake_rows["/tmp/a"],
                    fake_rows["/tmp/c"],
                    fake_rows["/tmp/b"],
                ],
            )


if __name__ == "__main__":
    unittest.main()
