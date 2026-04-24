import csv
import tempfile
import unittest
from pathlib import Path

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

    def test_summarize_manifest_token_stats_outputs_expected_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_a = root / "task_a_out"
            output_b = root / "task_b_out"

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

            manifest_path = root / "batch.yaml"
            manifest_path.write_text(
                "\n".join([
                    "tasks:",
                    "  - task_name: task_a",
                    f"    output_dir: {output_a}",
                    "  - task_name: task_b",
                    f"    output_dir: {output_b}",
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
                        "field": "prompt_tokens",
                        "total_records": 2,
                        "token_count_total": 2,
                        "token_sum_total": 24,
                        "avg_total_tokens": 30.0,
                    },
                    {
                        "task_name": "task_a",
                        "field": "completion_tokens",
                        "total_records": 2,
                        "token_count_total": 2,
                        "token_sum_total": 36,
                        "avg_total_tokens": 30.0,
                    },
                    {
                        "task_name": "task_a",
                        "field": "total_tokens",
                        "total_records": 2,
                        "token_count_total": 2,
                        "token_sum_total": 60,
                        "avg_total_tokens": 30.0,
                    },
                    {
                        "task_name": "task_b",
                        "field": "prompt_tokens",
                        "total_records": 1,
                        "token_count_total": 1,
                        "token_sum_total": 7,
                        "avg_total_tokens": 12.0,
                    },
                    {
                        "task_name": "task_b",
                        "field": "completion_tokens",
                        "total_records": 1,
                        "token_count_total": 1,
                        "token_sum_total": 5,
                        "avg_total_tokens": 12.0,
                    },
                    {
                        "task_name": "task_b",
                        "field": "total_tokens",
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
                        "field": "total_tokens",
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
            self.assertEqual(row, ["task_a", "total_tokens", "2", "2", "60", "30.0"])


if __name__ == "__main__":
    unittest.main()
