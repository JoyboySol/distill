import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.avg_correct_tokens import summarize_average_tokens


class AvgCorrectTokensTests(unittest.TestCase):

    def _write_rows(self, root: str, stream: str, rows):
        shard_dir = Path(root) / stream / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows),
                       shard_dir / "shard_00000.parquet")

    def test_correct_stream_averages_all_token_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_rows(
                tmpdir,
                "correct",
                [
                    {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30,
                        "is_correct": True,
                    },
                    {
                        "prompt_tokens": 20,
                        "completion_tokens": 40,
                        "total_tokens": 60,
                        "is_correct": True,
                    },
                ],
            )

            summary = summarize_average_tokens(tmpdir)

            self.assertEqual(summary["record_count"], 2)
            self.assertEqual(summary["counted_records"], 2)
            self.assertEqual(summary["average_tokens"]["prompt_tokens"], 15.0)
            self.assertEqual(summary["average_tokens"]["completion_tokens"],
                             30.0)
            self.assertEqual(summary["average_tokens"]["total_tokens"], 45.0)

    def test_all_stream_can_filter_by_correct_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_rows(
                tmpdir,
                "all",
                [
                    {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30,
                        "is_correct": True,
                    },
                    {
                        "prompt_tokens": 100,
                        "completion_tokens": 200,
                        "total_tokens": 300,
                        "is_correct": False,
                    },
                ],
            )

            summary = summarize_average_tokens(tmpdir,
                                               stream="all",
                                               only_correct_flag=True)

            self.assertEqual(summary["record_count"], 2)
            self.assertEqual(summary["counted_records"], 1)
            self.assertEqual(summary["average_tokens"]["total_tokens"], 30.0)


if __name__ == "__main__":
    unittest.main()
