import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from scripts.merge_correct_segments import merge_correct_segments


class MergeCorrectSegmentsTests(unittest.TestCase):

    def _write_segment(self, root: Path, index: int, rows) -> None:
        segment_dir = root / "correct" / "segments"
        segment_dir.mkdir(parents=True, exist_ok=True)
        target = segment_dir / f"segment_{index:06d}.jsonl"
        with target.open("w", encoding="utf-8") as f:
            for row in rows:
                import json
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def test_merge_builds_shards_and_token_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_segment(
                root,
                0,
                [{
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                    "is_correct": True,
                }],
            )
            self._write_segment(
                root,
                1,
                [{
                    "prompt_tokens": 20,
                    "completion_tokens": 40,
                    "total_tokens": 60,
                    "is_correct": True,
                }],
            )

            summary = merge_correct_segments(str(root), shard_size_mb=1)

            self.assertEqual(summary["merged_segments_this_run"], 2)
            self.assertEqual(summary["total_records"], 2)
            self.assertEqual(summary["token_sums"]["total_tokens"], 90)
            self.assertEqual(summary["average_tokens"]["total_tokens"], 45.0)

            shard_files = sorted((root / "correct" / "shards").glob("shard_*.parquet"))
            self.assertEqual(len(shard_files), 1)
            table = pq.read_table(shard_files[0])
            self.assertEqual(table.num_rows, 2)

    def test_rerun_is_incremental(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_segment(
                root,
                0,
                [{
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                    "is_correct": True,
                }],
            )

            first = merge_correct_segments(str(root), shard_size_mb=1)
            second = merge_correct_segments(str(root), shard_size_mb=1)

            self.assertEqual(first["merged_segments_this_run"], 1)
            self.assertEqual(second["merged_segments_this_run"], 0)
            self.assertEqual(second["shards_written_this_run"], 0)
            self.assertEqual(second["total_records"], 1)


if __name__ == "__main__":
    unittest.main()
