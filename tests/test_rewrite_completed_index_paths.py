import json
import tempfile
import unittest
from pathlib import Path

from scripts.rewrite_completed_index_paths import rewrite_completed_index_paths


class RewriteCompletedIndexPathsTests(unittest.TestCase):

    def test_dry_run_reports_matching_lines_without_writing_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "completed_index.jsonl"
            input_path.write_text(
                "\n".join([
                    "/old_root/a.parquet\t1\t0",
                    "/other_root/b.parquet\t2\t1",
                    "",
                ]),
                encoding="utf-8",
            )

            summary = rewrite_completed_index_paths(
                input_path=input_path,
                old_prefix="/old_root",
                new_prefix="/new_root",
                dry_run=True,
            )

            self.assertEqual(summary["total_lines"], 2)
            self.assertEqual(summary["matched_lines"], 1)
            self.assertEqual(summary["rewritten_lines"], 1)
            self.assertFalse(summary["wrote_output"])
            self.assertFalse((root / "completed_index.rewritten.jsonl").exists())
            self.assertEqual(
                input_path.read_text(encoding="utf-8"),
                "/old_root/a.parquet\t1\t0\n/other_root/b.parquet\t2\t1\n",
            )

    def test_rewrites_tsv_lines_to_output_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "completed_index.jsonl"
            output_path = root / "rewritten.jsonl"
            input_path.write_text(
                "\n".join([
                    "/mnt/source/data/a.parquet\t7\t0",
                    "/mnt/source/data/nested/b.parquet\t8\t3",
                    "/mnt/other/c.parquet\t9\t0",
                    "",
                ]),
                encoding="utf-8",
            )

            summary = rewrite_completed_index_paths(
                input_path=input_path,
                old_prefix="/mnt/source/data",
                new_prefix="/data/shared",
                output_path=output_path,
            )

            self.assertEqual(summary["matched_lines"], 2)
            self.assertEqual(summary["rewritten_lines"], 2)
            self.assertTrue(summary["wrote_output"])
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "/data/shared/a.parquet\t7\t0\n"
                "/data/shared/nested/b.parquet\t8\t3\n"
                "/mnt/other/c.parquet\t9\t0\n",
            )
            self.assertEqual(
                input_path.read_text(encoding="utf-8"),
                "/mnt/source/data/a.parquet\t7\t0\n"
                "/mnt/source/data/nested/b.parquet\t8\t3\n"
                "/mnt/other/c.parquet\t9\t0\n",
            )

    def test_in_place_rewrite_updates_legacy_json_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "completed_index.jsonl"
            input_path.write_text(
                "\n".join([
                    json.dumps({
                        "source_file": "/old_root/a.parquet",
                        "source_row": 4,
                        "rollout_index": 0,
                    }, ensure_ascii=False),
                    json.dumps({
                        "source_file": "/other_root/b.parquet",
                        "source_row": 5,
                        "rollout_index": 1,
                    }, ensure_ascii=False),
                    "",
                ]),
                encoding="utf-8",
            )

            summary = rewrite_completed_index_paths(
                input_path=input_path,
                old_prefix="/old_root",
                new_prefix="/new_root",
                in_place=True,
            )

            self.assertEqual(summary["matched_lines"], 1)
            self.assertEqual(summary["rewritten_lines"], 1)
            self.assertTrue(summary["wrote_output"])
            lines = input_path.read_text(encoding="utf-8").splitlines()
            first = json.loads(lines[0])
            second = json.loads(lines[1])
            self.assertEqual(first["source_file"], "/new_root/a.parquet")
            self.assertEqual(second["source_file"], "/other_root/b.parquet")


if __name__ == "__main__":
    unittest.main()
