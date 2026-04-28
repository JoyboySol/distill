import json
import csv
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.merge_correct_segments import merge_correct_segments, \
    merge_correct_segments_from_manifest, write_summary_csv


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

    def test_merge_state_is_backward_compatible_with_pipeline_only_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_segment(
                root,
                0,
                [{
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                    "is_correct": True,
                }],
            )
            state_path = root / "correct" / "merge_state.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps({"merged_segments": []}, ensure_ascii=False),
                encoding="utf-8",
            )

            summary = merge_correct_segments(str(root), shard_size_mb=1)

            self.assertEqual(summary["merged_segments_this_run"], 1)
            self.assertEqual(summary["pending_segment_count_before_run"], 1)
            self.assertEqual(summary["pending_segment_count"], 0)
            self.assertEqual(summary["total_records"], 1)
            normalized_state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertIn("stats", normalized_state)
            self.assertIn("next_shard_idx", normalized_state)

    def test_rebuilds_all_segments_when_legacy_state_marks_merged_but_no_shards_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_segment(
                root,
                0,
                [{
                    "prompt_tokens": 11,
                    "completion_tokens": 12,
                    "total_tokens": 23,
                    "is_correct": True,
                }],
            )
            state_path = root / "correct" / "merge_state.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps({"merged_segments": ["segment_000000.jsonl"]},
                           ensure_ascii=False),
                encoding="utf-8",
            )

            summary = merge_correct_segments(str(root), shard_size_mb=1)

            self.assertEqual(summary["merged_segments_this_run"], 1)
            self.assertEqual(summary["shards_written_this_run"], 1)
            self.assertEqual(summary["pending_segment_count_before_run"], 1)
            self.assertEqual(summary["pending_segment_count"], 0)
            self.assertEqual(summary["total_records"], 1)
            shard_files = sorted((root / "correct" / "shards").glob("shard_*.parquet"))
            self.assertEqual(len(shard_files), 1)

    def test_backfills_stats_from_existing_shards_when_legacy_state_lacks_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_dir = root / "correct" / "shards"
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_path = shard_dir / "shard_00000.parquet"
            table = pa.Table.from_pylist([{
                "prompt_tokens": 7,
                "completion_tokens": 8,
                "total_tokens": 15,
                "is_correct": True,
            }])
            pq.write_table(table, shard_path)

            segment_dir = root / "correct" / "segments"
            segment_dir.mkdir(parents=True, exist_ok=True)
            state_path = root / "correct" / "merge_state.json"
            state_path.write_text(
                json.dumps({"merged_segments": []}, ensure_ascii=False),
                encoding="utf-8",
            )

            summary = merge_correct_segments(str(root), shard_size_mb=1)

            self.assertEqual(summary["next_shard_idx"], 1)
            self.assertEqual(summary["total_records"], 1)
            self.assertEqual(summary["average_tokens"]["total_tokens"], 15.0)

    def test_optional_hf_upload_reuses_pipeline_state_paths_and_remote_dir_rule(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "merged_run"
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

            uploads = []

            class _FakeHfApi:

                def __init__(self, token=None):
                    self.token = token

                def create_repo(self, repo_id, repo_type="dataset", exist_ok=False):
                    uploads.append(("create_repo", repo_id, repo_type, exist_ok))

                def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type):
                    uploads.append((
                        "upload_file",
                        str(path_or_fileobj),
                        path_in_repo,
                        repo_id,
                        repo_type,
                    ))

            fake_module = types.SimpleNamespace(HfApi=_FakeHfApi)
            with patch.dict(sys.modules, {"huggingface_hub": fake_module}):
                first = merge_correct_segments(
                    str(root),
                    shard_size_mb=1,
                    upload_to_hf=True,
                    hf_repo_id="JoyboyGo/hf_data",
                    hf_token="token",
                    task_name="task_a",
                )
                second = merge_correct_segments(
                    str(root),
                    shard_size_mb=1,
                    upload_to_hf=True,
                    hf_repo_id="JoyboyGo/hf_data",
                    hf_token="token",
                    task_name="task_a",
                )

            self.assertEqual(first["hf_upload"]["uploaded"], 1)
            self.assertEqual(second["hf_upload"]["uploaded"], 0)

            shard_path = root / "correct" / "shards" / "shard_00000.parquet"
            self.assertIn(
                (
                    "upload_file",
                    str(shard_path),
                    "task_a/shard_00000.parquet",
                    "JoyboyGo/hf_data",
                    "dataset",
                ),
                uploads,
            )

            upload_state_path = root / ".hf_upload_state.json"
            state = json.loads(upload_state_path.read_text(encoding="utf-8"))
            self.assertEqual(
                state["uploaded_paths"],
                ["correct/shards/shard_00000.parquet"],
            )

            merge_state_path = root / "correct" / "merge_state.json"
            merge_state = json.loads(merge_state_path.read_text(encoding="utf-8"))
            self.assertEqual(merge_state["merged_segments"], ["segment_000000.jsonl"])

    def test_force_reupload_ignores_existing_hf_upload_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "merged_run"
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
            merge_correct_segments(str(root), shard_size_mb=1)

            upload_state_path = root / ".hf_upload_state.json"
            upload_state_path.write_text(
                json.dumps({
                    "uploaded_paths": ["correct/shards/shard_00000.parquet"]
                }, ensure_ascii=False),
                encoding="utf-8",
            )

            uploads = []

            class _FakeHfApi:

                def __init__(self, token=None):
                    self.token = token

                def create_repo(self, repo_id, repo_type="dataset", exist_ok=False):
                    uploads.append(("create_repo", repo_id, repo_type, exist_ok))

                def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type):
                    uploads.append((
                        "upload_file",
                        str(path_or_fileobj),
                        path_in_repo,
                        repo_id,
                        repo_type,
                    ))

            fake_module = types.SimpleNamespace(HfApi=_FakeHfApi)
            with patch.dict(sys.modules, {"huggingface_hub": fake_module}):
                summary = merge_correct_segments(
                    str(root),
                    shard_size_mb=1,
                    upload_to_hf=True,
                    hf_repo_id="JoyboyGo/hf_data",
                    hf_token="token",
                    task_name="task_a",
                    force_reupload=True,
                )

            self.assertEqual(summary["hf_upload"]["uploaded"], 1)
            self.assertEqual(summary["hf_upload"]["skipped"], 0)
            self.assertIn(
                (
                    "upload_file",
                    str(root / "correct" / "shards" / "shard_00000.parquet"),
                    "task_a/shard_00000.parquet",
                    "JoyboyGo/hf_data",
                    "dataset",
                ),
                uploads,
            )

    def test_force_merge_rebuilds_shards_and_resets_state_from_segments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "merged_run"
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

            first = merge_correct_segments(str(root), shard_size_mb=1)
            self.assertEqual(first["shards_written_this_run"], 1)

            shard_dir = root / "correct" / "shards"
            extra_shard = shard_dir / "shard_99999.parquet"
            extra_shard.write_bytes(b"stale")

            self._write_segment(
                root,
                1,
                [{
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                    "is_correct": True,
                }],
            )

            summary = merge_correct_segments(
                str(root),
                shard_size_mb=1,
                force_merge=True,
            )

            self.assertEqual(summary["merged_segments_this_run"], 2)
            self.assertEqual(summary["pending_segment_count_before_run"], 2)
            self.assertEqual(summary["pending_segment_count"], 0)
            self.assertEqual(summary["next_shard_idx"], 1)
            self.assertEqual(summary["total_records"], 2)
            shard_files = sorted(shard_dir.glob("shard_*.parquet"))
            self.assertEqual([path.name for path in shard_files], ["shard_00000.parquet"])
            table = pq.read_table(shard_files[0])
            self.assertEqual(table.num_rows, 2)

    def test_manifest_batch_merge_uses_task_output_dirs_and_shared_hf_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            root_a = tmp_root / "task_a_out"
            root_b = tmp_root / "task_b_out"
            self._write_segment(
                root_a,
                0,
                [{
                    "prompt_tokens": 4,
                    "completion_tokens": 5,
                    "total_tokens": 9,
                    "is_correct": True,
                }],
            )
            self._write_segment(
                root_b,
                0,
                [{
                    "prompt_tokens": 6,
                    "completion_tokens": 7,
                    "total_tokens": 13,
                    "is_correct": True,
                }],
            )

            manifest_path = tmp_root / "tasks.yaml"
            manifest_path.write_text(
                "\n".join([
                    "shard_size_mb: 1",
                    "upload_merged_shards: true",
                    "hf_repo_id: JoyboyGo/hf_data",
                    "hf_token: token",
                    "tasks:",
                    "  - task_name: task_a",
                    f"    output_dir: {root_a}",
                    "  - task_name: task_b",
                    f"    output_dir: {root_b}",
                ]),
                encoding="utf-8",
            )

            uploads = []

            class _FakeHfApi:

                def __init__(self, token=None):
                    self.token = token

                def create_repo(self, repo_id, repo_type="dataset", exist_ok=False):
                    uploads.append(("create_repo", repo_id, repo_type, exist_ok))

                def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type):
                    uploads.append((
                        "upload_file",
                        str(path_or_fileobj),
                        path_in_repo,
                        repo_id,
                        repo_type,
                    ))

            fake_module = types.SimpleNamespace(HfApi=_FakeHfApi)
            with patch.dict(sys.modules, {"huggingface_hub": fake_module}):
                summaries = merge_correct_segments_from_manifest(str(manifest_path))

            self.assertEqual(len(summaries), 2)
            self.assertEqual({summary["task_name"] for summary in summaries},
                             {"task_a", "task_b"})
            self.assertIn(
                (
                    "upload_file",
                    str(root_a / "correct" / "shards" / "shard_00000.parquet"),
                    "task_a/shard_00000.parquet",
                    "JoyboyGo/hf_data",
                    "dataset",
                ),
                uploads,
            )
            self.assertIn(
                (
                    "upload_file",
                    str(root_b / "correct" / "shards" / "shard_00000.parquet"),
                    "task_b/shard_00000.parquet",
                    "JoyboyGo/hf_data",
                    "dataset",
                ),
                uploads,
            )

    def test_write_summary_csv_flattens_key_counts_in_workdir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "merge_correct_segments_summary.csv"
            summaries = [{
                "task_name": "task_a",
                "output_dir": "/tmp/out",
                "stream": "correct",
                "total_segment_count": 10,
                "pending_segment_count_before_run": 3,
                "pending_segment_count": 0,
                "merged_segments_total": 10,
                "merged_segments_this_run": 3,
                "shards_written_this_run": 2,
                "next_shard_idx": 2,
                "total_records": 123,
                "merged_records_this_run": 45,
                "token_counts": {
                    "total_tokens": 123,
                },
                "token_sums": {
                    "total_tokens": 4567,
                },
                "average_tokens": {
                    "total_tokens": 37.1301,
                },
                "hf_upload": {
                    "uploaded": 2,
                    "skipped": 1,
                    "reason": "",
                },
            }]

            write_summary_csv(summaries, csv_path)

            with csv_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["task_name"], "task_a")
            self.assertEqual(row["total_segment_count"], "10")
            self.assertEqual(row["total_records"], "123")
            self.assertEqual(row["token_count_total"], "123")
            self.assertEqual(row["token_sum_total"], "4567")
            self.assertEqual(row["avg_total_tokens"], "37.1301")
            self.assertEqual(row["hf_uploaded"], "2")
            self.assertEqual(row["hf_skipped"], "1")

    def test_merge_serializes_dynamic_judge_detail_for_parquet(self):
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
                    "judge_detail": {
                        "constraint_results": [{
                            "source": "a",
                            "kind": "num_sentences",
                            "passed": True,
                            "detail": {
                                "expected": 5,
                                "actual": 5,
                            },
                        }],
                    },
                }, {
                    "prompt_tokens": 4,
                    "completion_tokens": 5,
                    "total_tokens": 9,
                    "is_correct": True,
                    "judge_detail": {
                        "constraint_results": [{
                            "source": "b",
                            "kind": "specific_ending",
                            "passed": False,
                            "detail": {
                                "expected": "done",
                            },
                        }],
                    },
                }],
            )

            summary = merge_correct_segments(str(root), shard_size_mb=1)

            self.assertEqual(summary["merged_segments_this_run"], 1)
            shard_files = sorted((root / "correct" / "shards").glob("shard_*.parquet"))
            self.assertEqual(len(shard_files), 1)
            rows = pq.read_table(shard_files[0]).to_pylist()
            self.assertEqual(len(rows), 2)
            self.assertIsInstance(rows[0]["judge_detail"], str)
            self.assertIn("constraint_results", rows[0]["judge_detail"])
            self.assertIsInstance(rows[1]["judge_detail"], str)


if __name__ == "__main__":
    unittest.main()
