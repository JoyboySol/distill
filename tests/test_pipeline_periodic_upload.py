import asyncio
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from distill.core.pipeline import DistillPipeline, ResultItem
from distill.runtime.settings import PipelineConfig


class PeriodicMergeUploadTests(unittest.TestCase):

    def _build_pipeline(self, output_dir: str, **overrides) -> DistillPipeline:
        config = PipelineConfig(
            input_dir="in",
            output_dir=output_dir,
            failure_log=str(Path(output_dir) / "failures.jsonl"),
            model_name="test-model",
            api_key="EMPTY",
            base_urls=["http://127.0.0.1:20001/v1"],
            max_concurrency=1,
            merge_every_n_writes=overrides.get("merge_every_n_writes", 0),
            upload_every_n_merges=overrides.get("upload_every_n_merges", 1),
            upload_merged_shards=overrides.get("upload_merged_shards", False),
            hf_repo_id=overrides.get("hf_repo_id"),
            hf_token=overrides.get("hf_token"),
            task_name=overrides.get("task_name", "task_a"),
        )
        return DistillPipeline(config)

    def _result_item(self, source_row: int) -> ResultItem:
        return ResultItem(
            source_file="/tmp/source.jsonl",
            source_row=source_row,
            estimated_size=128,
            record={
                "source_file": "/tmp/source.jsonl",
                "source_row": source_row,
                "rollout_index": 0,
                "is_correct": True,
                "generation_finish_reason": "stop",
            },
        )

    def test_writer_daemon_runs_periodic_merge_and_upload_before_shutdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = self._build_pipeline(
                tmpdir,
                merge_every_n_writes=2,
                upload_every_n_merges=1,
                upload_merged_shards=True,
                hf_repo_id="JoyboyGo/hf_data",
                hf_token="token",
            )
            calls = []

            async def fake_flush_segment_buffer(buffers, buffer_sizes, stream_name):
                calls.append(("flush", stream_name))
                buffers[stream_name] = []
                buffer_sizes[stream_name] = 0

            async def fake_merge_stream_segments(stream_name):
                calls.append(("merge", stream_name))
                return {
                    "merged_segments_this_run": 1 if stream_name == pipeline.STREAM_CORRECT else 0,
                    "shards_written_this_run": 1 if stream_name == pipeline.STREAM_CORRECT else 0,
                }

            async def fake_upload():
                calls.append(("upload", pipeline.STREAM_CORRECT))
                return {"uploaded": 1}

            pipeline._flush_segment_buffer = fake_flush_segment_buffer
            pipeline._merge_stream_segments = fake_merge_stream_segments
            pipeline._upload_pending_correct_shards = fake_upload

            class _Pbar:
                disable = False
                _resumed_tasks = 0
                _discovered_tasks = 0
                _written_tasks = 0
                _correct_tasks = 0
                _overlong_tasks = 0

                def set_postfix_str(self, text):
                    self.postfix = text

            async def run_writer():
                task = asyncio.create_task(pipeline.writer_daemon(_Pbar()))
                await pipeline.result_queue.put(self._result_item(0))
                await pipeline.result_queue.put(self._result_item(1))
                await pipeline.result_queue.put(None)
                await task

            asyncio.run(run_writer())

            self.assertGreaterEqual(calls.count(("merge", pipeline.STREAM_CORRECT)), 2)
            self.assertEqual(calls.count(("upload", pipeline.STREAM_CORRECT)), 2)

    def test_upload_pending_correct_shards_uses_single_level_remote_dir_and_is_incremental(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = self._build_pipeline(
                tmpdir,
                upload_merged_shards=True,
                hf_repo_id="JoyboyGo/hf_data",
                hf_token="token",
                task_name="task_a",
            )
            shard_dir = Path(tmpdir) / "correct" / "shards"
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_path = shard_dir / "shard_00000.parquet"
            shard_path.write_bytes(b"parquet")

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
                first = asyncio.run(pipeline._upload_pending_correct_shards())
                second = asyncio.run(pipeline._upload_pending_correct_shards())

            self.assertEqual(first["uploaded"], 1)
            self.assertEqual(second["uploaded"], 0)
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

            state_path = Path(tmpdir) / ".hf_upload_state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(
                state["uploaded_paths"],
                ["correct/shards/shard_00000.parquet"],
            )


if __name__ == "__main__":
    unittest.main()
