import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from distill.core.pipeline import DistillPipeline
from distill.runtime.settings import PipelineConfig


class PipelineRunShutdownTests(unittest.TestCase):

    def test_run_completes_with_periodic_merge_and_multiple_workers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "in"
            input_dir.mkdir(parents=True, exist_ok=True)
            input_path = input_dir / "sample.jsonl"
            input_path.write_text(
                json.dumps({"instruction": "Write hello world."},
                           ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            output_dir = Path(tmpdir) / "out"
            config = PipelineConfig(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                failure_log=str(Path(tmpdir) / "failures.jsonl"),
                model_name="test-model",
                api_key="EMPTY",
                base_urls=["http://127.0.0.1:20001/v1"],
                file_pattern="*.jsonl",
                input_content_field="instruction",
                judge_mode="none",
                treat_no_judge_as_correct=True,
                sample_limit=1,
                max_concurrency=4,
                judge_concurrency=2,
                merge_every_n_writes=1,
                segment_flush_interval_sec=0.0,
                batch_size=8,
            )
            pipeline = DistillPipeline(config)
            pipeline.failure_recorder.record_failure = AsyncMock()

            async def fake_generate(prompt: str):
                return {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        },
                        {
                            "role": "assistant",
                            "content": "hello world",
                            "reasoning_content": "brief reasoning",
                        },
                    ],
                    "finish_reason": "stop",
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
                }

            pipeline.llm_manager.generate = fake_generate

            summary = asyncio.run(asyncio.wait_for(pipeline.run(), timeout=5.0))

            self.assertEqual(summary["written_delta"], 1)
            self.assertEqual(summary["correct_delta"], 1)
            self.assertTrue((output_dir / "all" / "shards" /
                             "shard_00000.parquet").exists())
            self.assertTrue((output_dir / "correct" / "shards" /
                             "shard_00000.parquet").exists())

    def test_run_interrupt_cancels_inflight_requests_and_writes_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "in"
            input_dir.mkdir(parents=True, exist_ok=True)
            input_path = input_dir / "sample.jsonl"
            input_path.write_text(
                json.dumps({"instruction": "Write hello world."},
                           ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            output_dir = Path(tmpdir) / "out"
            config = PipelineConfig(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                failure_log=str(Path(tmpdir) / "failures.jsonl"),
                model_name="test-model",
                api_key="EMPTY",
                base_urls=["http://127.0.0.1:20001/v1"],
                file_pattern="*.jsonl",
                input_content_field="instruction",
                judge_mode="none",
                treat_no_judge_as_correct=True,
                sample_limit=1,
                max_concurrency=1,
                judge_concurrency=1,
                upload_merged_shards=True,
                hf_repo_id="JoyboyGo/hf_data",
                hf_token="token",
                segment_flush_interval_sec=0.0,
                batch_size=8,
            )
            pipeline = DistillPipeline(config)
            pipeline.failure_recorder.record_failure = AsyncMock()
            pipeline._upload_pending_correct_shards = AsyncMock(
                return_value={"uploaded": 0, "skipped": 0, "reason": "no_shards"})

            started = asyncio.Event()
            cancelled = asyncio.Event()

            async def slow_generate(prompt: str):
                started.set()
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
                return {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        },
                        {
                            "role": "assistant",
                            "content": "hello world",
                        },
                    ],
                    "finish_reason": "stop",
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
                }

            pipeline.llm_manager.generate = slow_generate

            async def run_and_interrupt():
                run_task = asyncio.create_task(pipeline.run())
                await asyncio.wait_for(started.wait(), timeout=2.0)
                pipeline._request_stop("SIGINT")
                return await asyncio.wait_for(run_task, timeout=2.0)

            summary = asyncio.run(run_and_interrupt())

            self.assertTrue(summary["interrupted"])
            self.assertTrue(cancelled.is_set())
            self.assertTrue(pipeline.llm_manager._stop_requested)
            pipeline._upload_pending_correct_shards.assert_awaited_once()
            self.assertTrue((output_dir / "all" / "judge_stats.json").exists())
            self.assertTrue((output_dir / "correct" / "judge_stats.json").exists())
            self.assertFalse((output_dir / "all" / "shards" /
                              "shard_00000.parquet").exists())


if __name__ == "__main__":
    unittest.main()
