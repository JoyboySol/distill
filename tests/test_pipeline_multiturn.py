import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
import json

from distill.common.utils import ensure_message_shape
from distill.core.pipeline import DistillPipeline, TaskItem
from distill.runtime.settings import PipelineConfig


class MultiTurnPipelineTests(unittest.IsolatedAsyncioTestCase):

    def _build_pipeline(self, output_dir: str) -> DistillPipeline:
        config = PipelineConfig(
            input_dir="in",
            output_dir=output_dir,
            failure_log=str(Path(output_dir) / "failures.jsonl"),
            model_name="test-model",
            api_key="EMPTY",
            base_urls=["http://127.0.0.1:20001/v1"],
            max_concurrency=1,
        )
        return DistillPipeline(config)

    async def test_worker_distills_multiturn_conversation_with_distilled_history(
            self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = self._build_pipeline(tmpdir)
            pipeline.failure_recorder.record_failure = AsyncMock()

            input_messages = [
                {
                    "role": "user",
                    "content": "Q1",
                },
                {
                    "role": "assistant",
                    "content": "raw A1",
                },
                {
                    "role": "user",
                    "content": "Q2",
                },
                {
                    "role": "assistant",
                    "content": "raw A2",
                },
            ]

            seen_contexts = []

            async def fake_generate_messages(messages):
                seen_contexts.append([(msg["role"], msg["content"])
                                      for msg in messages])
                if len(seen_contexts) == 1:
                    return {
                        "assistant_message": {
                            "role": "assistant",
                            "content": "distilled A1",
                            "reasoning_content": "r1",
                            "tool_calls": None,
                            "tool_call_id": None,
                            "name": None,
                        },
                        "finish_reason": "stop",
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 4,
                            "total_tokens": 14,
                        },
                    }
                return {
                    "assistant_message": {
                        "role": "assistant",
                        "content": "distilled A2",
                        "reasoning_content": "r2",
                        "tool_calls": None,
                        "tool_call_id": None,
                        "name": None,
                    },
                    "finish_reason": "stop",
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 5,
                        "total_tokens": 25,
                    },
                }

            pipeline.llm_manager.generate_messages = fake_generate_messages

            class _Pbar:

                def update(self, n):
                    return None

            await pipeline.task_queue.put(
                TaskItem(
                    source_file="/tmp/input.jsonl",
                    source_row=3,
                    rollout_index=0,
                    row_data={"messages": input_messages},
                    input_messages=input_messages,
                    task_mode="multi_turn",
                ))
            await pipeline.task_queue.put(None)

            await pipeline.worker(0, _Pbar())

            result = await pipeline.judge_queue.get()
            self.assertEqual(
                seen_contexts,
                [
                    [("user", "Q1")],
                    [("user", "Q1"), ("assistant", "distilled A1"),
                     ("user", "Q2")],
                ],
            )
            self.assertEqual(
                [msg["content"] for msg in result.messages],
                ["Q1", "distilled A1", "Q2", "distilled A2"],
            )
            self.assertEqual(
                [msg["reasoning_content"] for msg in result.messages
                 if msg["role"] == "assistant"],
                ["r1", "r2"],
            )
            self.assertEqual(result.distill_status, "success")
            self.assertEqual(result.assistant_turns_completed, 2)
            self.assertEqual(result.assistant_turns_total, 2)
            self.assertEqual(result.usage["prompt_tokens"], 30)
            self.assertEqual(result.usage["completion_tokens"], 9)
            self.assertEqual(result.usage["total_tokens"], 39)
            pipeline.failure_recorder.record_failure.assert_not_awaited()

    async def test_worker_keeps_successful_prefix_and_marks_partial_on_turn_failure(
            self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = self._build_pipeline(tmpdir)
            pipeline.failure_recorder.record_failure = AsyncMock()

            input_messages = [
                {
                    "role": "user",
                    "content": "Q1",
                },
                {
                    "role": "assistant",
                    "content": "raw A1",
                },
                {
                    "role": "user",
                    "content": "Q2",
                },
                {
                    "role": "assistant",
                    "content": "raw A2",
                },
                {
                    "role": "user",
                    "content": "Q3",
                },
            ]

            call_count = 0

            async def fake_generate_messages(messages):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return {
                        "assistant_message": {
                            "role": "assistant",
                            "content": "distilled A1",
                            "reasoning_content": "r1",
                            "tool_calls": None,
                            "tool_call_id": None,
                            "name": None,
                        },
                        "finish_reason": "stop",
                        "usage": {
                            "prompt_tokens": 7,
                            "completion_tokens": 3,
                            "total_tokens": 10,
                        },
                    }
                raise RuntimeError("second turn failed")

            pipeline.llm_manager.generate_messages = fake_generate_messages

            class _Pbar:

                def update(self, n):
                    return None

            await pipeline.task_queue.put(
                TaskItem(
                    source_file="/tmp/input.jsonl",
                    source_row=4,
                    rollout_index=0,
                    row_data={"messages": input_messages},
                    input_messages=input_messages,
                    task_mode="multi_turn",
                ))
            await pipeline.task_queue.put(None)

            await pipeline.worker(0, _Pbar())

            result = await pipeline.judge_queue.get()
            self.assertEqual(
                [msg["content"] for msg in result.messages],
                ["Q1", "distilled A1", "Q2"],
            )
            self.assertEqual(result.distill_status, "partial")
            self.assertEqual(result.distill_error,
                             "worker_error:RuntimeError:second turn failed")
            self.assertEqual(result.assistant_turns_completed, 1)
            self.assertEqual(result.assistant_turns_total, 2)
            pipeline.failure_recorder.record_failure.assert_not_awaited()

    async def test_worker_completes_trailing_user_turn_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = self._build_pipeline(tmpdir)
            pipeline.config.complete_trailing_user_turn = True
            pipeline.failure_recorder.record_failure = AsyncMock()

            input_messages = [
                {
                    "role": "user",
                    "content": "Q1",
                },
                {
                    "role": "assistant",
                    "content": "raw A1",
                },
                {
                    "role": "user",
                    "content": "Q2 follow-up",
                },
            ]

            seen_contexts = []

            async def fake_generate_messages(messages):
                seen_contexts.append([(msg["role"], msg["content"])
                                      for msg in messages])
                if len(seen_contexts) == 1:
                    return {
                        "assistant_message": {
                            "role": "assistant",
                            "content": "distilled A1",
                            "reasoning_content": "r1",
                            "tool_calls": None,
                            "tool_call_id": None,
                            "name": None,
                        },
                        "finish_reason": "stop",
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 4,
                            "total_tokens": 14,
                        },
                    }
                return {
                    "assistant_message": {
                        "role": "assistant",
                        "content": "distilled A2",
                        "reasoning_content": "r2",
                        "tool_calls": None,
                        "tool_call_id": None,
                        "name": None,
                    },
                    "finish_reason": "stop",
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 5,
                        "total_tokens": 17,
                    },
                }

            pipeline.llm_manager.generate_messages = fake_generate_messages

            class _Pbar:

                def update(self, n):
                    return None

            await pipeline.task_queue.put(
                TaskItem(
                    source_file="/tmp/input.jsonl",
                    source_row=8,
                    rollout_index=0,
                    row_data={"messages": input_messages},
                    input_messages=input_messages,
                    task_mode="multi_turn",
                ))
            await pipeline.task_queue.put(None)

            await pipeline.worker(0, _Pbar())

            result = await pipeline.judge_queue.get()
            self.assertEqual(
                seen_contexts,
                [
                    [("user", "Q1")],
                    [("user", "Q1"), ("assistant", "distilled A1"),
                     ("user", "Q2 follow-up")],
                ],
            )
            self.assertEqual(
                [msg["content"] for msg in result.messages],
                ["Q1", "distilled A1", "Q2 follow-up", "distilled A2"],
            )
            self.assertEqual(result.distill_status, "success")
            self.assertEqual(result.assistant_turns_completed, 2)
            self.assertEqual(result.assistant_turns_total, 2)
            self.assertEqual(result.usage["prompt_tokens"], 22)
            self.assertEqual(result.usage["completion_tokens"], 9)
            self.assertEqual(result.usage["total_tokens"], 31)

    def test_build_output_record_skips_judge_for_multi_turn_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = self._build_pipeline(tmpdir)
            task = TaskItem(
                source_file="/tmp/input.jsonl",
                source_row=5,
                rollout_index=0,
                row_data={"messages": []},
                input_messages=[],
                task_mode="multi_turn",
            )
            messages = [
                {
                    "role": "user",
                    "content": "Q1",
                },
                {
                    "role": "assistant",
                    "content": "A1",
                    "reasoning_content": "r1",
                },
            ]

            with patch("distill.core.pipeline.judge_output_with_timeout",
                       side_effect=AssertionError("judge should be skipped")):
                record = pipeline._build_output_record(
                    task,
                    messages,
                    finish_reason="stop",
                    usage={
                        "prompt_tokens": 11,
                        "completion_tokens": 4,
                        "total_tokens": 15,
                    },
                    distill_status="success",
                    distill_error=None,
                    assistant_turns_completed=1,
                    assistant_turns_total=1,
                )

            self.assertEqual(record["messages"],
                             [ensure_message_shape(message) for message in messages])
            self.assertEqual(record["judge_type"], "none")
            self.assertEqual(record["judge_backend"], "none")
            self.assertIsNone(record["is_correct"])
            self.assertEqual(record["judge_status"], "not_applicable")
            self.assertEqual(record["distill_status"], "success")

    def test_build_output_record_can_treat_no_judge_as_correct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = self._build_pipeline(tmpdir)
            pipeline.config.treat_no_judge_as_correct = True
            task = TaskItem(
                source_file="/tmp/input.jsonl",
                source_row=6,
                rollout_index=0,
                row_data={"messages": []},
                input_messages=[],
                task_mode="multi_turn",
            )
            messages = [
                {
                    "role": "user",
                    "content": "Q1",
                },
                {
                    "role": "assistant",
                    "content": "A1",
                },
            ]

            with patch("distill.core.pipeline.judge_output_with_timeout",
                       side_effect=AssertionError("judge should be skipped")):
                record = pipeline._build_output_record(
                    task,
                    messages,
                    finish_reason="stop",
                    usage={
                        "prompt_tokens": 11,
                        "completion_tokens": 4,
                        "total_tokens": 15,
                    },
                    distill_status="success",
                    distill_error=None,
                    assistant_turns_completed=1,
                    assistant_turns_total=1,
                )

            self.assertEqual(record["judge_type"], "none")
            self.assertEqual(record["judge_backend"], "none")
            self.assertTrue(record["is_correct"])
            self.assertEqual(record["judge_status"], "assumed_correct")

    def test_build_output_record_can_skip_single_turn_judge_via_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = self._build_pipeline(tmpdir)
            pipeline.config.judge_mode = "none"
            pipeline.config.treat_no_judge_as_correct = True
            task = TaskItem(
                source_file="/tmp/input.jsonl",
                source_row=7,
                rollout_index=0,
                row_data={"instruction": "Write code"},
                prompt="Write code",
                task_mode="single_turn",
            )
            messages = [
                {
                    "role": "user",
                    "content": "Write code",
                },
                {
                    "role": "assistant",
                    "content": "def solve(): pass",
                },
            ]

            with patch("distill.core.pipeline.judge_output_with_timeout",
                       side_effect=AssertionError("judge should be skipped")):
                with patch("distill.core.pipeline.judge_output",
                           side_effect=AssertionError("judge should be skipped")):
                    record = pipeline._build_output_record(
                        task,
                        messages,
                        finish_reason="stop",
                        usage={
                            "prompt_tokens": 11,
                            "completion_tokens": 4,
                            "total_tokens": 15,
                        },
                        distill_status="success",
                        distill_error=None,
                        assistant_turns_completed=1,
                        assistant_turns_total=1,
                    )

            self.assertEqual(record["judge_type"], "none")
            self.assertEqual(record["judge_backend"], "none")
            self.assertTrue(record["is_correct"])
            self.assertEqual(record["judge_status"], "assumed_correct")

    def test_build_output_record_does_not_treat_overlong_no_judge_as_correct(
            self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = self._build_pipeline(tmpdir)
            pipeline.config.judge_mode = "none"
            pipeline.config.treat_no_judge_as_correct = True
            task = TaskItem(
                source_file="/tmp/input.jsonl",
                source_row=8,
                rollout_index=0,
                row_data={"instruction": "Write code"},
                prompt="Write code",
                task_mode="single_turn",
            )
            messages = [
                {
                    "role": "user",
                    "content": "Write code",
                },
                {
                    "role": "assistant",
                    "content": "def solve(): pass",
                },
            ]

            with patch("distill.core.pipeline.judge_output_with_timeout",
                       side_effect=AssertionError("judge should be skipped")):
                with patch("distill.core.pipeline.judge_output",
                           side_effect=AssertionError("judge should be skipped")):
                    record = pipeline._build_output_record(
                        task,
                        messages,
                        finish_reason="length",
                        usage={
                            "prompt_tokens": 11,
                            "completion_tokens": 4,
                            "total_tokens": 15,
                        },
                        distill_status="success",
                        distill_error=None,
                        assistant_turns_completed=1,
                        assistant_turns_total=1,
                    )

            self.assertEqual(record["judge_type"], "none")
            self.assertEqual(record["judge_backend"], "none")
            self.assertIsNone(record["is_correct"])
            self.assertEqual(record["judge_status"], "not_applicable_overlong")

    def test_build_output_record_passes_explicit_judge_mode_to_single_turn_judge(
            self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = self._build_pipeline(tmpdir)
            pipeline.config.judge_mode = "instruction_following"
            task = TaskItem(
                source_file="/tmp/input.jsonl",
                source_row=9,
                rollout_index=0,
                row_data={"ground_truth": "[]"},
                prompt="Prompt",
                task_mode="single_turn",
            )
            messages = [
                {
                    "role": "user",
                    "content": "Prompt",
                },
                {
                    "role": "assistant",
                    "content": "Answer",
                },
            ]

            with patch("distill.core.pipeline.judge_output_with_timeout",
                       return_value={
                           "judge_type": "instruction_following",
                           "judge_backend": "instruction_following_v1",
                           "is_correct": True,
                           "judge_status": "pass",
                           "judge_detail": {},
                       }) as mock_judge:
                record = pipeline._build_output_record(
                    task,
                    messages,
                    finish_reason="stop",
                    usage={},
                    distill_status="success",
                    distill_error=None,
                    assistant_turns_completed=1,
                    assistant_turns_total=1,
                )

            self.assertEqual(record["judge_type"], "instruction_following")
            self.assertEqual(
                mock_judge.call_args.kwargs["judge_mode"],
                "instruction_following",
            )

    def test_prepare_task_input_detects_serialized_multiturn_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = self._build_pipeline(tmpdir)
            raw_input = json.dumps([
                {
                    "role": "user",
                    "content": "Q1",
                },
                {
                    "role": "assistant",
                    "content": "A1",
                },
            ])

            prepared = pipeline._prepare_task_input(raw_input)

            self.assertIsNotNone(prepared)
            self.assertEqual(prepared.task_mode, "multi_turn")
            self.assertEqual(
                [msg["role"] for msg in prepared.input_messages],
                ["user", "assistant"],
            )


if __name__ == "__main__":
    unittest.main()
