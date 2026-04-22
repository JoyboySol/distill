import unittest

from distill.core.judge import judge_output
from distill.core.pipeline import DistillPipeline, TaskItem
from distill.runtime.settings import PipelineConfig


class McqBoxedJudgeTests(unittest.TestCase):

    def test_mcq_boxed_judge_passes_when_boxed_choice_matches(self):
        row_data = {
            "messages": [
                {
                    "role": "system",
                    "content": "",
                },
                {
                    "role": "user",
                    "content": "Which option is correct?\nA: Wrong\nB: Right\nC: Also wrong",
                },
                {
                    "role": "assistant",
                    "content": "Reasoning...\n\\boxed{B}",
                },
            ]
        }
        messages = [{
            "role": "assistant",
            "content": "After checking each option, the answer is\n\\boxed{B}",
        }]

        result = judge_output(row_data, messages)

        self.assertEqual(result["judge_type"], "mcq_boxed")
        self.assertEqual(result["judge_backend"], "boxed_choice_v1")
        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")
        self.assertEqual(result["judge_detail"]["reference"], "B")
        self.assertEqual(result["judge_detail"]["prediction"], "B")

    def test_mcq_boxed_judge_reports_wrong_answer_when_prediction_differs(self):
        row_data = {
            "messages": [
                {
                    "role": "user",
                    "content": "Pick one.\nA: Wrong\nB: Wrong\nC: Correct\nD: Wrong",
                },
                {
                    "role": "assistant",
                    "content": "Correct answer:\n\\boxed{C}",
                },
            ]
        }
        messages = [{
            "role": "assistant",
            "content": "I choose\n\\boxed{A}",
        }]

        result = judge_output(row_data, messages)

        self.assertEqual(result["judge_type"], "mcq_boxed")
        self.assertFalse(result["is_correct"])
        self.assertEqual(result["judge_status"], "wrong_answer")
        self.assertEqual(result["judge_detail"]["reference"], "C")
        self.assertEqual(result["judge_detail"]["prediction"], "A")

    def test_mcq_boxed_judge_accepts_unboxed_answer_phrase(self):
        row_data = {
            "messages": [
                {
                    "role": "user",
                    "content": "Pick one.\nA: Wrong\nB: Wrong\nC: Correct\nD: Wrong",
                },
                {
                    "role": "assistant",
                    "content": "Correct answer:\n\\boxed{C}",
                },
            ]
        }
        messages = [{
            "role": "assistant",
            "content": "The correct answer is C: the third option.",
        }]

        result = judge_output(row_data, messages)

        self.assertEqual(result["judge_type"], "mcq_boxed")
        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")
        self.assertEqual(result["judge_detail"]["reference"], "C")
        self.assertEqual(result["judge_detail"]["prediction"], "C")

    def test_mcq_boxed_judge_accepts_unboxed_option_phrase(self):
        row_data = {
            "messages": [
                {
                    "role": "user",
                    "content": "Pick one.\nA: Wrong\nB: Wrong\nC: Correct\nD: Wrong",
                },
                {
                    "role": "assistant",
                    "content": "Correct answer:\n\\boxed{C}",
                },
            ]
        }
        messages = [{
            "role": "assistant",
            "content": "After reviewing the choices, Option C is correct because it matches the definition.",
        }]

        result = judge_output(row_data, messages)

        self.assertEqual(result["judge_type"], "mcq_boxed")
        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")
        self.assertEqual(result["judge_detail"]["reference"], "C")
        self.assertEqual(result["judge_detail"]["prediction"], "C")

    def test_multiturn_build_output_record_uses_mcq_boxed_judge(self):
        config = PipelineConfig(
            input_dir="in",
            output_dir="out",
            failure_log="failure.jsonl",
            model_name="test-model",
            api_key="EMPTY",
            base_urls=["http://127.0.0.1:20001/v1"],
            max_concurrency=1,
        )
        pipeline = DistillPipeline(config)
        task = TaskItem(
            source_file="/tmp/stem.parquet",
            source_row=0,
            rollout_index=0,
            row_data={
                "messages": [
                    {
                        "role": "system",
                        "content": "",
                    },
                    {
                        "role": "user",
                        "content": "Question?\nA: Wrong\nB: Correct\nC: Wrong",
                    },
                    {
                        "role": "assistant",
                        "content": "Reference\n\\boxed{B}",
                    },
                ]
            },
            input_messages=[
                {
                    "role": "system",
                    "content": "",
                },
                {
                    "role": "user",
                    "content": "Question?\nA: Wrong\nB: Correct\nC: Wrong",
                },
                {
                    "role": "assistant",
                    "content": "Reference\n\\boxed{B}",
                },
            ],
            task_mode="multi_turn",
        )

        record = pipeline._build_output_record(
            task,
            messages=[
                {
                    "role": "system",
                    "content": "",
                },
                {
                    "role": "user",
                    "content": "Question?\nA: Wrong\nB: Correct\nC: Wrong",
                },
                {
                    "role": "assistant",
                    "content": "Prediction\n\\boxed{B}",
                },
            ],
            finish_reason="stop",
            usage={},
            distill_status="success",
            distill_error=None,
            assistant_turns_completed=1,
            assistant_turns_total=1,
        )

        self.assertEqual(record["judge_type"], "mcq_boxed")
        self.assertEqual(record["judge_backend"], "boxed_choice_v1")
        self.assertTrue(record["is_correct"])
        self.assertEqual(record["judge_status"], "pass")


if __name__ == "__main__":
    unittest.main()
