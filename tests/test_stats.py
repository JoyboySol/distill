import unittest

from distill.tools.stats import summarize


class StatsTests(unittest.TestCase):

    def test_summarize_accepts_stringified_judge_detail(self):
        summary = summarize([{
            "judge_type": "instruction_following",
            "judge_backend": "instruction_following_v1",
            "judge_status": "pass",
            "generation_finish_reason": "stop",
            "is_correct": True,
            "judge_detail": "{\"unsupported_constraints\": [\"response language\"]}",
        }])

        self.assertEqual(summary["total_records"], 1)
        self.assertEqual(summary["correct_counts"]["true"], 1)
        self.assertEqual(summary["overall_accuracy"], 100.0)


if __name__ == "__main__":
    unittest.main()
