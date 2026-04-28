import unittest

from distill.core.judge import judge_output, judge_output_with_timeout


class JudgeRoutingTests(unittest.TestCase):

    def test_explicit_instruction_following_mode_skips_code_family(self):
        row_data = {
            "ground_truth": (
                "[{'instruction_id': ['last_word:last_word_answer'], "
                "'kwargs': [{'last_word': 'brief'}]}]"
            ),
            "dataset": "ifeval",
            "test_list": ["assert False"],
        }
        messages = [{
            "role": "assistant",
            "content": "final answer brief",
        }]

        auto_result = judge_output(row_data, messages)
        if_result = judge_output(row_data,
                                 messages,
                                 judge_mode="instruction_following")

        self.assertEqual(auto_result["judge_type"], "code_mbpp")
        self.assertEqual(auto_result["judge_status"], "failed")
        self.assertEqual(if_result["judge_type"], "instruction_following")
        self.assertEqual(if_result["judge_status"], "pass")

    def test_explicit_math_mode_disables_instruction_following_judge(self):
        row_data = {
            "ground_truth": (
                "[{'instruction_id': ['last_word:last_word_answer'], "
                "'kwargs': [{'last_word': 'brief'}]}]"
            ),
            "dataset": "ifeval",
        }
        messages = [{
            "role": "assistant",
            "content": "final answer brief",
        }]

        result = judge_output(row_data, messages, judge_mode="math")

        self.assertIsNone(result["judge_type"])
        self.assertEqual(result["judge_status"], "not_applicable")

    def test_timeout_wrapper_respects_explicit_judge_mode(self):
        row_data = {
            "ground_truth": (
                "[{'instruction_id': ['last_word:last_word_answer'], "
                "'kwargs': [{'last_word': 'brief'}]}]"
            ),
            "dataset": "ifeval",
            "test_list": ["assert False"],
        }
        messages = [{
            "role": "assistant",
            "content": "final answer brief",
        }]

        result = judge_output_with_timeout(
            row_data,
            messages,
            judge_mode="instruction_following",
            timeout=20,
        )

        self.assertEqual(result["judge_type"], "instruction_following")
        self.assertEqual(result["judge_status"], "pass")


if __name__ == "__main__":
    unittest.main()
