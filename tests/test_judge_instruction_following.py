import unittest

from distill.core.judge import judge_output, judge_output_with_timeout


class InstructionFollowingJudgeTests(unittest.TestCase):

    def test_structured_if_constraints_pass_when_all_supported_rules_match(self):
        row_data = {
            "ground_truth": (
                "[{'instruction_id': ['detectable_format:sentence_hyphens', "
                "'last_word:last_word_answer'], 'kwargs': [None, "
                "{'last_word': 'brief'}]}]"
            ),
            "dataset": "ifeval",
        }
        messages = [{
            "role": "assistant",
            "content": (
                "Kpanlogo-is-a-percussion-instrument-while-Shamisen-is-a-"
                "string-instrument-brief"
            ),
        }]

        result = judge_output(row_data, messages)

        self.assertEqual(result["judge_type"], "instruction_following")
        self.assertEqual(result["judge_backend"], "instruction_following_v1")
        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_structured_if_returns_not_applicable_for_unsupported_constraint(self):
        row_data = {
            "ground_truth": (
                "[{'instruction_id': ['count:lowercase_counting'], 'kwargs': "
                "[{'N': 3}]}]"
            ),
            "dataset": "ifeval",
        }
        messages = [{
            "role": "assistant",
            "content": "plain lowercase response",
        }]

        result = judge_output(row_data, messages)

        self.assertEqual(result["judge_type"], "instruction_following")
        self.assertEqual(result["judge_backend"], "instruction_following_v1")
        self.assertIsNone(result["is_correct"])
        self.assertEqual(result["judge_status"], "not_applicable")

    def test_tulu_prompt_constraints_pass_for_excluded_words_and_specific_ending(self):
        row_data = {
            "prompt": (
                "Provide a detailed analysis of Candace Parker's defensive "
                "techniques in her recent games, excluding the words "
                "\"aggressive\" and \"blocking\". End your response with "
                "\"Go team.\""
            ),
            "constraints": [
                "keywords:exclude words",
                "specific ending",
            ],
        }
        messages = [{
            "role": "assistant",
            "content": (
                "Her defensive timing stands out because she anticipates "
                "passing lanes and rotates early to help teammates. Go team."
            ),
        }]

        result = judge_output(row_data, messages)

        self.assertEqual(result["judge_type"], "instruction_following")
        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_prompt_constraints_fail_when_required_keyword_is_missing(self):
        row_data = {
            "prompt": (
                "Write a fan chant for the Punjab Panthers. Include the "
                "keywords \"roar,\" \"victory,\" and \"spirit.\" Use no "
                "commas in the chant."
            ),
            "constraints": [
                "include keywords",
                "punctuation:use no comma",
            ],
        }
        messages = [{
            "role": "assistant",
            "content": "Roar for victory tonight Panthers rise high",
        }]

        result = judge_output(row_data, messages)

        self.assertEqual(result["judge_type"], "instruction_following")
        self.assertFalse(result["is_correct"])
        self.assertEqual(result["judge_status"], "wrong_answer")

    def test_tulu_keywords_ignore_prompt_punctuation_noise(self):
        row_data = {
            "prompt": (
                "Write a fan chant for the Punjab Panthers. Include the "
                "keywords \"roar,\" \"victory,\" and \"spirit.\" Use no "
                "commas in the chant."
            ),
            "constraints": [
                "include keywords",
                "punctuation:use no comma",
            ],
        }
        messages = [{
            "role": "assistant",
            "content": "Roar for victory and spirit tonight Panthers rise high",
        }]

        result = judge_output(row_data, messages)

        self.assertEqual(result["judge_type"], "instruction_following")
        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_timeout_wrapper_preserves_instruction_following_result(self):
        row_data = {
            "ground_truth": (
                "[{'instruction_id': ['last_word:last_word_answer'], "
                "'kwargs': [{'last_word': 'brief'}]}]"
            ),
            "dataset": "ifeval",
        }
        messages = [{
            "role": "assistant",
            "content": "Concise answer brief",
        }]

        direct = judge_output(row_data, messages)
        guarded = judge_output_with_timeout(row_data, messages, timeout=20)

        self.assertEqual(guarded, direct)


if __name__ == "__main__":
    unittest.main()
