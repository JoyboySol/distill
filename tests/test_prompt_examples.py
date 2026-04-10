import unittest

from distill.core.judge import judge_output
from distill.core.prompt_examples import (build_prompt_example_sample,
                                          parse_prompt_examples)


class PromptExampleParserTests(unittest.TestCase):

    def test_parse_single_example_with_colons(self):
        prompt = """Problem statement.

Example
Input:
3
1 2
3 4
5 6

Output:
3
7
11

Explanation
Done.
"""
        parsed = parse_prompt_examples(prompt)
        self.assertEqual(parsed, [{
            "input": "3\n1 2\n3 4\n5 6",
            "output": "3\n7\n11",
        }])

    def test_parse_multiple_examples(self):
        prompt = """Description

Examples

Input
RRS

Output
2

Input
SSS

Output
0

Input
RSR

Output
1
"""
        parsed = parse_prompt_examples(prompt)
        self.assertEqual(len(parsed), 3)
        self.assertEqual(parsed[1]["input"], "SSS")
        self.assertEqual(parsed[1]["output"], "0")

    def test_build_prompt_example_sample_from_row(self):
        row_data = {
            "input": """Task

Example
Input:
1 2
Output:
3
"""
        }
        sample = build_prompt_example_sample(row_data)
        self.assertEqual(sample["inputs"], ["1 2"])
        self.assertEqual(sample["outputs"], ["3"])
        self.assertEqual(sample["source"], "prompt_examples")

    def test_parse_examples_stops_before_constraints_section(self):
        prompt = """Race cars.

Example
Input:
3
1
10
3
8 3 6
5
4 5 1 2 3

Output:
1
2
2

Constraints
1 <= T <= 100
"""
        parsed = parse_prompt_examples(prompt)
        self.assertEqual(parsed, [{
            "input": "3\n1\n10\n3\n8 3 6\n5\n4 5 1 2 3",
            "output": "1\n2\n2",
        }])


class PromptExampleJudgeTests(unittest.TestCase):

    def test_prompt_examples_judge_passes(self):
        row_data = {
            "input": """Add two integers.

Example
Input:
1 2

Output:
3
""",
            "solution": "a, b = map(int, input().split())\nprint(a + b)\n",
        }
        messages = [{
            "role": "assistant",
            "content": "```python\na, b = map(int, input().split())\nprint(a + b)\n```",
        }]

        result = judge_output(row_data, messages)

        self.assertEqual(result["judge_type"], "code_prompt_examples")
        self.assertEqual(result["judge_backend"], "prompt_examples_v1")
        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")
        self.assertEqual(result["judge_detail"]["example_count"], 1)
        self.assertEqual(result["judge_detail"]["oracle_status"], "pass")

    def test_prompt_examples_judge_reports_wrong_answer(self):
        row_data = {
            "input": """Add two integers.

Example
Input:
1 2

Output:
3
""",
            "solution": "a, b = map(int, input().split())\nprint(a + b)\n",
        }
        messages = [{
            "role": "assistant",
            "content": "```python\na, b = map(int, input().split())\nprint(a - b)\n```",
        }]

        result = judge_output(row_data, messages)

        self.assertFalse(result["is_correct"])
        self.assertEqual(result["judge_status"], "wrong_answer")

    def test_prompt_examples_judge_skips_unverified_examples(self):
        row_data = {
            "input": """Add two integers.

Example
Input:
1 2

Output:
999
""",
            "solution": "a, b = map(int, input().split())\nprint(a + b)\n",
        }
        messages = [{
            "role": "assistant",
            "content": "```python\na, b = map(int, input().split())\nprint(a + b)\n```",
        }]

        result = judge_output(row_data, messages)

        self.assertIsNone(result["is_correct"])
        self.assertEqual(result["judge_status"], "not_applicable")
        self.assertEqual(result["judge_detail"]["oracle_status"],
                         "wrong_answer")


if __name__ == "__main__":
    unittest.main()
