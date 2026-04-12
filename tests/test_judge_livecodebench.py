import json
import unittest

from distill.core.judge import (extract_code_text_last_block, judge_output,
                                judge_output_with_timeout)


class LiveCodeBenchJudgeTests(unittest.TestCase):

    def test_call_based_livecodebench_passes(self):
        row_data = {
            "input_output": json.dumps({
                "inputs": ["1\n2", "5\n7"],
                "outputs": ["3", "12"],
                "fn_name": "add",
            })
        }
        messages = [{
            "role": "assistant",
            "content": "```python\ndef add(a, b):\n    return a + b\n```",
        }]

        result = judge_output(row_data, messages)

        self.assertEqual(result["judge_type"],
                         "code_livecodebench_generation")
        self.assertEqual(result["judge_backend"], "livecodebench_v6")
        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")
        self.assertEqual(result["judge_detail"]["passed_tests"], 2)

    def test_call_based_livecodebench_reports_wrong_answer(self):
        row_data = {
            "input_output": json.dumps({
                "inputs": ["1\n2", "5\n7"],
                "outputs": ["3", "12"],
                "fn_name": "add",
            })
        }
        messages = [{
            "role": "assistant",
            "content": "```python\ndef add(a, b):\n    return a - b\n```",
        }]

        result = judge_output(row_data, messages)

        self.assertFalse(result["is_correct"])
        self.assertEqual(result["judge_status"], "wrong_answer")
        self.assertEqual(result["judge_detail"]["error_message"],
                         "Wrong Answer")

    def test_standard_input_livecodebench_passes(self):
        row_data = {
            "input_output": json.dumps({
                "inputs": ["1 2\n", "5 7\n"],
                "outputs": ["3", "12"],
                "fn_name": None,
            })
        }
        messages = [{
            "role": "assistant",
            "content": "```python\na, b = map(int, input().split())\nprint(a + b)\n```",
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")
        self.assertEqual(result["judge_detail"]["mode"], "standard_input")

    def test_public_private_tests_are_supported(self):
        row_data = {
            "public_test_cases": json.dumps([{
                "input": "2\n3",
                "output": "6",
            }]),
            "private_test_cases": json.dumps([{
                "input": "4\n5",
                "output": "20",
            }]),
            "metadata": json.dumps({
                "func_name": "mul"
            }),
        }
        messages = [{
            "role": "assistant",
            "content": "```python\ndef mul(a, b):\n    return a * b\n```",
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")
        self.assertEqual(result["judge_detail"]["test_source"],
                         "public_private_test_cases")

    def test_timeout_guard_matches_direct_judge_for_simple_case(self):
        row_data = {
            "input_output": json.dumps({
                "inputs": ["1\n2", "5\n7"],
                "outputs": ["3", "12"],
                "fn_name": "add",
            })
        }
        messages = [{
            "role": "assistant",
            "content": "```python\ndef add(a, b):\n    return a + b\n```",
        }]

        direct = judge_output(row_data, messages)
        guarded = judge_output_with_timeout(row_data,
                                            messages,
                                            timeout=20)

        self.assertEqual(guarded, direct)

    def test_existing_mbpp_path_keeps_priority(self):
        row_data = {
            "input_output": json.dumps({
                "inputs": ["1\n2"],
                "outputs": ["3"],
                "fn_name": "add",
            }),
            "test_list": ["assert add(1, 2) == 3"],
        }
        messages = [{
            "role": "assistant",
            "content": "```python\ndef add(a, b):\n    return a + b\n```",
        }]

        result = judge_output(row_data, messages)

        self.assertEqual(result["judge_type"], "code_mbpp")
        self.assertTrue(result["is_correct"])

    def test_extract_code_prefers_python_block_over_later_non_python_block(self):
        content = """Explanation

```python
a, b = map(int, input().split())
print(a + b)
```

Sample output
```
3
```
"""
        extracted = extract_code_text_last_block(content)
        self.assertIn("print(a + b)", extracted)
        self.assertNotEqual(extracted.strip(), "3")

    def test_extract_code_prefers_python_like_unlabeled_block(self):
        content = """Explanation

```
a, b = map(int, input().split())
print(a + b)
```

```
3
```
"""
        extracted = extract_code_text_last_block(content)
        self.assertIn("print(a + b)", extracted)
        self.assertNotEqual(extracted.strip(), "3")

    def test_code_task_without_reliable_tests_does_not_fall_back_to_math(self):
        row_data = {
            "dataset": "code_contests",
            "input": "Write a program. Input: n. Output: answer.",
            "solution": "import sys\nn = int(input())\nprint(n)\n",
        }
        messages = [{
            "role": "assistant",
            "content": "We read n and print it.\n```python\nn = int(input())\nprint(n)\n```",
        }]

        result = judge_output(row_data, messages)

        self.assertEqual(result["judge_type"], "code_unverified")
        self.assertEqual(result["judge_backend"], "code_safeguard_v1")
        self.assertIsNone(result["is_correct"])
        self.assertEqual(result["judge_status"], "not_applicable")


if __name__ == "__main__":
    unittest.main()
