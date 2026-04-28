import unittest
from unittest.mock import patch

from distill.commands.rejudge_saved_outputs import rejudge_record


class _FakeRowCache:

    def __init__(self, row_data):
        self._row_data = row_data

    def get_row(self, source_file: str, source_row: int):
        return self._row_data


class RejudgeSavedOutputsTests(unittest.TestCase):

    def test_force_pass_if_marks_instruction_following_record_correct(self):
        record = {
            "source_file": "/tmp/source.parquet",
            "source_row": 0,
            "judge_type": "instruction_following",
            "judge_backend": "instruction_following_v1",
            "judge_status": "wrong_answer",
            "is_correct": False,
            "judge_detail": {
                "reason": "old"
            },
            "messages": [{
                "role": "assistant",
                "content": "final answer summary",
            }],
        }
        row_cache = _FakeRowCache({
            "ground_truth":
            ("[{'instruction_id': ['last_word:last_word_answer'], "
             "'kwargs': [{'last_word': 'brief'}]}]"),
            "dataset": "ifeval",
        })

        changed, updated = rejudge_record(record,
                                          row_cache,
                                          force_pass_if=True)

        self.assertTrue(changed)
        self.assertEqual(updated["judge_type"], "instruction_following")
        self.assertEqual(updated["judge_backend"],
                         "instruction_following_force_pass_v1")
        self.assertEqual(updated["judge_status"], "pass")
        self.assertTrue(updated["is_correct"])
        self.assertTrue(updated["judge_detail"]["force_passed"])

    def test_force_pass_if_leaves_non_if_records_on_normal_judge_path(self):
        record = {
            "source_file": "/tmp/source.parquet",
            "source_row": 0,
            "judge_type": "mcq",
            "judge_backend": "mcq_v1",
            "judge_status": "wrong_answer",
            "is_correct": False,
            "judge_detail": None,
            "messages": [{
                "role": "assistant",
                "content": "B",
            }],
        }
        row_cache = _FakeRowCache({
            "answer": "A",
            "output": "B",
        })

        with patch("distill.commands.rejudge_saved_outputs.judge_output",
                   return_value={
                       "judge_type": "mcq",
                       "judge_backend": "mcq_v1",
                       "judge_status": "wrong_answer",
                       "is_correct": False,
                       "judge_detail": {
                           "source": "mock"
                       },
                   }) as mock_judge:
            changed, updated = rejudge_record(
                record,
                row_cache,
                judge_mode="math",
                force_pass_if=True,
            )

        self.assertFalse(changed)
        self.assertEqual(updated["judge_type"], "mcq")
        self.assertEqual(mock_judge.call_args.kwargs["judge_mode"], "math")

    def test_force_pass_if_does_not_mark_overlong_record_correct(self):
        record = {
            "source_file": "/tmp/source.parquet",
            "source_row": 0,
            "judge_type": "instruction_following",
            "judge_backend": "instruction_following_v1",
            "judge_status": "wrong_answer",
            "is_correct": False,
            "judge_detail": {
                "reason": "old"
            },
            "generation_finish_reason": "length",
            "messages": [{
                "role": "assistant",
                "content": "truncated answer",
            }],
        }
        row_cache = _FakeRowCache({
            "ground_truth":
            ("[{'instruction_id': ['last_word:last_word_answer'], "
             "'kwargs': [{'last_word': 'brief'}]}]"),
            "dataset": "ifeval",
        })

        changed, updated = rejudge_record(record,
                                          row_cache,
                                          force_pass_if=True)

        self.assertTrue(changed)
        self.assertEqual(updated["judge_type"], "instruction_following")
        self.assertEqual(updated["judge_backend"],
                         "instruction_following_force_pass_v1")
        self.assertEqual(updated["judge_status"], "not_applicable_overlong")
        self.assertIsNone(updated["is_correct"])
        self.assertTrue(updated["judge_detail"]["overlong_blocked"])
