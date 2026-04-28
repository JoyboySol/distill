import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from distill.commands.eval_existing_outputs import evaluate_rows, iter_rows


class EvalExistingOutputsTests(unittest.TestCase):

    def _write_rows(self, rows):
        tmpdir = tempfile.TemporaryDirectory()
        path = Path(tmpdir.name) / "sample.parquet"
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path)
        return tmpdir, path

    def test_iter_rows_keeps_instruction_following_columns(self):
        tmpdir, _ = self._write_rows([{
            "id": "if-1",
            "output": "short answer brief",
            "ground_truth":
            ("[{'instruction_id': ['last_word:last_word_answer'], "
             "'kwargs': [{'last_word': 'brief'}]}]"),
            "constraints": ["specific ending"],
            "prompt": "End your response with \"brief\".",
        }])
        self.addCleanup(tmpdir.cleanup)

        rows = list(iter_rows(Path(tmpdir.name), limit=10))

        self.assertEqual(len(rows), 1)
        _, _, row = rows[0]
        self.assertIn("ground_truth", row)
        self.assertIn("constraints", row)
        self.assertEqual(row["ground_truth"],
                         "[{'instruction_id': ['last_word:last_word_answer'], "
                         "'kwargs': [{'last_word': 'brief'}]}]")

    def test_evaluate_rows_supports_instruction_following_judge(self):
        tmpdir, _ = self._write_rows([{
            "id": "if-1",
            "output": "short answer brief",
            "ground_truth":
            ("[{'instruction_id': ['last_word:last_word_answer'], "
             "'kwargs': [{'last_word': 'brief'}]}]"),
            "dataset": "ifeval",
        }])
        self.addCleanup(tmpdir.cleanup)

        summary = evaluate_rows(Path(tmpdir.name), limit=10)

        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["judge_type_counter"],
                         {"instruction_following": 1})
        self.assertEqual(summary["backend_counter"],
                         {"instruction_following_v1": 1})
        self.assertEqual(summary["status_counter"], {"pass": 1})


if __name__ == "__main__":
    unittest.main()
