import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from distill.commands.profile_manifest_inputs import \
    summarize_manifest_input_profiles


def _write_parquet_rows(root: Path, rows: list[dict]) -> Path:
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "train-00000-of-00001.parquet"
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    return path


class ProfileManifestInputsTests(unittest.TestCase):

    def test_summarize_manifest_input_profiles_reports_constraint_distribution_and_supportability(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "task_in"
            _write_parquet_rows(
                input_dir,
                [
                    {
                        "prompt": (
                            "Write a short note about Alberta energy policy. "
                            "Include the keywords 'pipeline' and 'tax'."
                        ),
                        "constraints": ["include keywords"],
                    },
                    {
                        "prompt": (
                            "Write a 100-word manifesto in JSON format about "
                            "climate action."
                        ),
                        "constraints": ["format:use json format"],
                    },
                    {
                        "prompt": "Plain prompt with no structured constraints.",
                        "constraints": None,
                    },
                ],
            )

            manifest_path = root / "batch.yaml"
            manifest_path.write_text(
                "\n".join([
                    "tasks:",
                    "  - task_name: tulu_profiled",
                    f"    input_dir: {input_dir}",
                    "    file_pattern: data/train-*.parquet",
                    "    input_field: prompt",
                    "    judge_mode: instruction_following",
                    "",
                ]),
                encoding="utf-8",
            )

            rows = summarize_manifest_input_profiles(str(manifest_path))

            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0],
                {
                    "task_name": "tulu_profiled",
                    "input_dir": str(input_dir),
                    "file_pattern": "data/train-*.parquet",
                    "file_count": 1,
                    "estimated_row_count": 3,
                    "field_presence": {
                        "prompt": 3,
                        "constraints": 2,
                        "ground_truth": 0,
                    },
                    "constraint_counts": {
                        "include keywords": 1,
                        "format:use json format": 1,
                    },
                    "constraint_combo_counts": {
                        "include keywords": 1,
                        "format:use json format": 1,
                    },
                    "supportability": {
                        "supported_rows": 2,
                        "unsupported_rows": 0,
                        "no_constraint_rows": 1,
                    },
                },
            )

    def test_task_name_filter_selects_single_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_a = root / "task_a_in"
            input_b = root / "task_b_in"
            _write_parquet_rows(
                input_a,
                [{
                    "prompt": "Include the keywords 'north' and 'south'.",
                    "constraints": ["include keywords"],
                }],
            )
            _write_parquet_rows(
                input_b,
                [{
                    "prompt": "Use JSON.",
                    "constraints": ["format:use json format"],
                }],
            )

            manifest_path = root / "batch.yaml"
            manifest_path.write_text(
                "\n".join([
                    "tasks:",
                    "  - task_name: task_a",
                    f"    input_dir: {input_a}",
                    "    file_pattern: data/train-*.parquet",
                    "  - task_name: task_b",
                    f"    input_dir: {input_b}",
                    "    file_pattern: data/train-*.parquet",
                    "",
                ]),
                encoding="utf-8",
            )

            rows = summarize_manifest_input_profiles(
                str(manifest_path),
                task_name="task_b",
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["task_name"], "task_b")
            self.assertEqual(rows[0]["estimated_row_count"], 1)
