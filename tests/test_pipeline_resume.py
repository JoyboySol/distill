import json
import tempfile
import unittest
from pathlib import Path

from distill.core.pipeline import DistillPipeline
from distill.runtime.settings import PipelineConfig


class ResumeIndexTests(unittest.TestCase):

    def _build_pipeline(self, output_dir: str) -> DistillPipeline:
        config = PipelineConfig(
            input_dir="in",
            output_dir=output_dir,
            failure_log=str(Path(output_dir) / "failures.jsonl"),
            model_name="test-model",
            api_key="EMPTY",
            base_urls=["http://127.0.0.1:20001/v1"],
            max_concurrency=4,
        )
        return DistillPipeline(config)

    def test_append_completed_index_writes_tsv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = self._build_pipeline(tmpdir)
            pipeline._append_completed_index([
                {
                    "source_file": "/data/train-0001.parquet",
                    "source_row": 7,
                    "rollout_index": 2,
                }
            ])

            path = Path(tmpdir) / ".resume" / "completed_index.jsonl"
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                "/data/train-0001.parquet\t7\t2\n",
            )

    def test_load_completed_index_filters_to_current_input_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = self._build_pipeline(tmpdir)
            index_path = Path(tmpdir) / ".resume" / "completed_index.jsonl"
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.write_text(
                "/data/train-0001.parquet\t7\t0\n"
                "/data/train-0002.parquet\t8\t0\n",
                encoding="utf-8",
            )

            completed = pipeline._load_completed_index(
                {"/data/train-0002.parquet"})

            self.assertEqual(completed,
                             {"/data/train-0002.parquet::8::0"})

    def test_load_completed_index_supports_legacy_json_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = self._build_pipeline(tmpdir)
            index_path = Path(tmpdir) / ".resume" / "completed_index.jsonl"
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.write_text(
                json.dumps({
                    "source_file": "/data/train-legacy.parquet",
                    "source_row": 9,
                    "rollout_index": 1,
                }) + "\n",
                encoding="utf-8",
            )

            completed = pipeline._load_completed_index(
                {"/data/train-legacy.parquet"})

            self.assertEqual(completed,
                             {"/data/train-legacy.parquet::9::1"})

    def test_opencode_reasoning_prompt_is_wrapped_for_python_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = self._build_pipeline(tmpdir)
            pipeline.config.task_name = "opencode_reasoning_split_0"

            formatted = pipeline._format_task_prompt("Original problem body")

            self.assertIn("Solve this problem using Python.", formatted)
            self.assertTrue(formatted.endswith("Original problem body"))

    def test_non_opencode_reasoning_prompt_is_left_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = self._build_pipeline(tmpdir)
            pipeline.config.task_name = "math_task"

            formatted = pipeline._format_task_prompt("Original problem body")

            self.assertEqual(formatted, "Original problem body")


if __name__ == "__main__":
    unittest.main()
