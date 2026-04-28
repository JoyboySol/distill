import tempfile
import unittest
from pathlib import Path

from distill.cli import run_resolved_configs
from distill.runtime.settings import PipelineConfig


class RoundRobinSchedulerTests(unittest.TestCase):

    def _build_config(self,
                      task_name: str,
                      sample_limit,
                      task_schedule: str = "round_robin",
                      chunk_size: int = 2) -> PipelineConfig:
        tmpdir = tempfile.mkdtemp()
        return PipelineConfig(
            input_dir=str(Path(tmpdir) / "in"),
            output_dir=str(Path(tmpdir) / "out" / task_name),
            failure_log=str(Path(tmpdir) / f"{task_name}.jsonl"),
            model_name="test-model",
            api_key="EMPTY",
            base_urls=["http://127.0.0.1:20001/v1"],
            task_name=task_name,
            sample_limit=sample_limit,
            task_schedule=task_schedule,
            round_robin_chunk_size=chunk_size,
        )

    def test_round_robin_increases_sample_limit_in_chunks(self):
        configs = [
            self._build_config("task_a", 5, chunk_size=2),
            self._build_config("task_b", 3, chunk_size=2),
        ]
        calls = []

        class _FakePipeline:

            def __init__(self, config):
                self.config = config

            async def run(self):
                calls.append((self.config.task_name, self.config.sample_limit))
                return {"input_exhausted": False}

        run_resolved_configs(configs, pipeline_cls=_FakePipeline)

        self.assertEqual(calls, [
            ("task_a", 2),
            ("task_b", 2),
            ("task_a", 4),
            ("task_b", 3),
            ("task_a", 5),
        ])

    def test_round_robin_stops_unbounded_task_after_input_exhausts(self):
        configs = [
            self._build_config("task_a", None, chunk_size=2),
        ]
        calls = []

        class _FakePipeline:

            def __init__(self, config):
                self.config = config

            async def run(self):
                calls.append((self.config.task_name, self.config.sample_limit))
                return {
                    "input_exhausted": int(self.config.sample_limit or 0) >= 4
                }

        run_resolved_configs(configs, pipeline_cls=_FakePipeline)

        self.assertEqual(calls, [
            ("task_a", 2),
            ("task_a", 4),
        ])

    def test_round_robin_stops_all_scheduling_after_interrupt(self):
        configs = [
            self._build_config("task_a", 6, chunk_size=2),
            self._build_config("task_b", 6, chunk_size=2),
        ]
        calls = []

        class _FakePipeline:

            def __init__(self, config):
                self.config = config

            async def run(self):
                calls.append((self.config.task_name, self.config.sample_limit))
                if self.config.task_name == "task_a":
                    return {"input_exhausted": False, "interrupted": True}
                return {"input_exhausted": False, "interrupted": False}

        run_resolved_configs(configs, pipeline_cls=_FakePipeline)

        self.assertEqual(calls, [
            ("task_a", 2),
        ])

    def test_serial_stops_scheduling_after_interrupt(self):
        configs = [
            self._build_config("task_a", 6, task_schedule="serial"),
            self._build_config("task_b", 6, task_schedule="serial"),
        ]
        calls = []

        class _FakePipeline:

            def __init__(self, config):
                self.config = config

            async def run(self):
                calls.append((self.config.task_name, self.config.sample_limit))
                if self.config.task_name == "task_a":
                    return {"input_exhausted": False, "interrupted": True}
                return {"input_exhausted": False, "interrupted": False}

        run_resolved_configs(configs, pipeline_cls=_FakePipeline)

        self.assertEqual(calls, [
            ("task_a", 6),
        ])


if __name__ == "__main__":
    unittest.main()
