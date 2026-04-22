import tempfile
import unittest
from pathlib import Path

from distill.cli import DEFAULT_PIPELINE_VALUES, _build_config_from_values, \
    _normalize_config_keys, build_configs, build_parser


class CliConfigTests(unittest.TestCase):

    def test_normalize_config_keys_accepts_llm_timeout_alias(self):
        normalized = _normalize_config_keys({
            "llm-timeout": 7200,
            "vllm-ls-command": "ps -eo args=",
        })

        self.assertEqual(normalized["llm_timeout"], 7200)
        self.assertEqual(normalized["vllm_ls_command"], "ps -eo args=")

    def test_build_config_propagates_llm_timeout(self):
        values = dict(DEFAULT_PIPELINE_VALUES)
        values.update({
            "ports": "1597-1598",
            "llm_timeout": 7200,
            "sample_limit": 12,
            "judge_timeout_sec": 45,
            "vllm_ls_command": "ps -eo args=",
            "judge_mode": "none",
            "task_schedule": "round_robin",
            "round_robin_chunk_size": 500,
            "complete_trailing_user_turn": True,
            "merge_every_n_writes": 1000,
            "upload_every_n_merges": 2,
            "upload_merged_shards": True,
            "treat_no_judge_as_correct": True,
            "hf_repo_id": "JoyboyGo/hf_data",
            "hf_remote_prefix": "distill",
        })

        config = _build_config_from_values(values)

        self.assertEqual(config.llm_timeout, 7200)
        self.assertEqual(config.sample_limit, 12)
        self.assertEqual(config.judge_timeout_sec, 45)
        self.assertEqual(config.vllm_ls_command, "ps -eo args=")
        self.assertEqual(config.judge_mode, "none")
        self.assertEqual(config.task_schedule, "round_robin")
        self.assertEqual(config.round_robin_chunk_size, 500)
        self.assertTrue(config.complete_trailing_user_turn)
        self.assertEqual(config.merge_every_n_writes, 1000)
        self.assertEqual(config.upload_every_n_merges, 2)
        self.assertTrue(config.upload_merged_shards)
        self.assertTrue(config.treat_no_judge_as_correct)
        self.assertEqual(config.hf_repo_id, "JoyboyGo/hf_data")
        self.assertEqual(config.hf_remote_prefix, "distill")
        self.assertEqual(config.base_urls, [
            "http://127.0.0.1:1597/v1",
            "http://127.0.0.1:1598/v1",
        ])

    def test_manifest_upload_flag_is_not_overridden_when_cli_flag_is_absent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "task.yaml"
            config_path.write_text(
                "\n".join([
                    "task_name: test_task",
                    "input_dir: /tmp/in",
                    "output_dir: /tmp/out",
                    "failure_log: test_failed.jsonl",
                    "ports: 1597",
                    "upload_merged_shards: true",
                    "judge_mode: none",
                ]),
                encoding="utf-8",
            )

            parser = build_parser()
            args = parser.parse_args(["--config", str(config_path)])

            config = build_configs(args)[0]

            self.assertTrue(config.upload_merged_shards)
            self.assertEqual(config.judge_mode, "none")


if __name__ == "__main__":
    unittest.main()
