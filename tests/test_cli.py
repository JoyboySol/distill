import unittest

from distill.cli import DEFAULT_PIPELINE_VALUES, _build_config_from_values, \
    _normalize_config_keys


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
            "vllm_ls_command": "ps -eo args=",
        })

        config = _build_config_from_values(values)

        self.assertEqual(config.llm_timeout, 7200)
        self.assertEqual(config.vllm_ls_command, "ps -eo args=")
        self.assertEqual(config.base_urls, [
            "http://127.0.0.1:1597/v1",
            "http://127.0.0.1:1598/v1",
        ])


if __name__ == "__main__":
    unittest.main()
