import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from distill.commands import download


class DownloadCommandTests(unittest.TestCase):

    def test_direct_download_builds_expected_hfd_command(self):
        parser = download.build_parser()
        args = parser.parse_args([
            "--repo-id",
            "org/example-dataset",
            "--local-dir",
            "/tmp/download-target",
            "--include",
            "*.parquet",
            "train/*",
            "--exclude",
            "*.md",
            "--dataset",
            "--tool",
            "wget",
            "-x",
            "3",
            "-j",
            "4",
            "--revision",
            "dev",
            "--hf-username",
            "alice",
            "--hf-token",
            "secret",
            "--hfd-command",
            "/usr/local/bin/hfd",
        ])

        with patch("distill.commands.download.subprocess.run") as mock_run:
            summary = download.run_namespace(args)

        mock_run.assert_called_once_with([
            "/usr/local/bin/hfd",
            "org/example-dataset",
            "--include",
            "*.parquet",
            "train/*",
            "--exclude",
            "*.md",
            "--hf_username",
            "alice",
            "--hf_token",
            "secret",
            "--tool",
            "wget",
            "-x",
            "3",
            "-j",
            "4",
            "--dataset",
            "--local-dir",
            "/tmp/download-target",
            "--revision",
            "dev",
        ], check=True)
        self.assertEqual(summary["repo_id"], "org/example-dataset")
        self.assertEqual(summary["local_dir"], "/tmp/download-target")
        self.assertTrue(summary["dataset"])

    def test_manifest_download_uses_local_dir_then_falls_back_to_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "downloads.yaml"
            manifest_path.write_text(
                "\n".join([
                    "tool: aria2c",
                    "threads: 6",
                    "jobs: 7",
                    "dataset: true",
                    "revision: main",
                    "tasks:",
                    "  - task_name: ds_a",
                    "    repo_id: org/a",
                    "    local_dir: /tmp/a_local",
                    "    include:",
                    "      - train/*",
                    "  - task_name: ds_b",
                    "    repo_id: org/b",
                    "    output_dir: /tmp/b_output",
                    "",
                ]),
                encoding="utf-8",
            )

            with patch("distill.commands.download.subprocess.run") as mock_run:
                summaries = download.download_from_manifest(config_path=str(
                    manifest_path))

        self.assertEqual(mock_run.call_count, 2)
        first_call = mock_run.call_args_list[0].args[0]
        second_call = mock_run.call_args_list[1].args[0]
        self.assertIn("--local-dir", first_call)
        self.assertEqual(first_call[first_call.index("--local-dir") + 1],
                         "/tmp/a_local")
        self.assertIn("--local-dir", second_call)
        self.assertEqual(second_call[second_call.index("--local-dir") + 1],
                         "/tmp/b_output")
        self.assertEqual(summaries[0]["task_name"], "ds_a")
        self.assertEqual(summaries[1]["task_name"], "ds_b")
        self.assertEqual(summaries[1]["local_dir"], "/tmp/b_output")


if __name__ == "__main__":
    unittest.main()
