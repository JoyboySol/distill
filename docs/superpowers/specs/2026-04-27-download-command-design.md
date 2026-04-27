# Download Command Design

**Context**

The repository now has a unified `distill` CLI, but downloading Hugging Face
models and datasets still relies on an external shell utility, `hfd`. The user
already has a working local `hfd` installation and wants that capability
available from the project CLI, including manifest-driven batch execution.

## Goals

- Add `distill download` as a first-class CLI subcommand.
- Support both direct command-line arguments and manifest-driven task bundles.
- Reuse existing manifest resolution patterns: `--config`, `--config-name`,
  `--task`, and `--manifest-dir`.

## Decisions

- The command will shell out to `hfd` via `subprocess.run`.
- Download tasks use `local_dir` as the preferred manifest field for the target
  directory and fall back to `output_dir` when `local_dir` is absent.
- The repo identifier field is `repo_id`.
- Include/exclude patterns accept either YAML lists or CLI repeated values.
- A task may optionally set:
  `repo_id`, `local_dir`, `output_dir`, `include`, `exclude`, `hf_username`,
  `hf_token`, `tool`, `threads`, `jobs`, `dataset`, `revision`,
  `hfd_command`.

## Constraints

- The command must not break the existing root CLI routing.
- Tests should not depend on a real `hfd` binary or network access.
