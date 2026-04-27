# Unified CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the pipeline and common Python utilities behind a single
`distill` CLI while keeping old script entrypoints as compatibility shims.

**Architecture:** Keep the existing pipeline parser and runtime behavior in a
dedicated `run` command module, then add a root dispatcher that routes explicit
subcommands and still falls back to legacy pipeline invocation when users pass
old-style flags directly. Migrate standalone scripts into packaged command
modules and leave the old script paths as thin forwarders.

**Tech Stack:** Python, argparse, unittest, setuptools

---

### Task 1: Lock Root CLI Behavior With Tests

**Files:**
- Modify: `tests/test_cli.py`

- [ ] Add a test that the root parser recognizes explicit subcommands such as
  `run` and `stats`.
- [ ] Add a test that `distill.cli.main([...])` routes old-style pipeline args
  to the run command for backward compatibility.
- [ ] Add a test that `distill.cli.main(["list-configs", ...])` dispatches to
  the list-configs handler.

### Task 2: Extract The Pipeline Run Command

**Files:**
- Create: `distill/commands/__init__.py`
- Create: `distill/commands/run.py`
- Modify: `distill/cli.py`

- [ ] Move the existing pipeline parser/config builder/run logic into
  `distill/commands/run.py`.
- [ ] Re-export compatible helpers from `distill/cli.py` so current imports in
  tests and callers keep working.
- [ ] Add a root parser and dispatch flow in `distill/cli.py`.

### Task 3: Package Common Tool Commands

**Files:**
- Create: `distill/commands/*.py`
- Modify: `distill/tools/stats.py`

- [ ] Move each script-backed command into a packaged module with
  `build_parser()`, `run_namespace()`, and `main()` helpers.
- [ ] Point `stats` at the packaged implementation so the unified CLI and
  module entrypoint share logic.

### Task 4: Add Compatibility Shims And Packaging Entry Point

**Files:**
- Modify: `scripts/*.py`
- Create: `pyproject.toml`

- [ ] Replace old scripts with thin forwarders to the packaged command modules.
- [ ] Emit a deprecation warning from each compatibility shim.
- [ ] Add a console-script entrypoint for `distill`.

### Task 5: Update Docs And Verify

**Files:**
- Modify: `README.md`

- [ ] Document the new `distill <subcommand>` workflow and legacy compatibility.
- [ ] Run targeted CLI tests and command smoke checks.
