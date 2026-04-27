# Download Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `distill download` command that wraps the local `hfd` tool and supports manifest-based batch downloads.

**Architecture:** Implement the download wrapper as a packaged command module under `distill/commands/`, following the same manifest resolution style as the existing run command. Build shell arguments from normalized task values and execute `hfd` with `subprocess.run`, while tests patch the subprocess layer to verify behavior without real downloads.

**Tech Stack:** Python, argparse, subprocess, unittest, YAML manifests

---

### Task 1: Lock Download Behavior With Tests

**Files:**
- Create: `tests/test_download.py`
- Modify: `tests/test_cli.py`

- [ ] Add a failing test for direct CLI argument parsing and `hfd` command construction.
- [ ] Add a failing test for manifest-driven batch download dispatch.
- [ ] Add a failing root CLI test that accepts the `download` subcommand.

### Task 2: Implement The Packaged Download Command

**Files:**
- Create: `distill/commands/download.py`
- Modify: `distill/cli.py`

- [ ] Add parser builders and task normalization helpers.
- [ ] Add direct and manifest-driven execution flows.
- [ ] Register `download` in the root CLI.

### Task 3: Update Docs And Verify

**Files:**
- Modify: `README.md`

- [ ] Document direct and manifest-based usage for `distill download`.
- [ ] Run targeted tests and command help verification.
