# Round-Robin Scheduling And HF Upload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add manifest-level round-robin task scheduling, periodic merge, and incremental Hugging Face upload for merged correct shards.

**Architecture:** Keep the generation pipeline unchanged and add scheduling in the CLI orchestration layer. Extend the writer path with periodic merge triggers and a resumable upload state file so merge/upload can run incrementally during long jobs.

**Tech Stack:** Python, asyncio, pyarrow, huggingface_hub, unittest

---

### Task 1: Lock Scheduling Behavior With Tests

**Files:**
- Modify: `tests/test_cli.py`
- Create: `tests/test_round_robin_scheduler.py`

- [ ] Add config parsing assertions for new scheduler and upload fields.
- [ ] Add a round-robin test that verifies chunked sample-limit progression.
- [ ] Add a round-robin test that verifies unbounded tasks stop after input exhaustion.

### Task 2: Implement Manifest-Level Round Robin

**Files:**
- Modify: `distill/cli.py`
- Modify: `distill/runtime/settings.py`

- [ ] Add new config fields for scheduling and upload behavior.
- [ ] Add CLI argument parsing and YAML propagation.
- [ ] Add a round-robin runner that reuses `DistillPipeline.run()` summaries.

### Task 3: Lock Periodic Merge And Upload Behavior With Tests

**Files:**
- Create: `tests/test_pipeline_periodic_upload.py`

- [ ] Add a writer-daemon test proving periodic merge/upload runs before shutdown.
- [ ] Add an upload-state test proving only new correct shards are uploaded.

### Task 4: Implement Periodic Merge And Incremental HF Upload

**Files:**
- Modify: `distill/core/pipeline.py`

- [ ] Return run summaries needed by the manifest scheduler.
- [ ] Add periodic merge triggering based on write count.
- [ ] Add local HF upload state tracking.
- [ ] Upload only `correct` shards to a single remote directory level.

### Task 5: Update Examples And Verify

**Files:**
- Modify: `manifest/four_datasets_distill.yaml`
- Modify: `README.md`

- [ ] Add manifest placeholders for round-robin and HF upload config.
- [ ] Document serial vs round-robin task scheduling.
- [ ] Document periodic merge and correct-shard upload fields.
- [ ] Run targeted tests and full unittest discovery.
