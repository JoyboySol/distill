# Data Profiling And IF Judge Design

## Goal

Add an input-data profiling tool for manifest-backed training datasets and
improve the instruction-following judge so it can cover more high-frequency,
deterministic Tulu prompt constraints.

## Scope

This change intentionally focuses on two areas only:

- raw training-data profiling
- instruction-following judge coverage

It does not change the existing saved-output statistics tool or its current
accuracy fields.

## Current Problem

The repository can summarize generated outputs, but it lacks a first-class way
to inspect the input datasets before distillation. That makes it harder to
answer:

- which constraint families dominate a dataset
- how much of a task the current judge can verify
- which unsupported constraints should be prioritized next

At the same time, the current instruction-following judge supports only a small
subset of prompt-derived Tulu constraints, and some supported prompt parsing is
too narrow or overly permissive.

## Design

### 1. Add a dedicated raw-data profiling command

Create a CLI command that reads manifest task definitions directly from YAML and
profiles the selected input dataset without requiring generated outputs.

The profiler should report:

- task metadata such as `task_name`, `input_dir`, file count, and estimated row
  count
- field availability for columns such as `prompt`, `constraints`, and
  `ground_truth`
- constraint frequency counts
- constraint combination counts
- a judge-supportability estimate based on the current
  `instruction_following.py` parser

The output should be machine-readable JSON so it can be used in scripts and
manual debugging alike.

### 2. Keep the profiling logic separate from saved-output stats

The existing `distill.tools.stats` module should remain unchanged. Profiling is
about understanding source data, not evaluating generated results. The new
logic should live under the command surface and should not couple to shard-based
 output analysis.

### 3. Expand prompt-derived instruction-following coverage

Improve prompt parsing and checking for the Tulu prompt-based verifier, with the
first pass focused on high-frequency, deterministic constraints.

Priority areas:

- fix `include keywords` extraction so apostrophes inside words do not get
  misread as quoted phrases
- support broader `specific ending` prompt phrasings
- support `response language` for explicit English requests
- support `in english and capital`
- support `case:in english and lowercase`
- support `length constraints:number of paragraphs`
- support `length constraints:number of sentences`

### 4. Preserve conservative grading behavior

If any requested constraint is unsupported, the instruction-following judge
should still return `is_correct = None` with `judge_status = not_applicable`.
The improvement is to support more deterministic constraints, not to weaken the
current safety bar.

## Testing Strategy

Add tests in two layers:

- command-level tests for raw-data profiling summaries
- judge unit tests for each newly supported or fixed prompt-derived constraint

The implementation should follow TDD:

- write failing tests
- verify they fail for the intended reason
- implement the smallest change
- rerun targeted tests

## Non-Goals

- No changes to current output-statistics semantics
- No full replication of every Tulu or IFEval constraint in one pass
- No manifest schema changes
- No saved-output schema changes
