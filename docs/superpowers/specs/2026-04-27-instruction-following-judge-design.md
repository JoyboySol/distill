# Instruction-Following Judge Design

## Goal

Add an instruction-following judge for the repository's verifiable instruction
datasets while restructuring judge internals so future judge additions do not
keep growing `distill/core/judge.py` as a monolith.

## Scope

This change keeps the public judge API stable:

- `judge_output(row_data, messages, label_field=None)`
- `judge_output_with_timeout(row_data, messages, label_field=None, timeout=...)`

The pipeline, manifests, and saved output schema remain unchanged.

## Current Problem

`distill/core/judge.py` currently owns:

- result dispatch
- code extraction helpers
- math normalization and verification
- MCQ parsing
- timeout subprocess guard
- task-type detection

That coupling makes it awkward to add a new judge family and hard to test each
domain in isolation.

## Design

### 1. Thin public entrypoint

Keep `distill/core/judge.py` as the stable public module, but reduce it to:

- imports/re-exports
- shared timeout wrapper
- the top-level dispatcher call

### 2. Split by responsibility

Create a small judge package under `distill/core/judges/`:

- `shared.py`
  Common helpers for message normalization, assistant text extraction, and
  result helpers.
- `code.py`
  Code-task detection and existing MBPP/HumanEval/LiveCodeBench/prompt-example
  logic.
- `math.py`
  Math postprocessing and `math_verify` fallback flow.
- `mcq.py`
  Choice prompt detection and boxed/freeform option extraction.
- `instruction_following.py`
  New verifiable instruction-following logic.
- `dispatcher.py`
  Ordered judge dispatch and judge-type hint logic for timeout fallbacks.

### 3. New instruction-following judge

The new judge targets two dataset families:

- `allenai/IF_multi_constraints_upto5`
- `allenai/tulu-3-sft-personas-instruction-following`

It will only emit a correctness verdict when all requested constraints are
supported by the verifier. Otherwise it returns:

- `judge_type = instruction_following`
- `judge_backend = instruction_following_v1`
- `is_correct = None`
- `judge_status = not_applicable`

This avoids silently scoring partially-verified samples as correct.

### 4. Constraint sources

The judge supports two input modes:

- Structured mode for IF-style rows with `ground_truth`
- Prompt-derived mode for Tulu-style rows with `constraints` plus explicit prompt
  wording

Structured mode is preferred because it is higher confidence.

### 5. Initial supported constraint families

The first version focuses on high-value, deterministic checks:

- keyword include / exclude / once
- first word / last word
- repeat prompt / repeat phrase / copy span
- punctuation bans for common marks
- lowercase / uppercase word frequency
- number of paragraphs
- placeholder count
- quotation wrapping/presence
- title/header presence
- bullet-list count
- sentence-hyphen format
- specific ending phrase
- two-response separation when explicitly requested

Unsupported or weakly-specified constraints remain `not_applicable`.

### 6. Testing strategy

Add tests at two levels:

- domain-level unit tests for the new instruction-following verifier
- regression tests confirming existing code/MCQ/prompt-example behavior still
  routes correctly through the refactored dispatcher

## Non-Goals

- No manifest schema changes
- No saved output schema changes
- No attempt to fully replicate every IFEval constraint in one pass
- No external service dependencies for judging
