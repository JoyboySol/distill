# Data Profiling And IF Judge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a manifest-driven raw-data profiling command and improve prompt-derived instruction-following judge coverage for high-frequency Tulu constraints.

**Architecture:** Keep profiling logic independent from saved-output stats by adding a dedicated command that reads manifest task inputs directly. Expand `instruction_following.py` conservatively by improving prompt parsing and adding deterministic constraint checks while preserving `not_applicable` for unsupported cases.

**Tech Stack:** Python, PyArrow parquet reading, argparse CLI commands, unittest.

---

### Task 1: Add Raw-Data Profiling Command Tests

**Files:**
- Create: `tests/test_profile_manifest_inputs.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Cover:

- a manifest task with parquet inputs and `constraints`
- per-constraint counts and combination counts
- supportability summary derived from the instruction-following parser
- root CLI registration for the new subcommand

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m unittest tests.test_profile_manifest_inputs tests.test_cli -v`

Expected: FAIL because the new command module and CLI wiring do not exist yet.

- [ ] **Step 3: Commit**

```bash
git add tests/test_profile_manifest_inputs.py tests/test_cli.py
git commit -m "test: add raw data profiling command coverage"
```

### Task 2: Implement Raw-Data Profiling Command

**Files:**
- Create: `distill/commands/profile_manifest_inputs.py`
- Modify: `distill/commands/__init__.py`
- Modify: `distill/cli.py`

- [ ] **Step 1: Write minimal implementation**

Implement:

- manifest task loading and optional `task_name` filtering
- parquet/jsonl row estimation
- constraint counts and combination counts
- supportability summary using the current instruction-following parser
- JSON output and optional file save

- [ ] **Step 2: Run targeted tests**

Run: `./.venv/bin/python -m unittest tests.test_profile_manifest_inputs tests.test_cli -v`

Expected: PASS

- [ ] **Step 3: Refactor for clarity if needed**

Keep helpers small and focused:

- row iteration / estimation helpers
- constraint summarization helpers
- supportability summarization helpers

- [ ] **Step 4: Commit**

```bash
git add distill/commands/profile_manifest_inputs.py distill/commands/__init__.py distill/cli.py
git commit -m "feat: add manifest input profiling command"
```

### Task 3: Add Failing Judge Tests For High-Value Prompt Constraints

**Files:**
- Modify: `tests/test_judge_instruction_following.py`

- [ ] **Step 1: Write failing tests**

Add cases for:

- apostrophes not corrupting `include keywords`
- broader `specific ending` prompt phrasing
- `response language` in English
- `in english and capital`
- `case:in english and lowercase`
- `length constraints:number of paragraphs`
- `length constraints:number of sentences`

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m unittest tests.test_judge_instruction_following -v`

Expected: FAIL because the parser/checker does not support all of these cases yet.

- [ ] **Step 3: Commit**

```bash
git add tests/test_judge_instruction_following.py
git commit -m "test: cover prompt-derived instruction constraints"
```

### Task 4: Implement Judge Coverage Improvements

**Files:**
- Modify: `distill/core/judges/instruction_following.py`

- [ ] **Step 1: Implement the smallest parser and checker changes**

Add or improve support for:

- safer quoted-keyword extraction
- broader ending-phrase extraction
- English-language checks
- uppercase and lowercase English constraints
- paragraph counting
- sentence counting

- [ ] **Step 2: Run targeted judge tests**

Run: `./.venv/bin/python -m unittest tests.test_judge_instruction_following -v`

Expected: PASS

- [ ] **Step 3: Run nearby regression tests**

Run: `./.venv/bin/python -m unittest tests.test_judge_routing tests.test_prompt_examples tests.test_pipeline_multiturn -v`

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add distill/core/judges/instruction_following.py
git commit -m "feat: expand instruction-following judge coverage"
```

### Task 5: Final Verification

**Files:**
- Modify: `docs/superpowers/specs/2026-04-28-data-profiling-and-if-judge-design.md`
- Modify: `docs/superpowers/plans/2026-04-28-data-profiling-and-if-judge.md`

- [ ] **Step 1: Run focused verification**

Run:

`./.venv/bin/python -m unittest tests.test_profile_manifest_inputs tests.test_cli tests.test_judge_instruction_following tests.test_judge_routing tests.test_prompt_examples tests.test_pipeline_multiturn -v`

Expected: PASS

- [ ] **Step 2: Record any follow-up gaps**

Document remaining unsupported high-frequency constraints only if they remain
out of scope for this change.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-04-28-data-profiling-and-if-judge-design.md docs/superpowers/plans/2026-04-28-data-profiling-and-if-judge.md
git commit -m "docs: add profiling and judge improvement plan"
```

---

## Addendum: Force-Pass Historical IF Outputs And Add LLM Judge Mode

### Task 6: Add Failing Tests For Historical Force-Pass Support

**Files:**
- Modify: `tests/test_eval_existing_outputs.py`
- Create: `tests/test_rejudge_saved_outputs.py`

- [ ] **Step 1: Write the failing tests**

Cover:

- force-passing an instruction-following record updates `is_correct`,
  `judge_status`, `judge_backend`, and `judge_detail`
- non-IF records are left unchanged when IF-only force-pass is requested
- eval/rejudge surfaces can pass an explicit judge mode through to the judge

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m unittest tests.test_rejudge_saved_outputs tests.test_eval_existing_outputs -v`

Expected: FAIL because the new force-pass path and judge-mode plumbing do not
exist yet.

### Task 7: Implement Historical Force-Pass Support

**Files:**
- Modify: `distill/commands/rejudge_saved_outputs.py`
- Modify: `distill/commands/eval_existing_outputs.py`

- [ ] **Step 1: Implement the smallest CLI and record-rewrite changes**

Add:

- a `--force-pass-if` style switch for saved IF segments
- optional `--judge-mode` plumbing where needed
- deterministic metadata in `judge_backend` / `judge_detail` for auditability

- [ ] **Step 2: Run targeted tests**

Run: `./.venv/bin/python -m unittest tests.test_rejudge_saved_outputs tests.test_eval_existing_outputs -v`

Expected: PASS

### Task 8: Add Failing Tests For LLM Instruction-Following Judge

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_judge_routing.py`
- Create: `tests/test_judge_instruction_following_llm.py`

- [ ] **Step 1: Write the failing tests**

Cover:

- `instruction_following_llm` is an accepted judge mode
- dispatcher routes IF samples to the LLM judge when requested
- the LLM judge parses structured pass/fail output
- malformed LLM output degrades gracefully with a stable backend/status

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m unittest tests.test_cli tests.test_judge_routing tests.test_judge_instruction_following_llm -v`

Expected: FAIL because the new judge mode and LLM judge do not exist yet.

### Task 9: Implement LLM Instruction-Following Judge

**Files:**
- Create: `distill/core/judges/instruction_following_llm.py`
- Modify: `distill/common/judge_mode.py`
- Modify: `distill/core/judges/dispatcher.py`
- Modify: `distill/core/judge.py`
- Modify: `distill/commands/run.py`

- [ ] **Step 1: Implement the minimal end-to-end path**

Add:

- a synchronous wrapper around the existing async LLM manager for judge usage
- a structured prompt/response contract for IF judging
- stable result mapping to the repository’s judge result schema
- judge-mode parsing and routing for `instruction_following_llm`

- [ ] **Step 2: Run targeted tests**

Run: `./.venv/bin/python -m unittest tests.test_cli tests.test_judge_routing tests.test_judge_instruction_following_llm -v`

Expected: PASS

### Task 10: Final Verification For This Addendum

**Files:**
- Modify: `docs/superpowers/specs/2026-04-28-data-profiling-and-if-judge-design.md`
- Modify: `docs/superpowers/plans/2026-04-28-data-profiling-and-if-judge.md`

- [ ] **Step 1: Run focused verification**

Run:

`./.venv/bin/python -m unittest tests.test_rejudge_saved_outputs tests.test_eval_existing_outputs tests.test_cli tests.test_judge_routing tests.test_judge_instruction_following_llm tests.test_judge_instruction_following -v`

Expected: PASS

- [ ] **Step 2: Apply the force-pass rewrite to the two IF output directories**

Run the updated `rejudge-saved-outputs` command against the two IF-related
segment directories from `manifest/general.yaml`, then rebuild correct segments
and merged shards.
