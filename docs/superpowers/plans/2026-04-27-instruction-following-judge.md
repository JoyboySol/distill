# Instruction-Following Judge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a verifiable instruction-following judge and refactor judge internals into maintainable modules without changing the public API.

**Architecture:** Keep `distill/core/judge.py` as the stable entrypoint, move domain-specific logic into focused modules under `distill/core/judges/`, and add a dispatcher that routes to code, instruction-following, MCQ, and math judges in priority order. The new instruction-following judge only produces pass/fail when every requested constraint is supported.

**Tech Stack:** Python, `unittest`, existing distill pipeline/judge helpers

---

### Task 1: Write instruction-following failing tests

**Files:**
- Create: `tests/test_judge_instruction_following.py`

- [ ] **Step 1: Write the failing test**

Add tests covering:
- structured IF sample with `last_word` + `sentence_hyphens`
- structured IF sample with unsupported constraint returning `not_applicable`
- Tulu-style prompt with excluded words and specific ending
- Tulu-style prompt with missing required keyword returning `wrong_answer`

- [ ] **Step 2: Run test to verify it fails**

Run: `/mnt/ssd/lvzhihao/PostTrain/distill/.venv/bin/python -m unittest tests.test_judge_instruction_following`
Expected: FAIL because the new judge is not registered yet.

### Task 2: Extract shared/domain judge modules

**Files:**
- Create: `distill/core/judges/__init__.py`
- Create: `distill/core/judges/shared.py`
- Create: `distill/core/judges/code.py`
- Create: `distill/core/judges/math.py`
- Create: `distill/core/judges/mcq.py`
- Create: `distill/core/judges/dispatcher.py`
- Modify: `distill/core/judge.py`

- [ ] **Step 1: Move shared helpers and keep imports stable**
- [ ] **Step 2: Move code/math/mcq logic into focused modules**
- [ ] **Step 3: Add dispatcher helpers for judge selection and judge-type hints**
- [ ] **Step 4: Run existing judge regression tests**

Run: `/mnt/ssd/lvzhihao/PostTrain/distill/.venv/bin/python -m unittest tests.test_judge_mcq tests.test_judge_livecodebench tests.test_prompt_examples`
Expected: PASS

### Task 3: Implement instruction-following verifier

**Files:**
- Create: `distill/core/judges/instruction_following.py`
- Modify: `distill/core/judges/dispatcher.py`

- [ ] **Step 1: Implement structured IF constraint parsing from `ground_truth`**
- [ ] **Step 2: Implement prompt-derived Tulu constraint extraction for deterministic patterns**
- [ ] **Step 3: Return `not_applicable` for unsupported partial coverage**
- [ ] **Step 4: Register the new judge ahead of MCQ/math fallback**

### Task 4: Verify and document behavior

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a short README note describing the new judge family**
- [ ] **Step 2: Run focused test suite**

Run: `/mnt/ssd/lvzhihao/PostTrain/distill/.venv/bin/python -m unittest tests.test_judge_instruction_following tests.test_judge_mcq tests.test_judge_livecodebench tests.test_prompt_examples`
Expected: PASS

- [ ] **Step 3: Run one lightweight end-to-end smoke check through `judge_output_with_timeout`**
