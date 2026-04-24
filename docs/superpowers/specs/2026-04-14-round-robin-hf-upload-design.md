# Round-Robin Scheduling And HF Upload Design

**Context**

The distillation pipeline already supported:

- single-task serial execution
- per-task concurrency
- resumable segment-to-shard writes
- multi-turn conversation distillation

This design extends the existing behavior without rewriting the inner
`DistillPipeline` execution model.

## Goals

- Support manifest-level round-robin scheduling so multi-task runs produce a
  more even mix of distilled data.
- Support periodic segment merge instead of merge-only-on-exit.
- Upload merged `correct` shards to Hugging Face dataset storage.
- Keep existing resume semantics and per-task output layout intact.

## Decisions

- Round-robin lives in `distill/cli.py`, not inside `DistillPipeline`.
- Each round increases a task's effective `sample_limit` by
  `round_robin_chunk_size`.
- Existing completed-index resume logic remains the source of truth for
  deduplication.
- Periodic merge is triggered by `merge_every_n_writes`.
- Only `correct/shards/*.parquet` are eligible for HF upload.
- Remote upload keeps a single directory level:
  `<hf_remote_prefix or task_name>/shard_00000.parquet`
- Upload progress is tracked locally with `.hf_upload_state.json`.

## Constraints

- Upload failures must not break the distillation write path.
- Existing single-turn and multi-turn record formats must remain unchanged.
- Manifest defaults should still work for serial execution when round-robin is
  not configured.
