---
name: posttrain-distill-qwen
description: Run, monitor, and debug the PostTrain distillation workflow for Qwen3.6/Qwen3 math trajectories in /root/nas/share/vc0e4b0o65t5/lvzhihao/PostTrain/distill or /workspace/lvzhihao/PostTrain/distill. Use when asked to start full or test distillation, configure vLLM reasoning for Qwen, check boxed math answers, judge low correctness, clean stale trajectories, inspect outputs, or hand off operational knowledge to another agent.
---

# PostTrain Distill Qwen

## Operating Rules

- Work from `/root/nas/share/vc0e4b0o65t5/lvzhihao/PostTrain/distill` unless the user gives another checkout. `/workspace/lvzhihao/PostTrain/distill` is the same project path used by configs.
- Use `.venv/bin/python` and `.venv/bin/vllm`; do not assume system Python has the right vLLM.
- Treat long runs as production jobs. Start them detached with `setsid`, write PID/log/status files under `logs/`, and do not leave foreground sessions running.
- Before killing a run, stop the whole session group, not only the wrapper PID. `run.sh` starts child vLLM and distill processes.
- Preserve unrelated user edits. Do not use destructive git commands.

## Known Good Configuration

For `manifest/post_train/yulan_math.yaml`, keep these settings unless the user explicitly changes them:

```yaml
serve:
  model_path: /workspace/lvzhihao/PostTrain/distill/models/Qwen3.6-27B
  model_name: Qwen3.6-27B
  max_model_len: 65536
  gpus: "0 1 2 3"
  serve_extra_args: "--max-num-seqs 128"

model: Qwen3.6-27B
enable_thinking: true
max_tokens: 60000
temperature: 1.0
concurrency: 256
rollout_count: 8
```

`run.sh` already defaults to `--reasoning-parser qwen3`; verify serve logs still show `reasoning_parser: qwen3`.

Use a system prompt that requests only the final answer in `\boxed{}`. Intermediate boxed values make math judge extraction less reliable.

## Start Full Distillation

Clean the formal output only when the user asks for a clean restart:

```bash
rm -rf /workspace/lvzhihao/PostTrain/distill/trajectories/YuLan-Math \
       /workspace/lvzhihao/PostTrain/distill/failures/yulan_math_failed_0_END.jsonl
```

Start detached:

```bash
LOG=logs/distill_yulan_math_full_$(date -u +%Y%m%d_%H%M%S).log
STATUS=${LOG%.log}.status
setsid bash -c 'bash run.sh --config manifest/post_train/yulan_math.yaml >"$1" 2>&1; echo $? >"$2"' _ "$LOG" "$STATUS" &
PID=$!
echo "$PID" > logs/distill_yulan_math.pid
echo "$LOG" > logs/distill_yulan_math.latest_log
echo "$STATUS" > logs/distill_yulan_math.latest_status
```

Monitor:

```bash
ps -p "$(cat logs/distill_yulan_math.pid)" -o pid,ppid,sid,stat,etime,cmd
tail -n 80 "$(cat logs/distill_yulan_math.latest_log)"
STATUS=$(cat logs/distill_yulan_math.latest_status); [[ -f "$STATUS" ]] && cat "$STATUS" || echo running
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

## Stop A Run

Always stop the session group:

```bash
PID=$(cat logs/distill_yulan_math.pid)
SID=$(ps -p "$PID" -o sid= | tr -d ' ')
kill -TERM -"$SID"
sleep 8
if ps -s "$SID" >/dev/null 2>&1; then kill -KILL -"$SID"; fi
```

Then verify no stale processes remain:

```bash
ps -eo pid,ppid,sid,stat,cmd | rg 'vllm serve|distill --config|run.sh --config' | rg -v 'rg '
```

## Test Before Full Runs

Run a small isolated test when changing sampling, judge, prompt, reasoning, or vLLM versions:

```bash
rm -rf /workspace/lvzhihao/PostTrain/distill/trajectories/YuLan-Math-test-temp1 \
       /workspace/lvzhihao/PostTrain/distill/failures/yulan_math_test_failed_0_END.jsonl

LOG=logs/distill_yulan_math_temp1_test_$(date -u +%Y%m%d_%H%M%S).log
STATUS=${LOG%.log}.status
setsid bash -c 'bash run.sh --config manifest/post_train/yulan_math.yaml -- --sample-limit 4 --output-dir /workspace/lvzhihao/PostTrain/distill/trajectories/YuLan-Math-test-temp1 --failure-log yulan_math_test_failed.jsonl --concurrency 32 >"$1" 2>&1; echo $? >"$2"' _ "$LOG" "$STATUS" &
```

Inspect the resulting records:

```bash
.venv/bin/python /root/.codex/skills/posttrain-distill-qwen/scripts/check_yulan_outputs.py \
  /workspace/lvzhihao/PostTrain/distill/trajectories/YuLan-Math-test-temp1
```

Expected after the known-good fix: nonzero total, all or nearly all correct on early easy samples, every assistant message has a system prompt, nonempty `reasoning_content`, and `\boxed{}` in content.

## Reasoning and Judge Pitfalls

- vLLM Qwen reasoning needs both sides:
  - server: `--reasoning-parser qwen3`
  - request: `extra_body={"chat_template_kwargs": {"enable_thinking": true}}`
- Newer vLLM/OpenAI responses may use `message.reasoning`; older code may look for `message.reasoning_content`. Store either into the trajectory field `reasoning_content`.
- If outputs show correct math but low correctness, inspect `judge_detail.prediction` before blaming the model. The math judge must prefer boxed/final-answer windows, display math at the end, simple equation RHS, and numeric fallbacks.
- `max_tokens: 65536` can fail because prompt tokens plus generated tokens exceed the 65536 context. Use `60000`.
- `--max-num-seqs 128` is needed for this model on these GPUs because cache/page constraints fail at higher default concurrency.

## Quick Health Script

Use `scripts/check_yulan_outputs.py` for output summaries. It reports total rows, correctness counts, system prompt count, reasoning count, boxed count, per-input rollout diversity, and a few examples.
