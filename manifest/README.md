# Manifest Tasks

这个目录用于存放批量启动时使用的 YAML 任务配置。

- `rule_examples/`
  放单个 task 的示例 YAML，可以直接复制后改路径使用。
- 你自己的批量任务
  可以继续按业务拆子目录，比如 `daily/`、`math/`、`code/`。

推荐做法：

1. 每个任务单独一个 YAML。
2. 路径、模型、并发、range 都写在 YAML 里。
3. 临时覆盖项再通过 CLI 追加，例如 `--range-start` 或 `--ports`。

也支持单文件多 task，推荐在批量任务场景下使用：

```yaml
model: Nanbeige4.1-3B
ports:
  - 6758-6765
max_tokens: 12000

tasks:
  - task_name: task_a
    input_dir: /data/a
    output_dir: /out/a
    file_pattern: a.parquet

  - task_name: task_b
    input_dir: /data/b
    output_dir: /out/b
    file_pattern: b.parquet
    max_tokens: 16000
```

上面这种写法里，顶层字段会作为公共配置自动应用到所有 task。

常见命令：

```bash
./.venv/bin/python -m distill --list-configs
./.venv/bin/python -m distill --config-name local_parquet_task
./.venv/bin/python -m distill --config manifest/rule_examples/local_parquet_task.yaml
./.venv/bin/python -m distill --config manifest/rule_examples/batch_tasks.yaml
./.venv/bin/python -m distill --config manifest/rule_examples/batch_tasks.yaml --task openthoughts3_math_part2
```
