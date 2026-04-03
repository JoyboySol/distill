# Distill

这个目录是一个异步蒸馏流水线，用来把原始样本逐条喂给模型，拿到回答后整理成统一的 SFT 训练格式，并在写盘时附带基础判题结果。

它现在已经从单文件脚本拆成了几个模块，便于单独维护：

- `distill/distill.py`
  CLI 入口，只负责解析参数和启动流水线。
- `distill/config.py`
  配置定义和日志初始化。
- `distill/llm.py`
  异步 OpenAI 兼容客户端，请求重试和多后端负载分配。
- `distill/failure.py`
  失败记录、失败样本跳过。
- `distill/judge.py`
  `math` / `code` 判题逻辑。
- `distill/pipeline.py`
  `producer -> generation -> judge -> writer` 主流水线。
- `distill/stats.py`
  输出结果统计脚本。
- `distill/utils.py`
  通用序列化和消息规范化工具。

## 1. 项目目标

这条流水线主要做 6 件事：

1. 从 `parquet` 或 `jsonl` 输入中读取样本。
2. 从指定字段中抽出用户 prompt。
3. 调 OpenAI 兼容接口生成 assistant 回答。
4. 把生成和判题拆成两个独立 stage，避免慢 judge 阻塞模型请求。
5. 把输出整理成统一的 `sharegpt/sft` 风格记录。
6. 对 `math` 和 `code` 样本补充自动判题结果，并以 `segment jsonl -> parquet shard` 两阶段写盘。

它适合以下场景：

- 把问答数据批量过一遍 teacher model，生成蒸馏数据。
- 同时保留推理字段 `reasoning_content`，而不是把它丢掉。
- 边生成边做轻量质量标记，比如 `is_correct`。
- 支持断点续跑和失败样本记录。

## 2. 当前数据流

整条链路大致是：

`input file -> producer -> task_queue -> generation workers -> judge_queue -> judge workers -> result_queue -> writer -> segment jsonl -> parquet shard`

更具体一点：

1. `producer`
   扫描输入目录，按 `file_pattern` 和 `range` 取文件。
2. 逐条读取样本
   从 `input_content_field` 抽 prompt，并做 prompt 规范化。
3. `generation worker`
   发请求给模型，拿到 `[user, assistant]` 两条消息。
4. `judge worker`
   对样本做 `math` / `code` 判定。
5. `writer`
   先写小粒度 `jsonl segment`，最后再 merge 成 `parquet shard`。

这样做的主要目的有两个：

- 避免慢判题拖住生成吞吐
- 避免“大约 200MB 都在内存里，进程一断就全没了”

## 3. 输入要求

当前支持两类输入：

- `*.parquet`
- `*.jsonl`

默认会从 `question` 字段里取输入，也可以通过 `--input-field` 改掉。

支持的 prompt 形态包括：

- 纯字符串
- 序列化后的 JSON 字符串
- `OpenAI messages` 风格列表
- 含 `content` 字段的字典

如果某条样本无法抽出有效 prompt：

- 不会进入推理
- 会记录到 failure log
- 后续重跑时会自动跳过

## 4. 输出格式

每条输出都会被整理成一条标准记录，核心字段包括：

- `dataset_name`
- `sample_id`
- `dedup_hash`
- `content_chars`
- `turn_count`
- `has_reasoning`
- `messages`
- `metadata`
- `adapter_status`
- `adapter_name`
- `record_mode`
- `source_file`
- `source_row`
- `judge_type`
- `judge_backend`
- `is_correct`
- `judge_status`
- `judge_detail`

其中：

- `messages` 是真正的结构化 list，不是 JSON 字符串。
- `reasoning_content` 会直接保留在原消息字段里。
- `source_file + source_row` 是断点续跑和去重的基础主键。
- `judge_backend` 用来区分是 `math_verify`、`math_rule` 还是 `none`。

## 5. 判题逻辑

### 5.1 Code 判题

当前 code 判题分两类：

- `code_mbpp`
  如果样本里有 `test_list_2` 或 `test_list`，就把模型生成的代码和测试拼起来执行。
- `code_humaneval`
  如果样本里有 `prompt + test + entry_point` 这类字段，就走 HumanEval 风格执行。

返回结果会写成：

- `judge_type = code_mbpp | code_humaneval`
- `judge_backend = none`
- `judge_status = pass | wrong_answer | timeout | failed`
- `is_correct = True | False`

### 5.2 Math 判题

数学判题现在是双层结构：

1. 优先走 `math_verify`
2. 失败时回退到规则式判题

也就是：

- `judge_backend = math_verify`
  使用 `math_verify + latex2sympy2_extended` 做符号级验证。
- `judge_backend = math_rule`
  如果 `math_verify` 不可用，或者某条样本解析失败，就回退到规则判题。

规则判题会做：

- `\boxed{}` 抽取
- `final answer` / `answer is` 提取
- `frac / sqrt / unit / 空格 / $...$` 等归一化
- 宽松等价比较

当前策略上需要注意一点：

- `math_verify` 现在是单位敏感的，不会把 `5 kg` 和 `5` 判成一样。
- `math_rule` 仍然是更宽松的规则归一化逻辑。
- 因此两者可能出现分歧，`judge_backend` 会把实际采用的判题后端记录下来。

如果 `math_verify` 回退了，原因会写进：

- `judge_detail.verify_fallback_reason`

## 6. 断点续跑

断点续跑有两层保护：

### 6.1 已完成样本跳过

启动时会扫描输出目录里已有的：

- `all/shards/shard_*.parquet`
- `all/segments/segment_*.jsonl`

然后把每条记录的：

- `source_file`
- `source_row`

读出来，构建 completed set。

只要某条输入已经出现在 `all` 流的历史 segment 或 shard 中，后续就不会重复处理。

### 6.2 失败样本跳过

失败记录会写到 `failure_log`，并且自动加上当前 range 后缀，例如：

- `distill/failures/failed_tasks_0_END.jsonl`

每条失败日志包含：

- `source_file`
- `source_row`
- `reason`
- `time`

后续重跑时这些失败样本会直接跳过，避免一直卡在同一条坏数据上。

默认情况下，失败日志会落到项目目录下的 `failures/`，方便开发时直接查看。

如果你显式传了绝对路径 `--failure-log /abs/path/my_failures.jsonl`，
脚本会继续尊重这个绝对路径，只是同样会自动加上 range 后缀。

## 7. 两阶段写盘与正确样本分流

writer 现在不是直接把内存 buffer 憋到 `200MB` 再写 parquet，而是两阶段：

1. 先写中间 `jsonl segment`
2. 再 merge 成最终 `parquet shard`

这样做的收益是：

- 中断时不容易丢掉大块内存结果
- `jsonl` 适合追加式、恢复友好的中间写入
- `parquet` 仍然保留为最终训练产物

### 7.1 输出目录结构

当前会分成两条流：

- `all`
  全量结果流
- `correct`
  `is_correct=True` 的正确样本流

目录结构如下：

```text
output_dir/
  all/
    segments/
      segment_000000.jsonl
    shards/
      shard_00000.parquet
    merge_state.json
  correct/
    segments/
      segment_000000.jsonl
    shards/
      shard_00000.parquet
    merge_state.json
```

### 7.2 segment 写入

默认：

- `--segment-size-mb 4`

当内存 buffer 达到这个大小时，会先 flush 成一个小的 `segment_*.jsonl`。

这样即使进程意外退出，损失窗口也只会落在很小的 segment buffer 上，而不是一个大 shard。

### 7.3 shard merge

默认：

- `--shard-size-mb 200`

writer 在主流程收尾时，会把未 merge 的 segment 按目标大小合并成：

- `shard_00000.parquet`
- `shard_00001.parquet`
- ...

写盘特性：

- segment 和 shard 都先写 `.tmp` 再 `os.replace`
- 写失败会自动重试
- merge 状态记录在 `merge_state.json`

### 7.4 正确样本分流

只要某条记录满足：

- `is_correct == True`

它除了会进入 `all` 流，还会额外进入：

- `correct/segments`
- `correct/shards`

所以你后面可以直接拿：

- `all/shards`
  做全量训练或分析
- `correct/shards`
  做高质量子集训练或单独抽样

## 8. 运行方式

推荐直接使用当前目录提供的虚拟环境：

```bash
/mnt/ssd/lvzhihao/PostTrain/distill/.venv/bin/python \
  /mnt/ssd/lvzhihao/PostTrain/distill/distill/distill.py \
  --input-dir /path/to/input \
  --output-dir /path/to/output \
  --file-pattern "*.parquet" \
  --input-field question
```

### 常用参数

- `--input-dir`
  输入目录。
- `--output-dir`
  输出 shard 目录。
- `--file-pattern`
  输入 glob，例如 `*.parquet` 或 `*.jsonl`。
- `--input-field`
  从哪一列抽 prompt。
- `--label-field`
  显式指定判题参考答案列，例如 `label`。
- `--range-start`
  文件列表起始位置。
- `--range-end`
  文件列表结束位置。
- `--model`
  请求使用的模型名。
- `--ports`
  直接给本地端口列表，例如 `6758,6759,6761-6765`。
- `--concurrency`
  生成并发。
- `--judge-concurrency`
  判题并发。
- `--active-files`
  同时活跃读取的文件数。
- `--batch-size`
  每次读取批大小。
- `--segment-size-mb`
  中间 `jsonl segment` 的目标大小。
- `--shard-size-mb`
  目标 shard 大小。
- `--write-retries`
  写盘失败重试次数。

### 推荐启动命令

下面这条命令比较适合作为当前版本的标准启动方式：

```bash
/mnt/ssd/lvzhihao/PostTrain/distill/.venv/bin/python \
  /mnt/ssd/lvzhihao/PostTrain/distill/distill/distill.py \
  --input-dir /mnt/hdd/lvzhihao/data/KodCode-V1-SFT-4o/data \
  --output-dir /mnt/hdd/lvzhihao/output/KodCode-V1-SFT-4o \
  --file-pattern "*.parquet" \
  --input-field question \
  --label-field answer \
  --model Qwen3-30B-A3B-Thinking-2507 \
  --ports 6758-6765 \
  --concurrency 2048 \
  --judge-concurrency 32 \
  --active-files 6 \
  --batch-size 1000 \
  --segment-size-mb 4 \
  --shard-size-mb 200 \
  --write-retries 3
```

如果你更看重稳妥写盘而不是 shard 数量，可以把：

- `--segment-size-mb`
  再调小一点
- `--judge-concurrency`
  根据 CPU 和 code/math 判题负载调节

### 端口配置

现在主推荐用 `--ports`，不用再手写 URL：

#### 方式 1：连续端口范围

适合一组本地 vLLM / OpenAI 兼容服务连续起在多个端口上：

```bash
--ports 6758-6765
```

默认会展开成：

```text
http://localhost:6758/v1
...
http://localhost:6765/v1
```

#### 方式 2：离散端口 + 范围混写

```bash
--ports 6758,6759,6761-6765
```

如果命令里不传 `--ports`，程序会优先读取环境变量：

- `DISTILL_PORTS`

比如：

```bash
export DISTILL_PORTS="6758-6765"
```

如果这些都没传，才会回退到默认本地范围：

```text
http://localhost:6758/v1
...
http://localhost:6765/v1
```

### 示例：跑 jsonl

```bash
/mnt/ssd/lvzhihao/PostTrain/distill/.venv/bin/python \
  /mnt/ssd/lvzhihao/PostTrain/distill/distill/distill.py \
  --input-dir /data/my_jsonl \
  --output-dir /data/my_output \
  --file-pattern "*.jsonl" \
  --input-field prompt \
  --label-field label \
  --ports 6758-6765
```

### 示例：只跑文件列表的一段

```bash
/mnt/ssd/lvzhihao/PostTrain/distill/.venv/bin/python \
  /mnt/ssd/lvzhihao/PostTrain/distill/distill/distill.py \
  --input-dir /data/my_input \
  --output-dir /data/my_output \
  --range-start 100 \
  --range-end 200 \
  --ports 6758-6765
```

## 9. 统计脚本

跑完之后可以用 `stats.py` 汇总判题情况：

```bash
/mnt/ssd/lvzhihao/PostTrain/distill/.venv/bin/python \
  /mnt/ssd/lvzhihao/PostTrain/distill/distill/stats.py \
  --output-dir /path/to/output \
  --stream all
```

如果想把统计结果存成 JSON：

```bash
/mnt/ssd/lvzhihao/PostTrain/distill/.venv/bin/python \
  /mnt/ssd/lvzhihao/PostTrain/distill/distill/stats.py \
  --output-dir /path/to/output \
  --stream all \
  --save-json /path/to/output/judge_stats.json
```

如果你只想统计正确样本流：

```bash
/mnt/ssd/lvzhihao/PostTrain/distill/.venv/bin/python \
  /mnt/ssd/lvzhihao/PostTrain/distill/distill/stats.py \
  --output-dir /path/to/output \
  --stream correct
```

当前统计内容包括：

- `total_records`
- `judge_type_counts`
- `judge_backend_counts`
- `judge_status_counts`
- `correct_counts`
- `overall_accuracy`
- `backend_accuracy`
- `math_backend_summary`
- `math_verify_fallback_reasons`

特别适合看这些问题：

- 一共有多少数学样本真的走到了 `math_verify`
- 有多少回退到了 `math_rule`
- fallback 的主要原因是什么
- `code` 和 `math` 的判题通过率大概怎样
- `correct` 分流里到底保留了多少条

## 10. 依赖说明

当前 `.venv` 里已经验证可用的关键依赖包括：

- `openai`
- `tenacity`
- `pyarrow`
- `tqdm`
- `math_verify`
- `latex2sympy2_extended`

其中数学符号验证依赖：

- `math_verify`
- `latex2sympy2_extended`

如果这两个包不可用，项目仍然能跑，只是数学判题会自动回退到规则式判题。

## 11. 目前的设计取舍

这套实现现在偏向“先稳跑、再可分析”，所以做了几个明确取舍：

- 输出直接写最终 shard，而不是特别碎的临时 parquet
- 样本主键使用 `source_file + source_row`
- `messages` 保持结构化，而不是转成字符串
- 判题结果附着在记录上，而不是另外再生成一套 sidecar 文件
- 数学判题采用“符号验证优先，规则判题兜底”

## 12. 后续可以继续增强的方向

如果后面还要继续扩，这几个方向会比较自然：

- 增加 `--math-judge-backend auto|verify|rule`
- 增加更人类可读的文本统计 summary
- 把 logger 路径也做成 CLI 参数
- 为 `judge.py` 增加单元测试
- 为 `pipeline.py` 增加一组最小集成测试
- 如果后续样本 schema 稳定，可以引入更严格的输入 schema 校验

## 13. 一句话总结

这个项目现在已经不是“单个大脚本做所有事”了，而是一条模块化的蒸馏流水线：

- 前面负责读数据和调模型
- 中间把生成和判题拆成独立 stage
- 后面负责断点续跑、segment 持久化、shard merge 和正确样本分流
- 最后还能用统计脚本快速看整体质量

## 14. quick start

```bash
/mnt/ssd/lvzhihao/PostTrain/distill/.venv/bin/python \
  /mnt/ssd/lvzhihao/PostTrain/distill/distill/distill.py \
  --input-dir /mnt/hdd/huanglisheng/train_data/G-OPD-Training-Data/DeepMath-103K \
  --output-dir /mnt/hdd/lvzhihao/output/DeepMath-103K-distill \
  --file-pattern "slime_style_train_data.jsonl" \
  --input-field prompt \
  --label-field label \
  --model Qwen3-30B-A3B-Instruct-2507 \
  --ports 6758-6765 \
  --concurrency 1024 \
  --judge-concurrency 16 \
  --active-files 1 \
  --batch-size 1000 \
  --segment-size-mb 4 \
  --shard-size-mb 200 \
  --write-retries 3
```
