# distill

## YuLan Math 一站式启动

`manifest/post_train/yulan_math.yaml` 已经包含 `serve:` 配置，可以直接启动
vLLM 服务并运行蒸馏：

```bash
cd /mnt/ssd/lvzhihao/PostTrain/distill
bash run.sh --config manifest/post_train/yulan_math.yaml
```

小规模试跑：

```bash
cd /mnt/ssd/lvzhihao/PostTrain/distill
bash run.sh --config manifest/post_train/yulan_math.yaml -- --sample-limit 10
```

命令行参数和环境变量会覆盖 manifest 里的 `serve:` 配置，例如：

```bash
bash run.sh --config manifest/post_train/yulan_math.yaml --gpus "0 1 2 3 4 5 6 7"
```

# serve

## 8 卡

```bash
BASE_PORT=6758 \
MODEL_PATH="/mnt/hdd/Nanbeige4.1-3B" \
MODEL_NAME="Nanbeige4.1-3B" \
MAX_MODEL_LEN=65536 \
bash /mnt/ssd/lvzhihao/PostTrain/distill/scripts/serve/serve-qwen.sh
```

## 6 卡

```bash
BASE_PORT=1597 \
MODEL_PATH="/mnt/hdd/Nanbeige4.1-3B" \
MODEL_NAME="Nanbeige4.1-3B" \
MAX_MODEL_LEN=65536 \
GPUS_STR="0 1 2 3 4 5" \
bash /mnt/ssd/lvzhihao/PostTrain/distill/scripts/serve/serve-qwen.sh
```
