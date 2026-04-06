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