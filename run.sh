#!/bin/bash
set -e
echo "=== AGILLM-3 Training on Tenstorrent N300s ==="
echo "Started: $(date -u)"
exec python3 /workspace/n_tenstorrent_port.py train \
  --backend tt \
  --preset base \
  --steps 1000000 \
  --block 576 \
  --batch_size 4 \
  --save_dir /workspace/ckpts \
  --save_every_sec 43200 \
  --max_ckpts 3 \
  --tt_dtype bf16 \
  --ar_only
