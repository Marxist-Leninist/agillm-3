#!/bin/bash
set -e
echo "=== AGILLM-3 Training on Tenstorrent N300s ==="
echo "Started: $(date -u)"

# Setup HF token from env
if [ -n "$HF_TOKEN" ]; then
  mkdir -p /root/.cache/huggingface
  echo "$HF_TOKEN" > /root/.cache/huggingface/token
  echo "HF token configured"
fi

# Download checkpoint from HuggingFace if not present
CKPT="/workspace/ckpts/pretrain_step09131773.pt"
if [ ! -f "$CKPT" ]; then
  echo "[$(date -u)] Downloading checkpoint from HuggingFace..."
  curl -L -o "$CKPT" https://huggingface.co/OpenTransformer/AGILLM-3/resolve/main/pretrain_step09131773.pt
  echo "[$(date -u)] Download complete: $(ls -lh $CKPT)"
else
  echo "Checkpoint exists: $(ls -lh $CKPT)"
fi

# Create HF upload helper
mkdir -p /workspace/ckpts_tt
cat > /workspace/upload_ckpt.py << 'PYEOF'
import os, glob
from huggingface_hub import HfApi
REPO = "OpenTransformer/AGILLM-3"
def upload_latest():
    api = HfApi()
    pts = sorted(glob.glob("/workspace/ckpts_tt/*.pt"), key=os.path.getmtime)
    if not pts: print("No checkpoints"); return
    f = pts[-1]
    print(f"Uploading {os.path.basename(f)} ({os.path.getsize(f)/1e9:.2f}GB)...")
    api.upload_file(path_or_fileobj=f, path_in_repo=f"tenstorrent/{os.path.basename(f)}", repo_id=REPO)
    print("Done.")
if __name__ == "__main__": upload_latest()
PYEOF

echo "[$(date -u)] Starting training..."

# Start training
python3 /workspace/n_tenstorrent_port.py train \
  --backend tt \
  --preset base \
  --warmstart_from "$CKPT" \
  --steps 1000000 \
  --block 576 \
  --batch_size 1 \
  --save_dir /workspace/ckpts_tt \
  --save_every_sec 1800 \
  --tt_dtype bf16 \
  > /workspace/train.log 2>&1 &

TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"

# Upload to HF every 30 min while training runs
while kill -0 $TRAIN_PID 2>/dev/null; do
  sleep 1800
  if ls /workspace/ckpts_tt/*.pt 1>/dev/null 2>&1; then
    echo "[$(date -u)] Uploading checkpoint..."
    python3 /workspace/upload_ckpt.py 2>&1 || true
  fi
done

# Final upload
echo "[$(date -u)] Training ended. Final upload..."
python3 /workspace/upload_ckpt.py 2>&1 || true
echo "[$(date -u)] All done."

# Keep container alive
tail -f /dev/null
