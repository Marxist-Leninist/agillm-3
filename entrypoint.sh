#!/bin/bash
set -e

echo "[$(date)] AGILLM-3 Tenstorrent N300s Training Entrypoint"
echo "[$(date)] Setting up environment..."

# Install runtime deps
pip install -q huggingface_hub 2>/dev/null
apt-get update -qq && apt-get install -y -qq tmux 2>/dev/null

# Configure HF token
mkdir -p /root/.cache/huggingface
echo "$HF_TOKEN" > /root/.cache/huggingface/token

# Download latest checkpoint from HuggingFace
CKPT_FILE="/workspace/ckpts/pretrain_step09131773.pt"
if [ ! -f "$CKPT_FILE" ]; then
  echo "[$(date)] Downloading checkpoint from HuggingFace..."
  curl -L -o "$CKPT_FILE" https://huggingface.co/OpenTransformer/AGILLM-3/resolve/main/pretrain_step09131773.pt
  echo "[$(date)] Download complete: $(ls -lh $CKPT_FILE)"
else
  echo "[$(date)] Checkpoint already exists"
fi

# Create upload helper script
cat > /workspace/upload_ckpt.py << 'PYEOF'
import sys, os, glob
from huggingface_hub import HfApi
REPO = "OpenTransformer/AGILLM-3-Large-Tenstorrent"
CKPT_DIR = "/workspace/ckpts_tt"
def upload_latest():
    api = HfApi()
    pts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")), key=os.path.getmtime)
    if not pts:
        print("No checkpoints found"); return
    latest = pts[-1]
    fname = os.path.basename(latest)
    sz = os.path.getsize(latest) / 1e9
    print(f"Uploading {fname} ({sz:.2f} GB) to {REPO}...")
    api.upload_file(path_or_fileobj=latest, path_in_repo=fname, repo_id=REPO)
    print(f"Uploaded {fname} to HuggingFace.")
if __name__ == "__main__":
    upload_latest()
PYEOF

mkdir -p /workspace/ckpts_tt

echo "[$(date)] Starting training on TT N300s..."

# Start training in background
python3 /workspace/n_tenstorrent_port.py train \
  --backend tt \
  --preset base \
  --warmstart_from "$CKPT_FILE" \
  --steps 50000 \
  --batch_size 2 \
  --block 576 \
  --save_dir /workspace/ckpts_tt \
  --save_every 500 \
  --tt_dtype bf16 \
  --tt_optimization_level 1 \
  2>&1 | tee /workspace/train.log &

TRAIN_PID=$!
echo "[$(date)] Training PID: $TRAIN_PID"

# Background uploader: every 30 min push checkpoint to HF
while kill -0 $TRAIN_PID 2>/dev/null; do
  sleep 1800
  if ls /workspace/ckpts_tt/*.pt 1>/dev/null 2>&1; then
    echo "[$(date)] Uploading checkpoint to HuggingFace..."
    python3 /workspace/upload_ckpt.py 2>&1
  fi
done

# Final upload
echo "[$(date)] Training ended. Final upload..."
python3 /workspace/upload_ckpt.py 2>&1
echo "[$(date)] All done. Keeping container alive..."

# Keep alive after training
tail -f /dev/null
