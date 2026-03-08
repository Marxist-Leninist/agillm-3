#!/bin/bash
echo "=== AGILLM-3 Training on Tenstorrent N300s ==="
echo "Started: $(date -u)"

# Setup HF token from env
if [ -n "$HF_TOKEN" ]; then
  mkdir -p /root/.cache/huggingface
  echo "$HF_TOKEN" > /root/.cache/huggingface/token
  echo "HF token configured"
fi

# Wait for TT device to appear (can take 30-60s after container start)
echo "[$(date -u)] Waiting for TT device to appear..."
for i in $(seq 1 120); do
  if ls /dev/tenstorrent/* 1>/dev/null 2>&1; then
    echo "[$(date -u)] TT device found after ${i}s"
    break
  fi
  if [ $i -eq 120 ]; then
    echo "[$(date -u)] ERROR: No TT device after 120s. Listing /dev:"
    ls -la /dev/ | grep -i ten
    echo "Keeping container alive for debugging..."
    while true; do sleep 3600; done
  fi
  sleep 1
done

# Fix device permissions
echo "[$(date -u)] Fixing TT device permissions..."
for dev in /dev/tenstorrent/*; do
  chmod 666 "$dev" 2>/dev/null && echo "  Fixed: $dev -> $(ls -la $dev)"
done

# Extra wait for firmware to fully initialize after device appears
echo "[$(date -u)] Waiting 15s for TT firmware to stabilize..."
sleep 15
echo "[$(date -u)] Re-fixing permissions after firmware init..."
for dev in /dev/tenstorrent/*; do
  chmod 666 "$dev" 2>/dev/null && echo "  Fixed: $dev -> $(ls -la $dev)"
done

# Download checkpoint from HuggingFace if not present
mkdir -p /workspace/ckpts
CKPT="/workspace/ckpts/pretrain_step09131773.pt"
if [ ! -f "$CKPT" ]; then
  echo "[$(date -u)] Downloading checkpoint from HuggingFace..."
  pip install -q huggingface_hub 2>/dev/null
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
REPO = "OpenTransformer/AGILLM-3-Large-Tenstorrent"
def upload_latest():
    api = HfApi()
    pts = sorted(glob.glob("/workspace/ckpts_tt/*.pt"), key=os.path.getmtime)
    if not pts: print("No checkpoints"); return
    f = pts[-1]
    print(f"Uploading {os.path.basename(f)} ({os.path.getsize(f)/1e9:.2f}GB)...")
    api.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f), repo_id=REPO)
    print("Done.")
if __name__ == "__main__": upload_latest()
PYEOF

# Training with retry loop (up to 5 attempts)
export PJRT_DEVICE=TT
export XLA_STABLEHLO_COMPILE=1

MAX_RETRIES=5
for attempt in $(seq 1 $MAX_RETRIES); do
  echo "[$(date -u)] Training attempt $attempt/$MAX_RETRIES"

  # Re-fix device permissions before each attempt
  for dev in /dev/tenstorrent/*; do
    chmod 666 "$dev" 2>/dev/null
  done

  # Start training in background
  python3 /workspace/n_tenstorrent_port.py train \
    --backend tt \
    --preset base \
    --warmstart_from "$CKPT" \
    --steps 1000000 \
    --block 576 \
    --batch_size 2 \
    --save_dir /workspace/ckpts_tt \
    --save_every 500 \
    --tt_dtype bf16 \
    --tt_optimization_level 1 \
    2>&1 | tee /workspace/train.log &

  TRAIN_PID=$!
  echo "[$(date -u)] Training PID: $TRAIN_PID"

  # Upload to HF every 30 min while training runs
  while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 1800
    if ls /workspace/ckpts_tt/*.pt 1>/dev/null 2>&1; then
      echo "[$(date -u)] Uploading checkpoint to HF..."
      python3 /workspace/upload_ckpt.py 2>&1 || true
    fi
  done

  # Check if training exited cleanly
  wait $TRAIN_PID
  EXIT_CODE=$?
  if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date -u)] Training completed successfully!"
    break
  fi

  echo "[$(date -u)] Training crashed (exit $EXIT_CODE). Retry in 60s..."
  # Re-fix permissions and wait longer between retries
  for dev in /dev/tenstorrent/*; do
    chmod 666 "$dev" 2>/dev/null
  done
  sleep 60
done

# Final upload
echo "[$(date -u)] Final checkpoint upload..."
python3 /workspace/upload_ckpt.py 2>&1 || true
echo "[$(date -u)] All done."

# Keep container alive so we can exec in
while true; do sleep 3600; done
