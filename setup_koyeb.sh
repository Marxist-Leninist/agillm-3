#!/bin/bash
# setup_koyeb.sh — Tenstorrent N300s Koyeb Instance Setup
# For AGILLM-3 inference benchmarking on Wormhole hardware
# Run this after SSH/VSCode tunnel is established to Koyeb instance

set -e

echo "╔══════════════════════════════════════════════════╗"
echo "║  AGILLM-3 × Tenstorrent N300s Setup             ║"
echo "║  OpenTransformers Ltd — Inference Benchmark      ║"
echo "╚══════════════════════════════════════════════════╝"

# ── 1. Check TT hardware is visible ──
echo "[1/6] Checking Tenstorrent hardware..."
if ls /dev/tenstorrent* 2>/dev/null; then
    echo "  ✓ Tenstorrent device(s) found"
else
    echo "  ✗ No Tenstorrent devices — are you on a TT-N300S instance?"
    exit 1
fi

# ── 2. Install TT software stack (if not pre-installed) ──
echo "[2/6] Checking TT software stack..."
if python3 -c "import ttnn" 2>/dev/null; then
    echo "  ✓ ttnn already available"
else
    echo "  Installing TT-Metalium SDK..."
    pip install git+https://github.com/tenstorrent/tt-metal.git 2>/dev/null || {
        echo "  Note: tt-metal may already be in the container image"
    }
fi

# Check for torch_ttnn (deprecated but functional)
if python3 -c "import torch_ttnn" 2>/dev/null; then
    echo "  ✓ torch_ttnn available"
else
    echo "  Installing torch_ttnn (PyTorch 2.0 TT-NN compiler)..."
    pip install git+https://github.com/tenstorrent/pytorch2.0_ttnn.git 2>/dev/null || {
        echo "  Note: torch_ttnn install may need tt-metal first"
    }
fi

# ── 3. Install Python deps ──
echo "[3/6] Installing Python dependencies..."
pip install -q torch transformers datasets huggingface_hub 2>/dev/null

# ── 4. Create workspace ──
echo "[4/6] Setting up workspace..."
mkdir -p /workspace/ckpts /workspace/benchmarks /workspace/logs

# ── 5. Download a small AGILLM-3 checkpoint for testing ──
echo "[5/6] Checkpoint setup..."
echo "  To test inference, copy a checkpoint to /workspace/ckpts/"
echo "  Example: scp your_server:/path/to/checkpoint.pt /workspace/ckpts/"
echo "  Or use huggingface_hub to download if published"

# ── 6. Verify setup ──
echo "[6/6] Running verification..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
try:
    import ttnn
    print(f'  TT-NN: available')
    # Try opening device
    device = ttnn.open_device(device_id=0)
    print(f'  TT Device: opened successfully')
    ttnn.close_device(device)
    print(f'  TT Device: closed cleanly')
except Exception as e:
    print(f'  TT-NN: {e}')
try:
    import torch_ttnn
    print(f'  torch_ttnn: available')
except:
    print(f'  torch_ttnn: not available')
"

echo ""
echo "Setup complete. Next steps:"
echo "  1. Copy a checkpoint:  scp server:ckpts/pretrain_step*.pt /workspace/ckpts/"
echo "  2. Run inference bench: python3 n_tt.py infer --mode ar --ckpt /workspace/ckpts/your.pt --prompt 'Hello world'"
echo "  3. Run compile test:   python3 n_tt.py compile-test --ckpt /workspace/ckpts/your.pt"
echo ""
echo "IMPORTANT: Instance is STATELESS. Save results externally!"
