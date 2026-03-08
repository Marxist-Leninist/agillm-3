# AGILLM-3 — Joint AR+SAT Transformer
## OpenTransformers Ltd

698M parameter language model using joint autoregressive + semi-autoregressive (AR+SAT) architecture with tuneable expansion ratio attention.

### Architecture
- **Tuneable Attention MHA** with rank projection (U matrix) — configurable expansion ratio
- **ALiBi** positional encoding (no learned positional embeddings)
- **Joint AR+SAT** training — autoregressive + semi-autoregressive heads trained simultaneously
- **Variable stride SAT** — gated stride prediction for speculative generation

---

### Tenstorrent Port Scripts

| Script | Backend API | Author | Status |
|--------|------------|--------|--------|
| **`n_tenstorrent_port.py`** | TT-XLA / PJRT | GPT-5.4 Pro Extended Thinking | **Recommended — use this one** |
| `n_tenstorrent.py` | TT-XLA / PJRT | GPT-5.4 Pro Extended Thinking | Alternative port |
| `n_tt_singlefile.py` | TT-XLA / PJRT | GPT-5.4 Pro Standard | Alternative port |
| `n_tt.py` | torch_ttnn (deprecated) | Early exploration | Legacy / reference only |

**Use `n_tenstorrent_port.py`** — it has the most complete training path, SPMD support, and robust checkpoint cross-compatibility between CUDA and TT.

See [`n_tenstorrent_port_README.md`](n_tenstorrent_port_README.md) for detailed setup and usage.

---

### All Files

| File | Lines | Purpose |
|------|-------|---------|
| `n.py` | 1032 | Original CUDA trainer/inferencer (vast.ai) |
| `n_tenstorrent_port.py` | 1755 | **Primary TT-XLA port** — training + inference on N300s |
| `n_tenstorrent.py` | 1574 | Alternative TT-XLA port |
| `n_tt_singlefile.py` | 1637 | Alternative TT-XLA port (Pro Standard) |
| `n_tt.py` | 1300 | Legacy TT port using torch_ttnn |
| `nat_mamba.py` | 625 | Mamba/SSM architecture variant |
| `nat_mamba_final.py` | 477 | Final Mamba variant |
| `nat_mamba_simple.py` | 285 | Simplified Mamba variant |

---

### Key Features of n_tenstorrent_port.py
- `--backend auto|cuda|tt|cpu` — auto-detects available hardware
- Training via XLA-style optimizer stepping (`xm.optimizer_step`)
- Checkpoints always saved as CPU tensors — **load NVIDIA-trained checkpoints on TT and vice versa**
- Static-shape inference on TT (avoids XLA recompilation traps)
- TT tuning flags: `--tt_dtype`, `--tt_bfp8`, `--tt_weight_bfp8`, `--tt_optimization_level`, `--tt_trace`
- Experimental SPMD for 2-chip N300 (`--tt_spmd`)

---

### Quick Start — Koyeb Setup
```bash
# 1. Activate coupon at koyeb.com: TTDEPLOY25FADEV2W (1x N300s, 2 weeks)
# 2. Set up environment
python3 -m venv .xla-venv
source .xla-venv/bin/activate
pip install pjrt-plugin-tt --extra-index-url https://pypi.eng.aws.tenstorrent.com/
pip install torch datasets transformers sentencepiece safetensors
```

### Training on TT
```bash
python n_tenstorrent_port.py train \
  --backend tt \
  --preset nano_3x \
  --steps 10000 \
  --batch_size 4 \
  --block 576 \
  --save_dir /workspace/ckpts_tt \
  --tt_dtype bf16 \
  --tt_optimization_level 1
```

### Warm-Start from NVIDIA Checkpoint → Continue Training on TT
```bash
python n_tenstorrent_port.py train \
  --backend tt \
  --preset nano_3x \
  --warmstart_from /workspace/ckpts_expansion/final.pt \
  --steps 10000 \
  --batch_size 4 \
  --block 576 \
  --save_dir /workspace/ckpts_tt_resume \
  --tt_dtype bf16
```

### Inference on TT (using NVIDIA-trained checkpoint)
```bash
python n_tenstorrent_port.py infer \
  --backend tt \
  --mode ar \
  --ckpt /workspace/ckpts_expansion/final.pt \
  --prompt "The capital of France is" \
  --max_new 64 \
  --tt_dtype bf16
```

### Train on CUDA (vast.ai)
```bash
python n.py train --preset base --amp --compile --source ...
```

---

### Presets
`femto` -> `pico` -> `nano` -> `micro` -> `small` -> `base` -> `large` with expansion ratios from 1x to 96x.

### Koyeb Instance Is STATELESS
Save everything externally before the 2-week window ends!
```bash
scp -r /workspace/ckpts/ your_server:~/tt_ckpts/
scp -r /workspace/benchmarks/ your_server:~/tt_benchmarks/
```

### License
Apache-2.0
