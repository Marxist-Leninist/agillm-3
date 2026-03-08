# AGILLM-3 — Joint AR+SAT Transformer
## OpenTransformers Ltd

698M parameter language model using joint autoregressive + semi-autoregressive (AR+SAT) architecture with tuneable expansion ratio attention.

### Files

| File | Lines | Purpose |
|------|-------|---------|
| `n.py` | 1032 | Original CUDA trainer/inferencer (vast.ai) |
| `n_tenstorrent.py` | 1574 | **TT-XLA port** — training-first, runs on Tenstorrent N300s via Koyeb |
| `n_tt.py` | 1300 | Alternative TT port using torch_ttnn (inference-focused) |
| `nat_mamba.py` | 625 | Mamba/SSM architecture variant |
| `nat_mamba_final.py` | 477 | Final Mamba variant |
| `nat_mamba_simple.py` | 285 | Simplified Mamba variant |

### Architecture
- **Tuneable Attention MHA** with rank projection (U matrix) — configurable expansion ratio
- **ALiBi** positional encoding (no learned positional embeddings)
- **Joint AR+SAT** training — autoregressive + semi-autoregressive heads trained simultaneously
- **Variable stride SAT** — gated stride prediction for speculative generation

### Tenstorrent N300s Training (n_tenstorrent.py)

Training-first port using TT-XLA / pjrt-plugin-tt. Key design decisions:
- Conservative `torch_xla` device loop (matches tt-blacksmith training recipes)
- Model weights cast to **bfloat16** on TT, fp32 on CUDA
- Masks use `-1e9` instead of `-inf` (bf16 compatibility)
- Checkpoints saved as **portable CPU fp32** — cross-compatible TT <-> CUDA <-> CPU
- TT inference uses full-recompute path (dynamic KV-cache not supported on XLA)
- `--accelerator {auto,tt,cuda,cpu}` flag for runtime selection

```bash
# Train on Tenstorrent
python n_tenstorrent.py train --accelerator tt --preset base --block 576 --steps 1000

# Resume CUDA checkpoint on TT
python n_tenstorrent.py train --accelerator tt --resume ckpts/final.pt

# Train on CUDA (vast.ai)
python n.py train --preset base --amp --compile --source ...

# Infer on TT
python n_tenstorrent.py infer --accelerator tt --mode ar --ckpt ckpts/final.pt --prompt "Hello"
```

### Koyeb Setup (free 2-week N300s access)
1. Sign up at [koyeb.com](https://koyeb.com)
2. Join [Tenstorrent Discord](https://discord.gg/tenstorrent)
3. Activate coupon: `TTDEPLOY25FADEV2W`
4. Deploy with VSCode tunnel or Docker

### Install (TT-XLA)
```bash
pip install pjrt-plugin-tt --extra-index-url https://pypi.eng.aws.tenstorrent.com/
pip install transformers datasets
```

### Presets
`femto` -> `pico` -> `nano` -> `micro` -> `small` -> `base` -> `large` with expansion ratios from 1x to 96x.

### License
Apache-2.0
