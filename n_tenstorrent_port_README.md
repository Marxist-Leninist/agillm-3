# Tenstorrent port notes

## What changed

- Added `--backend auto|cuda|tt|cpu`
- Added Tenstorrent runtime setup through TT-XLA / PJRT
- Training path uses XLA-style optimizer stepping on TT (`xm.optimizer_step`)
- Checkpoints are always saved with CPU tensors so they can move between CUDA and TT
- TT inference avoids dynamic KV-cache assumptions and uses a static-shape path for robustness
- Added TT tuning flags:
  - `--tt_dtype fp32|bf16`
  - `--tt_bfp8`
  - `--tt_weight_bfp8`
  - `--tt_optimization_level`
  - `--tt_trace`
  - `--tt_spmd` (experimental)

## Koyeb setup sketch

```bash
python3 -m venv .xla-venv
source .xla-venv/bin/activate
pip install pjrt-plugin-tt --extra-index-url https://pypi.eng.aws.tenstorrent.com/
pip install torch datasets transformers sentencepiece safetensors
```

## Training example

```bash
python n_tenstorrent_port.py train \
  --backend tt \
  --preset nano_3x \
  --steps 10000 \
  --batch_size 4 \
  --block 576 \
  --save_dir /workspace/ckpts_expansion_tt \
  --tt_dtype bf16 \
  --tt_optimization_level 1
```

## Warm-start from NVIDIA checkpoint and continue training on TT

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

## Inference from NVIDIA-trained checkpoint on TT

```bash
python n_tenstorrent_port.py infer \
  --backend tt \
  --mode ar \
  --ckpt /workspace/ckpts_expansion/final.pt \
  --prompt "The capital of France is" \
  --max_new 64 \
  --tt_dtype bf16
```

## Experimental two-chip attempt on N300

```bash
python n_tenstorrent_port.py train \
  --backend tt \
  --tt_spmd \
  --batch_size 8 \
  --block 576 \
  --steps 10000
```

Use the SPMD flag carefully. It is intentionally marked experimental in the script.
