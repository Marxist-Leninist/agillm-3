# n_tenstorrent_port.py fix notes

## What was wrong

1. **TT init on Koyeb was failing before training started**
   - The failure you hit (`NoAccess error`, `ARC core failed to start`) is primarily an environment / device-access issue, not just a model bug.
   - On Koyeb, fix `/dev/tenstorrent/*` permissions in the container *before* launching Python.

2. **Custom TT compile options were being forced too early**
   - `torch_xla.set_custom_compile_options(...)` is a tuning knob, not required for bring-up.
   - Safer first boot: do not set compile options unless explicitly requested.

3. **TT mask numerics were too optimistic**
   - Using literal `-inf` with BF16 can be brittle on TT. Use a large finite negative (`-1e9`) on TT.

4. **Checkpoint warmstart path missed compiled-key cleanup**
   - `_safe_load_any` now strips `_orig_mod.` prefixes so CUDA-compiled checkpoints warmstart cleanly.

5. **TT synchronization was too weak around stepping / loss reads**
   - Added explicit sync before reading TT loss values and around TT optimizer stepping.

6. **`status` should not require transformers/datasets**
   - Tokenizer and dataset imports are now lazy.

## Startup snippet for Koyeb

```bash
#!/usr/bin/env bash
set -euo pipefail

# Koyeb / Tenstorrent bring-up
if [ -e /dev/tenstorrent/0 ]; then
  chmod 666 /dev/tenstorrent/* || true
fi

export PJRT_DEVICE=TT
export XLA_STABLEHLO_COMPILE=1

python /workspace/n_tenstorrent_port_fixed.py train \
  --backend tt \
  --preset base \
  --warmstart_from /workspace/ckpts/pretrain_step09131773.pt \
  --save_dir /workspace/ckpts \
  --save_every_sec 1800 \
  --tt_dtype bf16
```

## Safer first smoke test

```bash
python /workspace/n_tenstorrent_port_fixed.py train \
  --backend tt \
  --preset nano_3x \
  --steps 5 \
  --block 128 \
  --batch_size 1 \
  --save_dir /workspace/ckpts_smoke \
  --tt_dtype bf16
```

After that works, try adding:

```bash
--tt_optimization_level 0
```

and only then:

```bash
--tt_optimization_level 1
```
