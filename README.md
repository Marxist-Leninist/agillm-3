# AGILLM-3 × Tenstorrent N300s
## OpenTransformers Ltd — Inference Benchmark Package

### What This Is
Modified AGILLM-3 trainer/inferencer with a device abstraction layer that supports:
- **CUDA** (primary, for vast.ai training)
- **Tenstorrent Wormhole N300s** (experimental, via Koyeb)
- **CPU** (fallback)

### Reality Check
From Tenstorrent's own model support table:
- GPT-2 inference: 🚧 (partial, some ops fall back to CPU)
- Llama/Falcon/OPT/GPTNeo: ❌ (don't compile at all)
- Training throughput: ~10-100x slower than inference on TT
- pytorch2.0_ttnn is deprecated → TT-Forge is the future

**Bottom line**: This is for R&D benchmarking, not replacing vast.ai training.

### Koyeb Setup (2-week free access)
```bash
# 1. Activate coupon at koyeb.com: TTDEPLOY25FADEV2W
# 2. Deploy container or use VSCode tunnel
# 3. Run setup
bash setup_koyeb.sh
# 4. Copy a small checkpoint for testing
scp vast_instance:/workspace/ckpts/pretrain_step*.pt /workspace/ckpts/
```

### Usage
```bash
# Inference (auto-detects TT > CUDA > CPU)
python3 n_tt.py infer --mode ar --ckpt ckpts/model.pt --prompt "Hello world"

# Force TT backend
python3 n_tt.py --backend tt infer --mode ar --ckpt ckpts/model.pt --prompt "Hello"

# Test which ops compile to TT-NN
python3 n_tt.py compile-test --ckpt ckpts/model.pt

# Benchmark (N runs, reports tok/s stats)
python3 n_tt.py benchmark --ckpt ckpts/model.pt --runs 10

# Training (CUDA recommended, TT experimental)
python3 n_tt.py --backend cuda train --preset base --amp --source ...
```

### Key Modifications from n.py
1. `DeviceBackend` class abstracts CUDA/TT/CPU
2. ALiBi biases are cached (avoids dynamic computation TT can't handle)
3. Removed `@torch._dynamo.disable` decorators (TT needs to trace)
4. KV-cache disabled on TT (falls back to full recomputation)
5. `compile-test` command reports op compatibility
6. `benchmark` command saves JSON results for comparison
7. All mask functions take explicit `device` parameter

### What To Actually Test
1. Does the model compile? → `compile-test`
2. How fast is inference? → `benchmark` (compare to CUDA numbers)
3. Which ops fall back? → Check compile-test output
4. Is 24GB GDDR6 enough for 698M? → Yes, ~2.8GB fp16

### Instance Is STATELESS
Save everything externally before the 2-week window ends!
```bash
# Save benchmarks
scp -r /workspace/benchmarks/ your_server:~/tt_benchmarks/
```
