FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-amd64:latest

# AGILLM-3 on Tenstorrent N300s — Koyeb Deployment
# OpenTransformers Ltd

WORKDIR /workspace

# Python deps
RUN pip install --no-cache-dir \
    torch \
    transformers \
    datasets \
    huggingface_hub \
    && pip install git+https://github.com/tenstorrent/pytorch2.0_ttnn.git || true

# Copy scripts
COPY n_tt.py /workspace/n_tt.py
COPY setup_koyeb.sh /workspace/setup_koyeb.sh
RUN chmod +x /workspace/setup_koyeb.sh

# Create dirs
RUN mkdir -p /workspace/ckpts /workspace/benchmarks /workspace/logs

# Default: run setup then drop to shell
CMD ["/bin/bash"]
