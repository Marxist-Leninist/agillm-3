FROM ghcr.io/tenstorrent/tt-xla-slim:latest

# AGILLM-3 on Tenstorrent N300s — Koyeb Deployment
# OpenTransformers Ltd — TT-XLA training-first port

WORKDIR /workspace

# TT-XLA runtime deps
RUN pip install --no-cache-dir \
    torch \
    transformers \
    datasets \
    huggingface_hub \
    sentencepiece \
    safetensors \
    && pip install pjrt-plugin-tt --extra-index-url https://pypi.eng.aws.tenstorrent.com/ || true

# Copy training scripts
COPY n_tenstorrent_port.py /workspace/n_tenstorrent_port.py
COPY n.py /workspace/n.py
COPY entrypoint.sh /workspace/entrypoint.sh

# Create dirs
RUN mkdir -p /workspace/ckpts /workspace/ckpts_tt /workspace/logs

# Env defaults for TT runtime
ENV PJRT_DEVICE=TT
ENV XLA_STABLEHLO_COMPILE=1

# Run training on startup
CMD ["bash", "/workspace/entrypoint.sh"]
