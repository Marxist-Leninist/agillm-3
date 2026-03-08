FROM ghcr.io/tenstorrent/tt-xla-slim:latest

# AGILLM-3 on Tenstorrent N300s - Koyeb Deployment
# OpenTransformers Ltd - TT-XLA training-first port

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
COPY run.sh /workspace/run.sh
RUN chmod +x /workspace/run.sh

# Create dirs
RUN mkdir -p /workspace/ckpts /workspace/logs

# Env defaults for TT runtime
ENV PJRT_DEVICE=TT
ENV XLA_STABLEHLO_COMPILE=1

# Run training entrypoint
CMD ["/workspace/run.sh"]
