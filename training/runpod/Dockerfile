# Self-contained training image (alternative to running setup.sh on RunPod's PyTorch template).
# Build from the repository root:
#   docker build -f training/runpod/Dockerfile -t <you>/lasr-training .
# On RunPod: use the image in a pod template, mount a volume at /workspace, then run
#   bash /opt/lasr/training/runpod/run.sh
# (Models are written to /opt/lasr/models/; copy them out or set REPO paths via a clone instead.)
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ARG FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends curl xz-utils ca-certificates git \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL "$FFMPEG_URL" -o /tmp/ffmpeg.tar.xz \
 && tar -xJf /tmp/ffmpeg.tar.xz -C /tmp \
 && install -m 0755 /tmp/ffmpeg-*/bin/ffmpeg /tmp/ffmpeg-*/bin/ffprobe /usr/local/bin/ \
 && rm -rf /tmp/ffmpeg* \
 && ffmpeg -hide_banner -encoders | grep -E "libx264|libx265|libsvtav1"

WORKDIR /opt/lasr
COPY training/requirements.txt training/requirements.txt
RUN pip install --no-cache-dir -r training/requirements.txt
COPY training/ training/
RUN mkdir -p models

ENV SKIP_SETUP=1 WORK=/workspace/lasr
CMD ["bash", "/opt/lasr/training/runpod/run.sh"]
