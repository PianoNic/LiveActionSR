#!/usr/bin/env bash
# Prepare a RunPod PyTorch pod (or any Linux box with CUDA torch already installed):
# installs a static ffmpeg with libx264/libx265/libsvtav1/libvmaf if the system one lacks them,
# then the Python requirements. Safe to re-run.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING="$(dirname "$HERE")"
FFMPEG_URL="${FFMPEG_URL:-https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz}"

have_encoders() {
  command -v ffmpeg >/dev/null 2>&1 || return 1
  local enc
  enc="$(ffmpeg -hide_banner -encoders 2>/dev/null)"
  for e in libx264 libx265 libsvtav1; do
    grep -q " $e " <<<"$enc" || return 1
  done
}

if ! have_encoders; then
  echo "installing static ffmpeg (x264/x265/svt-av1/vmaf) from $FFMPEG_URL"
  command -v xz >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq xz-utils curl >/dev/null; }
  tmp="$(mktemp -d)"
  curl -fsSL "$FFMPEG_URL" -o "$tmp/ffmpeg.tar.xz"
  tar -xJf "$tmp/ffmpeg.tar.xz" -C "$tmp"
  install -m 0755 "$tmp"/ffmpeg-*/bin/ffmpeg "$tmp"/ffmpeg-*/bin/ffprobe /usr/local/bin/
  rm -rf "$tmp"
  hash -r
fi
ffmpeg -hide_banner -version | head -1

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
pip install -q -r "$TRAINING/requirements.txt"
