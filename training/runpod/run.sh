#!/usr/bin/env bash
# Full pipeline on one GPU pod: fetch data -> bake degradations -> train balanced + quality ->
# export JSON into <repo>/models/. Every step is resumable: re-running skips finished work
# (fetched clips, baked splits, exported models) and resumes training from last.pt.
#
#   bash training/runpod/run.sh
#   SCALE_POSITIONS=0.25 BAL_ITERS=50000 QUAL_ITERS=50000 bash training/runpod/run.sh   # cheap trial
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING="$(dirname "$HERE")"
REPO="$(dirname "$TRAINING")"

WORK="${WORK:-/workspace/lasr}"                 # put this on the pod's persistent volume
WORKERS="${WORKERS:-$(nproc)}"                  # bake processes (CPU bound)
LOADER_WORKERS="${LOADER_WORKERS:-$(( $(nproc) > 12 ? 10 : $(nproc) - 2 ))}"
SCALE_POSITIONS="${SCALE_POSITIONS:-1.0}"       # <1 for a smaller/cheaper dataset
VARIANTS="${VARIANTS:-4}"                       # random degradations per HR clip (train)
KEEP="${KEEP:-3}"                               # frames kept per encoded clip
BAL_ITERS="${BAL_ITERS:-200000}"
QUAL_ITERS="${QUAL_ITERS:-300000}"
PARALLEL="${PARALLEL:-0}"                       # 1 = train both models at once (GPU is underused by one)
INCLUDE_RESEARCH="${INCLUDE_RESEARCH:-0}"       # 1 = also research-only sources (see DATASETS.md)
INCLUDE_SHAREALIKE="${INCLUDE_SHAREALIKE:-0}"   # 1 = also CC BY-SA sources (see DATASETS.md)
SKIP_SETUP="${SKIP_SETUP:-0}"

cd "$TRAINING"
[ "$SKIP_SETUP" = 1 ] || bash "$HERE/setup.sh"
mkdir -p "$WORK" "$REPO/models"
stamp() { date '+%H:%M:%S'; }

echo "[$(stamp)] 1/4 fetch HR clips -> $WORK/hr"
research=()
[ "$INCLUDE_SHAREALIKE" = 1 ] && research=(--include-sharealike)
[ "$INCLUDE_RESEARCH" = 1 ] && research=(--include-research)
python data/fetch.py --out "$WORK/hr" --scale-positions "$SCALE_POSITIONS" --threads 16 "${research[@]}"

echo "[$(stamp)] 2/4 bake degradations -> $WORK/baked"
if [ ! -e "$WORK/baked/train/shard_0000_lr.npy" ]; then
  python data/bake.py --hr "$WORK/hr" --out "$WORK/baked" --split train \
    --variants "$VARIANTS" --keep "$KEEP" --config configs/base.yaml --workers "$WORKERS"
fi
if [ ! -e "$WORK/baked/val/shard_0000_lr.npy" ]; then
  python data/bake.py --hr "$WORK/hr" --out "$WORK/baked" --split val \
    --variants 2 --keep 3 --config configs/base.yaml --workers "$WORKERS"
fi

train() {  # name iterations
  local name="$1" iters="$2" out="$WORK/runs/$1" resume=()
  [ -e "$out/last.pt" ] && resume=(--resume "$out/last.pt")
  python train.py --config "configs/$name.yaml" "${resume[@]}" --set \
    "data.root=$WORK/baked" "log.out_dir=$out" "train.iterations=$iters" "train.workers=$LOADER_WORKERS" \
    2>&1 | tee -a "$WORK/train_$name.log"
}

echo "[$(stamp)] 3/4 train"
if [ "$PARALLEL" = 1 ]; then
  LOADER_WORKERS=$(( LOADER_WORKERS / 2 > 2 ? LOADER_WORKERS / 2 : 2 ))
  train balanced "$BAL_ITERS" & pid=$!
  train quality "$QUAL_ITERS"
  wait "$pid"
else
  train balanced "$BAL_ITERS"
  train quality "$QUAL_ITERS"
fi

echo "[$(stamp)] 4/4 export"
for name in balanced quality; do
  python export.py "$WORK/runs/$name/best.pt" --out "$REPO/models/$name.json" --onnx "$WORK/runs/$name/$name.onnx"
  cp "$WORK/runs/$name/val_latest.json" "$WORK/runs/$name/val_final.json"
done
echo "[$(stamp)] done. Models: $REPO/models/balanced.json, $REPO/models/quality.json"
echo "validation reports: $WORK/runs/*/val_final.json (gain_vs_lanczos_db must be > 0)"
