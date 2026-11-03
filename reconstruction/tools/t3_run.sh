#!/bin/bash
# T3: GVHMR on free stock clips (Mixkit, Pexels), then checks calibrated on REHAB24-6 squats.
# Resumable: finished downloads, clips and GVHMR outputs are skipped. One GVHMR job at a time, on the
# GPU with the most free memory, only when at least MINFREE MiB are free (another user's job holds most
# of both GPUs), with the preprocessing batch lowered to 4 (t3_gvhmr_lowmem.py, about 3 GB peak).
set -u
B=$HOME/fall2026; R=$B/runs/t3-internet; D=$B/data/internet; L=$B/logs/t3; T=${T:-$B/Fall-2026-Research/reconstruction/tools}  # 2026-11-02 run: T=$B/logs/t3/bin, same files
MINFREE=5000
mkdir -p "$R/clips" "$R/out" "$D/raw" "$L/jobs"
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
log() { echo "$(date +%FT%T) $*" | tee -a "$L/status.txt"; }
pick_gpu() {
  # among GPUs with enough free memory, prefer the least busy one, so another user's active job is not slowed
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits |
    awk -F', ' -v m=$MINFREE '$2 >= m {print $3, -$2, $1}' | sort -n -k1,1 -k2,2 | head -1 | awk '{print $3}'
}
watchdog() {  # protect the other user's job: if a GPU drops below 700 MiB free, stop MY process on it
  while :; do
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | while IFS=', ' read -r gi fr; do
      [ "${fr:-99999}" -lt 700 ] || continue
      for p in $(nvidia-smi -i "$gi" --query-compute-apps=pid --format=csv,noheader); do
        if [ "$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')" = "$USER" ]; then log "WATCHDOG GPU $gi ${fr} MiB free: stopping my pid $p"; kill "$p"; fi
      done
    done
    sleep 2
  done
}
watchdog & WD=$!
trap 'kill $WD 2>/dev/null' EXIT
log "=== start"
while read -r src id kind; do
  [ -z "${src:-}" ] && continue
  case $src in \#*) continue ;; esac
  n=${src}_${id}; raw=$D/raw/$n.mp4; clip=$R/clips/$n.mp4; out=$R/out/$n
  if [ ! -s "$raw" ]; then
    rm -f "$raw.part"
    if [ "$src" = mixkit ]; then
      for res in 1080 720 360; do
        curl -sS -f -L -A "$UA" -o "$raw.part" "https://assets.mixkit.co/videos/$id/$id-$res.mp4" && break
      done
    else
      curl -sS -f -L -A "$UA" -o "$raw.part" "https://www.pexels.com/download/video/$id/"
    fi
    if [ -s "$raw.part" ]; then mv "$raw.part" "$raw"; (cd "$D/raw" && sha256sum "$n.mp4" >> "$D/SHA256SUMS")
    else log "FAIL download $n"; rm -f "$raw.part"; continue; fi
  fi
  if [ ! -s "$clip" ]; then
    ffmpeg -nostdin -v error -y -i "$raw" -t 30 -vf "scale=-2:'min(1080,ih)',fps=30" -an -c:v libx264 -crf 16 "$clip" < /dev/null \
      || { log "FAIL ffmpeg $n"; continue; }
  fi
  if [ -s "$out/joints.npz" ]; then log "skip $n (done)"; continue; fi
  while :; do G=$(pick_gpu); [ -n "$G" ] && break; log "wait: no GPU with >= $MINFREE MiB free ($n)"; sleep 120; done
  log "run $n on GPU $G ($(nvidia-smi -i "$G" --query-gpu=memory.free --format=csv,noheader) free)"
  cd "$B/external/GVHMR"
  systemd-run --user --scope -q -p MemoryMax=32G -p MemorySwapMax=0 \
    env PATH="$B/envs/gvhmr/bin:$PATH" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 \
    /usr/bin/time -v bash -c "python $T/t3_gvhmr_lowmem.py --video $clip --output_root $R/out && python $B/Fall-2026-Research/reconstruction/tools/gvhmr_export_joints.py $out" \
    > "$L/jobs/$n.log" 2>&1 < /dev/null
  rc=$?
  peak=$(grep -oP 'Maximum resident set size \(kbytes\): \K[0-9]+' "$L/jobs/$n.log" | tail -1)
  log "done $n rc $rc peakRSS_MB $(( ${peak:-0} / 1024 ))"
done < "$T/t3_clips.txt"
log "metrics"
cd "$B/external/GVHMR"
"$B/envs/gvhmr/bin/python" -I "$T/t3_internet_metrics.py" --internet "$R" --clips "$T/t3_clips.txt" --rehab "$B/data/rehab24-6" \
  --rehab-chunks "$B/runs/t2-chunks/out-truef" --out "$R/metrics" > "$L/metrics.log" 2>&1
log "metrics rc $?"
log "=== ALL DONE"
