#!/bin/bash
# T3 person-choice rerun: multi-person clips plus single-person controls, with t3_gvhmr_pick.py.
set -u
B=$HOME/fall2026; R=$B/runs/t3-internet; L=$B/logs/t3; T=$B/logs/t3/bin; OUT=$R/out-pick
mkdir -p "$OUT" "$L/jobs-pick"
log() { echo "$(date +%FT%T) $*" | tee -a "$L/status-pick.txt"; }
watchdog() { while :; do nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | while IFS=', ' read -r gi fr; do
  [ "${fr:-99999}" -lt 700 ] || continue
  for p in $(nvidia-smi -i "$gi" --query-compute-apps=pid --format=csv,noheader); do
    [ "$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')" = "$USER" ] && { log "WATCHDOG GPU $gi ${fr} MiB free: stopping my pid $p"; kill "$p"; }
  done; done; sleep 2; done; }
watchdog & WD=$!; trap 'kill $WD 2>/dev/null' EXIT
pick_gpu() { nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits |
  awk -F', ' '$2 >= 5000 {print $3, -$2, $1}' | sort -n -k1,1 -k2,2 | head -1 | awk '{print $3}'; }
log "=== start"
# default: the 9 clips of 2026-11-03 02:10; with a clip list (src id kind per line), those clips
if [ $# -ge 1 ]; then NAMES=$(grep -v '^#' "$1" | awk 'NF>=2 {print $1"_"$2}'); else
  NAMES="mixkit_23914 mixkit_24305 mixkit_24304 mixkit_44438 mixkit_49038 mixkit_52106 mixkit_36062 mixkit_52117 pexels_7426039"; fi
for n in $NAMES; do
  [ -s "$OUT/$n/joints.npz" ] && { log "skip $n"; continue; }
  while :; do G=$(pick_gpu); [ -n "$G" ] && break; log "wait GPU ($n)"; sleep 120; done
  log "run $n on GPU $G"
  cd "$B/external/GVHMR"
  systemd-run --user --scope -q -p MemoryMax=32G -p MemorySwapMax=0 \
    env PATH="$B/envs/gvhmr/bin:$PATH" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
    bash -c "python $T/t3_gvhmr_pick.py --video $R/clips/$n.mp4 --output_root $OUT && python $B/Fall-2026-Research/reconstruction/tools/gvhmr_export_joints.py $OUT/$n" \
    > "$L/jobs-pick/$n.log" 2>&1 < /dev/null
  log "done $n rc $? $(grep -a -o 'T3PICK.*' "$L/jobs-pick/$n.log" | head -1)"
done
log "=== ALL DONE"
