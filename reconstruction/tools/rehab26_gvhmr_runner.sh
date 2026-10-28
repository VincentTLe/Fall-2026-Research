#!/usr/bin/env bash
# Resumable GVHMR runner for T5 on physical GPU 1 only. Run by the enabled user unit t5-gvhmr.service.
# - starts a chunk only while no other job (cmdline without /runs/t5/) holds GPU 1
# - watchdog: any of our processes seen on GPU 0 is killed and logged as WRONG-GPU (chunk is retried later)
# - finished chunks (mp33.npz present) are skipped, so a restart or reboot only loses the chunks in flight
set -u
P=${T5_PAR:-6}
LIMIT=${T5_LIMIT:-0}   # >0: launch at most this many chunks (canary), then wait and exit
LAUNCHED=0
T=$(dirname "$(readlink -f "$0")")
R=$HOME/fall2026/runs/t5/gvhmr; LG=$HOME/fall2026/logs/t5-gvhmr; mkdir -p $LG
GPU0=GPU-a1eda44a-2eaf-3321-a457-772d62cd483f; GPU1=GPU-dc002436-5d9a-3b94-b629-7af3611b3a1c
log() { echo "$(date '+%F %T') $*" >> $LG/runner.log; }
foreign_on_gpu1() {
  for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F', ' -v g=$GPU1 '$2==g{print $1}'); do
    tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q '/runs/t5/' || { echo $p; return 0; }
  done; return 1; }
gpu1_free_stably() {   # no active p2-* user unit and no foreign process on GPU 1 for 5 checks, 60 s apart
  systemctl --user list-units --state=active --plain --no-legend 'p2-*' | grep -q . && return 1
  for i in 1 2 3 4 5; do foreign_on_gpu1 >/dev/null && return 1; [ $i -lt 5 ] && sleep 60; done; return 0; }
watchdog() { while sleep 20; do
  for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F', ' -v g=$GPU0 '$2==g{print $1}'); do
    if tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q '/runs/t5/'; then log "WRONG-GPU pid $p on GPU0, killing"; kill $p; fi
  done
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ' >> $LG/gpu.log; echo >> $LG/gpu.log
done; }
watchdog & WD=$!
trap 'kill $WD 2>/dev/null' EXIT
log "runner start, parallel $P"
for pass in 1 2 3; do       # later passes retry failed chunks
  while read -r V N S E; do
    [ -s $R/out/$N/mp33.npz ] && continue
    NOK=$(grep -l "RESULT .* rc 0 peak" $LG/*-f*.log 2>/dev/null | wc -l); NBAD=$(grep -lE "RESULT .*(FAIL|rc [1-9])" $LG/*-f*.log 2>/dev/null | wc -l)
    if [ $NOK -eq 0 ] && [ $NBAD -ge 3 ]; then log "FAIL-FAST: 3 failed chunks and none ok, stopping"; touch $R/STOP; fi
    [ -e $R/STOP ] && { log "STOP file, exiting"; wait; exit 0; }
    if [ -z "${GATE_OK:-}" ]; then
      until gpu1_free_stably; do log "GPU1 not free (p2 unit or foreign pid), waiting"; sleep 120; done
      GATE_OK=1; log "GPU1 free for 5 min, starting"
    fi
    while f=$(foreign_on_gpu1); do log "GPU1 has foreign pid $f, waiting"; sleep 120; done
    while [ $(jobs -rp | wc -l) -gt $P ]; do sleep 5; done   # watchdog is one of the jobs
    bash $T/rehab26_gvhmr_job.sh $V $N $S $E < /dev/null &
    LAUNCHED=$((LAUNCHED+1)); log "launched $N"
    if [ $LIMIT -gt 0 ] && [ $LAUNCHED -ge $LIMIT ]; then wait $!; log "canary done: $(tail -1 $LG/$N.log)"; exit 0; fi
    sleep 15
  done < $R/jobs.txt
  wait $(jobs -rp | grep -v "^$WD$")
  log "pass $pass done: $(ls $R/out/*/mp33.npz 2>/dev/null | wc -l) / $(wc -l < $R/jobs.txt) chunks"
done
log "RUNNER_DONE"
