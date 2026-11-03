#!/bin/bash
# T4b: two GaussianAvatar runs on REHAB24-6 PM_037, started by the enabled user unit t4b-avatar.service.
#   run 1 c17-twocam: camera-17 images, poses from T2c's two-camera SMPL fit (new cameras) moved into camera 17
#   run 2 c18-gvhmr : camera-18 images (transposed video), GVHMR poses
# Same recipe as T4 Step 3 (logs/p3-rehab-resumable.sh): stage 1, 200 epochs, --pose_op_start_iter 10, --save_epoch 20;
# eval 120..200 with raw test poses; FPS; turntable; then (extra, ~3 min) the refined-test-pose diagnostic and the
# novel-pose and GT|render videos.
# GPU use, to protect the other user's jobs:
#   - before each run: poll every 5 min until one GPU has >= 24 GB free (GaussianAvatar needs ~21 GB); use that GPU
#   - while any GPU step runs: a watchdog reads that GPU's free memory every 5 s; below 1 GB it stops the step's
#     scope and this script exits 0 ("STOPPED"), so the unit does not restart. Restart by hand to resume.
# Resumable: training resumes from the newest loadable checkpoint; every later step is skipped when its output exists.
set -u
H=/net/home/tle/fall2026; GA=$H/external/GaussianAvatar; PY=$H/envs/gs-avatar/bin/python
TOOLS=$H/wt-t4b/reconstruction/tools; FPS=$H/logs/p3-ga-fps.py; REF=$H/logs/p3-ga-refine-test-poses.py
TURN=$H/logs/p3-rehab-make-turn.py
LOGS=$H/logs/t4b; ST=$H/logs/t4b-status.txt; R=/home/tle/fall2026/runs/t4b
NEED_MIB=24576; MIN_FREE_MIB=1024
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=8 PYTHONPATH=$GA
cd $GA
mkdir -p $LOGS

status() { echo "$(date -Is) $*" | tee -a $ST; }
free_mib() { nvidia-smi -i $1 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '; }

wait_gpu() {  # sets G to a GPU with >= NEED_MIB free; polls every 5 min, logs once an hour
  local n=0
  while true; do
    for g in 0 1; do
      local f=$(free_mib $g)
      if [ -n "$f" ] && [ "$f" -ge $NEED_MIB ]; then G=$g; export CUDA_VISIBLE_DEVICES=$g
        status "      GPU $g has $f MiB free -> using it"; return 0; fi
    done
    [ $((n % 12)) -eq 0 ] && status "WAIT  no GPU with ${NEED_MIB} MiB free ($(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tr '\n' ' '))"
    n=$((n + 1)); sleep 300
  done
}

run() {  # name cmd...: CPU-only steps and GPU steps alike; GPU steps are watched
  local name=$1; shift; local log=$LOGS/$RUN-$name.log unit=t4b-$RUN-$name-$$
  status "START $RUN $name (log $log)"; { echo "+ $(date -Is) cwd $PWD GPU ${G:-none}"; echo "+ $*"; } >> $log
  systemd-run --user --scope -q --unit=$unit -p MemoryMax=64G -p MemorySwapMax=0 nice -n 19 /usr/bin/time -v "$@" >> $log 2>&1 &
  local pid=$! low=0
  while kill -0 $pid 2>/dev/null; do
    local f=$(free_mib $G)
    if [ -n "$f" ] && [ "$f" -lt $MIN_FREE_MIB ]; then
      low=1; status "STOP  GPU $G free memory $f MiB < $MIN_FREE_MIB during $RUN $name: stopping it"
      systemctl --user stop $unit.scope; break
    fi
    sleep 5
  done
  wait $pid; local rc=$?
  echo "+ exit $rc $(date -Is)" >> $log; status "END   $RUN $name exit=$rc"
  if [ $low -eq 1 ]; then status "=== STOPPED (low GPU memory). Resume: systemctl --user restart t4b-avatar"; exit 0; fi
  return $rc
}

latest_ok() { for e in $(ls $1/net 2>/dev/null | sed -n 's/^iteration_//p' | sort -rn); do
    $PY -c "import torch,sys; torch.load(sys.argv[1], map_location='cpu')" $1/net/iteration_$e/net.pth 2>/dev/null && { echo $e; return; }
    status "      $1 iteration_$e does not load, trying an older one"; done; }
video() { [ -f $2 ] || run $1 ffmpeg -v error -y -framerate $3 -i $4/%05d.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
            -c:v libx264 -pix_fmt yuv420p -crf 18 $2; }

one_run() {
  RUN=$1; local D=$R/$RUN M=$R/$RUN/stage1
  if [ -f $M/DONE ]; then status "      $RUN already done"; return 0; fi
  status "=== $RUN"
  wait_gpu
  # ---- train
  local last=$(latest_ok $M)
  if [ "$last" != "200" ]; then
    local resume=""; [ -n "$last" ] && resume="--checkpoint_epochs $last"
    status "      $RUN train: newest loadable checkpoint '${last:-none}', training to 200"
    ( while true; do nvidia-smi -i $G --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv,noheader; sleep 2; done ) \
      >> $LOGS/$RUN-train-gpumem.csv & local SAMPLER=$!
    run train $PY train.py -s $D -m $M --train_stage 1 --pose_op_start_iter 10 --save_epoch 20 $resume
    local rc=$?; kill $SAMPLER
    [ $rc -eq 0 ] && [ -f $M/net/iteration_200/net.pth ] || { status "ABORT $RUN training rc=$rc"; exit 1; }
  fi
  # ---- eval on the held-out repetitions (raw test poses)
  for e in 120 140 160 180 200; do
    [ -f $M/net/iteration_$e/net.pth ] || continue
    [ -f $M/metrics_epoch$e.json ] && continue
    rm -f $GA/results.txt
    run eval-e$e $PY eval.py -s $D -m $M --epoch $e && cp $GA/results.txt $M/results_epoch$e.txt && \
      status "      $RUN epoch $e (raw test poses): $(tr '\n' ' ' < $M/results_epoch$e.txt)"
    run metrics-e$e $PY -I $TOOLS/image_metrics.py $M/test_free/ours_$e $M/test_free/gt_image --device cuda --json $M/metrics_epoch$e.json
  done
  # ---- speed
  grep -qs 'frames .* fps' $LOGS/$RUN-fps.log || run fps $PY $FPS -s $D -m $M --epoch 200
  status "      $RUN FPS: $(grep -hs 'frames .* fps' $LOGS/$RUN-fps.log | tail -1)"
  # ---- 360-degree turntable (prepared on CPU: $D/turn_pose)
  if [ ! -f $M/turntable_epoch200.mp4 ]; then
    rm -rf $M/novel_pose
    run turntable $PY render_novel_pose.py -s $D -m $M --epoch 200 --test_folder $D/turn_pose && \
      video turntable-video $M/turntable_epoch200.mp4 30 $M/novel_pose/ours_200 && \
      { rm -rf $M/novel_pose_turntable; mv $M/novel_pose $M/novel_pose_turntable; }
  fi
  # ---- extras (as T4): refined-test-pose diagnostic (upper bound, fit on the test images), novel-pose and test videos
  if [ ! -f $M/metrics_epoch200_refined_diagnostic.json ]; then
    run refine-diagnostic $PY $REF -s $D -m $M --epoch 200 --protocol diagnostic && \
    run refine-metrics $PY -I $TOOLS/image_metrics.py $M/test_refined_diagnostic/ours_200 $M/test_refined_diagnostic/gt_image \
        --device cuda --json $M/metrics_epoch200_refined_diagnostic.json
  fi
  if [ ! -f $M/novel_pose_epoch200.mp4 ]; then
    rm -rf $M/novel_pose
    run novelpose $PY render_novel_pose.py -s $D -m $M --epoch 200 && \
      video novelpose-video $M/novel_pose_epoch200.mp4 30 $M/novel_pose/ours_200 && \
      { rm -rf $M/novel_pose_testpose; mv $M/novel_pose $M/novel_pose_testpose; }
  fi
  if [ ! -f $M/test_gt_vs_render_epoch200.mp4 ] && [ -d $M/test_free/ours_200 ]; then
    run testvideo ffmpeg -v error -y -framerate 6 -i $M/test_free/gt_image/%05d.png -framerate 6 -i $M/test_free/ours_200/%05d.png \
      -filter_complex "[0][1]hstack,pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p -crf 18 $M/test_gt_vs_render_epoch200.mp4
  fi
  [ -f $M/metrics_epoch200.json ] && [ -f $M/turntable_epoch200.mp4 ] && touch $M/DONE
  status "      $RUN finished: $(cat $M/results_epoch200.txt 2>/dev/null | tr '\n' ' ')"
}

status "=== t4b_avatar_runs.sh started, PID $$, boot $(uptime -s)"
one_run c17-twocam
one_run c18-gvhmr
if [ -f $R/c17-twocam/stage1/DONE ] && [ -f $R/c18-gvhmr/stage1/DONE ]; then
  systemctl --user disable t4b-avatar.service 2>&1 | tee -a $ST
  status "=== ALL DONE"
else
  status "=== finished with missing outputs (no DONE marker); see the END lines above"; exit 1
fi
