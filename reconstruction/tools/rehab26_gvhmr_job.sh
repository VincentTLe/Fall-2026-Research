#!/usr/bin/env bash
# One GVHMR chunk for REHAB26 (T5), modelled on P2's p2-chunk-job.sh.
# usage: rehab26_gvhmr_job.sh <video> <name> <start> <end>    (inclusive frame indices of the video)
set -u
V=$1; N=$2; S=$3; E=$4
R=$HOME/fall2026/runs/t5/gvhmr; C=$R/clips/$N.mp4; L=$HOME/fall2026/logs/t5-gvhmr/$N.log
T=$(dirname "$(readlink -f "$0")")
mkdir -p $R/clips $R/out $HOME/fall2026/logs/t5-gvhmr; exec > $L 2>&1
echo "JOB $N start $(date +%T)"
if [ -s $R/out/$N/mp33.npz ]; then echo "already done"; echo "RESULT $N rc 0 skip"; exit 0; fi
rm -rf $R/out/$N
[ -s $C ] || ffmpeg -nostdin -v error -y -i $V -vf "trim=start_frame=$S:end_frame=$((E+1)),setpts=PTS-STARTPTS" -an -c:v libx264 -crf 12 -r 30 $C
NF=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $C)
echo "CLIP frames $NF expected $((E-S+1))"; [ "$NF" = "$((E-S+1))" ] || { echo "RESULT $N FAIL clipframes"; rm -f $C; exit 1; }
cd $HOME/fall2026/external/GVHMR
systemd-run --user --scope -q -p MemoryMax=32G -p MemorySwapMax=0 \
  env PATH=$HOME/fall2026/envs/gvhmr/bin:$PATH CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 \
  /usr/bin/time -v bash -c "python $T/gvhmr_run.py --video $C --output_root $R/out && python $T/gvhmr_export_mp.py $R/out/$N" 2>&1 \
  | tr '\r' '\n' | grep -vE 'it/s\]|s/it\]|^\s*$|Warning|warn|register_new_resolver|migration|^\s*OmegaConf|^\s*cfg\.hydra'
RC=${PIPESTATUS[0]}
[ $RC = 0 ] && [ -s $R/out/$N/mp33.npz ] && rm -f $C
PEAK=$(grep -oP 'Maximum resident set size \(kbytes\): \K[0-9]+' $L | tail -1)
echo "RESULT $N rc $RC peakRSS_MB $(( ${PEAK:-0} / 1024 )) end $(date +%T)"
