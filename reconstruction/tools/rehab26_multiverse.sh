#!/usr/bin/env bash
# Multiverse over the paper-silent choices of the separability metric (resumable: skips finished configs).
# Usage: rehab26_multiverse.sh <out_dir>
set -u
OUT=${1:?out dir}; mkdir -p "$OUT"
PY=$HOME/fall2026/envs/gvhmr/bin/python
T=$(dirname "$(readlink -f "$0")")
DATA=$HOME/fall2026/data/rehab26-viewangles
SPEC=$HOME/fall2026/runs/t5/spec
for tau in mean pred; do for floor in q1_tau none; do for add in false true; do for f6 in xy yz; do
for d7 in F1 F5; do for asp in 1080,1920 1,1; do for pl in true false; do
  name="tau-${tau}_floor-${floor}_add-${add}_f6-${f6}_dtw7-${d7}_asp-${asp/,/x}_plane-${pl}"
  [ -s "$OUT/$name.graded.csv" ] && continue
  "$PY" -I "$T/rehab26_psi_run.py" --data "$DATA" --seeds 0-4 --procs 16 --out "$OUT/$name.csv" \
     --cfg tau=$tau floor=$floor include_additional=$add f6_ex7_plane=$f6 dtw_feature=1:F5,7:$d7,6:F15 \
           aspect=$asp plane3d=$pl \
  && "$PY" -I "$T/rehab26_grade.py" "$OUT/$name.csv" "$SPEC" --label "$name"
done; done; done; done; done; done; done
echo MULTIVERSE_DONE
