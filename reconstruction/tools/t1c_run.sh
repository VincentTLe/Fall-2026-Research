#!/usr/bin/env bash
# T1c: 7 and 8 spread cameras (+ 8 bunched) on Panoptic, triangulation only, same procedure as T1 (CPU).
set -uo pipefail
cd ~/fall2026; export PATH=$HOME/fall2026/envs/easymocap/bin:$PATH
T=Fall-2026-Research/reconstruction/tools; SRC=data/panoptic/emc_pose1_650; VAR=data/panoptic/variants; RUN=runs/p1-panoptic
while IFS="|" read -r name views gap; do
  [ -e "$VAR/$name" ] && { echo "exists, skipping $name"; continue; }
  python $T/make_variant.py $SRC $VAR/$name --views $views >$RUN/$name.variant.txt
  nice -n 19 python -u $T/run_mv1p.py --data $VAR/$name --out $RUN/$name --views $views --device cpu --threads 4 \
    --use-annots --no-vis --fps 29.97 --label "$name" >$RUN/$name.log 2>&1 || { echo "FAILED $name"; continue; }
  for opt in "--no-ankles" ""; do
    python $T/panoptic_to_easymocap.py eval $SRC/dome/keypoints3d $RUN/$name/triangulated --frames 392 $opt \
      --json $RUN/$name.eval${opt:+10}.json | head -1 | sed "s/^/$name [$views] gap $gap ${opt:-12 joints}: /"
  done
done < logs/t1c-subsets.txt
echo T1C-DONE
