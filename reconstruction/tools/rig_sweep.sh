#!/usr/bin/env bash
# Compare camera layouts, camera counts, 2D noise and sync offsets on synthetic data
# with known 3D ground truth, using EasyMocap's own triangulation (apps/demo/mv1p.py).
#
# Usage: EASYMOCAP=/path/to/EasyMocap bash rig_sweep.sh [WORKDIR]
# Needs only the EasyMocap Python env: no SMPL model, no detector weights, no GPU.
# mv1p.py stops with "data/smplx/smpl does not exist" after triangulation; that is expected
# here, because keypoints3d/ is already written by then.
set -uo pipefail
TOOLS="$(cd "$(dirname "$0")" && pwd)"
: "${EASYMOCAP:?set EASYMOCAP to the EasyMocap repo root}"
WORK="${1:-/tmp/rig_sweep}"
mkdir -p "$WORK"

triangulate() {  # $1 = dataset dir
  rm -rf "$1/output"
  (cd "$EASYMOCAP" && python apps/demo/mv1p.py "$1" --out "$1/output" --body body25 >"$1/mv1p.log" 2>&1)
  [ -d "$1/output/keypoints3d" ] || { echo "triangulation failed, see $1/mv1p.log" >&2; return 1; }
}

score() {  # $1 = label, $2 = dataset dir, $3 = gt dir
  local out mp rel miss rec
  out="$(python "$TOOLS/compare_keypoints3d.py" "$3" "$2/output/keypoints3d")"
  mp=$(awk '/^MPJPE/ {print $3}' <<<"$out")
  rel=$(awk '/^root-rel/ {print $4}' <<<"$out")
  miss=$(sed -n 's/.*missing_in_test=\([0-9]*\).*/\1/p' <<<"$out")
  rec=$(awk '/^joints reconstructed/ {print $3}' <<<"$out")
  printf '| %s | %s | %s | %s | %s |\n' "$1" "$mp" "$rel" "$rec" "$miss"
}

run() {  # $1 = label, $2 = name, rest = synth_rig.py args
  local label="$1" name="$2"; shift 2
  python "$TOOLS/synth_rig.py" "$WORK/$name" "$@" >/dev/null && triangulate "$WORK/$name" \
    && score "$label" "$WORK/$name" "$WORK/$name/gt/keypoints3d"
}

echo "| Variant | MPJPE (mm) | Root-rel. MPJPE (mm) | Joints reconstructed | Frames missing |"
echo "|---|---|---|---|---|"
run "5 cams, even 360°, 3 px noise"          even5    --cams 5
run "5 cams, even 360°, no occlusion model"   noocc5   --cams 5 --no-occlusion
run "5 cams, front 180°"                      front5   --cams 5 --arc 180
run "5 cams, clustered 90°"                   clust5   --cams 5 --arc 90
run "4 cams, even 360°"                       even4    --cams 4
run "3 cams, even 360°"                       even3    --cams 3
run "6 cams, even 360°"                       even6    --cams 6
run "5 cams, even, 8 px noise"                noise8   --cams 5 --noise 8
run "5 cams, even, 15% joints missing"        drop15   --cams 5 --dropout 0.15
run "5 cams, heights 0.6/1.6/2.4 m mixed"     mixedh   --positions "3,0,1.6; 0.93,2.85,2.4; -2.43,1.76,0.6; -2.43,-1.76,2.4; 0.93,-2.85,0.6"

# Sync: camera 03 lags by k frames (60 fps, squat period 2 s)
for k in 1 2 4 8; do
  python "$TOOLS/make_variant.py" "$WORK/even5" "$WORK/lag$k" --shift 03:$k >/dev/null \
    && triangulate "$WORK/lag$k" && score "5 cams, even, cam 03 lags $k frame(s)" "$WORK/lag$k" "$WORK/even5/gt/keypoints3d"
done
# Same lag with faster motion (1 s squats)
python "$TOOLS/synth_rig.py" "$WORK/fast5" --cams 5 --period 1.0 >/dev/null
for k in 2 4; do
  python "$TOOLS/make_variant.py" "$WORK/fast5" "$WORK/fastlag$k" --shift 03:$k >/dev/null \
    && triangulate "$WORK/fastlag$k" && score "5 cams, fast squat, cam 03 lags $k" "$WORK/fastlag$k" "$WORK/fast5/gt/keypoints3d"
done
