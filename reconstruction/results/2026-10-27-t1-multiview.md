# T1 multi-view reconstruction (P1), 2026-10-27

## Camera count and layout: CMU Panoptic, against the dome reference

`171204_pose1`, HD frames 650–1041 (392 frames, 29.97 fps; one person, subject 0), EasyMocap `e6006fd`
iterative triangulation (no SMPL), 2D from one YOLOv5m + HRNet-W48 pass on all 31 HD cameras, reused for every
row. Scored against the Panoptic `joints19` skeletons, which the dense dome reconstructs itself (no markers):
the **dome reference**, not mocap ground truth. Main score: **10 limb joints** (shoulders, elbows, wrists, hips,
knees). Ankles are left out of it because 12 of the 31 HD cameras never frame the feet and 2 more only about half
the time (see Checks). The 12-joint
column adds the ankles. "Missing" = share of dome joints EasyMocap did not reconstruct. No frames were missing in
any run.

| Layout | Cameras (azimuth order) | MPJPE, 10 joints (mm) | PA-MPJPE, 10 j. (mm) | Missing, 10 j. | MPJPE, 12 j. (mm) | Missing, 12 j. |
|---|---|---|---|---|---|---|
| All 31 HD | 00–30 | **11.7** | 10.2 | 0.0% | 14.4 | 0.0% |
| 6 spread (A / B / C) | 18 12 20 00 11 16 / 03 05 21 29 08 23 / 03 05 21 29 27 30 | **14.8** / 18.7 / 16.8 | 13.4 / 15.4 / 14.8 | 0.0 / 0.0 / 0.0% | 16.1 / 20.0 / 20.1 | 0.0 / 0.0 / 0.3% |
| 5 spread (A / B / C) | 03 05 00 08 23 / 18 12 21 11 16 / 22 20 29 27 30 | **16.2** / 16.3 / 18.0 | 14.0 / 14.0 / 15.8 | 0.1 / 0.0 / 0.2% | 17.2 / 18.0 / 18.3 | 15.1 / 0.0 / 14.9% |
| 4 spread (A / B / C) | 12 21 08 30 / 03 20 29 16 / 18 05 00 27 | **20.0** / 18.3 / 18.6 | 16.3 / 16.5 / 16.5 | 0.1 / 3.3 / 1.0% | 22.5 / 18.3 / 19.5 | 0.1 / 16.6 / 17.0% |
| 3 spread (A / B / C) | 18 20 08 / 12 29 23 / 22 00 16 | **24.6**\* / 22.7\* / 21.8 | 16.5 / 17.6 / 17.2 | 23.5 / 28.3 / 4.3% | 24.6 / 22.9 / 22.0 | 36.0 / 38.2 / 19.4% |
| 5 clustered, behind the subject (87° arc) | 18 03 16 23 30 | **26.0** | 19.2 | 0.0% | 23.7 | 0.0% |
| 5 clustered, in front (104° arc) | 20 21 00 29 11 | **24.6** | 17.8 | 0.0% | 27.0 | 12.4% |

\* Not comparable with the other rows: these runs dropped a quarter of the joints (mostly left knee/hip and one arm),
so the MPJPE is over the easier joints that remained.

Means over the three spread subsets (10 joints): 3 cams ≈ 23.0 mm, 4 ≈ 19.0, 5 ≈ 16.8, 6 ≈ 16.8, all 31 = 11.7.
The 3-camera mean is optimistic: two of the three 3-camera subsets lost 23–28% of the limb joints (marked \*), so
3 cameras are worse than this number suggests.
"A" is the subset with the smallest largest azimuth gap among mid-height cameras (0.5–1.5 m above the hip), B and C
the next best differing from A by ≥ 2 cameras.

## Sync tolerance: one camera of the 5-spread rig shifted

Same data, 5-spread layout A (03 05 00 08 23). One camera at a time shows frame i+k while the others show frame i
(`make_variant.py --shift CAM:k`). Each camera was shifted in turn; the table gives the mean over the five and the
range. 10 limb joints, frames 0–391. At 29.97 fps one frame is 33 ms.

| Shift k (frames / ms) | MPJPE per shifted camera 03 / 05 / 00 / 08 / 23 (mm) | Mean MPJPE (mm) | Change vs. synced | Mean PA-MPJPE (mm) | Joints missing |
|---|---|---|---|---|---|
| 0 / 0 | — | 16.2 | — | 14.0 | 0.1% |
| 1 / 33 | 16.5 / 16.6 / 16.6 / 16.6 / 16.4 | 16.6 | +0.3 (+0.2…+0.4) | 14.4 | 0.0–0.1% |
| 2 / 67 | 17.0 / 17.0 / 17.4 / 17.1 / 17.1 | 17.1 | +0.9 (+0.7…+1.2) | 14.7 | 0.1–0.2% |
| 4 / 133 | 18.2 / 17.5 / 18.1 / 17.6 / 18.6 | 18.0 | +1.8 (+1.3…+2.3) | 15.4 | 0.0–0.4% |
| 8 / 267 | 18.6 / 17.7 / 18.8 / 18.3 / 19.3 | 18.5 | +2.3 (+1.5…+3.1) | 15.8 | 0.2–7.9% |

**Compared with the synthetic sweep** (`2026-09-29-synthetic-rig-sweep.md`: 60 fps, 2 s squat, 8.9 mm baseline):
at matched time offsets the real-data cost is similar for small offsets and smaller for large ones. 33 ms costs
+0.4 mm synthetic vs. +0.3 mm real; 67 ms costs +1.4 vs. +0.9; 133 ms costs +4.3 vs. +1.8. On real data the cost
levels off. At 8 frames, joints start to go missing (up to 7.9%) because EasyMocap's iterative triangulation rejects
the shifted view once it disagrees by more than 25 px, so the rig falls back towards 4 good cameras (~19 mm, close to
the 4-camera rows above) instead of averaging in a wrong ray. The motion here is a slow range-of-motion exercise
(root-relative joint speed about 1 cm per frame, ~30 cm/s), so faster motion would cost more per frame. The
synthetic recommendation (residual offset ≤ 1 frame at 60 fps, ≤ 2 tolerable for slow motion) holds: on real data
a one-frame offset at 30 fps cost 0.3 mm, 2% of the error.

## 1. Result

On CMU Panoptic (`171204_pose1`, 392 frames), EasyMocap's triangulation is within **11.7 mm** of the dome reference
with all 31 HD cameras, **≈16.8 mm with 5 or 6 spread cameras**, ≈19 mm with 4 and ≈23 mm with 3 (10 limb joints,
mean of three subsets each). Two of the three 3-camera subsets lost 23–28% of the limb joints, so 3 cameras are
worse than 23 mm suggests. Clustering 5 cameras in a ~90–100° arc costs as much as dropping to 3 spread cameras
(24.6–26.0 mm), and shifting one of 5 cameras by 1/2/4/8 frames costs +0.3/+0.9/+1.8/+2.3 mm. The full `zju-ls-feng` run
(23 cameras, 800 frames, SMPL, 4 overlay videos) took 0.96 s per frame on GPU, with reprojection median 5.55 px and
p95 21.4 px; the p95 tail comes mostly from low-confidence 2D points (56% of the tail) and from wrists/elbows, not
from any badly calibrated camera.

## 2. Setup

| Field | Value |
|---|---|
| Machine | `lambdavector2`, Ubuntu 24.04.4, Threadripper PRO 5955WX, 251 GiB RAM; **GPU 1**, NVIDIA RTX A6000 (49 GB), driver 595.84. GPU 0 idle, not used. As in `2026-10-27-server-setup.md`, which agrees with what was used here |
| Env | `~/fall2026/envs/easymocap` from P0: Python 3.9.23, torch 2.0.1+cu117, numpy 1.23.5, opencv 4.11.0.86. **Added in this session** (YOLOv5 needs them, versions as on the laptop): pandas 1.5.3, seaborn 0.12.2, gitpython 3.1.44, thop 0.1.1.post2209072238. `pip check` clean |
| EasyMocap | `e6006fd3814d5f8ad45a7ce965beb9bcc90767f2`, `~/fall2026/external/EasyMocap` |
| YOLOv5 | v7.0 `915bbf294bb74c859f0b41f1c23bc395014ea679`, weights `yolov5m.pt` |
| 2D keypoints | HRNet-W48 COCO 384×288 (`pose_hrnet_w48_384x288.pth`); Step 3 also ViTPose+-B (`vitpose+_base.pth`) |
| panoptic-toolbox | `246ebcfeeeab9d52fb4843b6698abe2d1a66c5e1` |
| SMPL | Tan's neutral model, SHA-256 `4924f235…5688` (checked again after the PARE deletion) |
| Data | `zju-ls-feng` (SHA-256 `6a89a831…16ea`, 23 videos × 800 frames, 1024², 25 fps); Panoptic `171204_pose1` HD frames 650–1049, 31 cameras × 400 JPEG, calibration and `joints19` keypoints. Every file, hash and count is in `DATA_LOG.md` |
| Repo | branch `server/t1-20261027`, from `6fc763e` |

## 3. Commands

Logs are in `~/fall2026/logs/2026-10-27-p1-*.log`; `T=Fall-2026-Research/reconstruction/tools`,
`export CUDA_VISIBLE_DEVICES=1`.

```bash
# downloads (hashes in DATA_LOG.md)
git clone --branch v7.0 --depth 1 https://github.com/ultralytics/yolov5.git external/yolov5
curl -L -o models/detectors/yolov5m.pt https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt
python -m gdown 11ezQ6a_MxIRtj26WqhH3V3-xPI3XqYAw -O models/detectors/pose_hrnet_w48_384x288.pth
curl -L -o data/downloads/zju-ls-feng.zip 'https://www.dropbox.com/s/24mb7r921b1g9a7/zju-ls-feng.zip?dl=1'
pip install pandas==1.5.3 seaborn==0.12.2 gitpython==3.1.44 thop==0.1.1.post2209072238 -c external/EasyMocap/constraints.txt

# Step 1: reproduce the laptop subset, GPU and CPU
python $T/run_mv1p.py --data data/zju-ls-feng --out runs/p1-step1-repro-{cuda,cpu} --views 1 7 13 19 \
  --end 100 --step 2 --device {cuda,cpu} --threads 4 --smpl models/smpl/SMPL_NEUTRAL.pkl --render-views 1

# Step 2: all-view reference
python $T/run_mv1p.py --data data/zju-ls-feng --out runs/p1-zju-full-23cam --device cuda \
  --smpl models/smpl/SMPL_NEUTRAL.pkl --render-views 1 7 13 19 --write-annots --label all-view-reference

# annots-reuse check
python $T/make_variant.py data/zju-ls-feng data/zju-variants/c4 --views 1 7 13 19
python $T/run_mv1p.py --data data/zju-variants/c4 --out runs/p1-zju-c4-annots --views 1 7 13 19 \
  --end 100 --step 2 --device cpu --use-annots --no-vis

# Step 4: Panoptic
#   frames: for c in 00..30: ffmpeg -i http://domedb.perception.cs.cmu.edu/webdata/dataset/171204_pose1/videos/hd_shared_crf20/hd_00_$c.mp4 \
#             -vf "select=between(n\,650\,1049)" -vsync 0 -frames:v 400 -q:v 2 -start_number 0 hdImgs/00_$c/%06d.jpg
python $T/panoptic_to_easymocap.py convert data/panoptic/171204_pose1 data/panoptic/emc_pose1_650 --start 650 --end 1050
python $T/panoptic_to_easymocap.py layout data/panoptic/emc_pose1_650
python $T/run_mv1p.py --data data/panoptic/emc_pose1_650 --out runs/p1-panoptic-all31 --device cuda --fps 29.97 \
  --vis-views 00 08 --write-annots
python $T/panoptic_to_easymocap.py check data/panoptic/emc_pose1_650 --views 00 08 --frame 200 --image runs/p1-panoptic-all31/projcheck.jpg
python $T/panoptic_to_easymocap.py eval .../dome/keypoints3d runs/p1-panoptic-all31/triangulated --offset {-3..3}
bash logs/p1-panoptic-variants.sh     # per variant: make_variant.py [--shift CAM:k] -> run_mv1p.py --use-annots --no-vis --device cpu
                                      #              -> panoptic_to_easymocap.py eval --frames 392 [--no-ankles]
bash logs/p1-panoptic-alt.sh          # the B and C spread subsets

# Step 3: feet
python $T/run_mv1p.py --data data/zju-ls-feng --out runs/p1-step3-vitpose-c4 --views 1 7 13 19 --end 100 --step 2 \
  --device cuda --keypoints2d vitpose --smpl models/smpl/SMPL_NEUTRAL.pkl --render-views 1
python $T/run_mv1p.py --data data/panoptic/variants/s5 --out runs/p1-step3-vitpose-panoptic-s5 --views 03 05 00 08 23 \
  --device cuda --keypoints2d vitpose --fps 29.97
# HRNet body (slots 0-18) + ViTPose feet (19-24) merged into data/zju-variants/c4-feet/annots, then
python $T/run_mv1p.py --data data/zju-variants/c4-feet --out runs/p1-step3-merged-feet-c4 --views 1 7 13 19 \
  --end 100 --step 2 --device cuda --use-annots --smpl models/smpl/SMPL_NEUTRAL.pkl --render-views 1 7
```

## 4. Numbers

### Step 1: new runner vs. laptop (zju-ls-feng, cameras 1 7 13 19, source frames 0,2,…,98)

| Run | Reproj. median (px) | p95 (px) | Comparisons | Valid joints | Time |
|---|---|---|---|---|---|
| Laptop, `local_cpu.py`, CPU (Sep 29 / Oct 27) | 5.65 | 19.69 | 3792 | 19.00/19 | 7.28 s per step (fresh detection) |
| Server, `run_mv1p.py`, CPU, 4 threads | 5.650 | 19.690 | 3792 | 19.00/19 | 1.86 s per step; 2 min 4 s total with SMPL |
| Server, `run_mv1p.py`, GPU | 5.685 | 19.719 | 3793 | 19.00/19 | 0.17 s per step; 52 s total with SMPL |

The server CPU run matches the laptop to every printed digit, so the runner and install are equivalent. GPU vs. CPU on
the same machine: 98% of 2D joints agree within 0.5 px. The other 2.1% move by about 2 or 4 px, one HRNet heatmap
step (4 px in the crop), where near-tied heatmap peaks or the ±0.25-cell refinement flip under GPU arithmetic. One
low-confidence joint moved 39.5 px. Bounding boxes differ by ≤ 0.1 px. Effect on 3D joints: median per-frame max
4.8 mm; on SMPL pose parameters ≤ 0.016 rad.

### Step 2: full zju-ls-feng (23 cameras, 800 frames, GPU)

| Quantity | Value |
|---|---|
| Time per frame (YOLO + HRNet on 23 images + triangulation) | 0.963 s mean, 0.963 s median (≈42 ms per image; GPU utilisation ~33%, so CPU-side image I/O dominates) |
| SMPL fit, all 800 frames | ≈37 s (fit + render 915 s, minus 4 renders × ≈220 s at 3.65 frames/s) |
| Wall time, whole command | 29 min 22 s (incl. frame extraction and 4 overlay videos) |
| Reprojection (3D and 2D confidence > 0.3) | median **5.55 px**, p95 **21.41 px**, 348,762 joint/view comparisons |
| Valid joints | 19.00 / 19 every frame (25-slot BODY25: the 6 foot slots stay empty with HRNet) |
| Outputs | 800 `triangulated/`, 800 `smpl/`, `vis_keypoints2d.mp4`, `vis_repro.mp4` (cameras 1, 7, 13, 19), `vis_render_{1,7,13,19}.mp4`; all decode 800 frames at 25 fps |

Per camera (cameras 1–23 form one ring, ~2 m high, ~3 m radius; azimuth of camera n ≈ 181° + 14.5°·(n−1)):

| Camera | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| median (px) | 4.55 | 4.50 | 4.50 | 4.64 | 4.83 | 4.89 | 5.23 | 5.53 | 5.56 | 5.60 | 5.79 | 5.90 |
| p95 (px) | 18.2 | 17.9 | 16.7 | 16.8 | 16.8 | 17.2 | 19.3 | 19.7 | 20.6 | 21.0 | 22.2 | 22.6 |
| share of tail | 2.7% | 2.8% | 2.0% | 1.9% | 1.9% | 2.2% | 3.2% | 3.3% | 3.8% | 4.1% | 4.7% | 4.9% |

| Camera | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| median (px) | 5.93 | 5.90 | 6.33 | 6.46 | 6.70 | 6.37 | 6.19 | 6.24 | 6.21 | 5.86 | 5.45 |
| p95 (px) | 22.0 | 24.0 | 27.0 | 26.7 | 27.3 | 27.3 | 23.7 | 22.8 | 23.5 | 21.9 | 19.8 |
| share of tail | 4.7% | 5.7% | 6.8% | 6.9% | 7.5% | 6.8% | 5.3% | 5.0% | 5.4% | 4.6% | 3.7% |

Per joint (all cameras; "tail" = comparisons above the overall p95 of 21.4 px):

| Joint | median (px) | p95 (px) | share of tail | | Joint | median (px) | p95 (px) | share of tail |
|---|---|---|---|---|---|---|---|---|
| Nose | 4.39 | 27.2 | 9.4% | | RHip | 12.13 | 22.8 | 7.6% |
| Neck | 4.46 | 9.8 | 0.0% | | RKnee | 7.22 | 19.3 | 3.9% |
| RShoulder | 6.14 | 14.7 | 0.4% | | RAnkle | 3.35 | 9.6 | 0.4% |
| RElbow | 5.24 | 47.9 | 10.0% | | LHip | 11.38 | 24.1 | 9.4% |
| RWrist | 3.82 | 76.2 | 9.2% | | LKnee | 7.43 | 17.9 | 2.5% |
| LShoulder | 6.28 | 16.0 | 0.9% | | LAnkle | 3.73 | 10.4 | 1.0% |
| LElbow | 5.52 | 40.6 | 9.1% | | REye | 5.22 | 26.0 | 8.3% |
| LWrist | 3.90 | 65.2 | 9.4% | | LEye | 5.30 | 30.0 | 12.9% |
| MidHip | 9.85 | 17.4 | 1.2% | | REar / LEar | 4.63 / 4.21 | 17.2 / 18.4 | 1.8 / 2.7% |

Where the p95 tail comes from:
- **Low 2D confidence.** Points with HRNet confidence 0.3–0.7 are 5.5% of comparisons but 56% of the tail
  (median error 19–32 px; ≥ 0.9 confidence: median 4.2 px, p95 12.1 px).
- **Elbows and wrists.** 21% of comparisons, 38% of the tail. The worst camera/joint cells are single wrists with
  p95 of 120–150 px (e.g. camera 18 left wrist 147 px, camera 23 right wrist 138 px): wrong wrist detections that
  the iterative triangulation did not reject. Low confidence or arm joint together explain 62% of the tail.
- **Not a calibration problem.** With high-confidence (≥ 0.9) shoulders/hips/knees/ankles only, every camera's
  median is 4.2–5.9 px. Cameras 14–18 look worse overall because they hold more low-confidence views.
- **Not a few bad frames.** The worst 10% of frames hold only 14% of the tail; tail counts per 100-frame block are
  1,919–2,466.
- **Hips are a separate, systematic effect.** Hip medians are 11–12 px (other joints 3–7 px) but they barely enter
  the tail: where a detector puts the hip on a clothed body depends on the viewpoint.

### Step 4 extras: per-joint error, 5-spread layout A (03 05 00 08 23), vs. dome reference

| Joint (BODY25 ← joints19) | All 31 (mm) | 5-spread (mm) | 5-spread: frames reconstructed |
|---|---|---|---|
| RShoulder ← rShoulder | 14.5 | 18.4 | 100% |
| RElbow ← rElbow | 8.6 | 10.1 | 100% |
| RWrist ← rWrist | 7.9 | 10.5 | 100% |
| LShoulder ← lShoulder | 10.4 | 15.8 | 100% |
| LElbow ← lElbow | 7.9 | 9.1 | 100% |
| LWrist ← lWrist | 7.7 | 9.1 | 100% |
| RHip ← rHip | 19.0 | 29.5 | 99% |
| RKnee ← rKnee | 10.9 | 17.8 | 100% |
| RAnkle ← rAnkle | 21.5 | 78.2 | 14% |
| LHip ← lHip | 17.3 | 25.3 | 100% |
| LKnee ← lKnee | 13.2 | 16.5 | 100% |
| LAnkle ← lAnkle | 33.8 | 34.1 | 5% |
| Nose ← Nose | 5.7 | 13.5 | 100% |
| Neck ← Neck | 9.2 | 7.9 | 100% |
| MidHip ← BodyCenter | 15.5 | 25.0 | 100% |
| REye / LEye ← rEye / lEye | 13.7 / 10.6 | 18.4 / 13.2 | 100% |
| REar / LEar ← rEar / lEar | 8.3 / 6.6 | 12.8 / 9.9 | 100% |

Joint mapping (all 19 joints19 points have a BODY25 slot; BODY25 feet 19–24 have none): Neck→1, Nose→0,
BodyCenter→8 MidHip, lShoulder/lElbow/lWrist→5/6/7, lHip/lKnee/lAnkle→12/13/14, rShoulder/rElbow/rWrist→2/3/4,
rHip/rKnee/rAnkle→9/10/11, lEye→16, lEar→18, rEye→15, rEar→17. Scored sets: 12 = shoulders, elbows, wrists, hips,
knees, ankles; 10 = the same without ankles. Not scored as main joints: Neck and MidHip, which EasyMocap synthesises
as midpoints (shoulders, COCO hips) while the dome defines its own points, and the face points. Units: Panoptic cm,
EasyMocap m (calibration `t` and skeletons divided by 100). Axes: Panoptic world kept (y points down), so
triangulation and dome share one frame. Hips stay at 17–19 mm even with 31 cameras: COCO-style hips and the dome's
hips are not the same anatomical point, so part of the hip error is a definition difference, not reconstruction error.

### Step 3: feet

| Run (2D) | Data | Valid BODY25 joints | Reproj. median / p95 (px) | vs. reference |
|---|---|---|---|---|
| HRNet | ZJU 4 cams, 50 frames | 19.0 / 25 | 5.69 / 19.72 (19 joints) | agreement with all-23 HRNet run: 12.9 mm |
| ViTPose+-B (COCO-17 + 6 feet) | ZJU 4 cams, 50 frames | **25.0 / 25** | 6.86 / 18.75 (19 joints); feet 4.1–6.3 / 14.9–35.6 | agreement: 19.1 mm (biased towards HRNet, see text) |
| HRNet | Panoptic 5-spread A, 400 frames | 17.2 / 19\*\* | 7.77 / 34.95 (19 joints) | MPJPE 16.2 mm, PA 14.0 (10 joints, dome, 392 frames) |
| ViTPose+-B | Panoptic 5-spread A, 400 frames | 17.1 / 19\*\* | 9.37 / 34.72 (19 joints) | MPJPE **21.7 mm**, PA 14.9 (10 joints, dome, 392 frames); RAnkle 390 mm in 194 frames |
| HRNet body + ViTPose feet | ZJU 4 cams, 50 frames | 25.0 / 25 | 5.66 (25 joints) | body joints identical to HRNet (≤ 0.0007 mm) |

\*\* Out of 19, mean over frames; the missing ones are mostly ankles from cameras that do not frame the feet.

EasyMocap at the pinned commit **does** support a detector with feet: `myeasymocap.backbone.vitpose.vit_moe.MyViT`
(ViTPose+-B with mixture of experts) takes COCO-17 from its main head and the 6 COCO-WholeBody foot points from the
whole-body head (outputs 17–22), mapped to BODY25 19–24 (L big toe, L small toe, L heel, R big toe, R small toe,
R heel; mapping checked in `topdown_keypoints.coco23tobody25`). It drops in for `MyHRNet` without code changes to
EasyMocap; `run_mv1p.py --keypoints2d vitpose` uses it. On the dome-scored test, though, its body joints are worse
than HRNet's: +5.5 mm MPJPE on the 10 limb joints, mostly hips and knees (34 / 27 mm vs. 30 / 18 mm). PA-MPJPE is
almost the same, so it is mostly a consistent offset. Likely factors: 256×192 input vs. HRNet's 384×288, and a
different hip definition in its training data (not tested). The ZJU agreement numbers favour HRNet by construction,
because the reference was made with HRNet.

Because triangulation is per joint, HRNet for the body and ViTPose for the six foot slots gives exactly HRNet's body
plus feet. With SMPL, this visibly improves the feet (see Checks) and changes nothing else.

## 5. Checks

- **Laptop reproduction:** the server CPU run equals the laptop receipt (5.650 / 19.690 px, 3792); GPU differences
  explained above.
- **Overlays looked at.** ZJU Step 1: `vis_render_1` frames 0 and 25 (T-pose; raised right arm; mesh on the
  feet). ZJU Step 2: frames 0, 91 (the frame with the most tail comparisons), 400 and 799 on all four rendered
  cameras 1, 7, 13, 19. The mesh follows a T-pose, a raised right arm, both arms crossed in front, and a turn with
  one arm out, in all views. No limb swaps, floating, or missing frames in these. Clothing extends beyond the mesh
  as expected. Foot orientation is unconstrained without foot points.
- **2D reuse is real:** the `--use-annots` run on a `make_variant.py` copy made no detector caches, its 2D equals the
  fresh GPU detections to 3×10⁻⁵ px (float32 JSON rounding), and its 3D to < 0.001 mm. Its reprojection stats
  equal the Step 1 GPU run (5.685 / 19.718 px, 3793).
- **Panoptic calibration conversion:** dome joints projected through the converted `intri.yml`/`extri.yml` into
  HD cameras 00 and 08 land on the HRNet detections. Camera 00 (front): limb joints median 6.9 px, face 3–4 px, wrists
  4.4–4.6 px. Camera 08 (side): limb median 17.3 px. Frame 200 of camera 00 drawn and looked at
  (`runs/p1-panoptic-all31/projcheck_00.jpg`, green = dome, red = HRNet): all visible joints coincide.
- **The ankle outlier explained:** in that frame the ankles are outside the image (camera 00 frames the subject from
  mid-shin up). Over 40 frames per camera, the dome ankles are inside the image in ≥ 99% of frames for 17 cameras,
  in about half for 2 (22, 30), and in ≤ 2% for 12 (00 05 08 09 14 15 20 24 25 26 27 29). HRNet still reports
  "confident" (> 0.3) ankles in 28–49% of frames for most of those 12: guesses at the image border.
  That is why the main score leaves out ankles.
- **Units and frame alignment:** all-31 triangulation vs. dome at frame offsets −3…+3 gives 33.4, 25.9, 18.6,
  **14.4**, 20.5, 27.9, 35.5 mm. The minimum is at offset 0 and the errors are in mm, not m or cm, so the frame
  numbering (video frame n = skeleton n) and the cm→m conversion are right.
- **Subject orientation:** facing azimuth 188° (10–90%: 175–190°) from the shoulder line, with the nose in front of
  the neck in every frame. The 310°–37° cluster is therefore behind the subject.
- **Single-person check:** `171204_pose1` has 0 or 1 bodies per frame across all 27,561 skeleton files. Frames
  650–1049 have one body (id 0) with every joint confidence ≥ 0.22.
- **Feet:** cropped camera 1 and camera 7 (side) views, output frames 0 and 25 (source 0 and 50), HRNet-only vs.
  HRNet + ViTPose feet (`runs/p1-step3-merged-feet-c4/feet_compare*.jpg`). Without foot points, the far foot floats
  above its shoe in the side view and the feet point straight ahead in the front view. With them, both feet sit in
  the shoes from heel to toe and turn out like the shoes.
- **ViTPose+ checkpoint:** SHA-256 `de043d2e…3545` matches Tan's value. It loads with
  `torch.load(..., weights_only=True)` without error: one `state_dict`, 377 tensors, including the 17-output main
  head and the 133-output whole-body head. The runner forces `weights_only=True` for EasyMocap's own load
  (`SafeViT`), so the file was never unpickled any other way.

## 6. Failed or skipped

- **Accidental download of third-party SMPL files (my bug).** The first version of `run_mv1p.py` resolved the
  `--smpl` path after changing into the EasyMocap folder, so a relative path pointed at nothing. EasyMocap's
  `SMPLLoader` then ran its fallback: it downloaded the 1.32 GB PARE data archive from Google Drive and unzipped it into
  `external/EasyMocap/models/pare/`. That included `SMPL_MALE.pkl`, `SMPL_FEMALE.pkl` and
  `neutral_smpl_with_cocoplus_reg.pkl`, which were never used. `unzip` stopped at the prompt to overwrite P0's
  `SMPL_NEUTRAL.pkl` symlink, and I killed the process, so the link was not replaced. Fix: all paths are resolved
  before `chdir`, and the runner refuses to start the fit if the model or regressor file is missing.
  **Deleted after Step 2, with Tan's approval:** `pare-github-data.zip` and the 15 extracted files. P0's symlink
  `models/pare/data/body_models/smpl/SMPL_NEUTRAL.pkl` was kept. Afterwards both `SMPL_NEUTRAL.pkl` links resolve to
  `~/fall2026/models/smpl/SMPL_NEUTRAL.pkl`, and its SHA-256 is still `4924f235…5688` (log
  `2026-10-27-p1-pare-delete.log`).
- P0's result file was missing from my first `git pull` and appeared later on `main` as `6fc763e`. Until then I
  took the environment from P0's logs; the file agrees with them.
- First smoke test failed: `No module named 'pandas'` (YOLOv5 dependency). Installed the four packages listed in Setup.
- A shell `pkill`-style loop matched its own command line and killed itself (exit 144). Nothing else was affected;
  I checked the processes again by hand.
- The official ViTPose+ OneDrive link returns 401/403 to scripts. The Hugging Face conversion
  (`usyd-community/vitpose-plus-base`) has only the 17-point head, so it has no feet. Tan downloaded the original file
  in a browser.
- MediaPipe Pose (heels and one toe point per foot) was not tried. EasyMocap's wrapper uses the legacy `mp.solutions`
  API, which the installed mediapipe 1.1.1 no longer has. ViTPose+ made it unnecessary.
- Two wrong crops of camera 7 while checking the feet (guessed pixel coordinates, then a render-index vs.
  source-frame mix-up). Redone by projecting the ankles; the comparison above uses the correct crop.

## 7. Not tested

- SMPL on Panoptic (steps 4–5 score triangulated joints only), and SMPL accuracy anywhere: ZJU has no ground truth.
- Camera layouts at other heights. All spread subsets use the 0.5–1.5 m above-hip ring, about 1.4–2.4 m above the floor.
- More than one sequence or subject. All Panoptic numbers are one subject, one 13 s range-of-motion clip.
- Feet against any reference: the dome has no foot points. The feet were judged by eye on 2 frames × 2 cameras.
- ViTPose+ with flip testing or at a larger input size. A different keypoint confidence threshold for triangulation.
- Two cameras shifted at once, or sub-frame offsets.
- `local_cpu.py` was not changed and not re-run on the server.

## 8. Next step

Use the Panoptic set to settle what the team's rig needs. Repeat the 4/5/6-camera rows at two other camera heights,
and on one fast sequence (e.g. a dance or sports sequence from the Panoptic list). Fit SMPL on the 5-spread rig and
score the SMPL joints against the dome, since SMPL is what the telepresence avatar uses. Then add the HRNet body +
ViTPose feet combination to the runner as one option instead of a hand-built merge.
