# T4b: two more GaussianAvatar runs on REHAB24-6 (better poses; another camera), 2026-11-02

**Status: PENDING.** Data prepared and checked on CPU. Training has not started, because both GPUs are held by another
user's jobs (about 43 GB used of 49 GB on each). The enabled user unit `t4b-avatar.service` waits for a GPU with
24 GB free, then runs both. Numbers marked PENDING get filled in from its outputs.

Follow-up to T4 Step 3 (`2026-10-27-t4-gaussian.md`). There, the avatar trained on camera 17 with GVHMR poses scored
18.54 dB on held-out repetitions with raw test poses, and 28.06 dB once the test poses were refined on the test images.
So pose error, not appearance, is the limit.

| Run | Images | Poses (train and test, raw) | Question |
|---|---|---|---|
| T4 Step 3 (done) | camera 17 | GVHMR, camera 17 | baseline: 18.54 / 0.9430 / 0.0663 |
| **1 `c17-twocam`** | camera 17 | T2c's two-camera SMPL fit, moved into camera 17 | does depth from a second camera close the 18.5 → 28.1 dB gap? |
| **2 `c18-gvhmr`** | camera 18 (the transposed, 1080×1920 video) | GVHMR, camera 18 | same method, other view: how much does the view matter? |

## 1. Result

PENDING (training not started).

What is already known from the checks before training:
- **The two-camera poses line up with the person better than GVHMR's do**, judged against the SAM masks. Projected
  silhouette IoU is 0.835 on average against 0.813 for GVHMR (T4), and better in 311 of 389 frames. On the 35 test
  frames, it's 0.827 against 0.768.
- They fail differently, though. GVHMR is a little off everywhere. The two-camera fit is closer most of the time but
  jumps about 1 m off in 3 frames between repetitions (see 4).
- **The camera-18 GVHMR poses fit their view worse** (IoU 0.779 mean, test frames 0.751) than camera 17's (0.813), so
  run 2 starts from noisier poses as well as a different view.

## 2. Setup

Same recipe as T4 Step 3 unless noted:
- recording PM_037 (exercise 5, person 4), every 5th frame of 230..2170 = 389 frames;
- test = last 3 repetitions (frames ≥ 2000, 35 frames); GaussianAvatar `d981c62` + the T4 local patch;
- stage 1, 200 epochs, `--pose_op_start_iter 10`, `--save_epoch 20`;
- eval at epochs 120–200 with raw test poses (`eval.py`, cross-checked with `image_metrics.py`);
- FPS; 360° turntable of training frame 01200.
- Extras, about 5 min of GPU each, as in T4: the refined-test-pose diagnostic (upper bound only), the novel-pose video
  and the GT|render test video.

| Item | Run 1 `c17-twocam` | Run 2 `c18-gvhmr` |
|---|---|---|
| Images, masks | T4's camera-17 frames and SAM 2.1 masks (symlinked, unchanged) | new: every 5th frame cut from P2's camera-18 GVHMR clips (`logs/t4b/extract-frames-c18.sh`, same global frames as camera 17); SAM 2.1 hiera-base+ on CPU, prompt = GVHMR box +5% (`tools/rehab_sam2_masks.py`, 8 threads, 13 min). Scores 0.971–0.982 (median 0.982) |
| Poses | T2c `runs/t2c/fit/PM_037` (EasyMocap, HRNet 2D from both cameras, authors' **new** cameras). Vertices from EasyMocap's own `SMPLModel.vertices`, moved by the new camera-17 extrinsics (`tools/twocam_smpl_to_camera.py`), then the T4 SMPL fit to those vertices (`gvhmr_to_gaussianavatar.py --target-verts`) | P2's true-focal GVHMR rerun (`runs/t2-chunks/out-truef`, camera-18 clips), T4 conversion unchanged |
| SMPL fit to source vertices | RMS **0.33 mm**, max 2.1 mm (same body model on both sides) | RMS 6.8 mm, p99 16.0, max 34.3 mm (SMPL-X → SMPL, as T4's 10.2 mm) |
| Camera K | new camera-17 calibration: fx 2181.73, fy 2181.82, c (960, 540). T4 used the old one (fx 2181.65); difference < 0.1 px | **GVHMR's own K** (f 2176.09, c (540, 960)), so the poses are projected with the camera they were estimated for. The new camera-18 calibration has fx 2171.30 / fy 2181.82, which differs by more than T4's 1 px tolerance |
| Crop | x 195:1111, y 228:1080 → 916×852 (same as T4) | x 0:1080, y 557:1765 → 1080×1208 |
| Frames dropped from training | 3: 00960, 00965, 00980 (IoU 0.11, 0.59, 0.03) | 3: 00730, 00735, 00740 (IoU 0.58, 0.48, 0.51) |
| Train / test | **351** / 35 | **351** / 35 |
| Turntable | f = calibration fx, 1024² canvas | f × 0.9 (1958.5), because the person is closer to camera 18 (4.05 m against 4.85 m) and the feet would touch the canvas edge |

**Frame rule.** One rule for all runs: drop a training frame if its projected input pose overlaps the SAM mask with
IoU < 0.6 (`tools/ga_drop_frames.py`). It uses no mocap. On T4's camera-17 GVHMR data it would drop nothing (minimum
IoU 0.702), so T4 Step 3 already satisfies it. No test frame is below 0.6 in either run (minimum 0.684 and 0.681), so
the test sets are T4's 35 frames, unchanged.

**GPU and safety** (`tools/t4b_avatar_runs.sh`, unit `tools/t4b-avatar.service`, installed in `~/.config/systemd/user/`):
- Before each run, the unit polls every 5 min until a GPU has ≥ 24 GB free, then uses that GPU (GaussianAvatar
  needed 21.0 GB in T4).
- During every step, a watchdog reads that GPU's free memory every 5 s. Below 1 GB it stops the step's scope, writes
  `=== STOPPED`, and the unit exits without restarting.
- Each step runs in `systemd-run --scope -p MemoryMax=64G -p MemorySwapMax=0`, `nice -n 19`, `OMP_NUM_THREADS=8`.
- It resumes from the newest loadable checkpoint and skips finished steps. It disables itself after `=== ALL DONE`.
- Tested before enabling: a dummy job under a forced threshold was stopped within the first check, and the script
  exited 0.

Status: `~/fall2026/logs/t4b-status.txt`. Per-step logs: `~/fall2026/logs/t4b/`. Outputs:
`~/fall2026/runs/t4b/{c17-twocam,c18-gvhmr}/stage1/`.

## 3. Commands (preparation, all CPU, `nice -n 19`, 8 threads; logs `~/fall2026/logs/t4b/2026-11-02-*.log`)

```bash
T=~/fall2026/wt-t4b/reconstruction/tools; R=~/fall2026/runs/t4b
# run 1
ln -s runs/p3-ga-rehab/all/{images,masks} $R/c17-twocam/all/; cp runs/p3-ga-rehab/all/frame_map.csv $R/c17-twocam/all/
envs/easymocap/bin/python $T/twocam_smpl_to_camera.py --fit runs/t2c/fit/PM_037 --cams runs/t2c/data/PM_037 \
  --frame-map $R/c17-twocam/all/frame_map.csv --start 230 --cam 17 --smpl models/smpl/SMPL_NEUTRAL.pkl --out $R/c17-twocam/twocam_c17.npz
cd external/GVHMR; envs/gvhmr/bin/python $T/gvhmr_to_gaussianavatar.py $R/c17-twocam --chunks runs/t2-chunks/out-truef \
  --calib runs/t2b/calib-new/calib_c17.npz --seg data/rehab24-6/Segmentation.csv --video PM_037 \
  --target-verts $R/c17-twocam/twocam_c17.npz --threads 8 --check
# run 2
bash logs/t4b/extract-frames-c18.sh
envs/sam2/bin/python -I $T/rehab_sam2_masks.py $R/c18-gvhmr 8
cd external/GVHMR; envs/gvhmr/bin/python $T/gvhmr_to_gaussianavatar.py $R/c18-gvhmr --chunks runs/t2-chunks/out-truef \
  --calib runs/t2b/calib-new/calib_c18.npz --seg data/rehab24-6/Segmentation.csv --video PM_037 --k-from-gvhmr --threads 8 --check
# both
envs/gs-avatar/bin/python -I $T/ga_drop_frames.py $R/<run> --split {train,test} --min-iou 0.6
cd external/GaussianAvatar; python $T/ga_cano_maps.py $R/<run>/train $R/<run>/test
python logs/p3-rehab-make-turn.py $R/<run>/train $R/<run>/turn_pose 01200
systemctl --user enable --now t4b-avatar.service
```

## 4. Numbers

### Checks before training: projected input pose vs SAM mask (silhouette IoU, all 389 frames)

| Poses | mean | median | min | test frames (35) mean / min |
|---|---|---|---|---|
| T4: GVHMR, camera 17 | 0.813 | 0.814 | 0.702 | 0.768 / — |
| **Run 1: two-camera SMPL, camera 17** | **0.835** | **0.840** | 0.032 (00980) | **0.827** / 0.684 |
| Run 1 without the 3 dropped frames | 0.840 | 0.841 | — | same |
| **Run 2: GVHMR, camera 18** | 0.779 | 0.783 | 0.479 (00735) | 0.751 / 0.681 |

The test-frame mean for T4 is recomputed here from T4's `check/iou.json`.

**Overlays viewed** (`runs/t4b/*/check/`, kept out of git because faces are visible):
- Run 1, frame 01685 (lunge): mesh outline on the person's outline (IoU 0.823).
- Run 1, frame 00980: the mesh is about a body-width to the left, in a pose that doesn't match. A second person's arm
  is at the left edge of camera 17 between repetitions. This is T2c's known wrong-box failure: the cached HRNet
  detection picks the wrong person, and triangulation follows it. The same holds for 00960 and 00965.
- Run 2, frame 00735: the right person and a matching upper body, but the legs are misplaced: both mesh legs sit
  left of and below the real ones. An ordinary GVHMR pose error, not a wrong detection.

### Results (PENDING)

| Run | Epoch 200 raw test poses PSNR / SSIM / LPIPS | epochs 120–180 | refined diagnostic | train time | peak GPU | FPS |
|---|---|---|---|---|---|---|
| T4 Step 3 (camera 17, GVHMR) | 18.54 / 0.9430 / 0.0663 | 18.70 → 18.55 | 28.06 / 0.9778 / 0.0197 | 2 h 00 min | 21.0 GB | 38.2 |
| 1 `c17-twocam` | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| 2 `c18-gvhmr` | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

How to read them when they arrive:
- Run 1 against T4 uses the same images and the same test frames, so any change is down to the poses.
- Run 2 against T4 is not a clean "number of views" test. It changes the view and the pose quality at once (IoU 0.779
  against 0.813), and the crop is larger (1080×1208), which changes how much background enters PSNR.
- Both runs train on one camera's images. "Number of views" in the sense of training on cameras 17 and 18 together
  is not run here: GaussianAvatar's loader takes one camera.

## 5. Checks done

- Same global frames in both runs: camera-18 `frame_map.csv` equals camera 17's (389 frames, 230..2170).
- EasyMocap SMPL convention: `poses` holds 69 values (joints 1–23, no root). The root is `Rh` about the model origin,
  then `Th`. My first conversion treated the first 3 values as the root, and the fit started at 361–413 mm. After
  the fix it starts at 14.8 mm (shape not yet fitted) and ends at 0.33 mm.
- The root rotation is now initialised with a per-frame Kabsch fit to the target vertices (`--target-verts` only), so
  it doesn't depend on the source's convention. The GVHMR path is unchanged.
- The SAM script clipped boxes to 1920×1080. It now clips to the actual image size, which matters for camera 18's
  1080×1920 frames.
- The watchdog was tested on a dummy job (see 2).

## 6. Failed or skipped

- No training yet (GPUs busy, by instruction).
- Run 2's camera-18 SAM masks were checked by score and area only. Of the 6 overlays written, I viewed only
  frame 00735, where the mask outlines the person correctly.
- The two-camera poses for run 1 come from HRNet 2D (T2c). Triangulating GVHMR's ViTPose 2D was better in T2c
  (42.9 against 53.3 mm). A fit on ViTPose 2D would probably give better poses still, but it was not done.
