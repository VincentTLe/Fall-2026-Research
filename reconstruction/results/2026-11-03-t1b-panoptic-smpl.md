# T1b: SMPL body model on CMU Panoptic, per camera layout, 2026-11-03

Follow-up to `2026-10-27-t1-multiview.md`, which scored triangulated joints only. Here EasyMocap's SMPL fit is added on
top of the same triangulation, for six of T1's layouts, and scored against the dome reference.

## 1. Result

With EasyMocap's default `mv1p` SMPL fit, the body model makes every layout **worse**. Every layout with 4 or more
cameras ends up at about the same error, MPJPE 43–47 mm and PA-MPJPE about 33 mm on the 10 limb joints, whether it uses
4 cameras or all 31; with 3 cameras it reaches 63.6 / 48.5 mm. Triangulation alone gives 11.7–26.0 mm. So with this fit, the body model hides the layout differences instead of
reducing them.

The loss comes from the fit, not from the cameras. The fitted body sits about 40 mm from the triangulated points it was
fitted to, worst at the hips and knees (about 50 mm). In the all-31 fit, the shape stays close to SMPL's mean body. Part of the gap is
also joint definition: SMPL's J_regressor joints are rotation centres, while the dome's joints are keypoint-style. Its
shoulders are 67 mm off with 31 cameras, against 42 mm through EasyMocap's BODY25 regressor on the same fit. But even
that regressor is 42 mm off overall, so definition doesn't explain most of it.

**Does SMPL fill in the joints the 3-camera rig lost?** Yes in count, no in quality. Layout 3-spread A lost 23.5 % of
the limb joints (922 joint-frames, mostly the left knee, left hip and right elbow and wrist). The SMPL fit supplies all of
them, at 117 mm mean error (wrists about 270 mm, elbows about 130 mm, left knee 85 mm). Triangulation's error on the
joints it kept is 24.6 mm. A filled-in joint here is a guess from the pose prior and temporal smoothing, not a
measurement.

## 2. Setup

| Field | Value |
|---|---|
| Host | `lambdavector2`, **CPU only** (both GPUs held by another user's job), `CUDA_VISIBLE_DEVICES=` |
| Data | Panoptic `171204_pose1`, HD frames 650–1049 (T1's `data/panoptic/variants/<layout>`). Fitted all 400 frames, scored frames 0–391 as in T1 |
| 2D input | T1's HRNet-W48 annots (one YOLOv5m + HRNet pass on all 31 cameras), reused through `--use-annots`. **No re-detection** |
| Triangulation | EasyMocap iterative (T1's default), unchanged, so the fit's input is exactly T1's triangulation |
| Fit | EasyMocap `e6006fd` `run_mv1p.py --use-annots --smpl --device cpu --threads 8 --no-vis --render-views` (config `detect_triangulate_fitSMPL.yml`, default weights) with Tan's `SMPL_NEUTRAL.pkl` (SMPL v1.1.0) |
| Layouts | all 31; 6 spread A (18 12 20 00 11 16); 5 spread A (03 05 00 08 23); 4 spread A (12 21 08 30); 3 spread A (18 20 08); 5 bunched behind the subject (18 03 16 23 30) |
| Scoring | new `panoptic_smpl_eval.py`. Puts the SMPL J_regressor joints (on the fitted vertices) into BODY25 slots and scores them with T1's `panoptic_to_easymocap.py eval --no-ankles --frames 392` (10 limb joints: shoulders, elbows, wrists, hips, knees) |
| Run | one enabled user unit `t1b-smpl` (now disabled), each layout in `systemd-run --scope -p MemoryMax=32G -p MemorySwapMax=0`, `nice -n 19`. 3 min 24 s in total (21–56 s per layout); peak RSS ≤ 0.88 GB |

```
# per layout, in ~/fall2026/logs/t1b-pipeline/run.sh
python $T/run_mv1p.py --data data/panoptic/variants/<layout> --out runs/t1b/<layout> --views <cams> --device cpu --threads 8 \
    --use-annots --smpl models/smpl/SMPL_NEUTRAL.pkl --render-views --no-vis --fps 29.97
python $T/panoptic_smpl_eval.py --run runs/t1b/<layout> --smpl models/smpl/SMPL_NEUTRAL.pkl \
    --ref data/panoptic/emc_pose1_650/dome/keypoints3d --frames 392
```

## 4. Numbers

These use 10 limb joints and frames 0–391, against the dome reference. Each cell is MPJPE / PA-MPJPE in mm. "Missing"
is the share of the 10 limb joints with no estimate. The triangulation column reproduces T1 exactly.

| Layout | Triangulation (T1) | Missing | **SMPL, J_regressor** | Missing | SMPL, triangulated joints only | SMPL, EasyMocap BODY25 regressor |
|---|---|---|---|---|---|---|
| All 31 | **11.7** / 10.2 | 0.0 % | 44.8 / 32.9 | 0.0 % | 44.8 / 32.9 | 41.9 / 28.6 |
| 6 spread A | **14.8** / 13.4 | 0.0 % | 45.0 / 33.3 | 0.0 % | 45.0 / 33.3 | 42.6 / 28.9 |
| 5 spread A | **16.2** / 14.0 | 0.1 % | 43.5 / 33.2 | 0.0 % | 43.5 / 33.2 | 43.3 / 30.0 |
| 4 spread A | **20.0** / 16.3 | 0.1 % | 47.2 / 33.3 | 0.0 % | 47.2 / 33.3 | 44.8 / 29.2 |
| 3 spread A | **24.6**\* / 16.5 | 23.5 % | 63.6 / 48.5 | 0.0 % | 47.3 / 30.5 | 62.0 / 46.4 |
| 5 bunched, behind | **26.0** / 19.2 | 0.0 % | 45.2 / 33.5 | 0.0 % | 45.2 / 33.5 | 45.2 / 31.1 |

\* Over the 76.5 % of joints that triangulation kept (T1). "Triangulated joints only" scores SMPL on exactly those
joints. The 3-spread SMPL row (63.6 mm) includes the filled-in joints.

Per joint, all 31 cameras, MPJPE in mm:

| Joint | Shoulder R / L | Elbow R / L | Wrist R / L | Hip R / L | Knee R / L |
|---|---|---|---|---|---|
| Triangulation | 14.5 / 10.4 | 8.6 / 7.9 | 7.9 / 7.7 | 19.0 / 17.3 | 10.9 / 13.2 |
| SMPL, J_regressor | 67.5 / 66.9 | 35.0 / 40.1 | 25.2 / 23.3 | 41.6 / 52.9 | 47.9 / 48.2 |
| SMPL, BODY25 regressor | 42.5 / 41.5 | 32.5 / 35.5 | 25.5 / 23.8 | 58.6 / 54.9 | 52.1 / 51.8 |

Joints filled in by SMPL in 3-spread A, MPJPE in mm (number of joint-frames in brackets): right elbow 127.8 (121), left
elbow 136.4 (51), right wrist 267.3 (125), left wrist 278.3 (27), left hip 53.4 (178), left knee 85.1 (372), right
shoulder 60.7 (28), and 75.8 / 69.1 / 46.8 for the left shoulder, right hip and right knee (2–16 each). Overall:
116.6 mm over 922 joint-frames.

## 5. Checks

- **Same input as T1:** the triangulation scored in this run gives T1's numbers in every layout (for example 24.6 /
  16.5 mm with 23.5 % missing for 3-spread A, and 11.7 / 10.2 mm for all 31).
- **Fit vs. its own input:** EasyMocap's BODY25 keypoints of the fit are 40.6 mm (all 31) and 39.8 mm (5 spread) from
  the triangulated points, frames 0–391, 10 limb joints. By joint, hips and knees are 49–57 mm off and wrists 20–25 mm.
  The fit also moves less than its input (mean per-frame joint motion 9.5 vs. 12.6 mm with all 31), which looks like
  heavy smoothing. In the all-31 fit, the shape stays near the mean (first shape coefficient 0.27, the rest within ±0.05). On
  REHAB24-6 (T2c/T2d) the same fit stayed about 27 mm from its input.
- **Joint definition:** shoulders improve from 67 to 42 mm when scored through EasyMocap's BODY25 regressor instead of
  the SMPL J_regressor; elbows and wrists hardly change, and hips and knees get worse. So the J_regressor joints and the
  dome's keypoint-style joints differ most at the shoulders.

## 6. Failed or skipped

- **The default fit isn't tuned for this subject.** I didn't change EasyMocap's fit weights: shape regularisation,
  smoothing, and the 3D keypoint weight with `norm_info: 0.02`. A stronger 3D term or weaker smoothing would probably
  bring the fit closer to the triangulation. That would be a separate, tuned experiment, and it would need a held-out
  sequence so it doesn't tune on the reference.
- **Only the A subsets** of the spread layouts and only the bunched-behind layout, as asked. The B / C subsets and the
  in-front bunched layout weren't fitted.
- **Ankles** are left out of the main score, as in T1.
- **No GPU, by instruction.** The CPU run took 3.5 min.
- **My first scoring attempt** for the BODY25-regressor variant failed, because EasyMocap's `keypoints3d` files have no
  confidence column. I fixed it in `panoptic_smpl_eval.py`, which now writes confidence 1.
