# T2d: two cameras plus SMPL from ViTPose 2D, all 65 REHAB24-6 recordings, 2026-11-02

Follow-up to `2026-11-02-t2c-twocam-smpl.md`. T2c fitted SMPL to cached HRNet 2D on 12 recordings, and a few wrong boxes
spoiled the fit. Here the input is GVHMR's ViTPose 2D from both cameras, a view-agreement filter runs before
triangulation, and all 65 recordings are fitted.

## 1. Result

On the paper's benchmark (all 65 recordings, 1,072 repetitions × 2 cameras, 12 joints), the ViTPose two-camera SMPL fit
reaches a **BestFit of 5.09 cm per frame, 5.29 cm per repetition, and a BBox of 6.00 cm**. Single-camera GVHMR gets
6.00 / 7.16 / 7.94 cm, and the paper's best is 8.4 cm (BBox 11.7). The filter did its job against wrong boxes: per-frame
and per-repetition BestFit now differ by only 0.2 cm (T2c: 0.65 cm). The result is a two-camera method, so it isn't a
like-for-like entry in the paper's single-camera table.

On the T2 metrics (183 clean repetitions, 10 limb joints) the fit is **worse** than T2c and than plain ViTPose
triangulation: abs **56.1 mm**, PA **46.0 mm**. T2c got 57.3 / 41.2 mm and ViTPose triangulation 42.9 / 37.0 mm. The
cause is the filter, not the fit. Where the filter kept all 10 limb joints (14,384 of the 18,584 frames), the SMPL fit
scores PA 34.2 mm, against 31.9 mm for its own filtered triangulation. Where it dropped or lacked at least one limb joint
(4,387 frames), SMPL fills the joint in from its pose prior and temporal smoothing, and PA rises to 86.9 mm. Those are
mostly hidden-arm frames (right wrist 93.9 mm; PM_023 and PM_021). There the views disagree by more than 40 px, yet the
unfiltered triangulation was still close to the truth. The threshold was fixed before scoring, so I report it as is.
I haven't checked why the paper-protocol BestFit stays good despite this.

## 2. Setup

| Field | Value |
|---|---|
| Host | `lambdavector2`, **CPU only** (both GPUs held by another user's job), `CUDA_VISIBLE_DEVICES=` |
| 2D input | GVHMR's ViTPose-H COCO-17 from both cameras (`runs/t2b/chunks/out`; for the 12 P2 recordings it symlinks to `runs/t2-chunks/out-truef`), repetition spans only. Mapped to BODY25 (neck and mid-hip as pair means), confidence < 0.3 unused |
| Cameras | the authors' new cameras (`runs/t2b/emc-cams-new`) |
| View-agreement filter | **fixed before any scoring:** triangulate each joint from both views (EasyMocap `batch_triangulate`), reproject; if the error exceeds **40 px** in either view, drop that joint in that frame, then triangulate again |
| Fit | the `at_final` SMPL stages of EasyMocap `e6006fd` `config/mv1p/detect_triangulate_fitSMPL.yml` (init, shape, RT, pose refinement with smoothing and GMM prior), unchanged. I drove the stages directly from the 2D arrays, because `run_mv1p.py`'s dataset reader needs extracted images, which exist for only 12 recordings. One fit per contiguous block of frames with 2D in both views (65 blocks, one per recording) |
| Body model | Tan's `SMPL_NEUTRAL.pkl` (SMPL v1.1.0, SHA-256 in `DATA_LOG.md`); scored joints = SMPL J_regressor on the fitted vertices, as for GVHMR and T2c |
| Run | one enabled user unit `t2d-smpl` (now disabled), each recording in `systemd-run --scope -p MemoryMax=32G -p MemorySwapMax=0`, `nice -n 19`, 8 threads. 22:00–23:58, 171,326 frames; peak RSS ≤ 6.1 GB |
| Tool | new `reconstruction/tools/rehab_vitpose_smpl.py`; scoring with the T2c `--smpl` options |

```
T=~/fall2026/wt-t2d/reconstruction/tools   # logs: ~/fall2026/logs/t2d-pipeline/
python $T/rehab_vitpose_smpl.py --data data/rehab24-6 --chunks runs/t2b/chunks/out --cams runs/t2b/emc-cams-new \
    --smpl models/smpl/SMPL_NEUTRAL.pkl --out runs/t2d/fit --threads 8 --rec <rec>          # per recording, in the unit
python $T/rehab_paper_protocol.py --chunks runs/t2b/chunks/out --calib-old runs/t2b/calib-old --smpl runs/t2d/fit \
    --rec <65> --out runs/t2d/paper65_smpl.json
python $T/rehab_twocam.py --chunks runs/t2-chunks/out-truef --calib runs/t2b/calib-new --emc runs/t2b/emc-new \
    --emc-data data/rehab24-6-emc --smpl runs/t2d/fit --rec <12> --out runs/t2d/twocam_smpl.json
```

## 4. Numbers

**Paper protocol.** These use all 65 recordings: 2,144 repetition × camera rows, 256,892 frames, 12 joints including
hips, and IoU ≥ 0.1 (100.00 % of frames valid), in cm and frame-weighted.

| Method | BestFit per frame | BestFit per repetition | BBox |
|---|---|---|---|
| **Two-camera SMPL, ViTPose 2D (T2d)** | **5.09** | **5.29** | **6.00** |
| GVHMR, single camera (T2b) | 6.00 | 7.16 | 7.94 |
| Paper's best, single camera | 8.4 (7.7 with ideal 2D) | — | 11.7 |

Video-weighted, T2d gives 5.10 / 5.29 / 5.95 cm, and 5.03 / 5.23 / 5.94 cm without the mocap-erroneous repetitions.
BestFit is the same in both cameras (5.09 cm); BBox gives 5.94 cm (camera 17) and 6.05 cm (camera 18). On T2c's 12
recordings, T2d gives 5.20 / 5.43 cm, against T2c's 4.83 / 5.48 cm.

Per exercise (BestFit per frame, cm):

| Exercise | T2d (two cameras) | GVHMR camera 17 | GVHMR camera 18 |
|---|---|---|---|
| 1 | 5.75 | **4.73** | 7.22 |
| 2 | **4.82** | 5.34 | 6.42 |
| 3 | **5.29** | 5.73 | 7.57 |
| 4 | **5.21** | 5.90 | 6.73 |
| 5 | **5.29** | 5.82 | 6.53 |
| 6 | **4.18** | 5.14 | 5.70 |

**T2 metrics.** These use 183 clean repetitions, 18,584 frames, the 10 limb joints, and the same frames for all
methods, in mm. The other rows are the T2b / T2c values, reproduced by this run.

| Method | abs | PA | root | cent |
|---|---|---|---|---|
| Two-camera triangulation, ViTPose 2D (no filter) | **42.9** | **37.0** | **62.9** | **40.8** |
| Two-camera triangulation, HRNet 2D | 53.3 | 43.1 | 73.1 | 47.7 |
| Two-camera SMPL, HRNet 2D (T2c) | 57.3 | 41.2 | 94.0 | 49.9 |
| **Two-camera SMPL, ViTPose 2D + 40 px filter (T2d)** | 56.1 | 46.0 | 94.7 | 54.6 |
| GVHMR, camera 17 | 196.6 | 46.8 | 97.5 | 66.4 |
| GVHMR, camera 18 | 159.6 | 69.6 | 119.7 | 92.7 |

Per joint, T2d abs in mm:

| Shoulder L / R | Elbow L / R | Wrist L / R | Knee L / R | Ankle L / R |
|---|---|---|---|---|
| 42.2 / 49.8 | 62.3 / 73.7 | 54.0 / **93.9** | 36.4 / 41.0 | 50.1 / 57.0 |

Per recording, PA is clearly worse than ViTPose triangulation in PM_023 (89.8 vs 44.0), PM_021 (85.2 vs 64.4), PM_032
(54.7 vs 33.1), PM_124 (47.1 vs 33.9) and PM_018 (53.0 vs 39.3). It is 6.4 mm worse in PM_118, within 3 mm in four
recordings, and better in PM_010 and PM_040.

## 5. Checks

- **Filter statistics** (all 65 recordings): it dropped 44,480 of 3,249,832 joint observations (1.37 %), in 24,786 of
  171,326 frames (14.5 %). No block was too short to fit.
- **Fit follows its input:** on PM_018 (test run), the SMPL limb joints sit a median 26.8 mm from the filtered
  triangulation (p99 71.9 mm). T2c gave 27.5 mm.
- **Where the T2 loss comes from:** on the clean repetitions, PA is 34.2 mm for SMPL vs. 31.9 mm for its own input where
  all 10 limb joints survive (14,384 frames), and 86.9 mm where at least one doesn't (4,387 frames). See 1.
- **Same frames as before:** the triangulation, HRNet and GVHMR rows of the T2 table equal T2b / T2c exactly (183
  repetitions, 18,584 frames).
- **No wrong-box blow-ups:** per-repetition BestFit is 0.20 cm above per-frame overall and 0.15–0.28 cm per exercise. Both cameras
  give the same BestFit (100 % valid frames), unlike T2c.

## 6. Failed or skipped

- **The 40 px filter is too strict for hidden arms.** It removes the joint the SMPL fit most needs, and the fit's prior
  then does worse than the unfiltered triangulation. I didn't change the threshold after scoring. Options for a next
  run, untested: a looser or camera-specific threshold; dropping only the less confident view instead of the joint; or
  keeping the joint with lowered weight.
- **No GPU, by instruction.** The CPU run took 2 h, so there was no cost problem.
- **`run_mv1p.py` itself wasn't used.** Its image reader needs extracted frames, which exist for only 12 recordings. The
  fit stages and their settings are mv1p's own, unchanged.
- **BBox is my reconstruction** of the paper's rule (T2b). For a metric two-camera fit, its scale step does little.
