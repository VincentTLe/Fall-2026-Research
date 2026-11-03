# T2c: two cameras plus an SMPL body model on REHAB24-6, 2026-11-02

Follow-up to `2026-10-27-t2-monocular.md` and `2026-10-29-t2b-calibration-benchmark.md`. Question: does fitting SMPL
to the two-camera keypoints beat plain triangulation, and how does it compare with single-camera GVHMR?

## 1. Result

Fitting SMPL to two cameras makes the body shape slightly better and the absolute position slightly worse. On the 183
clean repetitions, PA-MPJPE improves from 43.1 mm (EasyMocap triangulation, its own input) to **41.2 mm**, while the error
without alignment gets worse, from 53.3 to **57.3 mm**. Both stay behind the triangulation of GVHMR's ViTPose 2D
(42.9 / 37.0 mm), so the better 2D detector matters more than the body model. The two-camera SMPL fit beats
single-camera GVHMR from either camera in all four metrics: PA 41.2 mm against 46.8 (horizontal camera) and 69.6
(vertical), and 57.3 mm without alignment against 196.6 and 159.6. GVHMR on the horizontal camera is still better in PA
for exercise 1 (33.9 vs 38.9 mm).

On the paper's protocol (12 recordings, 12 joints including hips), the SMPL fit reaches a **BestFit of 4.83 cm** per frame
and **5.48 cm** per repetition, against GVHMR's 6.28 / 7.49 cm on the same recordings. The paper's best on all 65
recordings is 8.4 cm (7.7 cm with ideal 2D). This is a two-camera result on 12 recordings, so it isn't a like-for-like
entry for the paper's single-camera table.

## 2. Setup

| Field | Value |
|---|---|
| Host | `lambdavector2`, **CPU only** (both GPUs busy with another user's job); `CUDA_VISIBLE_DEVICES=` checked in the process environment |
| Method | EasyMocap `e6006fd`, `mv1p` config `detect_triangulate_fitSMPL.yml` via `run_mv1p.py --use-annots --smpl --device cpu --threads 8 --triangulate-mode naive --no-vis --render-views` (no rendering) |
| 2D input | the cached P2 HRNet-W48 detections (`runs/t2-emc/<rec>/detections/`), written back as EasyMocap `annots/` by the new `rehab_emc_smpl.py prep`. **No re-detection.** |
| Cameras | the authors' new cameras (`runs/t2b/emc-cams-new`, from T2b) |
| Frames | P2's span per recording (first to last annotated repetition, 31,797 frames). SMPL is fitted over the whole span (EasyMocap's temporal smoothing), scored inside repetitions only. |
| Body model | Tan's `SMPL_NEUTRAL.pkl` (SMPL v1.1.0, SHA-256 in `DATA_LOG.md`), EasyMocap's `J_regressor_body25.npy` for the fit |
| Scored joints | SMPL J_regressor (24) on the fitted vertices, the same joint definition as GVHMR (`rehab_emc_smpl.py joints`) |
| Recordings | the 12 P2 recordings (PM_010, 014, 018, 021, 022, 023, 030, 032, 037, 040, 118, 124) |
| Run | one enabled user unit `t2c-smpl` (now disabled), each recording in `systemd-run --scope -p MemoryMax=32G -p MemorySwapMax=0`, `nice -n 19`, 8 threads. 18 min in total; peak RSS 0.84–1.22 GB per recording |
| Scoring | `rehab_twocam.py --smpl` (T2 metrics) and `rehab_paper_protocol.py --smpl` (T2b paper protocol), both new options |

```
T=~/fall2026/wt-t2c/reconstruction/tools   # logs: ~/fall2026/logs/t2c-pipeline/
python $T/rehab_emc_smpl.py prep --emc runs/t2-emc --emc-data data/rehab24-6-emc --cams runs/t2b/emc-cams-new --out runs/t2c/data --rec <rec>
python $T/run_mv1p.py --data runs/t2c/data/<rec> --out runs/t2c/fit/<rec> --views 17 18 --device cpu --threads 8 --use-annots \
    --smpl models/smpl/SMPL_NEUTRAL.pkl --render-views --no-vis --triangulate-mode naive --fps 30
python $T/rehab_emc_smpl.py joints --fit runs/t2c/fit --emc-data data/rehab24-6-emc --smpl models/smpl/SMPL_NEUTRAL.pkl --data data/rehab24-6 --rec <rec>
python $T/rehab_twocam.py --chunks runs/t2-chunks/out-truef --calib runs/t2b/calib-new --emc runs/t2b/emc-new \
    --emc-data data/rehab24-6-emc --smpl runs/t2c/fit --rec <12> --out runs/t2c/twocam_smpl.json
python $T/rehab_paper_protocol.py --chunks runs/t2-chunks/out-truef --calib-old runs/t2b/calib-old --smpl runs/t2c/fit \
    --rec <12> --out runs/t2c/paper12_smpl.json
```

## 4. Numbers

**T2 metrics.** These use 183 clean repetitions (no mocap error, at most one extra person per camera), 18,584 frames, and
the 10 limb joints, in mm and frame-weighted. All methods are scored on the same frames. Triangulation and GVHMR rows
are the T2b new-camera values, reproduced exactly by this run.

| Method | abs | PA | root | cent |
|---|---|---|---|---|
| Two-camera triangulation, ViTPose 2D | **42.9** | **37.0** | **62.9** | **40.8** |
| Two-camera triangulation, EasyMocap HRNet 2D | 53.3 | 43.1 | 73.1 | 47.7 |
| **Two-camera SMPL fit, EasyMocap HRNet 2D (T2c)** | 57.3 | 41.2 | 94.0 | 49.9 |
| GVHMR, camera 17 (horizontal) | 196.6 | 46.8 | 97.5 | 66.4 |
| GVHMR, camera 18 (vertical) | 159.6 | 69.6 | 119.7 | 92.7 |

"root" is relative to the mid-hip. SMPL hips sit lower than Motive's (P2), so root is worse for both SMPL-based methods
(the SMPL fit and GVHMR) for that reason alone.

Per exercise (PA, mm):

| Exercise | reps | tri ViTPose | tri HRNet | SMPL fit | GVHMR c17 | GVHMR c18 |
|---|---|---|---|---|---|---|
| 1 | 28 | 36.6 | 37.5 | 38.9 | 33.9 | 78.9 |
| 2 | 36 | 32.7 | 36.9 | 31.1 | 43.7 | 60.6 |
| 3 | 23 | 43.0 | 54.5 | 44.9 | 49.8 | 66.5 |
| 4 | 30 | 35.7 | 39.8 | 43.5 | 55.3 | 73.0 |
| 5 | 31 | 46.1 | 59.8 | 57.3 | 49.8 | 97.1 |
| 6 | 35 | 33.3 | 39.4 | 42.8 | 53.0 | 53.0 |

The SMPL fit beats its own input triangulation in PA in 97 of 183 repetitions, and without alignment in 53. Per joint
(abs, mm), the fit is a little worse than its input for all 10 joints. The largest increases are the left ankle
(46.2 → 57.7), the right shoulder (45.0 → 52.6) and the right ankle (55.0 → 62.5). The wrists stay within 2 mm:
50.5 → 52.7 and 69.5 → 70.3.

**Paper protocol.** These use the 12 recordings, all 216 repetitions × 2 cameras (432 rows, 45,422 frames), the 12 joints
including hips, and IoU ≥ 0.1, in cm and frame-weighted. GVHMR rows come from T2b's `paper65.json`, restricted to the
same 12 recordings.

| Method | BestFit per frame | BestFit per repetition | BBox | valid frames |
|---|---|---|---|---|
| **Two-camera SMPL fit (T2c)** | **4.83** | **5.48** | 5.90 | 99.84 % |
| GVHMR (single camera) | 6.28 | 7.49 | — | 100.00 % |
| Paper's best (all 65 recordings, single camera) | 8.4 (7.7 ideal 2D) | — | 11.7 | — |

For the SMPL fit, the camera matters only through the IoU filter, because the 3D is shared. Per frame it scores 4.80
(camera 17) and 4.85 cm (camera 18); per repetition 5.09 and 5.87 cm (see 5 for why these two differ). Without
repetitions that have more than one extra person in either camera (366 rows), it scores 4.76 / 5.36 cm, and GVHMR
6.15 / 7.23 cm. The SMPL BBox (5.90 cm) uses the reconstructed BBox rule from T2b. It's comparable with GVHMR only
loosely, because the SMPL fit's scale is already metric.

## 5. Checks

- **Same 3D input as T2b:** on a 60-frame test of PM_018, the triangulation inside this run equals T2b's new-camera
  EasyMocap re-triangulation (`runs/t2b/emc-new`) to within 1e-7 m.
- **Same frames as T2b:** the T2 scoring reproduces T2b's triangulation and GVHMR rows exactly (183 repetitions, 18,584
  frames, 42.9 / 37.0 mm, and so on). The paper-protocol GVHMR value on these 12 recordings (6.28 cm) is T2b's.
- **Rounding:** EasyMocap writes SMPL parameters with 3 decimals. Joints recomputed from the written parameters differ
  from EasyMocap's own full-precision output by 0.57 mm on average (1.4 mm at most).
- **Fit vs. input:** the fitted SMPL BODY25 joints sit a median 27.5 mm from the triangulated points they were fitted to
  (PM_018 test).
- **Gross errors:** in 78 of 22,711 repetition frames (0.3 %), the SMPL joints are more than 30 cm from the mocap: 39 in
  PM_010, 33 in PM_118, 5 in PM_022 and 1 in PM_032. In PM_118, the cached camera-17 detection in those frames is a box at
  the right image edge (x ≈ 1742–1919), not the patient. In PM_010 repetition 5, a second person is in camera 17
  (`extra_person_in_cam17 = 3`). Triangulating a wrong box against the right one gives points up to 2.5 m off, and SMPL
  follows them. The camera-17 IoU filter removes most of these frames; the camera-18 filter does not. That's why the
  per-repetition BestFit differs between the cameras (one Procrustes per repetition is very sensitive to a few wrong
  frames, while per-frame BestFit is not). For example, PM_118 repetition 2 scores 6.0 cm in camera 17 and 36.2 cm in
  camera 18. I left these frames in: removing them would need a rule that doesn't look at the mocap, and the cached
  detections pick one box per view without checking that both views agree.
- Self-tests `rehab_twocam.py --selftest` and `rehab_paper_protocol.py --selftest` pass after the changes.

## 6. Failed or skipped

- **No GPU, by instruction.** Everything ran on CPU. The fit was fast anyway (18 min for 12 recordings).
- **No re-detection, by instruction.** The cached HRNet 2D was reused as is, so its wrong-box frames (see 5) remain. A
  cross-view consistency check on the detections, or ViTPose 2D as the fit input, would probably help, since
  triangulating GVHMR's ViTPose 2D is already better (42.9 / 37.0 mm). I didn't try either here.
- **Only the 12 P2 recordings.** I didn't run the paper protocol on all 65, because the cached two-camera 2D exists only
  for these 12.
- **My mistake in a check:** my first check of the rounding error reported 0.9 m, because I indexed EasyMocap's keypoint
  output wrongly (it already returns only the 25 BODY25 joints). After the fix, the value is 0.57 mm. The scoring code
  doesn't use that path; it applies the SMPL J_regressor to the vertices.
