# T2b: the authors' camera parameters, and GVHMR on the REHAB24-6 benchmark, 2026-10-29

Follow-up to `2026-10-27-t2-monocular.md`. New inputs: camera parameters sent by the dataset authors (email,
2026-10-29), an "old" estimate (used in their publications and for the dataset's 2D joints) and a "new", more careful
estimate; and author copies of "REHAB24-6: Physical Therapy Dataset for Analyzing Pose Estimation Methods" (SISAP 2024)
and "Pose Estimation Analysis and Fine-Tuning on the REHAB24-6 Rehabilitation Dataset" (Information Systems 2025,
revised version). Neither the camera files nor the papers are in git; this file gives only the convention and derived
comparisons.

## 1. Result

On the paper's own benchmark protocol (all 65 recordings, both cameras, all 1,072 repetitions, the paper's 12 joints),
GVHMR, which is not in the paper, reaches a **BestFit 3D MPJPE of 6.00 cm** (BBox 7.94 cm). The paper's best results are
8.4 cm (MediaPipe Pose full) and 7.7 cm with ideal ground-truth 2D input. The paper doesn't say whether BestFit is fitted
per frame; with one fit per repetition instead, GVHMR gives 7.16 cm, still below both. Two caveats: BBox is my
reconstruction of a transformation the paper only describes in words, and the GVHMR weights come from an unofficial mirror.

The authors' old cameras are exactly the cameras I had recovered by DLT. Their new cameras match the real video better:
against two independent 2D detectors the median joint distance drops from 13.5 to 11.5 px (horizontal camera) and from
24.9 to 21.3 px (vertical camera), and two-camera triangulation with no alignment improves from 47.5 to 42.9 mm. The
change is modest, about 2–3 px, so it explains only a small part of the paper's 2D errors.

## 2. Setup

| Field | Value |
|---|---|
| Host | `lambdavector2`, 2× RTX A6000; all jobs on GPU 1 (`GPU-dc002436`); P3 on GPU 0 |
| GVHMR | `ee960bb`, weights from the **unofficial HF mirror** `camenduru/GVHMR` (official Google Drive was rate-limited on 10-27; Tan's choice; hashes in `DATA_LOG.md`). Static camera, focal 2181.7 px (camera 17) / 2176.09 px (camera 18) as in the P2 true-focal run |
| EasyMocap | `e6006fd`, YOLOv5m + HRNet-W48, naive 2-view triangulation (cached P2 detections) |
| Data | REHAB24-6 (Zenodo 13305826, CC BY-NC 4.0), all 65 recordings; camera parameters sent by the dataset authors (email, 2026-10-29), SHA-256 in `DATA_LOG.md` (match on the server) |
| This repo | worktree `~/fall2026/wt-t2b`, branch `server/t2b-20261029` from `origin/main` `9db2138` |
| New tools | `rehab_authors_calib.py` (A), `rehab_calib_compare.py` (B), `rehab_emc_retriangulate.py` (C), `rehab_paper_protocol.py` (D, with `--selftest`) in `reconstruction/tools/` |

## 3. Commands

Logs: `~/fall2026/logs/2026-10-29-t2b-*.log`, `logs/t2b-pipeline/`, `logs/t2b-chunks/`. Heavy jobs ran in
`systemd-run --user --scope -p MemoryMax=16G|32G -p MemorySwapMax=0`, GPU jobs with `CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=1`.

```bash
git fetch origin; git worktree add -b server/t2b-20261029 ~/fall2026/wt-t2b origin/main
sha256sum data/rehab24-6/authors_calib/*.json data/papers/*.pdf        # all 7 match DATA_LOG.md
# A
python conv_search.py            # (scratchpad) all axis-swap / transpose / centre-vs-translation / sign / flip combinations
python $T/rehab_authors_calib.py --authors data/rehab24-6/authors_calib --data data/rehab24-6 --dlt runs/t2-gvhmr --out runs/t2b
# B
python $T/rehab_calib_compare.py --data data/rehab24-6 --chunks runs/t2-chunks/out --emc runs/t2-emc \
    --emc-data data/rehab24-6-emc --calib runs/t2b --rec <12> --out runs/t2b/compare.json
python <paper-style 12-joint HRNet 2D error, old vs new>              # logs/2026-10-29-t2b-B-table4-style.log
# C
python $T/rehab_emc_retriangulate.py --cams 'data/rehab24-6-emc/{rec}' ... --check     # reproduces EasyMocap: 5e-8 m
python $T/rehab_emc_retriangulate.py --calib runs/t2b/calib-new --cams-out runs/t2b/emc-cams-new --emc runs/t2-emc --out runs/t2b/emc-new --rec <12>
python $T/rehab_eval.py   --chunks runs/t2-chunks/out-truef --calib runs/t2b/calib-{old,new} --rec <12>
python $T/rehab_twocam.py --chunks runs/t2-chunks/out-truef --calib runs/t2b/calib-{old,new} --emc runs/t2-emc|runs/t2b/emc-new ...
# D
pdftotext -layout <both papers>                                        # protocol: Sec. 4.2.1-4.2.3, 6.1-6.4
python $T/rehab_paper_protocol.py --selftest
python $T/rehab_paper_protocol.py --chunks runs/t2-chunks/out-truef --calib-old runs/t2b/calib-old --rec <12> --out runs/t2b/paper12.json
systemctl --user enable --now t2b-gvhmr65    # logs/t2b-pipeline/run.sh: 334 new chunks (53 recordings), 6 parallel on GPU 1,
                                             #   then rehab_paper_protocol.py on all 65 -> runs/t2b/paper65_summary.txt
python <IoU distribution over all 65 recordings>                       # logs/2026-10-29-t2b-D-iou-check.log
python $T/rehab_paper_protocol.py --chunks runs/t2b/chunks/out ... --rec <65>   # rerun after adding the per-repetition BestFit
python $T/rehab_paper_protocol.py --summary runs/t2b/paper65.json > runs/t2b/paper65_summary.txt
```

## 4. Numbers

### A. Camera convention

The authors' "old" parameters reproduce the dataset's 2D joints of all 65 recordings to ≤ 9.1e-13 px with this
convention, and no other combination comes within 900 px:

- world: the `3d_joints` axes with y and z swapped, X′ = M·X;
- `T` is the camera centre in that swapped world; `R` maps world to camera, with the camera looking along −z:
  x_cam = −R·(X′ − T);
- pinhole: u = f_x·x/z + c_x, v = f_y·y/z + c_y, no distortion, f_x = f_y.

In the `3d_joints` world this is P = K·[A | −A·C] with A = −R·M and C = M·T. **The old cameras are my DLT cameras**:
centre difference 0.0000 mm, rotation difference 0.00000°, projections equal to 1e-12 px (both cameras). Neither
file's R is an exact rotation: camera 17 singular values 0.9999/0.9999/0.9998, camera 18 0.9956/0.9956/0.9904 (new:
0.996/0.996/0.9912). For camera 18 this is where my DLT's f_x 2170.4 ≠ f_y 2181.8 and skew −0.25 came from. Where a
proper rotation is needed (EasyMocap files, rotating GVHMR output), P is factored into an upper-triangular K′ and a
rotation R′ (RQ), which reproduces P exactly (EasyMocap yml: 0.0002 px after 6-decimal rounding).

**New vs. old:**

| Camera | Centre moves | Rotation changes | Joint projections move (median / p95) |
|---|---|---|---|
| 17 (horizontal) | 12.7 cm | 0.50° | 6.7 / 13.0 px |
| 18 (vertical) | 10.6 cm | 1.24° | 10.6 / 26.4 px |

(Every 30th frame of all 65 recordings, all 26 joints. The rotation alone would be ~19 / 47 px at f = 2182 px; the
centre shift partly compensates.)

### B. Which calibration matches the real video?

Mocap joints projected with each calibration, compared with two detectors that never saw the dataset's 2D: GVHMR's
ViTPose-H (COCO-17) and EasyMocap's HRNet-W48. 12 P2 recordings, both cameras, frames inside repetitions, detector
confidence > 0.5, projection inside the image. Joints: shoulders, elbows, wrists, knees, ankles (hips left out:
Motive's hips are placed differently). "Constant" = |mean 2D residual| per joint and recording, "varying" = RMS
around that mean (n-weighted over joints and recordings).

| Camera | Detector | Joint obs. | Median px, old → new | Constant, old → new | Varying, old → new |
|---|---|---|---|---|---|
| 17 | ViTPose | 226,373 | 13.5 → **11.5** | 12.1 → 9.8 | 8.7 → 8.3 |
| 17 | HRNet | 225,163 | 15.2 → **13.1** | 13.8 → 11.4 | 19.9 → 19.5 |
| 18 | ViTPose | ~222,950 | 24.9 → **21.3** | 25.5 → 22.4 | 26.3 → 25.7 |
| 18 | HRNet | ~223,600 | 26.2 → **23.0** | 27.9 → 25.1 | 32.9 → 32.2 |

Per joint (median px, old/new), ViTPose: camera 17 R_wrist 17.3/11.5, R_elbow 16.7/12.1, R_ankle 16.1/10.4, but
L_knee 8.7/11.5; camera 18 L_wrist 28.4/20.8, L_ankle 27.8/20.5, but R_knee 14.1/17.4. HRNet shows the same pattern.
Per recording, the constant part drops in 20 of 24 recording × camera pairs for ViTPose; it rises in PM_037 camera 18
(11.6 → 14.3) and is unchanged in three. Overlays (PM_118 camera 18 frame 333, PM_014 camera 17 frame 921; kept out of
git, faces): the new projection sits on the visible knees, ankles and wrists, the old one a few pixels off (lower on the
legs in the vertical camera).

**So the new calibration is closer to the truth**, in the part of the error a camera can change (the constant offset),
for both detectors and both cameras. About 22 px of constant offset remains in camera 18; the detectors mark COCO-style
surface points while Motive's joints are skeleton centres, so not all of that is calibration.

**What this means for the paper's 2D numbers.** Measured like Table 4 of the Information Systems paper (12 joints with
hips, mean px, frames inside repetitions, no IoU filter), HRNet-W48 on my 12 recordings scores 20.3 px (horizontal) and
35.9 px (vertical) against the old-projected ground truth, in the same range as their 18–24 and 30–38 px; against the
new projection it would be 18.5 and 34.1 px (per exercise: horizontal 15.2→13.8, 16.2→12.6, 34.0→33.8, 16.0→15.3,
16.8→14.7, 25.8→24.2; vertical 33.2→29.8, 41.3→38.8, 28.0→27.0, 35.2→33.0, 47.4→48.0, 27.3→26.2). So roughly 2 px
(5–10%) of their 2D error is the rough calibration; most of it is detector error and joint-definition offsets. Hips get
worse with the new camera on the vertical view (31.0 → 34.9 px), which fits hips being defined differently rather than
the camera being worse.

Their fine-tuned HRNet (Table 7, 7–19 px) was trained and tested on old-projected ground truth, which itself is 6.7 /
10.6 px (median) and up to 13 / 26 px (p95) away from the better estimate. A fine-tuned error at or below that level
partly means the network learned the old projection's offsets; how close it is to the true image positions can't be
read from those numbers.

### C. Rerun with the new calibration (12 recordings, clean repetitions, GVHMR true focal)

PA does not use the calibration and is unchanged. Two-camera triangulation improves; GVHMR's camera-dependent numbers
move in both directions.

| Metric (mm) | Old | New |
|---|---|---|
| GVHMR camera 17, 12 joints: PA / root / cent / abs | 46.34 / 99.22 / 67.14 / 183.52 | **46.34** / 99.76 / 68.03 / 192.57 |
| GVHMR camera 18, 12 joints: PA / root / cent / abs | 69.86 / 126.93 / 98.69 / 178.00 | **69.86** / 126.04 / 97.60 / 159.48 |
| Two cameras, ViTPose + calibration, 10 joints: abs / PA / root / cent | 47.5 / 36.9 / 64.9 / 43.5 | **42.9** / 37.0 / 62.9 / 40.8 |
| Two cameras, EasyMocap HRNet, 10 joints: abs / PA / root / cent | 57.2 / 43.1 / 74.5 / 50.0 | **53.3** / 43.1 / 73.1 / 47.7 |
| GVHMR camera 17 on the same 10 joints/frames: PA / root / cent / abs | 46.8 / 97.0 / 65.6 / 182.4 | 46.8 / 97.5 / 66.4 / 196.6 |
| GVHMR camera 18 on the same 10 joints/frames: PA / root / cent / abs | 69.6 / 120.5 / 93.6 / 182.8 | 69.6 / 119.7 / 92.7 / 159.6 |

(183 repetitions clean in both cameras, 18,584 frames for the two-camera rows.) Two-camera abs per exercise, old → new:
ViTPose 47.9→42.6, 46.0→37.2, 49.2→47.8, 46.1→44.1, 53.2→51.7, 44.7→39.9; EasyMocap 50.9→45.5, 50.9→42.8, 62.9→62.5,
50.4→48.4, 65.8→65.0, 70.2→67.0. Every exercise improves, which is independent support for the new cameras: triangulation
uses nothing but the two cameras and the 2D detections.

### D. GVHMR on the paper's benchmark

**Protocol as written** (Information Systems 2025, Sec. 4.2.1–4.2.3, 6.1–6.4; the SISAP paper repeats it):

| Item | Paper | Choice here |
|---|---|---|
| Joints | 12: shoulders, elbows, wrists, **hips**, knees, ankles ("greatest common subset") | same; GVHMR SMPL-24 16/17, 18/19, 20/21, 1/2, 4/5, 7/8 ↔ REHAB 7/12, 8/13, 9/14, 16/21, 17/22, 18/23 |
| Frames | "exercise repetitions only" (stated for fine-tuning input) | frames inside every annotated repetition |
| Invalid frames | undetected frames, plus IoU of 2D boxes around GT and predicted **leg and shoulder** joints below a "low threshold" | IoU < 0.1 (value **not given**; 0.05 and 0.3 also reported); GT 2D = dataset 2D (old cameras), predicted 2D = GVHMR's own 2D input (ViTPose); "leg" = hips, knees, ankles |
| BestFit | Procrustes: scale, translation, rotation | per frame (granularity **not stated**) |
| BBox | GT rotated with the approximate (old) camera; no rotation fitted; x/y shift from 2D vs. 3D box centres relative to box size; only scale and depth shift fitted | my reading (formula **not given**): least-squares scale; predicted 3D box centre placed at GT 3D box centre + (2D centre offset ÷ GT 2D box size) × GT 3D box size; depth shifted by the mean depth difference; reported 3D, 2D (x/y) and depth (z) parts |
| Aggregation | "across all videos (i.e., all exercises and cameras)"; MPJPE = mean "across all poses and video sequences" | frame-weighted over all frames; video-weighted (each recording × camera once) also reported |
| `mocap_erroneous` | **not mentioned** | kept (main); dropped also reported |
| Units | 3D MPJPE in cm | cm |
| Test set | Table 5 does not say whether it is all data or a split (Table 7 is cross-subject) | all 65 recordings, both cameras |

**12 P2 recordings** (432 repetition × camera, 45,422 frames, valid 100% at IoU 0.1):
BestFit **6.28 cm** (video-weighted 6.36), BBox 3D 8.17 cm (2D part 5.27, depth part 5.21); camera 17 BestFit 5.26,
camera 18 7.31 cm.

**All 65 recordings** (the paper's setting): 2,144 repetition × camera, 256,892 frames, 99.997% valid at IoU 0.1 (six
frames have no prediction: the last frame of a repetition in recordings whose video is one frame shorter than the
mocap; one frame has IoU < 0.1). Conditions: GVHMR `ee960bb` with the unofficial mirror weights, static camera, focal
2181.7 / 2176.09 px, chunks of ≤ 1000 frames over the repetition spans, RTX A6000 (GPU 1).

| 3D MPJPE (cm) | BestFit (per frame) | BestFit (per repetition) | BBox 3D | BBox 2D part | BBox depth part |
|---|---|---|---|---|---|
| **GVHMR, all, frame-weighted** | **6.00** | 7.16 | 7.94 | 5.26 | 4.92 |
| GVHMR, all, video-weighted | 6.08 | 7.27 | 8.07 | 5.28 | 5.07 |
| GVHMR, `mocap_erroneous` dropped (2,114 rows) | 5.95 | 7.11 | 7.88 | 5.22 | 4.88 |
| GVHMR, horizontal camera (17) | 5.36 | 6.66 | 7.01 | 4.73 | 4.22 |
| GVHMR, vertical camera (18) | 6.65 | 7.67 | 8.87 | 5.80 | 5.62 |
| Paper, MPP-full | 8.4 | – | 11.7 | 6.6 | 8.4 |
| Paper, MPP-heavy | 8.7 | – | 12.0 | 6.6 | 8.7 |
| Paper, MHFormer-351 + HRNet | 9.0 | – | 14.6 | 6.4 | 12.1 |
| Paper, MotionBERT-lite + HRNet | 9.2 | – | 13.6 | 6.0 | 11.1 |
| Paper, MotionBERT-lite + ideal GT 2D | 7.7 | – | 11.3 | 3.8 | 10.0 |

IoU threshold 0.05, 0.1 or 0.3 changes nothing at two decimals (90 frames have IoU < 0.3, 0.035%); GVHMR follows one
tracked person through each chunk, and in these recordings that is always the subject. The 12 P2 recordings alone gave
6.28 cm (§ above), so the subset was representative.

BestFit per joint (cm, all, frame-weighted): shoulders 4.88 / 5.00, elbows 5.27 / 5.57, **wrists 7.51 / 8.63**,
**hips 7.34 / 7.28**, knees 4.43 / 4.33, ankles 5.90 / 5.90 (left / right). Wrists are worst, as in P2; hips are high
because Motive places them differently from SMPL.

**Per exercise, camera and view**, next to the paper's Table 6. Their Table 6 lists, for each exercise, the single best
of their models (chosen per exercise), so it favours them; GVHMR is one model everywhere.

| Ex | Camera | View | Reps | GVHMR BestFit | GVHMR BestFit per rep. | GVHMR BBox | Paper best BestFit | Paper best BBox |
|---|---|---|---|---|---|---|---|---|
| 1 | hor. | front | 88 | 4.87 | 5.53 | 6.88 | 7.9 | 11.3 |
| 1 | hor. | half-profile | 90 | 4.58 | 5.57 | 5.89 | 5.7 | 8.4 |
| 1 | ver. | half-profile | 90 | 5.16 | 5.91 | 7.94 | 7.4 | 19.3 |
| 1 | ver. | profile | 88 | 9.19 | 9.34 | 10.54 | 13.5 | 19.0 |
| 2 | hor. | front | 109 | 5.24 | 6.15 | 8.01 | 6.3 | 9.2 |
| 2 | hor. | half-profile | 99 | 5.47 | 7.07 | 6.90 | 6.5 | 10.2 |
| 2 | ver. | half-profile | 99 | 4.92 | 5.55 | 7.39 | 7.6 | 21.2 |
| 2 | ver. | profile | 109 | 7.60 | 8.40 | 9.82 | 11.3 | 21.0 |
| 3 | hor. | profile | 107 | 5.73 | 7.48 | 7.45 | 8.1 | 11.4 |
| 3 | ver. | front | 107 | 7.57 | 10.32 | 11.63 | 10.7 | 14.5 |
| 4 | hor. | front | 116 | 5.97 | 6.70 | 7.31 | 5.5 | 8.1 |
| 4 | hor. | half-profile | 94 | 5.80 | 6.75 | 7.34 | 8.2 | 11.6 |
| 4 | ver. | half-profile | 94 | 6.16 | 6.72 | 7.77 | 7.2 | 13.1 |
| 4 | ver. | profile | 116 | 7.17 | 8.88 | 9.47 | 12.0 | 16.1 |
| 5 | hor. | front | 88 | 5.74 | 7.94 | 6.91 | 11.3 | 13.2 |
| 5 | hor. | half-profile | 86 | 5.90 | 6.98 | 6.97 | 8.8 | 9.7 |
| 5 | ver. | half-profile | 86 | 5.98 | 6.75 | 7.34 | 12.8 | 14.9 |
| 5 | ver. | profile | 88 | 7.06 | 8.18 | 9.25 | 10.0 | 12.3 |
| 6 | hor. | front | 98 | 5.04 | 7.57 | 6.09 | 10.4 | 12.4 |
| 6 | hor. | half-profile | 97 | 5.24 | 6.59 | 6.91 | 7.3 | 9.0 |
| 6 | ver. | half-profile | 97 | 5.09 | 6.63 | 7.04 | 10.4 | 12.4 |
| 6 | ver. | profile | 98 | 6.28 | 7.15 | 7.80 | 12.1 | 14.5 |

GVHMR is lower than the paper's best model in 21 of 22 cells with per-frame BestFit; the exception is Ex4 horizontal
front (5.97 vs. 5.5). With per-repetition BestFit it is lower in 20 of 22 (exceptions: Ex4 horizontal front 6.70 vs.
5.5, Ex2 horizontal half-profile 7.07 vs. 6.5). GVHMR's worst cell, like the paper's, is the
vertical camera in profile for Ex1 (9.19 cm), the hidden-arm case found in P2. Per exercise (both cameras, BestFit):
Ex1 5.97, Ex2 5.88, Ex3 6.65, Ex4 6.31, Ex5 6.18, Ex6 5.42 cm.

**Run cost:** 334 new chunks (53 recordings, 279,058 span frames) in 5 h 48 min (15:31–21:19), 6 in parallel on GPU 1;
peak RSS 6.94–7.21 GB per chunk, peak GPU ≤ 4.7 GB per job, GPU 1 total ≤ 28.0 GB, my total RSS ≤ 23.3 GB (limits 40 GB
and 150 GB never hit); 11 GB of outputs in `runs/t2b/chunks/`.

## 5. Checks

- **Inputs:** SHA-256 of the 4 camera files and 3 PDFs match `DATA_LOG.md` on main.
- **Convention:** a full search over axis swap, R vs. Rᵀ, T as centre vs. translation, overall sign and image flips; only
  one combination reproduces the dataset 2D (2e-13 / 5e-13 px), the next best is > 900 px off. Then checked on every
  frame and joint of all 65 recordings (≤ 9.1e-13 px).
- **EasyMocap re-triangulation:** with the yml files EasyMocap used, my call into EasyMocap's own `read_camera`,
  `Undistort.points` and `batch_triangulate` reproduces its triangulated output to 5e-8 m (PM_032, PM_118: 5,182 frames),
  so the new-camera numbers are what EasyMocap would give, without re-detecting.
- **New yml files** reproduce the new P to 0.0002 px.
- **PA unchanged** under old and new (46.34 / 69.86 mm to two decimals), as it must be.
- **Paper protocol code:** `rehab_paper_protocol.py --selftest` (BestFit exact under scale + rotation + translation;
  BBox recovers a pure scale + depth shift exactly; a 2D box shifted by 10% of its width moves the 3D prediction by 10%
  of the 3D box width; IoU of two half-overlapping boxes = 1/3) passes. Cross-check with P2: camera 17/18 BestFit 5.26 /
  7.31 cm on all repetitions with hips, vs. P2's 46.3 / 69.9 mm on clean repetitions without hips.
- **Plausibility against the paper:** my HRNet 2D error with old ground truth (20.3 / 35.9 px) lies inside their
  HRNet range (18–24 / 30–38 px), so my 2D ground truth, joint set and frames match theirs closely enough to compare.
- **IoU filter is active, not idle:** over all 256,892 frames the leg + shoulder box IoU has percentiles 0.093 (min),
  0.32 (0.1%), 0.44 (1%), 0.58 (5%), 0.85 (median); 90 frames are below 0.3 and one below 0.1 (PM_005 camera 17,
  frame 1037). "100%" in an earlier summary was rounding (fixed to two decimals).
- **Joint mapping and units on the full run:** same SMPL-24 → REHAB mapping and metres → cm as the 12-recording run, which
  agrees with P2 (see above); the 65-recording BestFit (6.00 cm) is close to the 12-recording value (6.28 cm).
- **GPU placement:** 690 watch samples, 0 with a GVHMR job on GPU 0; every chunk's RESULT line has rc 0 (334/334).

## 6. Failed or skipped

- **The laptop's "≈ px" for new vs. old (19 / 47 px)** overstates the change: it is the rotation alone; with the centre
  shift, joints move 6.7 / 10.6 px (median).
- **The long run waited one usage-limit reset:** the scripts were written at ~13:00, but the unit was started only at
  15:31 after the limit reset; nothing had run in between (no logs, no outputs).
- **My first progress watch for the long run printed nothing** for 30 min (filter bug); the run itself was fine (24 chunks
  done, rc 0), checked directly and replaced by a simpler poll.
- **BBox is my reconstruction** of a transformation the paper describes only in words; treat its numbers as approximate.
- Step 5 of P2 (YouTube) is still open (needs clip URLs from Tan).
- **Comparability with the paper is limited by three unstated choices** (BestFit granularity, IoU threshold, BBox
  formula). I report the per-frame and per-repetition BestFit to bracket the first; the second makes no difference for
  GVHMR; the third makes my BBox numbers approximate. The paper's models also differ in input: they run per camera with
  their own 2D detectors, GVHMR with ViTPose-H; Table 6's "best model" is chosen per exercise on the same data.
- **GVHMR weights are from an unofficial mirror** (`camenduru/GVHMR`, see P2 and `DATA_LOG.md`); results could differ
  slightly from the official checkpoint if the files differ (only the YOLO file could be checked against an official
  source).

## 7. Not tested

- Distortion: both files say "without distortion"; I did not test whether a lens model would explain the remaining
  camera-18 offset.
- Their exact IoU threshold, BestFit granularity and BBox formula: not given in either paper; matching them exactly would
  need the authors' evaluation code.
- GVHMR with the new cameras' focal (both files have the same f; only the extrinsics changed).
- Fine-tuned HRNet (weights not available to us).
- GPU 0 / more parallel jobs.

## 8. Next step

Ask the dataset authors for their evaluation code (or just the IoU threshold, BestFit granularity and BBox formula) so
the GVHMR comparison can be made exact, and send them the calibration finding: the new cameras fit the video better,
which matters for their 2D ground truth and fine-tuning numbers. On our side, the two-camera result with the new
cameras (42.9 mm without any alignment) is the strongest number we have; the next run would fit SMPL to the two
views (EasyMocap `mv1p` with SMPL) on the same repetitions and score it with this protocol, to compare "two cameras +
body model" directly against GVHMR's 6.00 cm.
