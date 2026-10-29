# T2 single-camera reconstruction on REHAB24-6 (P2), 2026-10-27/28

## 1. Result

GVHMR (one camera, static-camera mode) on 399 clean repetitions of 12 REHAB24-6 recordings (all six exercises, lights
on and off) gives **PA-MPJPE 46.3 mm from the horizontal camera 17 and 69.8 mm from the vertical camera 18** on 12 limb
joints. With the true camera rotation and a mid-hip root it is 99.2 / 126.9 mm, partly because Motive and SMPL place the
hips ~9–13 cm apart. Wrists are the worst joints (61–132 mm PA), and a far-side arm hidden in profile view is the main
failure. Two cameras do much better on the same frames: triangulating GVHMR's own ViTPose 2D through the recovered
calibration gives **47.5 mm with no alignment at all** (PA 36.9 mm), against single-camera PA 46.8 / 69.6 mm on those
10 joints. Giving GVHMR the true focal length changes only the unaligned error (camera 17: 197 → 184 mm), and the six
worst camera-18 chunks are hidden-arm occlusions, not wrong tracks (camera 18 PA 63.4 mm without them).

## 2. Setup

| Field | Value |
|---|---|
| Host | `lambdavector2`, Ubuntu 24.04.4 LTS, kernel 6.8.0-137-generic (6.8.0-147 after the 10-28 reboot); 2× RTX A6000 49140 MiB, driver 595.84 (CUDA 13.2 driver API). All my jobs on physical GPU 1 (`GPU-dc002436`), except 20 GVHMR chunks on GPU 0 (§6) |
| GVHMR | https://github.com/zju3dv/GVHMR at `ee960bb6e2ea2d381aa97f08e9b71ef320b624b1` (2026-09-15, "Revert 'Adopt Project Registration License v1.0'"), clean working tree; license: research / non-profit only |
| GVHMR env | `~/fall2026/envs/gvhmr` (6.3 GB): Python 3.10.22 (conda-forge), torch 2.3.0+cu121 (cuDNN 8902), torchvision 0.18.0, numpy 1.23.5, pytorch3d 0.7.6, lightning 2.3.0, ultralytics 8.2.42, opencv 4.11.0; `pip check` clean |
| GVHMR weights | **unofficial mirror** https://huggingface.co/camenduru/GVHMR (official Google Drive refused all five files, "Too many users have viewed or downloaded this file recently"; Tan chose the mirror). SHA-256 in `DATA_LOG.md`; only `yolov8x.pt` could be matched to an official source (Ultralytics release v0.0.0, identical) |
| Body models | Tan's files in `~/fall2026/models/` (mode 600), **symlinked** into `external/GVHMR/inputs/checkpoints/body_models/{smplx,smpl}/` (that folder is gitignored by GVHMR; nothing copied, nothing committed). SMPL-X v1.1 (Tan: its `version.txt` says "Version 1.0"; that file is not on the server), SMPL v1.1.0 |
| Body model SHA-256 | SMPLX_NEUTRAL `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992`, SMPLX_MALE `ab318e3f37d2bfaae26abf4e6fab445c2a610e1d63714794d60379cc263bc2a5`, SMPLX_FEMALE `b2a3686c9d6d218ff6822fba411c607a3c8125a70af340f384ce68bebecabe0e`, SMPL_NEUTRAL `4924f235e63f7c5d5b690acedf736419c2edb846a2d69fc0956169615fa75688` (same as P0), SMPL_MALE `ed4d55bb3041fefc6f73b70694d6c8edc1020c0d07340be5cc651cae2c6a6ae3`, SMPL_FEMALE `6d4a1791b6b94880397e1a3a4539b703a228d2150c57de7b288389a8115f4ef0` |
| EasyMocap (2-camera) | `e6006fd` in `envs/easymocap` (P0), YOLOv5m v7.0 + HRNet-W48, via `reconstruction/tools/run_mv1p.py --triangulate-mode naive` |
| Data | REHAB24-6, Zenodo record 13305826, CC BY-NC 4.0: `videos.zip`, `2d_joints.zip`, `3d_joints.zip`, `Segmentation.csv/.txt`, `joints_names.txt`, all six match the Zenodo MD5; 130 mp4, 260 + 130 npy; SHA-256 in `DATA_LOG.md` |
| Subset | 12 recordings, two per exercise: lights on PM_032 (Ex1, subject 4), PM_040 (Ex2, s5), PM_030 (Ex3, s3), PM_124 (Ex4, s7), PM_037 (Ex5, s4), PM_118 (Ex6, s9); lights off PM_023 (Ex1, s3), PM_014 (Ex2, s2), PM_010 (Ex3, s1), PM_018 (Ex4, s2), PM_021 (Ex5, s2), PM_022 (Ex6, s2). Lights-off recordings exist only for subjects 1–3. Only the frames from the first to the last annotated repetition: 31,797 frames per camera, 432 repetitions, none with `mocap_erroneous`. **Clean** = `extra_person_in_cam ≤ 1` for the camera used: 185 repetitions (19,022 frames) for camera 17 and 214 (22,460) for camera 18 |
| Calibration | Not in the dataset. Recovered by DLT from the dataset's 2D↔3D joints: one 3×4 matrix per camera reproduces all 4.85 M projected joints of all 65 recordings to ≤ 2.2e-12 px. Camera 17: f 2181.7 px, principal point (960.0, 540.0), centre (−3.925, 1.070, 2.691) m. Camera 18: fx 2170.4, fy 2181.8, skew −0.25, principal point (540.0, 960.0), centre (2.045, 0.785, 2.842) m |
| This repo | branch `server/t2-20261027` from `main` at `c4c21a7` |

## 3. Commands

Logs are in `~/fall2026/logs/2026-10-27-p2-*.log`, `logs/p2-chunks/`, `logs/p2-emc/` and `logs/p2-pipeline/`. `T` is
`~/fall2026/Fall-2026-Research/reconstruction/tools`. Every heavy job after the OOM ran as
`systemd-run --user --scope -p MemoryMax=32G -p MemorySwapMax=0 [env CUDA_DEVICE_ORDER=PCI_BUS_ID] CUDA_VISIBLE_DEVICES=1 /usr/bin/time -v ...`.

```bash
# Step 1: data
curl -s https://zenodo.org/api/records/13305826                   # file list + MD5
bash logs/p2-rehab-fetch.sh                                       # curl 6 files, md5sum -c, sha256sum
unzip -l {videos,2d_joints,3d_joints}.zip; unzip -d ...           # structure, no calibration file
ffprobe -count_frames ...; python <inspect npy, bone lengths, overlay 2D on frame 600>
python verify_claims.py   # (scratchpad) 30 vs 120 fps, one DLT per camera on all 65 recordings -> runs/t2-gvhmr/calib_c{17,18}.npz,
                          #  frame-0 skeletons, mocap_erroneous count, cam18 hands outside the image

# Step 2: GVHMR
git clone --recursive https://github.com/zju3dv/GVHMR external/GVHMR          # ee960bb
conda create -p envs/gvhmr python=3.10 pip --override-channels -c conda-forge
pip install -r requirements.txt                                     # failed: chumpy build
pip install "numpy==1.23.5" "setuptools>=68,<80" wheel              # FIX 1 (same as P0)
pip install --no-build-isolation git+https://github.com/mattloper/chumpy.git
pip install -r requirements-nochumpy.txt; pip install -e .; pip check
ln -s ~/fall2026/models/smplx/SMPLX_{NEUTRAL,MALE,FEMALE}.npz inputs/checkpoints/body_models/smplx/
ln -s ~/fall2026/models/smpl/SMPL_{NEUTRAL,MALE,FEMALE}.pkl  inputs/checkpoints/body_models/smpl/
gdown --folder <official Drive folder>                              # refused (quota), see §6
bash logs/p2-gvhmr-weights-hf.sh                                    # mirror download + sha256sum -c
python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s  # demo, exit 0

# Step 3: GVHMR on REHAB24-6
bash logs/p2-gvhmr-rehab.sh            # first attempt, whole videos: PM_101 c17/c18 done, PM_040 c17 OOM-killed (§6)
bash logs/p2-memtest.sh clip300.mp4 / clip1000.mp4                  # peak memory, capped scope
python <make runs/t2-chunks/jobs.txt>  # 12 recordings x 2 cameras, repetition spans, 72 chunks <= 1000 frames
xargs -P 2 -L 1 bash logs/p2-chunk-job.sh < jobs.txt               # per chunk: ffmpeg trim (frame-exact, crf 12),
                                       #   $T/gvhmr_run.py (demo pipeline, -s, no rendering), $T/gvhmr_export_joints.py
python $T/rehab_eval.py --data data/rehab24-6 --chunks runs/t2-chunks/out --calib runs/t2-gvhmr --rec <12> --out reps.json
python $T/rehab_sync_check.py --data ... --chunks ... --rec <12> --trace PM_118 17 11

# Step 4: two cameras
python $T/rehab_to_easymocap.py --rec <rec> --out data/rehab24-6-emc/<rec>   # yml from calib + frames of the span
python $T/run_mv1p.py --data data/rehab24-6-emc/<rec> --out runs/t2-emc/<rec> --views 17 18 --device cuda --fps 30 \
    --no-vis --triangulate-mode naive
python $T/rehab_twocam.py --data ... --chunks ... --calib ... --rec <12> --emc runs/t2-emc --emc-data data/rehab24-6-emc
```

```bash
# True focal length (10-28): same chunks, preprocessing hard-linked from the default run, prediction only
bash logs/p2-pipeline/truef.sh          # xargs -P 6 logs/p2-pipeline/truef-job.sh < jobs.txt; per chunk:
                                        #   $T/gvhmr_run.py --video clips/<chunk>.mp4 --output_root out-truef --f_px 2181.7|2176.09
                                        #   $T/gvhmr_export_joints.py; then rehab_eval.py and rehab_twocam.py on out-truef
python $T/rehab_report.py --default runs/t2-results --truef runs/t2-results-truef --seg data/rehab24-6/Segmentation.csv
```

Steps S1–S5 after 21:23 on 10-27 ran unattended in `logs/p2-pipeline/pipeline.sh` (transient unit `p2-t2-pipeline`).
After the reboot, the rest ran in the enabled user units `p2-t2-resume` (`resume.sh`) and `p2-t2-truef` (`truef.sh`),
both disabled again once finished.

## 4. Numbers

Conditions for every table: REHAB24-6, the 12 recordings in §2, frames inside annotated repetitions only, clean
repetitions only (no mocap error, `extra_person ≤ 1` in the camera used), 30 fps, GVHMR `ee960bb` (static camera,
mirror weights), RTX A6000. Errors in mm, frame-weighted means over repetitions. Tables from
`reconstruction/tools/rehab_report.py` (output in `runs/t2-results-truef/report.md`).

Metrics, all on the **12 limb joints** (knees, ankles, toes, shoulders, elbows, wrists; mapping below):
- **PA**: PA-MPJPE, per-frame similarity alignment (rotation, translation, scale).
- **root**: MPJPE after mid-hip root alignment. GVHMR's camera-frame joints are rotated into the mocap world with the
  camera rotation recovered by DLT, so no rotation or scale is fitted. Both roots are the midpoint of their own
  skeleton's two hip joints. Motive's hips sit ~9–13 cm higher than SMPL's (§5), so this metric carries that offset.
- **cent**: like root, but aligning the centroid of the 12 joints, which doesn't depend on either pelvis definition.
- **abs**: no alignment at all: GVHMR's camera-frame joints placed in the world with the recovered camera (R, centre).
  This includes GVHMR's depth error, which depends on the focal length it assumes.

**Focal length.** By default GVHMR assumes f = image diagonal = 2202.9 px. The recovered cameras have 2181.7 px
(camera 17) and fx 2170.4 / fy 2181.8 px (camera 18; GVHMR models square pixels, so the rerun used the geometric mean,
2176.1 px). All 72 chunks were rerun with the true focal: `gvhmr_run.py --f_px`, which converts to GVHMR's
full-frame `f_mm` = 42.850 / 42.740 with the formula in `hmr_cam.py`. The rerun reused each chunk's preprocessing,
which doesn't depend on K in static-camera mode. Result: PA, root and cent change by ≤ 0.6 mm in every cell; only
abs changes (by up to 40 mm). So PA, root and cent are fair for one-vs-two-camera comparisons with either focal, and
abs is reported with the true focal, with the default focal in brackets.

**Main table: GVHMR, one camera at a time, per exercise** (true focal; default-focal abs in brackets)

| Exercise | Camera | Reps | Frames | PA | root | cent | abs |
|---|---|---|---|---|---|---|---|
| Ex1 arm abduction | 17 | 28 | 3223 | 32.4 | 83.0 | 64.5 | 313.7 (351.8) |
| Ex1 arm abduction | 18 | 31 | 3591 | 72.8 | 103.5 | 90.9 | 303.7 (343.6) |
| Ex2 arm VW | 17 | 36 | 4874 | 39.6 | 102.5 | 66.1 | 115.1 (138.7) |
| Ex2 arm VW | 18 | 41 | 6133 | 62.1 | 125.0 | 90.2 | 142.6 (167.0) |
| Ex3 push-ups | 17 | 25 | 3010 | 53.4 | 97.4 | 65.8 | 159.1 (171.8) |
| Ex3 push-ups | 18 | 27 | 3241 | 63.0 | 156.6 | 125.0 | 164.2 (182.9) |
| Ex4 leg abduction | 17 | 30 | 2881 | 55.4 | 120.4 | 66.9 | 123.6 (115.6) |
| Ex4 leg abduction | 18 | 30 | 2881 | 79.3 | 143.4 | 98.2 | 136.7 (142.4) |
| Ex5 lunge | 17 | 31 | 2257 | 55.4 | 85.2 | 71.8 | 293.4 (321.1) |
| Ex5 lunge | 18 | 40 | 3084 | 100.6 | 131.6 | 120.9 | 239.8 (257.0) |
| Ex6 squat | 17 | 35 | 2777 | 50.0 | 103.7 | 70.0 | 151.9 (128.0) |
| Ex6 squat | 18 | 45 | 3530 | 52.2 | 109.5 | 78.3 | 104.1 (112.1) |
| **All** | **17** | **185** | **19022** | **46.3** | **99.2** | **67.1** | **183.5 (196.6)** |
| **All** | **18** | **214** | **22460** | **69.9** | **126.9** | **98.7** | **178.0 (198.1)** |

Without the six outlier chunks (§5; all camera 18): camera 18, 177 reps, 17,628 frames, PA 63.4, root 124.4,
cent 95.4, abs 178.6. Camera 17 has no outlier chunk.

**Five worst metric joints (PA, mm, clean, true focal)**
- Camera 17: L_wrist 68.2, R_wrist 60.8, L_elbow 48.9, R_shoulder 47.4, R_ankle 47.3
- Camera 18: R_wrist 131.7, L_wrist 107.9, R_elbow 76.9, L_elbow 67.2, R_ankle 60.8

The joints left out of the metric are worse for a known reason. PA errors after fitting on the 12 metric joints:
L_hip 84.1/90.9, R_hip 76.4/82.4, neck 78.6/97.8, head 70.7/87.4 (camera 17/18). These four are placed differently
in Motive and SMPL.

Per joint, the per-repetition mean residual is 85–95% of the PA error (e.g. camera 18 R_wrist 117.6 of 131.7). So
within a repetition the error is mostly a steady offset, not jitter. That offset mixes joint-definition differences
with consistently wrong poses (for example a hidden arm placed wrongly for the whole repetition), and I can't
separate the two here.

**By view of the camera used** (true focal, clean)

| Camera | View | Reps | Frames | PA | root | cent | abs |
|---|---|---|---|---|---|---|---|
| 17 | front | 86 | 8913 | 46.7 | 106.3 | 75.1 | 217.5 |
| 17 | half-profile | 74 | 7099 | 42.8 | 91.1 | 57.8 | 151.2 |
| 17 | profile (push-ups) | 25 | 3010 | 53.4 | 97.4 | 65.8 | 159.1 |
| 18 | front (push-ups) | 27 | 3241 | 63.0 | 156.6 | 125.0 | 164.2 |
| 18 | half-profile | 85 | 8115 | 45.2 | 106.3 | 74.9 | 195.4 |
| 18 | profile | 102 | 11104 | 89.9 | 133.4 | 108.4 | 169.3 |

**By lights** (true focal, clean). Lights-off recordings are only from subjects 1–3, so lighting is confounded with
the subject; the per-recording table below separates them.

| Camera | Lights | Reps | Frames | PA | root | cent | abs |
|---|---|---|---|---|---|---|---|
| 17 | off | 75 | 8428 | 50.6 | 94.8 | 68.6 | 128.6 |
| 17 | on | 110 | 10594 | 42.9 | 102.7 | 66.0 | 227.2 |
| 18 | off | 97 | 10723 | 82.2 | 131.3 | 112.5 | 152.7 |
| 18 | on | 117 | 11737 | 58.6 | 123.0 | 86.0 | 201.2 |

**Per recording and camera** (true focal, clean)

| Ex | Recording | Subject | Lights | Camera | Reps | Frames | PA | root | cent | abs |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PM_023 | 3 | off | 17 | 8 | 1044 | 40.0 | 104.2 | 64.4 | 70.0 |
| 1 | PM_023 | 3 | off | 18 | 10 | 1281 | 109.1 | 139.2 | 116.5 | 175.6 |
| 1 | PM_032 | 4 | on | 17 | 20 | 2179 | 28.8 | 72.9 | 64.6 | 430.5 |
| 1 | PM_032 | 4 | on | 18 | 21 | 2310 | 52.6 | 83.6 | 76.7 | 374.7 |
| 2 | PM_014 | 2 | off | 17 | 18 | 2591 | 43.1 | 89.3 | 67.7 | 96.2 |
| 2 | PM_014 | 2 | off | 18 | 21 | 3209 | 59.8 | 112.7 | 95.3 | 134.4 |
| 2 | PM_040 | 5 | on | 17 | 18 | 2283 | 35.6 | 117.4 | 64.3 | 136.5 |
| 2 | PM_040 | 5 | on | 18 | 20 | 2924 | 64.5 | 138.4 | 84.5 | 151.5 |
| 3 | PM_010 | 1 | off | 17 | 14 | 1598 | 56.5 | 96.3 | 70.8 | 197.1 |
| 3 | PM_010 | 1 | off | 18 | 14 | 1593 | 70.5 | 162.6 | 132.6 | 190.7 |
| 3 | PM_030 | 3 | on | 17 | 11 | 1412 | 49.9 | 98.7 | 60.1 | 116.1 |
| 3 | PM_030 | 3 | on | 18 | 13 | 1648 | 55.7 | 150.9 | 117.5 | 138.6 |
| 4 | PM_018 | 2 | off | 17 | 10 | 942 | 51.2 | 94.8 | 58.7 | 101.1 |
| 4 | PM_018 | 2 | off | 18 | 10 | 942 | 101.8 | 134.5 | 116.5 | 158.7 |
| 4 | PM_124 | 7 | on | 17 | 20 | 1939 | 57.4 | 132.9 | 70.8 | 134.5 |
| 4 | PM_124 | 7 | on | 18 | 20 | 1939 | 68.4 | 147.7 | 89.3 | 126.0 |
| 5 | PM_021 | 2 | off | 17 | 11 | 1092 | 62.5 | 98.9 | 76.2 | 153.3 |
| 5 | PM_021 | 2 | off | 18 | 20 | 1919 | 128.5 | 158.3 | 144.5 | 167.4 |
| 5 | PM_037 | 4 | on | 17 | 20 | 1165 | 48.7 | 72.3 | 67.7 | 424.6 |
| 5 | PM_037 | 4 | on | 18 | 20 | 1165 | 54.5 | 87.7 | 82.1 | 358.9 |
| 6 | PM_022 | 2 | off | 17 | 14 | 1161 | 57.4 | 92.9 | 72.1 | 158.5 |
| 6 | PM_022 | 2 | off | 18 | 22 | 1779 | 53.2 | 100.1 | 86.2 | 115.8 |
| 6 | PM_118 | 9 | on | 17 | 21 | 1616 | 44.6 | 111.5 | 68.5 | 147.3 |
| 6 | PM_118 | 9 | on | 18 | 23 | 1751 | 51.2 | 119.0 | 70.4 | 92.3 |

The large abs values (360–430 mm) come from PM_032 and PM_037 (subject 4), so they likely reflect GVHMR's body size
or depth for that one subject, not lighting.

**One camera vs. two cameras, same frames** (repetitions clean in both cameras, 183 reps, 18,584 frames; 10 joints,
without toes because COCO has none). Two-camera methods: ViTPose 2D (GVHMR's own, from both cameras) triangulated
linearly through the recovered calibration; and EasyMocap `e6006fd`, YOLOv5m + HRNet-W48, naive 2-view triangulation.

| Method | abs | PA | root | cent |
|---|---|---|---|---|
| Two cameras, ViTPose + DLT calibration | **47.5** | **36.9** | 64.9 | **43.5** |
| Two cameras, EasyMocap HRNet | 57.2 | 43.1 | 74.5 | 50.0 |
| GVHMR camera 17 alone | 182.4 (193.6) | 46.8 | 97.0 | 65.6 |
| GVHMR camera 18 alone | 182.8 (204.6) | 69.6 | 120.5 | 93.6 |

Excluding repetitions that overlap an outlier chunk (151 reps, 14,987 frames): ViTPose two-camera abs 46.0 / PA 35.8,
EasyMocap 56.6 / 42.3, GVHMR camera 17 PA 47.1, camera 18 PA 63.1.

The two-camera numbers need no alignment at all, yet are as good as or better than the single-camera numbers after a
per-frame similarity alignment. For two-camera triangulation, "root" is worse than "abs" (64.9 vs. 47.5) because
aligning COCO hips to Motive's higher hips moves every joint by the definition offset; "cent" avoids that.

Per exercise (true focal, clean, same frames):

| Ex | Reps | Frames | 2-cam ViTPose abs | 2-cam ViTPose PA | EasyMocap abs | EasyMocap PA | GVHMR c17 PA | GVHMR c18 PA | GVHMR c17 abs | GVHMR c18 abs |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 28 | 3223 | 47.9 | 36.5 | 50.9 | 37.3 | 33.9 | 78.9 | 309.7 | 306.5 |
| 2 | 36 | 4795 | 46.0 | 32.7 | 50.9 | 36.9 | 43.7 | 60.6 | 109.3 | 143.3 |
| 3 | 23 | 2753 | 49.2 | 43.1 | 62.9 | 54.9 | 49.8 | 66.5 | 158.3 | 172.6 |
| 4 | 30 | 2877 | 46.1 | 35.5 | 50.4 | 39.7 | 55.3 | 73.0 | 119.5 | 127.2 |
| 5 | 31 | 2159 | 53.2 | 45.9 | 65.8 | 59.8 | 49.8 | 97.1 | 302.2 | 284.1 |
| 6 | 35 | 2777 | 44.7 | 33.0 | 70.2 | 39.3 | 53.0 | 53.0 | 157.0 | 96.5 |

**Joint mapping** (SMPL-24 from GVHMR's own SMPL-X → SMPL → J_regressor chain; REHAB24-6 26-joint Motive skeleton;
COCO-17 for the 2D/two-camera parts)

| Joint | SMPL-24 | REHAB24-6 | COCO-17 | Used for |
|---|---|---|---|---|
| L/R knee | 4 / 5 | LeftLeg 17 / RightLeg 22 | 13 / 14 | metric |
| L/R ankle | 7 / 8 | LeftFoot 18 / RightFoot 23 | 15 / 16 | metric |
| L/R toe | 10 / 11 | LeftToeBase 19 / RightToeBase 24 | – | metric (single camera only) |
| L/R shoulder | 16 / 17 | LeftArm 7 / RightArm 12 | 5 / 6 | metric |
| L/R elbow | 18 / 19 | LeftForeArm 8 / RightForeArm 13 | 7 / 8 | metric |
| L/R wrist | 20 / 21 | LeftHand 9 / RightHand 14 | 9 / 10 | metric |
| L/R hip | 1 / 2 | LeftUpLeg 16 / RightUpLeg 21 | 11 / 12 | root only; per-joint table |
| neck, head | 12, 15 | Neck 3, Head 4 | – | per-joint table only |
| not mapped | pelvis, spines, collars, hands | Hips, Spine, Spine1, Head_end, *Shoulder (clavicles), *_end | eyes, ears, nose | – |

**Run cost** (GPU 1): GVHMR, 72 chunks, 63,594 frames, peak RSS 6.87–7.06 GB per chunk, about 3.3 min per 900-frame
chunk with 2 in parallel. True-focal prediction only: 1.67–1.71 GB RSS and ≤ 1.84 GB GPU per job, 6 in parallel,
72 chunks in 6 min. EasyMocap: 3.49–3.61 GB RSS, 0.10–0.17 s per frame pair, 19/19 joints on every frame, median
reprojection 3.6–7.7 px against its own detections (consistency, not accuracy).

## 5. Checks

- **Units and axes, one frame then all:** mocap bone lengths are constant (fitted skeleton): thigh 0.454 m, height
  1.80 m on PM_000, so the files are in metres despite the Zenodo text's "virtual centimetres". With the recovered
  cameras, the mocap mid-hip in camera coordinates and GVHMR's agree in sign and size. PM_032 camera 17, frame 500:
  mocap (−0.49, 0.21, 4.74) m vs. GVHMR (−0.53, 0.27, 5.32) m, a depth ratio of 1.06–1.15 on 8 sampled frames in each
  camera, which is the abs error.
- **Calibration:** one DLT per camera, fitted on 3 random frames of each of the 1072 repetitions, reproduces all 4.85 M
  projected joints in all 65 recordings to ≤ 2.2e-12 px, and the EasyMocap yml files written from it reproduce the
  dataset 2D to ≤ 0.001 px. So the dataset's 2D joints are exact projections through one fixed camera per view. That
  camera is the authors' own hand-fitted model, not a lens calibration; I have no independent check of how well it
  matches the real lens.
- **Joint mapping:** checked on PM_101 before reporting. Motive's LeftUpLeg/RightUpLeg sit level with Hips (y 1.00 m)
  and give a 0.52 m thigh, against SMPL's 0.39 m. After PA alignment, GVHMR's hips sat 93 mm below Motive's, nearly
  constant. Hence the 12-joint metric set and the "cent" metric. "Left" is the subject's left: on a front view the
  Left* joints project on the image's right side (overlay of PM_000, frame 600).
- **Time alignment:** video frame i = mocap frame i. The videos have the same frame count as the mocap or one fewer
  (checked by decoding). An independent YOLOv8-pose check by Tan's other session found sync within ±1 frame. My check
  (`rehab_sync_check.py`, mean-removed 2D trajectories, 24 recording-camera pairs) gives lag 0 for 17 pairs and ±1
  for 7, on curves that vary by only 0.1–3 px over ±2 frames, so ±1 frame is the resolution of this test. Traces: in
  PM_032 camera 17, the right wrist peaks at frames 313–319 in both ViTPose and the projected mocap; in PM_118
  camera 17 (squat), the left hip bottoms out at frame 248 in both.
- **Right person tracked:** the projected mocap mid-hip lies inside GVHMR's tracked box in 100% of frames for 71/72
  chunks. The exception, PM_010 camera 17 frames 854–1159 (box stuck while the subject moves), falls between two
  repetitions and isn't scored.
- **Outlier chunks (Tan's question):** six chunks, all camera 18, have a fast-joint 2D median > 40 px: PM_023 frames
  35–804 (169 px), PM_032 90–1016 (111 px), PM_040 140–969 (71 px), PM_014 200–1183 (69 px), PM_124 200–1139 (42 px),
  PM_018 680–1280 (41 px). Not tracking errors: the tracked box overlaps the projected-mocap box in every frame (IoU
  median 0.54–0.74, never < 0.3), no frame has an all-joint median > 50 px, and the error sits on one arm (e.g.
  R_elbow 133–156 px, R_wrist 117–171 px in PM_040/PM_014). Frames viewed: PM_023 f57, PM_014 f297, PM_040 f554,
  PM_032 f545. The subject stands in profile to camera 18 with the far arm hidden behind the body, and ViTPose
  confidently places it on the near arm. No retracking was needed. Numbers are given with and without these chunks
  (§4); the rule uses only 2D detections against the dataset's 2D, no 3D errors.
- **Metric code:** `rehab_eval.py --selftest` (exact PA under a scaled, rotated, moving body; 10% scale error caught by
  the root metric; a 9 cm root offset spreads exactly 90 mm; a 2-frame lag recovered) and `rehab_twocam.py --selftest`
  (DLT and triangulation exact on synthetic cameras) both pass. The EasyMocap scoring path reproduced a standalone
  50-frame check exactly (55.0 mm, same per joint).
- **Focal rerun:** every true-focal chunk's saved K has f 2181.7 (36 camera-17 chunks) or 2176.09 (36 camera-18
  chunks). The two-camera numbers are identical in both runs (asserted in `rehab_report.py`).
- **GPU placement:** after the fix, every GVHMR and EasyMocap job was checked on GPU-dc002436 (pipeline `gpu.log`: 0
  P2 processes ever seen on GPU 0 after 21:01 on 10-27).
- **Frames looked at:** overlays of PM_000 (dataset 2D, both cameras), the GVHMR tennis demo, PM_032 camera 18 frame
  545 (hidden arm), and the four outlier frames above. They show faces and are kept out of git
  (`/tmp/claude-1017/-net-home-tle-fall2026/c4d89c3e-a120-49f7-bc52-17bee22f66ee/scratchpad/`).

## 6. Failed or skipped

- **Out-of-memory kill that stopped the session (19:34:47).** My first approach ran GVHMR's demo on whole videos. On
  PM_040-Camera17 (3471 frames, 1920×1080) the kernel OOM killer hit the job, and systemd then stopped the whole tmux
  scope, including Claude. The machine's 251 GB is shared. What I know about the cause: GVHMR's ViTPose and feature
  extractors load the whole video into RAM (`read_video_np` at half resolution, then a blurred copy and float crops,
  a few MB per frame), so memory grows with video length. Measured afterwards in capped scopes: peak RSS 7.06 GB at
  300 frames and 7.05 GB at 1000 frames (the fixed model-loading cost dominates), and about 2.2 MB more per frame at the
  end of preprocessing. Extrapolated, that is about 10 GB for 3471 frames, which alone does not explain exhausting
  251 GB. The kernel's OOM report is not readable without sudo, so I can't say what other processes held at 19:34.
  **Fix:** run GVHMR only on the repetition spans, cut into chunks of at most 1000 frames (frame indices kept in the
  file names, decoded frame count checked against the expected count), and run every heavy job in its own
  `systemd-run --user --scope -p MemoryMax=32G -p MemorySwapMax=0` scope, with `/usr/bin/time -v` peak RSS logged per
  job. I tested the cap first: a 3 GiB allocation under a 2 G cap was killed alone (rc 143) and the shell survived.
  Peak RSS of the 72 GVHMR chunks: 6.87–7.06 GB; true-focal prediction-only jobs: 1.67–1.71 GB. No job hit its cap.
- **My own memory bug, caught by the cap.** The first DLT used a full SVD (`U` of 167k × 167k, about 220 GB). Its 16 G
  scope killed it after 11 s with nothing else affected. Fixed with a thin SVD.
- **GVHMR chunks on the wrong GPU (found by Tan).** The 20 chunks that finished before 21:01 ran on physical GPU 0
  (`GPU-a1eda44a`) next to P3's training, which slowed from 2.85 to 1.46 it/s, while GPU 1 sat idle. Cause:
  ultralytics `select_device()` (`ultralytics/utils/torch_utils.py`, line ~127) sets `CUDA_VISIBLE_DEVICES="0"` when
  GVHMR's YOLO tracker starts. `demo.py`'s `__main__` queries CUDA first, which fixes the device before that; my
  `gvhmr_run.py` skipped `__main__`, so CUDA started after the override. Fix: `torch.cuda.init()` before preprocessing.
  The next job (PID 3401293) was on `GPU-dc002436` (GPU 1), and so were all later ones (pipeline `gpu.log`: 0
  GVHMR/EasyMocap processes on GPU 0 after the fix; the first watcher also logged P3's training, which belongs on
  GPU 0, so those lines are not P2's). Only speed was affected; the outputs are the same computation. The whole-video PM_101 runs and the
  memory tests used `demo.py` and stayed on GPU 1. EasyMocap (YOLOv5 has the same override) calls CUDA at startup in
  `run_mv1p.py` and was checked on GPU 1 with detection running.
- **Server reboot (2026-10-28 08:12).** Every process stopped; the last pipeline log line was 21:51:27 on 10-27. The
  transient unit `p2-t2-pipeline` (started with `systemd-run --user`) did not survive. Done at that point: 52/72 GVHMR
  chunks and 5/12 EasyMocap recordings (PM_118 was mid-run; its partial output was deleted and redone). Fix:
  `loginctl enable-linger tle` (worked without sudo, `Linger=yes`), and the rest moved into an idempotent script run by
  an **enabled** user unit (`~/.config/systemd/user/p2-t2-resume.service`, `WantedBy=default.target`), which skips
  finished work and removes partial outputs, so another reboot only costs the jobs in flight. Restarted 15:07:51.
- **GVHMR weights:** official Google Drive refused all five files (quota); unofficial HF mirror used with Tan's
  agreement (§2).
- **chumpy build** failed under build isolation (as in P0); fixed by installing it from git with
  `--no-build-isolation`. One fix, no other install problems; WHAM was not needed.
- **EasyMocap with two cameras:** its default `iterative` triangulation requires at least 3 views per joint and
  returned all zeros. Added `--triangulate-mode naive` (plain linear triangulation, `min_view=2`) to `run_mv1p.py`.
  With two views no outlier can be voted out, so one wrong 2D detection goes straight into the 3D point.
- **Time offset per chunk was not usable** as an estimate: in arm exercises the fastest joint is often the occluded
  arm, and one chunk "chose" +3 at 110 px for all offsets. The offset is fixed at 0 (§5 has the evidence).
- **My first selection** (one recording per exercise, whole videos) was replaced after the dataset notes: evaluation is
  per repetition, main numbers on clean repetitions only, two recordings per exercise to cover lights on/off. The
  whole-video PM_101 runs are not in the tables.
- **Two of my commands killed themselves:** `pkill -f` patterns matched the invoking shell. No other process was
  affected; I then killed my leftover EasyMocap job by PID and deleted its partial output in `runs/t2-emc/PM_040`.
- **Step 5 (YouTube, T3) skipped:** it needs clip URLs from Tan, or Tan's OK to pick clips.
- **Rendering:** chunk runs skip GVHMR's rendering (about 40% of the run time). Only the PM_101 whole-video runs have
  overlay videos (`runs/t2-gvhmr/PM_101-*`).

## 7. Not tested

- Other GVHMR settings: DPVO / moving-camera mode, the SMPL-X hands and face outputs, the non-square pixels of camera 18
  (GVHMR takes one focal length; the rerun used the geometric mean of fx and fy).
- The other 53 recordings and subjects 6, 8 and 10. Lighting can't be separated from the subject in this dataset, since
  lights-off recordings exist only for subjects 1–3.
- Chunk-boundary effects: GVHMR is a temporal model, and each chunk starts cold. Not measured.
- The 41-marker data (`*_markers.zip`, not downloaded).
- WHAM (not needed: GVHMR installed with one fix).
- Panoptic HD camera with GVHMR (in the T2 plan of EXPLORATION.md, not in this prompt).

## 8. Next step

The main single-camera failure is a far-side arm hidden in profile, and the second camera sees exactly that arm. So
the next run I would do is a multi-view body fit on the same repetitions: EasyMocap `mv1p` with SMPL on the two-camera
HRNet output (already computed in `runs/t2-emc/`), scored with the same 12 joints. That tells us how much of the
two-camera gain survives into a body model, which is what telepresence needs. After that: GVHMR on the remaining 53
recordings for per-subject numbers, and T3 once Tan sends 2–3 squat clips (or says I may pick them).
