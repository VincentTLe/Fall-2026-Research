# T5: does GVHMR change "2D beats monocular 3D" for rehabilitation assessment? (REHAB26-ViewAngles), 2026-10-28

## 1. Result

**Reproduction.** I reimplemented the separability metric Ψ of Jánošová et al. (arXiv 2609.35726v2, Sec. 4.1) from the
text and ran it on the dataset's own MediaPipe skeletons. It reproduces their Table 3 MPP 2D/3D rows with deviation. With
the settings fixed before any comparison, 80 of 90 cells are within 20% of the paper (median grade B). The mean absolute
deviation is 0.055 Ψ and the worst cell is squat 3D at +90°: 0.36 against 0.21. Their conclusion that 2D beats MediaPipe
3D holds for the squat in all 64 metric variants consistent with the paper's wording, but only in 48/64 for arm abduction
and 60/64 for shoulder extension. For those two exercises the 2D–3D gap is about as large as the effect of choices the
paper does not state.

**GVHMR** (unofficial weight mirror, as P2) was scored with the same metric on the same 255 sequences (all 15 Table 3
viewpoints, all 5 persons). It is better than both: Ψ 0.611 / 0.589 / 0.551 against MPP 2D 0.493 / 0.499 / 0.486 for
Ex1 / Ex2 / Ex3. That is +0.12 / +0.09 / +0.07, with 95% person-bootstrap CIs excluding 0, and it wins 38 of 45
viewpoint cells against MPP 2D and 44 of 45 against MPP 3D. On the arm exercises, part of the gain comes from GVHMR's
better 2D detector (ViTPose 2D: +0.04 / +0.05 over MPP 2D), and the 3D step adds +0.08 / +0.04 on top. So with this
modern 3D method, the paper's conclusion does not carry over: monocular 3D is not worse than 2D, it is better.

## 2. Setup

| Field | Value |
|---|---|
| Machine | `lambdavector2`, Ubuntu 24.04.5 LTS, kernel 6.8.0-147; 2× RTX A6000 49140 MiB, driver 595.99.02 (CUDA 13.2). **All T5 GPU work on physical GPU 1** (`GPU-dc002436`), started 16:15 after P2's unit finished; 0 WRONG-GPU events from the watchdog. GPU 0 never used (P3, and another user's job). Log `logs/2026-10-28-t5-env.log` |
| Metric code | `reconstruction/tools/rehab_psi.py` (one implementation for every number), Python 3.10.22, numpy 1.23.5, scipy 1.15.3, pandas 2.3.3 (in `envs/gvhmr`, nothing installed) |
| GVHMR | `external/GVHMR` at `ee960bb6e2ea2d381aa97f08e9b71ef320b624b1`, env `envs/gvhmr` (P2's; torch 2.3.0+cu121). **Weights: unofficial Hugging Face mirror `camenduru/GVHMR`, the files P2 downloaded** (official Google Drive refused; SHA-256 in `DATA_LOG.md`). Run via `gvhmr_run.py` (copied unchanged from P2: static camera `-s`, GVHMR's default focal length, `torch.cuda.init()` before ultralytics) |
| Paper | arXiv 2609.35726v2 (HTML, fetched 2026-10-28). Section 4.1 and Tables 1 and 3 read from the page text. Table 3 MPP rows frozen in `runs/t5/spec/table3_targets.csv`, SHA-256 `41e965ce…a738e` (checked by the grader), together with the grading bands and pass rule in `runs/t5/spec/SPEC.md` (`a153d72d…5429`), both written before any scoring code |
| Data | REHAB26-ViewAngles, Zenodo 23018128, CC BY-NC 4.0. All 6 files match the Zenodo MD5; SHA-256 in `DATA_LOG.md`. 45 recordings (3 exercises × 5 persons) × 3 cameras, 1080×1920 portrait at 30 fps, 11 orientations each, 1665 repetitions (832 correct, 833 incorrect; 13 of 165 orientation blocks have 6–7 correct or 6 incorrect instead of 5+5). 90 MediaPipe npy (T, 33, 4): 2D = normalised image x, y, z and visibility; 3D = world landmarks in m |
| Exercise IDs | paper Ex1 = dataset 1 (arm abduction), paper Ex2 = dataset **7** (shoulder extension), paper Ex3 = dataset **6** (squat) |
| Viewing angle | from `Segmentation.txt`: cam1 −o, cam2 45 − o, cam3 90 − o (o = orientation label). Table 3 0° has 15 sequences (one per camera per person); the other columns have 5 |
| Repo | branch `server/t5-20261028`, from `origin/main` at `9733991` |

## 3. Commands

`T` = `~/fall2026/wt-t5/reconstruction/tools`, `PY` = `~/fall2026/envs/gvhmr/bin/python -I`. Logs are in
`~/fall2026/logs/2026-10-28-t5-*.log` and `logs/t5-gvhmr/`.

```bash
curl -s https://zenodo.org/api/records/23018128                       # file list + MD5
curl -L --fail -o raw/<f> https://zenodo.org/api/records/23018128/files/<f>/content; md5sum; sha256sum
systemd-run --user --unit=t5-videos-dl ... curl ... videos.zip        # 8.4 GB, 2.0 TB free
unzip -Z1 <zip> | grep -E '(^/|\.\./)'                                # no unsafe paths; then unzip
PY $T/rehab26_inspect.py data/rehab26-viewangles                      # shapes vs segmentation, labels
ffprobe -count_packets ... (45 videos);  PY $T/rehab26_frame_sync.py ... # video longer than skeleton: where?
PY $T/rehab_psi.py --selftest
PY $T/rehab26_psi_run.py --data D --seeds 0-9 --procs 16 --out runs/t5/psi/mpp_default.csv --cfg aspect=1080,1920
PY $T/rehab26_grade.py runs/t5/psi/mpp_default.csv runs/t5/spec --md runs/t5/psi/mpp_default.md
$T/rehab26_multiverse.sh runs/t5/multiverse       # enabled unit t5-multiverse; 128 configs x 5 seeds
PY $T/rehab26_psi_checks.py D                     # feature ranges, fault injection
PY $T/rehab26_psi_run.py ... --shuffle-labels     # null model per method
PY $T/rehab26_psi_run.py ... --cfg smooth=9 / smooth=15
PY $T/rehab26_gvhmr_plan.py D runs/t5/gvhmr/jobs.txt   # 744 chunks <= 1000 frames, whole reps, Table 3 viewpoints first
# enabled unit t5-gvhmr: rehab26_gvhmr_runner.sh -> rehab26_gvhmr_job.sh per chunk:
#   ffmpeg trim (frame-exact, crf 12) + frame count check; systemd-run --user --scope -p MemoryMax=32G -p MemorySwapMax=0
#   env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python gvhmr_run.py && python gvhmr_export_mp.py
PY $T/rehab26_gvhmr_collect.py D runs/t5/gvhmr/out runs/t5/gvhmr/jobs.txt runs/t5/gvhmr/mp33
PY $T/rehab26_psi_run.py --source npy:runs/t5/gvhmr/mp33:_incam --dims 3d --skip-incomplete ...   (also _glob, _vitpose2d 2d)
PY $T/rehab26_compare.py --method MPP2D=..:2d MPP3D=..:3d GVHMR_incam=.. GVHMR_glob=.. --md runs/t5/psi/compare_main.md
```

## 4. Numbers

### 4.1 Table 3 reproduction (MPP 2D and 3D, all three exercises, 15 viewpoints)

These are the defaults fixed before comparing. The paper names only F5 as the DTW feature for Ex1; I chose F1 (sagittal
arm–torso angle) for Ex2 and F15 for Ex3. τ is the t-margin on the mean, t₀.₉₇₅,ₙ₋₁·s/√n, floored at the first
quartile of τ over frames. Only the intended error's features are scored. 2D angles are in pixels (x·1080, y·1920). 3D
uses MediaPipe world landmarks rotated about the vertical so the hips lie on +x, with angles in the frontal (xy),
sagittal (yz) or transverse (xz) plane per feature. Viewpoint value = mean over sequences, averaged over 10 seeds of the
random reference (0–9). Grades: A ≤2%, B ≤20%, C ≤40%, D ≤60%, F >60% relative deviation.

| Ex | Row | −90 | −60 | −40 | −30 | −20 | −15 | −10 | 0 | +10 | +15 | +20 | +30 | +40 | +60 | +90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2D paper | 0.36 | 0.39 | 0.55 | 0.57 | 0.56 | 0.57 | 0.54 | 0.58 | 0.60 | 0.66 | 0.62 | 0.61 | 0.61 | 0.63 | 0.41 |
| 1 | 2D ours | 0.36 | 0.33 | 0.50 | 0.49 | 0.52 | 0.54 | 0.51 | 0.50 | 0.50 | 0.54 | 0.49 | 0.57 | 0.55 | 0.60 | 0.35 |
| 1 | grade | A | B | B | B | B | B | B | B | B | B | C | B | B | B | B |
| 1 | 3D paper | 0.31 | 0.44 | 0.49 | 0.46 | 0.51 | 0.57 | 0.50 | 0.47 | 0.54 | 0.55 | 0.57 | 0.59 | 0.47 | 0.51 | 0.42 |
| 1 | 3D ours | 0.22 | 0.38 | 0.43 | 0.46 | 0.46 | 0.47 | 0.48 | 0.47 | 0.52 | 0.55 | 0.52 | 0.56 | 0.50 | 0.51 | 0.36 |
| 1 | grade | C | B | B | A | B | B | B | A | B | A | B | B | B | A | B |
| 2 | 2D paper | 0.51 | 0.42 | 0.44 | 0.57 | 0.58 | 0.56 | 0.44 | 0.55 | 0.61 | 0.59 | 0.60 | 0.56 | 0.60 | 0.52 | 0.42 |
| 2 | 2D ours | 0.50 | 0.38 | 0.36 | 0.54 | 0.40 | 0.52 | 0.45 | 0.49 | 0.52 | 0.52 | 0.52 | 0.59 | 0.65 | 0.55 | 0.50 |
| 2 | grade | B | B | B | B | C | B | A | B | B | B | B | B | B | B | B |
| 2 | 3D paper | 0.49 | 0.36 | 0.40 | 0.55 | 0.46 | 0.60 | 0.45 | 0.52 | 0.55 | 0.61 | 0.61 | 0.65 | 0.60 | 0.59 | 0.47 |
| 2 | 3D ours | 0.43 | 0.35 | 0.30 | 0.42 | 0.42 | 0.35 | 0.36 | 0.39 | 0.46 | 0.49 | 0.54 | 0.53 | 0.60 | 0.57 | 0.46 |
| 2 | grade | B | B | C | C | B | D | B | C | B | B | B | B | A | B | B |
| 3 | 2D paper | 0.53 | 0.56 | 0.55 | 0.51 | 0.40 | 0.48 | 0.33 | 0.43 | 0.51 | 0.52 | 0.52 | 0.55 | 0.54 | 0.54 | 0.51 |
| 3 | 2D ours | 0.48 | 0.53 | 0.57 | 0.51 | 0.47 | 0.55 | 0.39 | 0.43 | 0.48 | 0.45 | 0.52 | 0.55 | 0.50 | 0.51 | 0.45 |
| 3 | grade | B | B | B | A | B | B | B | A | B | B | A | A | B | B | B |
| 3 | 3D paper | 0.28 | 0.36 | 0.47 | 0.45 | 0.42 | 0.48 | 0.49 | 0.45 | 0.48 | 0.45 | 0.46 | 0.48 | 0.37 | 0.42 | 0.21 |
| 3 | 3D ours | 0.43 | 0.47 | 0.49 | 0.48 | 0.45 | 0.39 | 0.45 | 0.46 | 0.55 | 0.48 | 0.47 | 0.51 | 0.44 | 0.43 | 0.36 |
| 3 | grade | D | C | B | B | B | B | B | A | B | B | A | B | B | B | F |

Grade vector: A 13, B 67, C 7, D 2, F 1; median B. Mean absolute deviation 0.055, mean signed −0.029 (ours slightly
lower), correlation over the 90 cells 0.69. The seed range of a cell's mean (min–max over 10 seeds) is usually
0.01–0.04, up to 0.12 (Ex1 3D −90). Per-seed ranges are in `runs/t5/psi/mpp_default.md`.

The 2D − 3D gap (mean over the 15 viewpoints), which is the question of this track:

| | Ex1 | Ex2 | Ex3 |
|---|---|---|---|
| paper | +0.057 | +0.004 | +0.081 |
| ours, default | +0.033 | +0.054 | +0.036 |

The rule I set in `SPEC.md` before coding (median grade B or better, and the sign of the 2D − 3D gap matching for all
three exercises) is met. So I went on to Step 3.

### 4.2 Multiverse over the paper-silent choices (MediaPipe, 5 seeds)

There are 2⁷ = 128 configs: τ as mean margin or prediction interval; Q1 floor or none; additional errors in or out;
F6 plane for Ex2 (xy or yz); Ex2 DTW feature F1 or F5; 2D aspect in pixels or normalised; 3D plane projection or full
3D angle. No setting was chosen by its agreement with the table. The τ = prediction-interval reading gives Ψ far below
the paper (0–1 cells A/B, mean |dev| 0.25). The paper calls τ "the margin of error" of a 95% CI, which is the mean
reading, so that half is reported separately. Summary in `runs/t5/multiverse/summary.csv`, log
`logs/2026-10-28-t5-multiverse-summary.log`.

| Set | n | A/B cells of 90 (min/median/max) | mean \|dev\| (min/median/max) | Ex1: 2D>3D | Ex2: 2D>3D | Ex3: 2D>3D | all three |
|---|---|---|---|---|---|---|---|
| τ = mean margin (paper's wording) | 64 | 62 / 71 / 81 | 0.050 / 0.065 / 0.080 | 48/64 | 60/64 | 64/64 | 48/64 |
| all | 128 | 0 / 32 / 81 | 0.050 / 0.148 / 0.266 | 96/128 | 124/128 | 128/128 | 96/128 |

In the τ = mean set, the 2D − 3D gap ranges over −0.009…+0.086 for Ex1, −0.014…+0.078 for Ex2 and +0.030…+0.080 for
Ex3. It moves mainly with the 2D aspect (pixels vs. normalised) and with the 3D plane projection. With normalised 2D
and plane projection, Ex1 and Ex2 come out almost even (−0.006, +0.006).

### 4.3 GVHMR vs. MediaPipe on the same sequences

GVHMR numbers use the **unofficial weight mirror** (§2). There are 255 sequences: every (person, orientation, camera)
at a Table 3 viewpoint, 85 per exercise, with the paired MediaPipe values from §4.1 (10 seeds, defaults). The remaining
240 sequences, at viewpoints outside Table 3, were still running when this was written. GVHMR `incam` = camera
coordinates. `glob` = GVHMR's gravity-aligned world turned to y down. Both get the same hip normalisation as MPP 3D.
CIs come from 2000 bootstrap resamples of the 5 persons. `runs/t5/psi/compare_main.md`, `compare_decomp.md`.

| Exercise | n seq | MPP 2D | MPP 3D | ViTPose 2D | GVHMR incam | GVHMR glob | GVHMR incam − MPP 2D | GVHMR incam − MPP 3D | GVHMR incam − ViTPose 2D |
|---|---|---|---|---|---|---|---|---|---|
| Ex1 arm abduction | 85 | 0.493 | 0.460 | 0.530 | 0.611 | 0.612 | +0.118 [+0.096, +0.140] | +0.151 [+0.143, +0.159] | +0.081 [+0.051, +0.107] |
| Ex2 shoulder extension | 85 | 0.499 | 0.439 | 0.548 | 0.589 | 0.587 | +0.091 [+0.026, +0.139] | +0.151 [+0.078, +0.203] | +0.041 [−0.045, +0.110] |
| Ex3 squat | 85 | 0.486 | 0.457 | – | 0.551 | 0.540 | +0.065 [+0.044, +0.079] | +0.093 [+0.058, +0.116] | – |

ViTPose 2D is GVHMR's own COCO-17 2D input, mapped to MediaPipe indices. It has no heels or toes, so it has no squat value.

Null model (labels shuffled within each block, same sequences) and Ψ above that floor:

| Exercise | floor MPP 2D | floor MPP 3D | floor GVHMR | Ψ − floor: MPP 2D | MPP 3D | ViTPose 2D | GVHMR |
|---|---|---|---|---|---|---|---|
| Ex1 | 0.107 | 0.100 | 0.091 | 0.386 | 0.360 | 0.423 | 0.520 |
| Ex2 | 0.109 | 0.099 | 0.100 | 0.390 | 0.340 | 0.446 | 0.489 |
| Ex3 | 0.104 | 0.113 | 0.126 | 0.382 | 0.344 | – | 0.425 |

Per viewpoint (5 sequences each, 15 at 0°), GVHMR incam beats MPP 2D in 38 of 45 cells and MPP 3D in 44 of 45. Its
lowest values include the profile views (Ex1 −90: 0.49, Ex3 +90: 0.44) and Ex2 −10 (0.47). The cells where MPP 2D is higher are Ex2 −30,
+30, +40, Ex3 −40, −15, +90 and one tie (Ex1 +60). The full per-viewpoint table is in `runs/t5/psi/compare_main.md`.
The seed range of a GVHMR viewpoint mean has a median of 0.024 and a maximum of 0.107.

Excerpt at 0°, ±90° (n = 15 at 0°, 5 at ±90°):

| Exercise | Viewpoint | MPP 2D | MPP 3D | GVHMR incam |
|---|---|---|---|---|
| Ex1 | −90 / 0 / +90 | 0.36 / 0.50 / 0.35 | 0.22 / 0.47 / 0.36 | 0.49 / 0.60 / 0.52 |
| Ex2 | −90 / 0 / +90 | 0.50 / 0.49 / 0.50 | 0.43 / 0.39 / 0.46 | 0.70 / 0.59 / 0.59 |
| Ex3 | −90 / 0 / +90 | 0.48 / 0.43 / 0.45 | 0.43 / 0.46 / 0.36 | 0.51 / 0.57 / 0.44 |

GVHMR cost: 383 chunks in about 5 h 15 min on GPU 1 (6 parallel, about 1.1 chunks/min), peak RSS 6.93–6.98 GB per
chunk, no failed chunk.

## 5. Checks

- **Known-answer self-test** (`rehab_psi.py --selftest`): angles, DTW on a time-stretched sine, Eq. 1 on hand-built
  values (0.375 exactly), and 3D normalisation (hip line on +x, lengths kept). A strongly perturbed rep gives Ψ 0.967;
  a clean held-out rep gives 0.113.
- **Fault injection on real data** (`logs/2026-10-28-t5-psi-checks.log`, 396 sequences): true labels give mean Ψ 0.500;
  shuffled labels 0.115; each error scored on another error's features 0.367. The shuffled value shows that **Ψ's real
  noise floor is about 0.1, not 0**: the model is built from the same correct reps it scores. The paper treats 0 as the
  baseline. "Wrong features" stays high because the features move together during an error.
- **Feature ranges** (frontal, correct reps): abduction F5 98–176° (5th–95th percentile), elbow F4 154–179°, squat knee
  F15 down to 115° (2D) and 92° (3D), trunk F6 ≈ 178°. These are anatomically plausible.
- **Data inspection**: all segmentation ranges lie inside the skeleton rows, and the files contain no NaNs. 29 of 45 videos have
  1–741 more frames than skeleton rows. Video motion energy cross-correlated with MediaPipe joint speed gives lag 0 (±1)
  through whole recordings (`logs/2026-10-28-t5-frame-sync.log`), so the extra frames are at the end and the frame
  numbers apply to both. For GVHMR (383 chunks): in 287 of the 303 chunks with a clear signal (corr > 0.5), the ViTPose
  wrist track and the MediaPipe wrist track agree best at a lag within ±1 frame. The median wrist disagreement is
  11.8 px. 13 chunks are above 30 px: 6 at the ±90° profile views (where one wrist is hidden; the worst is 203 px,
  ex1_Terka_cam1 o=90), 6 camera-1 views at −30° to −60°, and 1 at −15°.
- **GVHMR joints** (`runs/t5/gvhmr/mp33/checks.csv`): median upper arm 0.26 m, forearm 0.25 m, thigh 0.39 m,
  shin 0.41 m. Feet are 1.27 m below the shoulders in camera y (y down, as MediaPipe). Units: m in both 3D sources.
  Angles do not depend on scale.
- **Is the GVHMR gain an artefact?** It is not a floor effect: GVHMR's shuffled-label floor equals MediaPipe's for Ex1
  and Ex2. For the squat it is 0.02 higher, so the squat gain is about +0.04 above the floor rather than +0.065. It is
  not temporal smoothing: Savitzky–Golay on the MediaPipe joints (9 and 15 frames) raises MPP Ψ by only +0.006–0.010
  (`logs/2026-10-28-t5-psi-mpp-smooth.log`). It is not the output frame: incam and glob agree within 0.011.
- **Joint mapping GVHMR → MediaPipe** (`gvhmr_export_mp.py`): MP 11/12 shoulders ← SMPL J16/J17, 13/14 elbows ← J18/J19,
  15/16 wrists ← J20/J21, 23/24 hips ← J1/J2, 25/26 knees ← J4/J5, 27/28 ankles ← J7/J8. From SMPL vertices
  (`smplx` `vertex_ids['smplh']`): 0 nose ← v332, 29/30 heels ← v3387/v6787, 31/32 foot index ← v3216/v6617. SMPL's hip
  joints are not the same landmarks as MediaPipe's (P2 found the SMPL and mocap hips 9–13 cm apart; I did not measure
  the offset to MediaPipe). Angles using the
  hips (F1, F2, F6, F7, F9, F13, F15) are therefore measured on slightly different landmarks.

## 6. Failed or skipped

- **Step 4 (REHAB24-6 Ex1/Ex6, MPJPE vs Ψ): skipped.** REHAB24-6 labels repetitions only as correct or incorrect, with
  no error type, so the "relevant features" of Eq. 3 and Table 1 cannot be defined. Its mocap skeleton has no heel
  joint (F13/F14). P2's four Ex1/Ex6 recordings hold about 16 incorrect repetitions in total (PM_022 has 1). Any
  MPJPE–Ψ relation from that would mean nothing.
- **Independent verifier** (the reproduction protocol asks for a separate agent with injected faults): not done in this
  session. The fault injections above were chosen by me, the author.
- My own bugs, all caught before results: a relative path in the first job list (one canary chunk failed); ffmpeg
  reading the runner's job list from stdin (fixed with `-nostdin` and `< /dev/null` before any chunk ran); and a
  one-shot GPU gate that passed during a gap between P2's jobs. It was replaced by "no `p2-*` unit active and no
  foreign process for 5 min" before any GPU work started.
- The paper does not state the DTW library or variant, the τ formula, which values the first-quartile floor applies
  to, the joints of each feature, the DTW feature for Ex2/Ex3, or the number of seeds. My choices are in
  `rehab_psi.Config`, and the consequential ones were run as the multiverse in §4.2.

## 7. Not tested

- GVHMR on the 240 sequences at viewpoints outside Table 3 (still running in `t5-gvhmr`, about 6 more hours).
- GVHMR with the true focal length (unknown for the iPhone 16e video mode), and the official weights.
- GVHMR under the multiverse variants (only the default config), and the GVHMR ViTPose 2D for the squat (no foot points).
- Chunk-boundary effects (each chunk starts cold; chunks hold whole reps).
- Multi-camera methods of Table 3, YOLO26, and the paper's per-feature analyses (RQ1–RQ3).
- Whether the paper's Table 3 used `additional_errors` or nominal vs. measured orientations (the CSV has only nominal).

## 8. Next step

When `t5-gvhmr` finishes, rerun collect → psi_run → compare to add the other 240 sequences and check that the gain
holds at the non-table viewpoints. Then ask an independent reviewer to audit `rehab_psi.py` against Sec. 4.1, with
faults they choose, before this goes into the project plan. If it holds, GVHMR (or at least ViTPose 2D) is a better
single-camera front end for exercise assessment than MediaPipe. The next useful measurement is a two-camera fusion of
GVHMR, to compare with the paper's best multi-camera rows (Ortho. 0.58–0.76).
