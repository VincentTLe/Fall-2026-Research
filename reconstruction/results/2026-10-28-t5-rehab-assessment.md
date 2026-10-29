# T5: does GVHMR change "2D beats monocular 3D" for rehabilitation assessment? (REHAB26-ViewAngles), 2026-10-28

## 1. Result

**Reproduction.** I reimplemented the separability metric Ψ of Jánošová et al. (arXiv 2609.35726v2, Sec. 4.1) from the
text and ran it on the dataset's own MediaPipe skeletons. It reproduces their Table 3 MPP 2D/3D rows **only to grade B**
(within 20% per cell). With the settings fixed before any comparison, 80 of 90 cells are within 20% of the paper. The
mean absolute deviation is 0.055 Ψ and the correlation over cells only 0.69. The worst cell is squat 3D at +90°: 0.36
against 0.21. Their conclusion that 2D beats MediaPipe 3D holds for the squat in all 64 metric variants consistent with
the paper's wording, but only in 48/64 for arm abduction and 60/64 for shoulder extension.

**GVHMR**, with the **unofficial weight mirror** that P2 used, was scored with the same metric on all 495 sequences
(5 persons × 3 cameras × 11 orientations × 3 exercises).
- **Arm exercises: better than MediaPipe 2D.** Ψ 0.602 vs 0.490 for arm abduction (+0.11, 95% CI over persons
  +0.09…+0.13) and 0.579 vs 0.489 for shoulder extension (+0.09, +0.06…+0.12). The gain is the same on the 240
  sequences outside Table 3, which the first version of this file did not use, and it holds in all 8 metric variants
  tested (§4.4).
- **Squat: not established.** At the default the gain is +0.04 (+0.01…+0.07). That is about +0.02 after subtracting
  GVHMR's higher shuffled-label floor. It is +0.017 (CI includes 0) on the held-out 240 sequences. It turns into a
  **loss** of −0.09 if 3D angles are measured in full 3D instead of the per-plane projection the paper describes. The
  first version of this file reported +0.065 for the squat on the Table 3 viewpoints alone; that overstated it.
- **Against MediaPipe 3D, GVHMR is better** on all three exercises (+0.14, +0.13, +0.09), in 85 of 87 viewpoint cells.

So with GVHMR, the paper's "2D beats monocular 3D" does not hold for the two arm exercises. For the squat, where the
paper found the clearest 2D advantage, GVHMR only closes the gap to 2D, and whether it is ahead depends on how the 3D
angles are measured. For shoulder extension, most of the gain comes from GVHMR's better 2D detector (ViTPose), not from
3D. This contradicts a published claim, so the caveats in §1a apply to every number here.

### 1a. Caveats

1. **The baseline is reproduced only to grade B**, not exactly. Cell-level agreement is moderate (r = 0.69), and my
   2D − 3D gaps differ from the paper's in size (Ex2: +0.054 vs +0.004; Ex3: +0.036 vs +0.081). My Ψ is therefore not
   guaranteed to be the paper's Ψ. The GVHMR comparison is internal: all methods are scored with my implementation on
   the same sequences.
2. **Unofficial weights.** GVHMR used the `camenduru/GVHMR` Hugging Face mirror (P2's files); only `yolov8x.pt` was
   matched to an official source. The official checkpoints may differ. GVHMR also ran with its default focal length
   (the real one is unknown).
3. **5 persons.** All CIs resample the 5 persons. With 5 people a 95% bootstrap interval is rough, and the people are
   the dataset's volunteers, not patients.
4. **Metric choices the paper leaves open**: the DTW variant, the DTW feature for Ex2/Ex3, the τ formula, what the
   first-quartile floor applies to, the joints of each feature, and the measurement plane of each 3D feature. The
   2D − 3D gap for MediaPipe moves with these (§4.2). For GVHMR the squat result changes sign with one of them (§4.4).
5. **Ψ has a noise floor of about 0.1**, not 0 (§5), and this floor differs slightly between methods (it is 0.018
   higher for GVHMR on the squat).
6. The independent review (§4.5) found no definite bug, but it was one reviewer, run as an AI agent, reviewing against
   the arXiv text.

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
# 10-29, after all 744 chunks
systemctl --user stop t5-gvhmr t5-multiverse; systemctl --user disable t5-gvhmr t5-multiverse
PY $T/rehab26_gvhmr_collect.py ...; PY $T/rehab26_psi_run.py ... --out runs/t5/psi/all_{gvhmr_incam,gvhmr_glob,vitpose2d}[_shuffled].csv
PY $T/rehab26_compare.py ... --md runs/t5/psi/all_compare_main.md   (+ decomposition, floors, non-Table-3 subset)
# independent review agent (read-only), then: strengthened selftest; reviewer's inject.py re-run on it
PY $T/rehab26_psi_run.py ... --seeds 0-4 --cfg <one variant> -> runs/t5/sens/{mpp,gvhmr}_<variant>.csv   (8 variants)
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

### 4.3 GVHMR vs. MediaPipe on the same sequences (all 495)

GVHMR numbers use the **unofficial weight mirror** (§2). Sequences are every (person, orientation, camera), 165 per
exercise, paired with the MediaPipe values from §4.1 (10 seeds, defaults). GVHMR `incam` = camera coordinates.
`glob` = GVHMR's gravity-aligned world turned to y down. Both get the same hip normalisation as MPP 3D. CIs come from
2000 bootstrap resamples of the 5 persons. Files `runs/t5/psi/all_compare_main.md` and `all_compare_decomp.md`; log
`logs/2026-10-29-t5-compare-all.log`.

| Exercise | n seq | MPP 2D | MPP 3D | ViTPose 2D | GVHMR incam | GVHMR glob | GVHMR incam − MPP 2D | GVHMR incam − MPP 3D | GVHMR incam − ViTPose 2D |
|---|---|---|---|---|---|---|---|---|---|
| Ex1 arm abduction | 165 | 0.490 | 0.460 | 0.529 | 0.602 | 0.602 | +0.112 [+0.094, +0.131] | +0.142 [+0.132, +0.152] | +0.073 [+0.045, +0.097] |
| Ex2 shoulder extension | 165 | 0.489 | 0.443 | 0.552 | 0.579 | 0.578 | +0.091 [+0.055, +0.124] | +0.136 [+0.088, +0.171] | +0.028 [−0.038, +0.083] |
| Ex3 squat | 165 | 0.502 | 0.450 | – | 0.543 | 0.535 | +0.042 [+0.008, +0.068] | +0.093 [+0.063, +0.122] | – |

ViTPose 2D is GVHMR's own COCO-17 2D input, mapped to MediaPipe indices. It has no heels or toes, so it has no squat value.

The same comparison on only the **240 sequences at viewpoints outside Table 3**. These were not available when the first
version of this file was written, so they are a held-out check:

| Exercise | n seq | MPP 2D | MPP 3D | GVHMR incam | GVHMR − MPP 2D | GVHMR − MPP 3D |
|---|---|---|---|---|---|---|
| Ex1 | 80 | 0.486 | 0.459 | 0.593 | +0.106 [+0.091, +0.122] | +0.133 [+0.114, +0.150] |
| Ex2 | 80 | 0.479 | 0.447 | 0.569 | +0.090 [+0.059, +0.130] | +0.121 [+0.094, +0.153] |
| Ex3 | 80 | 0.518 | 0.443 | 0.536 | +0.017 [−0.031, +0.058] | +0.093 [+0.049, +0.131] |

Null model (labels shuffled within each block, all 495 sequences) and Ψ above that floor:

| Exercise | floor MPP 2D | floor MPP 3D | floor ViTPose 2D | floor GVHMR | Ψ − floor: MPP 2D | MPP 3D | ViTPose 2D | GVHMR |
|---|---|---|---|---|---|---|---|---|
| Ex1 | 0.100 | 0.098 | 0.097 | 0.093 | 0.390 | 0.362 | 0.432 | 0.509 |
| Ex2 | 0.113 | 0.104 | 0.113 | 0.105 | 0.376 | 0.339 | 0.439 | 0.474 |
| Ex3 | 0.112 | 0.117 | – | 0.130 | 0.390 | 0.333 | – | 0.413 |

Over all 29 viewpoints (87 exercise × viewpoint cells), GVHMR incam beats MPP 2D in 72 cells and MPP 3D in 85. MPP 2D is
equal or higher in 15 cells: 9 squat cells (−45, −40, −35, −25, −15, +45, +50, +70, +90), 5 shoulder-extension cells
(−30, +30, +35, +40, +45) and one tie (Ex1 +60). The Table 3 viewpoint values are unchanged from the first version.
At 0° and ±90° (n = 15 at 0°, 5 at ±90°):

| Exercise | Viewpoint | MPP 2D | MPP 3D | GVHMR incam |
|---|---|---|---|---|
| Ex1 | −90 / 0 / +90 | 0.36 / 0.50 / 0.35 | 0.22 / 0.47 / 0.36 | 0.49 / 0.60 / 0.52 |
| Ex2 | −90 / 0 / +90 | 0.50 / 0.49 / 0.50 | 0.43 / 0.39 / 0.46 | 0.70 / 0.59 / 0.59 |
| Ex3 | −90 / 0 / +90 | 0.48 / 0.43 / 0.45 | 0.43 / 0.46 / 0.36 | 0.51 / 0.57 / 0.44 |

The seed range of a GVHMR viewpoint mean (Table 3 viewpoints, 10 seeds) has a median of 0.024 and a maximum of 0.107.
GVHMR cost: 744 chunks from 16:15 to 02:06 (9 h 51 min) on GPU 1, 6 parallel. Peak RSS 6.93–6.98 GB per chunk.
0 failed chunks, 0 WRONG-GPU events.

### 4.4 Does the GVHMR comparison depend on the open metric choices?

Each row changes one choice from the default. All 495 sequences, 5 seeds (0–4), MPP and GVHMR scored with the same
setting. `full3d` = angle between the full 3D vectors instead of the per-plane projection of Sec. 4.3. `f6_yz` =
Ex2 torso-forward error measured in the sagittal plane; the review flagged the default frontal plane as nearly blind to
forward bending. `asp_norm` = 2D angles in normalised instead of pixel coordinates (affects MPP 2D only). Log
`logs/2026-10-29-t5-sensitivity.log`, files `runs/t5/sens/`.

| Variant | Ex1 GVHMR − MPP 2D | Ex2 GVHMR − MPP 2D | Ex3 GVHMR − MPP 2D | Ex3 GVHMR − MPP 3D |
|---|---|---|---|---|
| default | +0.112 [+0.093, +0.130] | +0.090 [+0.053, +0.125] | +0.042 [+0.007, +0.068] | +0.093 |
| f6_yz | +0.112 [+0.093, +0.131] | +0.136 [+0.092, +0.176] | +0.042 [+0.007, +0.069] | +0.093 |
| full3d | +0.023 [+0.008, +0.043] | +0.117 [+0.099, +0.133] | **−0.090 [−0.121, −0.067]** | −0.018 |
| floor none | +0.109 [+0.092, +0.127] | +0.092 [+0.057, +0.126] | +0.038 [+0.003, +0.067] | +0.085 |
| Ex2 DTW on F5 | +0.112 [+0.093, +0.131] | +0.056 [+0.021, +0.091] | +0.042 [+0.007, +0.069] | +0.093 |
| additional errors | +0.112 [+0.094, +0.130] | +0.091 [+0.056, +0.125] | +0.042 [+0.007, +0.069] | +0.093 |
| τ prediction interval | +0.106 [+0.079, +0.135] | +0.079 [+0.035, +0.123] | +0.046 [+0.036, +0.055] | +0.093 |
| asp_norm | +0.143 [+0.126, +0.159] | +0.131 [+0.097, +0.165] | +0.018 [−0.024, +0.058] | +0.093 |

For Ex1 and Ex2 the gain over MPP 2D is positive with a CI excluding 0 in all 8 rows, though it shrinks to +0.023 for
Ex1 under `full3d`. For Ex3 it is small and positive in 6 rows, includes 0 in one, and reverses in `full3d`. There, 3D
angles of the knee, foot and trunk mix the planes, and GVHMR is then even below MPP 3D.

### 4.5 Independent review

A separate agent, which had not seen the code or my results, was given the paper URL, read Sec. 4.1, 4.3, 5.1 and
Table 1 itself, and reviewed `rehab_psi.py`, `rehab26_psi_run.py`, `gvhmr_export_mp.py` and `rehab26_gvhmr_collect.py`.
It wrote its own fault-injection script.

- **Definite bugs: none found.** It checked Eq. 1–3 (including the min(1, ·) cap and the /τ normalisation; it read
  Eq. 1 from the page's LaTeX), the τ statistic (ddof 1, t₀.₉₇₅,ₙ₋₁, s/√n), the averaging over (erroneous rep,
  feature) pairs, DTW and warping, slices (inclusive last frame), the viewpoint formula, seeding, the left/right joint
  indices, and that every block has ≥ 5 correct and ≥ 5 incorrect reps.
- **GVHMR → MediaPipe mapping: correct.** The SMPL joint order was checked against `smplx`, and the vertex IDs against
  `vertex_ids['smplh']`. On real output, the projected GVHMR joints match same-side ViTPose keypoints (e.g. wrist 14 px
  vs 495 px for the swapped side). Heels lie behind the ankles and toes in front (ankle at 0.22–0.23 of heel→toe; foot
  0.26–0.27 m). The COCO→MediaPipe map and the glob conversion (a proper 180° rotation, not a mirror) are correct.
- **Choices it flagged, not bugs:** Ex2's torso-forward error is scored by F6 in the frontal plane, and F1's sagittal
  projection becomes short near full abduction. Both are now tested in §4.4: f6_yz raises the Ex2 gain, and full3d is
  the row that changes conclusions. Other flagged choices: F3/F7/F15 definitions, Q1 floor, warp averaging, equal
  weight per (rep, feature), and a separate reference per camera.
- **Fixed after the review** (neither changes a reported number):
  - (1) **The self-test was weak.** It caught 6 of the reviewer's 23 injected faults and never ran `features()` or the
    τ statistics. I added hand-computed known answers for the features (both sides, planes, vertical, full 3D, 2D
    aspect), τ and the Q1 floor, warping, and Eq. 1 with τ ≠ 1. It now catches 21/23 of the reviewer's faults
    (`logs/2026-10-29-t5-selftest-faults.log`). The two it still misses are the Eq. 3 baseline excluding the
    reference, and the DTW tie-break order. The reviewer checked the first directly, and the second only matters on
    exact ties. These tests were written after seeing the fault list, so this is not a blind re-test.
  - (2) **`Config`'s default 2D aspect is (1, 1)**, while all reported runs passed pixel aspect explicitly. The REHAB26
    driver now defaults to 1080×1920. A run without `--cfg` reproduces the reported default run exactly (264/264 rows
    identical).

## 5. Checks

- **Known-answer self-test** (`rehab_psi.py --selftest`): angles, DTW on a time-stretched sine, Eq. 1 on hand-built
  values (τ = 1 and τ = 2), τ and the Q1 floor, warping of a 2× stretched rep, all features on a hand-built pose
  (frontal, sagittal, full 3D, 2D with two aspects), and 3D normalisation (hip line on +x, lengths kept). It catches
  21 of 23 faults chosen by the reviewer (§4.5). A strongly perturbed rep gives Ψ 0.967;
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
  numbers apply to both. For GVHMR (all 744 chunks): in 548 of the 579 chunks with a clear signal (corr > 0.5), the ViTPose
  wrist track and the MediaPipe wrist track agree best at a lag within ±1 frame. The median wrist disagreement is
  12.1 px; 25 chunks are above 30 px (on the first 383 chunks: 6 of 13 at the ±90° profile views, the worst 203 px,
  ex1_Terka_cam1 o=90).
- **GVHMR joints** (`runs/t5/gvhmr/mp33/checks.csv`): median upper arm 0.26 m, forearm 0.25 m, thigh 0.39 m,
  shin 0.41 m. Feet are 1.27 m below the shoulders in camera y (y down, as MediaPipe). Units: m in both 3D sources.
  Angles do not depend on scale.
- **Is the GVHMR gain an artefact?** It is not a floor effect for Ex1 and Ex2: GVHMR's shuffled-label floor equals
  MediaPipe's. For the squat it is 0.018 higher (all 495 sequences), so the squat gain above the floor is about +0.02. It is
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
- **Independent verifier:** done on 10-29 (§4.5). There was one reviewer, and the faults in the re-test were known to me
  when I wrote the new tests.
- My own bugs, all caught before results: a relative path in the first job list (one canary chunk failed); ffmpeg
  reading the runner's job list from stdin (fixed with `-nostdin` and `< /dev/null` before any chunk ran); and a
  one-shot GPU gate that passed during a gap between P2's jobs. It was replaced by "no `p2-*` unit active and no
  foreign process for 5 min" before any GPU work started.
- The paper does not state the DTW library or variant, the τ formula, which values the first-quartile floor applies
  to, the joints of each feature, the DTW feature for Ex2/Ex3, or the number of seeds. My choices are in
  `rehab_psi.Config`, and the consequential ones were run as the multiverse in §4.2.

## 7. Not tested

- GVHMR with the official weights, and with the true focal length (unknown for the iPhone 16e video mode).
- The full 64-config multiverse for GVHMR. Only 8 one-at-a-time variants were run (§4.4), and no combinations.
- GVHMR's ViTPose 2D for the squat (no foot points).
- Chunk-boundary effects (each chunk starts cold; chunks hold whole reps).
- Multi-camera methods of Table 3, YOLO26, and the paper's per-feature analyses (RQ1–RQ3).
- Whether the paper's Table 3 used `additional_errors`, or nominal vs. measured orientations (the CSV has only nominal).
- Other people or patients: 5 healthy volunteers only.

## 8. Next step

Ask the paper's authors (or check their code, if released) for the per-feature planes, the DTW feature for Ex2/Ex3 and
the τ floor. Those choices decide the squat result and limit the reproduction to grade B. With them, rerun §4.1 and
§4.4. If the arm-exercise gain survives, ViTPose 2D or GVHMR is a better single-camera front end than MediaPipe for
these exercises. The next measurement is a two-camera fusion of GVHMR, to compare with the paper's best multi-camera
rows (Ortho. 0.58–0.76). Rerunning GVHMR with official weights, if the Google Drive quota allows, would remove caveat 2.
