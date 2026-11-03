# T5b: two-camera exercise assessment on REHAB26 without calibration, 2026-11-02

## 1. Result

**Reproduction.** I implemented the paper's two calibration-free two-camera strategies (Jánošová et al.,
arXiv 2609.35726v2, Sec. 4.4) for cameras 1 and 3 (90° apart) and scored them with T5's `rehab_psi.py` (same defaults,
10 seeds):
- **Orthogonal 2D:** each feature taken from the more suitable of the two 2D views.
- **Weighted fusion:** each camera's 3D pose normalised, then averaged with squared-visibility weights.

Both MediaPipe rows reproduce Table 3 to about the level T5 reached for one camera: 20 of 21 cells at grade A or B for
each. The worst cells are Ortho Ex2 −20° (0.59 vs 0.76, C) and Merged Ex3 0° (0.48 vs 0.32, D).

**Two cameras vs one.** The paper's finding holds here: orthogonal 2D beats one MediaPipe 2D camera on the same blocks
by +0.10 / +0.07 / +0.05 Ψ (Ex1 / Ex2 / Ex3; CIs over persons exclude 0).

**With GVHMR** (unofficial weight mirror, as in T5), two cameras fused with equal weights give the highest Ψ for the arm
exercises: 0.622 and 0.680, against 0.574 and 0.550 for MediaPipe orthogonal 2D. For the squat it only ties MediaPipe
orthogonal 2D (+0.008, CI −0.05…+0.07).

**Not done.** Triangulation and RUMPL need camera calibration that the dataset does not provide, so they were not
attempted.

## 2. Setup

| Field | Value |
|---|---|
| Machine | `lambdavector2`, CPU only (both GPUs held by another user's job); 16 processes |
| Code | branch `server/t5b-20261102` from `origin/main` `7d44d52`; new `reconstruction/tools/rehab26_twocam.py`, `rehab26_twocam_report.py`; `rehab_psi.py` unchanged from T5 (`d3211a0`) |
| Inputs (no new downloads, no new GPU runs) | REHAB26 MediaPipe 2D/3D (Zenodo 23018128, MD5-checked in T5); T5's ViTPose 2D and GVHMR incam per camera (`runs/t5/gvhmr/mp33/`, all 744 chunks; **GVHMR weights = unofficial HF mirror `camenduru/GVHMR`**) |
| Paper | Sec. 4.4, 5.1 and the Table 3 caption reread on 2026-11-02 |
| Frozen before scoring | `runs/t5b/spec/twocam_targets.csv` (SHA-256 `affe820f…d3`) holds the Table 3 Ortho and Merged rows. `SPEC.md` (`404f8b52…87`) holds the rules below |
| Two-camera sequence | (recording, orientation), cameras 1 and 3. The dataset states the same frame numbers apply to all cameras. Table 3 column = cam1 angle (−o); cam3 = column + 90° |

Rules the paper does not state, fixed in `SPEC.md` before scoring:

- **Ortho.** The paper says only that the view for each feature "is determined in advance". In each sequence, the
  "frontal" camera is whichever of cam1/cam3 has the smaller |viewing angle| (tie at o = 45 → cam1).
  - From the frontal camera: frontal-plane and transverse features (F2–F7, F9, F10).
  - From the other camera: sagittal features (F1, F8, F11–F15).
  - The DTW feature comes from its assigned camera.
  - An alternative with fixed cameras (frontal features always cam1, sagittal always cam3) is reported but not graded.
- **Merged.** The paper does not state how the frontal rotation or the height scale is found. Each camera's pose:
  1. mid-hip moved to the origin;
  2. rotated about the vertical so the hips lie on +x (the same step as T5's single-camera 3D);
  3. scaled to unit median body length (mid-hip→mid-shoulder + thigh + shin);
  4. combined with Eq. 4, using MediaPipe visibility squared as weights.

  Calibration is not needed.
- **GVHMR two cameras.** Primary: the same fusion of GVHMR incam with equal weights, since GVHMR gives no per-joint
  confidence. Secondary: per-feature camera as in Ortho, on GVHMR 3D features.
- **ViTPose 2D Ortho.** Ex1 and Ex2 only (no foot keypoints).

## 4. Numbers

### 4.1 MediaPipe two-camera rows against Table 3

Viewpoint value = mean over the 5 persons, averaged over seeds 0–9. The seed range (min–max of the viewpoint mean)
is in brackets. Grades as in T5: A ≤ 2%, B ≤ 20%, C ≤ 40%, D ≤ 60%, F > 60%.

| Ex | Row | −90 | −60 | −40 | −30 | −20 | −10 | 0 |
|---|---|---|---|---|---|---|---|---|
| 1 | Ortho paper | 0.58 | 0.61 | 0.65 | 0.66 | 0.64 | 0.63 | 0.66 |
| 1 | Ortho ours | 0.59 (0.58–0.60) | 0.62 (0.61–0.63) | 0.59 (0.58–0.61) | 0.59 (0.57–0.60) | 0.56 (0.55–0.57) | 0.58 (0.57–0.59) | 0.53 (0.52–0.56) |
| 1 | grade | A | A | B | B | B | B | B |
| 2 | Ortho paper | 0.65 | 0.58 | 0.62 | 0.70 | 0.76 | 0.61 | 0.65 |
| 2 | Ortho ours | 0.62 (0.61–0.64) | 0.47 (0.43–0.50) | 0.53 (0.52–0.54) | 0.63 (0.62–0.64) | 0.59 (0.58–0.61) | 0.54 (0.51–0.55) | 0.56 (0.54–0.58) |
| 2 | grade | B | B | B | B | C | B | B |
| 3 | Ortho paper | 0.61 | 0.61 | 0.62 | 0.58 | 0.53 | 0.58 | 0.61 |
| 3 | Ortho ours | 0.57 (0.56–0.58) | 0.53 (0.53–0.55) | 0.52 (0.51–0.53) | 0.51 (0.50–0.51) | 0.56 (0.55–0.57) | 0.56 (0.55–0.58) | 0.59 (0.58–0.60) |
| 3 | grade | B | B | B | B | B | B | B |
| 1 | Merged paper | 0.55 | 0.63 | 0.63 | 0.61 | 0.58 | 0.62 | 0.51 |
| 1 | Merged ours | 0.48 (0.47–0.49) | 0.55 (0.54–0.56) | 0.54 (0.54–0.55) | 0.53 (0.52–0.54) | 0.55 (0.54–0.57) | 0.53 (0.51–0.54) | 0.45 (0.40–0.48) |
| 1 | grade | B | B | B | B | B | B | B |
| 2 | Merged paper | 0.60 | 0.51 | 0.59 | 0.64 | 0.49 | 0.56 | 0.57 |
| 2 | Merged ours | 0.50 (0.49–0.51) | 0.49 (0.46–0.52) | 0.58 (0.56–0.59) | 0.57 (0.56–0.59) | 0.51 (0.49–0.53) | 0.53 (0.52–0.54) | 0.49 (0.47–0.51) |
| 2 | grade | B | B | B | B | B | B | B |
| 3 | Merged paper | 0.45 | 0.52 | 0.49 | 0.48 | 0.45 | 0.48 | 0.32 |
| 3 | Merged ours | 0.48 (0.45–0.49) | 0.49 (0.48–0.51) | 0.50 (0.50–0.51) | 0.49 (0.48–0.50) | 0.46 (0.44–0.47) | 0.49 (0.48–0.51) | 0.48 (0.47–0.49) |
| 3 | grade | B | B | B | B | A | B | D |

- **Ortho:** A 2, B 18, C 1. Worst cell Ex2 −20 (22%). Mean |dev| 0.066; ours is lower in 18 of 21 cells.
- **Merged:** A 1, B 19, D 1. Worst cell Ex3 0° (51%). Mean |dev| 0.053.

### 4.2 Two cameras next to T5's one-camera rows

There are 55 blocks per exercise (5 persons × 11 orientations), the same blocks for every row. "1 cam" = T5's
single-camera Ψ, averaged over cam1 and cam3 of the block. CIs come from 2000 bootstrap resamples of the 5 persons.
Output `runs/t5b/psi/report.md`, log `logs/2026-11-02-t5b-report.log`.

| Row | Ex1 arm abduction | Ex2 shoulder extension | Ex3 squat |
|---|---|---|---|
| MPP 2D, 1 cam | 0.471 | 0.484 | 0.499 |
| MPP 3D, 1 cam | 0.445 | 0.438 | 0.450 |
| ViTPose 2D, 1 cam | 0.516 | 0.552 | – |
| GVHMR, 1 cam | 0.592 | 0.576 | 0.536 |
| **MPP 2D Ortho** | 0.574 | 0.550 | 0.548 |
| MPP 2D Ortho, fixed cameras (alternative rule) | 0.467 | 0.501 | 0.509 |
| **MPP 3D Merged** | 0.518 | 0.531 | 0.482 |
| **ViTPose 2D Ortho** | 0.611 | 0.608 | – |
| **GVHMR Merged, equal weights** (primary GVHMR rule) | 0.622 | 0.680 | 0.556 |
| GVHMR per-feature camera (secondary) | 0.620 | 0.618 | 0.545 |

| Paired difference | Ex1 | Ex2 | Ex3 |
|---|---|---|---|
| MPP 2D Ortho − MPP 2D 1 cam | +0.102 [+0.078, +0.127] | +0.066 [+0.060, +0.072] | +0.049 [+0.025, +0.073] |
| ViTPose 2D Ortho − ViTPose 2D 1 cam | +0.095 [+0.083, +0.107] | +0.056 [+0.046, +0.066] | – |
| MPP 3D Merged − MPP 3D 1 cam | +0.072 [+0.056, +0.089] | +0.094 [+0.071, +0.112] | +0.032 [+0.013, +0.052] |
| GVHMR Merged − GVHMR 1 cam | +0.030 [+0.017, +0.045] | +0.104 [+0.080, +0.134] | +0.019 [−0.004, +0.035] |
| ViTPose 2D Ortho − MPP 2D Ortho | +0.037 [+0.021, +0.054] | +0.058 [+0.032, +0.083] | – |
| GVHMR Merged − MPP 2D Ortho | +0.048 [+0.017, +0.080] | +0.130 [+0.106, +0.153] | +0.008 [−0.051, +0.068] |
| GVHMR per-feature camera − MPP 2D Ortho | +0.047 [+0.013, +0.079] | +0.069 [+0.038, +0.099] | −0.003 [−0.047, +0.036] |

## 5. Checks

- **Fusion geometry** (`logs/2026-11-02-t5b-fusion-check.log`). After the per-camera normalisation, the median
  cam1–cam3 distance of 8 limb and head joints is:
  - GVHMR: 0.03–0.07 body lengths;
  - MediaPipe 3D: 0.07–0.16 body lengths.

  This holds over all 11 orientations of ex1_Honza and ex6_Mima. A mirrored or wrongly rotated view would differ by
  about half a body length, so the hip-based rotation needs no calibration.
- **Coverage.** Each MediaPipe/GVHMR row has 165 blocks × 10 seeds = 1650 rows, and ViTPose 1100 (Ex1, Ex2). No
  block was skipped for missing frames.
- **Same metric as T5.** `rehab_psi.py` is unchanged from T5 (self-test 21/23 of the reviewer's faults). Seeds use
  T5's hashing with camera tag "13". The targets file hash is checked by the report script.
- **Seed spread.** A viewpoint mean varies by at most 0.08 across seeds (Merged Ex1 0°: 0.40–0.48), usually 0.01–0.03.
- **The rule matters.** With fixed cameras, Ortho gives no gain over one camera (0.467 / 0.501 / 0.509) and lies far
  below the paper's Ortho row. The per-orientation frontal/lateral rule gives values near the paper's. This supports
  the rule, but the paper does not state it.

## 6. Failed or skipped

- **Triangulation (DLT) and RUMPL: skipped.** The paper says triangulation "requires explicit camera calibration", and
  RUMPL uses "the known camera calibration matrices". REHAB26 provides no calibration, and the prompt rules out
  inventing one. The two strategies done here needed none, so I did not have to stop.
- **Rules the paper leaves open:** which view each Ortho feature uses, how the Merged frontal rotation and height scale
  are found, and how weights apply to GVHMR. My choices are in §2, frozen before scoring. Only the fixed-camera
  alternative was run for Ortho.
- **Caveats carried over from T5:**
  - the metric reproduces the paper only to grade B;
  - GVHMR used the unofficial weight mirror and its default focal length;
  - there are 5 persons, so the CIs are rough;
  - the open metric choices of T5 §4.4 (e.g. full-3D angles reverse the squat) were not rerun for the two-camera rows.
- **Synchronisation.** I relied on the dataset's statement that the cameras share frame numbers; it was not
  rechecked here. T5's lag check covered each camera against its own skeleton, not cam1 against cam3.
