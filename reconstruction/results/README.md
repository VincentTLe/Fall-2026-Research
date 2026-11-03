# Results index

There is one file per experiment, newest last. Each file is laid out the same way:
- **1. Result:** the answer in a few sentences;
- **2–3. Setup and commands:** what was run;
- **4. Numbers;**
- **5. Checks:** how the numbers were verified;
- **6. Failed or skipped;**
- **7. Not tested;**
- **8. Next step.**

Section 1 is enough for a quick read. The overview and figures are in the [top-level README](../../README.md).

| Date | File | Question | Result in one line |
|---|---|---|---|
| 2026-09-29 | [Synthetic rig sweep](2026-09-29-synthetic-rig-sweep.md) | Q1, in simulation | 5 even cameras give 8.9 mm. 2D keypoint noise matters more than layout; keep sync within 1 frame at 60 fps |
| 2026-10-27 | [GPU server setup](2026-10-27-server-setup.md) | Infrastructure | EasyMocap runs on the school server (RTX A6000) and reproduces the sweep exactly |
| 2026-10-27 | [T1 multi-view](2026-10-27-t1-multiview.md) | Q1, real data | CMU Panoptic: about 17 mm with 5–6 spread cameras, 11.7 mm with 31. 5 cameras bunched on one side give 25–26 mm. A 1-frame offset adds 0.3 mm |
| 2026-10-27 | [T2 one camera vs. two](2026-10-27-t2-monocular.md) | Q2 | REHAB24-6, 12 recordings: one camera (GVHMR) has PA-MPJPE 46–70 mm; two cameras are at 47.5 mm with no alignment |
| 2026-10-27 | [T4 Gaussian avatars](2026-10-27-t4-gaussian.md) | Q3 | GaussianAvatar reproduced (28.80 vs. the authors' 28.83 dB) and renders at 38 fps. On a phone video and on one REHAB24-6 camera (18.5 dB raw poses vs. 28.1 dB corrected), pose error limits quality |
| 2026-10-28 | [T5 exercise assessment](2026-10-28-t5-rehab-assessment.md) | Q2, assessment | REHAB26: GVHMR separates correct from incorrect arm exercises better than MediaPipe 2D (+0.11, +0.09 Ψ). The squat is not settled |
| 2026-10-29 | [T2b cameras and benchmark](2026-10-29-t2b-calibration-benchmark.md) | Q2, benchmark | GVHMR has 6.0 cm error on the REHAB24-6 benchmark, against 8.4 cm for the paper's best. The authors' newer cameras fit the video better: two cameras reach 42.9 mm |
| 2026-10-29 | [Summary for the Nov 3 meeting](2026-10-29-summary.md) | All | What Q1–Q4 show and don't show, where files disagree, the smallest next experiment for each gap, a 7-slide outline |
| 2026-11-02 | [T2c two cameras + SMPL](2026-11-02-t2c-twocam-smpl.md) | Q2 | An SMPL fit to two-camera HRNet 2D gives 4.83 cm on the paper's protocol, against GVHMR's 6.28 (12 recordings). Body shape improves slightly (41.2 vs. 43.1 mm after alignment), position does not (57.3 vs. 53.3 mm) |
| 2026-11-02 | [T2d two cameras + SMPL, ViTPose](2026-11-02-t2d-twocam-smpl-vitpose.md) | Q2 | All 65 recordings: 5.09 cm BestFit on the paper's protocol, against GVHMR's 6.00 and the paper's 8.4. Its view-agreement filter hurts the hidden-arm frames on the T2 metrics (PA 46.0 mm) |
| 2026-11-02 | [T5b two-camera assessment](2026-11-02-t5b-twocam-assessment.md) | Q2, assessment | Calibration-free two-camera methods on REHAB26: the paper's orthogonal 2D reproduces. GVHMR fused from two views is best on the arm exercises (0.62 / 0.68 vs. 0.57 / 0.55); the squat is a tie |
| 2026-11-02 | [T4b avatar runs (pending)](2026-11-02-t4b-avatar-views.md) | Q3 | Prepared and checked: two-camera poses fit the person's outline better (IoU 0.835 vs. 0.813). Training waits for a GPU with 24 GB free |
| 2026-11-03 | [T1b SMPL on Panoptic](2026-11-03-t1b-panoptic-smpl.md) | Q1 | EasyMocap's default SMPL fit makes every layout worse (43–47 mm with 4+ cameras, against 11.7–26 for triangulation). The joints it fills in for 3 cameras are about 117 mm off |
| 2026-11-02 | [T3 internet video](2026-11-02-t3-internet-video.md) | Q4 | 41 stock clips in two rounds. GVHMR is consistent with the image on 20. The counter, fixed before round 2, is exact on 4 of 4 new clips where the right person is tracked. In 2 of 3 clips with a coach it followed the coach. Choosing the person who moves most fixed both |
| 2026-11-02 | [Depth noise](2026-11-02-depth-noise.md) | Q2, Q3 | GVHMR's depth error is mostly a constant offset plus slow drift (17 of 72 mm is jitter), so smoothing does not help. Depth must come from a second camera |
| — | [DATA_LOG.md](DATA_LOG.md) | Provenance | Every dataset and model downloaded, with license and checksum |
| — | [figures/](figures/) | Overview | Summary figures, drawn by [`../tools/make_summary_figures.py`](../tools/make_summary_figures.py) |

Metrics, in one line each:
- **MPJPE:** mean distance between estimated and true 3D joints.
- **PA-MPJPE:** the same, after the best rotation, shift and scale, so it measures pose shape only.
- **Ψ:** how well correct and incorrect repetitions are told apart (about 0.1 is chance; higher is better).
- **PSNR / SSIM / LPIPS:** image similarity between a rendered avatar and the real frame. Higher PSNR and SSIM are better; lower LPIPS is better.
