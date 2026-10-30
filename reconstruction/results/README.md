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
| 2026-10-27 | [T4 Gaussian avatars](2026-10-27-t4-gaussian.md) | Q3 | GaussianAvatar reproduced (28.80 vs. the authors' 28.83 dB) and renders at 38 fps. On a phone video, pose error limits quality |
| 2026-10-28 | [T5 exercise assessment](2026-10-28-t5-rehab-assessment.md) | Q2, assessment | REHAB26: GVHMR separates correct from incorrect arm exercises better than MediaPipe 2D (+0.11, +0.09 Ψ). The squat is not settled |
| 2026-10-29 | [T2b cameras and benchmark](2026-10-29-t2b-calibration-benchmark.md) | Q2, benchmark | GVHMR has 6.0 cm error on the REHAB24-6 benchmark, against 8.4 cm for the paper's best. The authors' newer cameras fit the video better: two cameras reach 42.9 mm |
| — | [DATA_LOG.md](DATA_LOG.md) | Provenance | Every dataset and model downloaded, with license and checksum |
| — | [figures/](figures/) | Overview | Summary figures, drawn by [`../tools/make_summary_figures.py`](../tools/make_summary_figures.py) |

Metrics, in one line each:
- **MPJPE:** mean distance between estimated and true 3D joints.
- **PA-MPJPE:** the same, after the best rotation, shift and scale, so it measures pose shape only.
- **Ψ:** how well correct and incorrect repetitions are told apart (about 0.1 is chance; higher is better).
- **PSNR / SSIM / LPIPS:** image similarity between a rendered avatar and the real frame. Higher PSNR and SSIM are better; lower LPIPS is better.
