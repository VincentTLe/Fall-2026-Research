# Reconstruction exploration plan (Oct 27 – early Nov 2026)

Prof. Cao's advice (October 2026): go beyond the ZJU sample. Try reconstruction on videos from the internet, from
one camera and from many cameras, and try Gaussian Splatting. He will look at the results to judge whether the work
could become a paper.

This file is the plan for that. It is separate from the semester plan in [`README.md`](README.md), which covers the
team's GoPro rig. The work below needs only public data, so it does not wait on the other teams. When they have
recordings, the same scripts run on our own videos.

Each method below links its paper or code. Check every number quoted from a paper against the paper before it goes
on a slide.

---

## 1. What already works (Sep 29 – Oct 27)

| Stage | Status | Evidence |
|---|---|---|
| EasyMocap install, Windows CPU | Verified | [`SETUP_WINDOWS.md`](SETUP_WINDOWS.md) |
| Detection + triangulation, `zju-ls-feng`, 4 cameras, 50 frames | Verified, reprojection median 5.65 px | [`LOCAL_RUN_STATUS.md`](LOCAL_RUN_STATUS.md) |
| SMPL fit + overlay on the same subset | Verified Oct 27: 50 SMPL frames, mesh overlay on camera 1 | [`LOCAL_RUN_STATUS.md`](LOCAL_RUN_STATUS.md) |
| Synthetic camera layout / sync sweep | Done, simulated data only | [`results/2026-09-29-synthetic-rig-sweep.md`](results/2026-09-29-synthetic-rig-sweep.md) |

Known gaps carried into this plan: only 4 of 23 cameras were used, the COCO-17 detector gives no foot points, the
reprojection p95 (19.69 px) is unexplained, and nothing has been measured against true 3D ground truth.

---

## 2. Questions that could become a paper

These are hypotheses to test, not results.

- **Q1. Cameras vs. accuracy on real data.** How much 3D accuracy does a sparse rig (3, 4, 5, 6 cameras) lose against a
  dense reference? Does the synthetic sweep's trend (more views help most between 3 and 5) hold on real video?
- **Q2. One camera vs. a few cameras for rehab motion.** On rehabilitation exercises with motion-capture ground truth,
  how far are current single-camera methods from multi-view triangulation + SMPL? Which joints fail (feet, hands,
  occluded side)?
- **Q3. Avatar quality from cheap input.** Can a Gaussian-splatting avatar from one phone video, or from our few
  GoPros, give a usable telepresence view? How does quality change with the number of input views?
- **Q4. Internet video as data.** Can unsynchronized, uncalibrated online videos of the same exercise be combined
  into one 3D reconstruction, following the ZJU "Motion Capture from Internet Videos" approach?

Q1 and Q2 have true ground truth available in public data, so they can produce real numbers by early November. Q3
and Q4 are qualitative at first.

---

## 3. Datasets (chosen Oct 27)

| Dataset | What it gives | Ground truth | Access (checked Oct 26) | Used for |
|---|---|---|---|---|
| `zju-ls-feng` (EasyMocap sample) | 23 calibrated cameras, 800 frames | None beyond an all-view reference | Already on the laptop | Full EasyMocap run, SMPL, baseline |
| [CMU Panoptic](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox) | 31 HD + 480 VGA cameras, calibration per sequence | 3D skeletons (`joints19`) reconstructed from the dense dome | `getData.sh` script, no registration; non-commercial research use | Q1: pick 3–6 of the 31 HD cameras, compare against the dome skeletons |
| [REHAB24-6](https://zenodo.org/records/13305826) | 6 rehab exercises (squat, lunge, ...), 10 people, 2 RGB cameras, 30 fps, correctness labels | 26 joints in 3D from 16 OptiTrack cameras | Direct download from Zenodo, 5.7 GB, CC BY-NC 4.0 | Q2: one camera vs. both cameras vs. mocap. **No calibration file is listed**, so check what is in the archive first |
| [NeuMan](https://github.com/apple/ml-neuman) | Phone videos of a person in a scene | None | Direct download links in the repo | Q3: Gaussian avatar from one video |
| [iMoCap](https://github.com/zju3dv/iMoCap) | Internet videos of the same action | None | GitHub | Q4 (stretch) |
| YouTube clips (a few) | Real-world monocular video | None | Download per clip | Qualitative only. Do not publish frames with identifiable faces |

Not used now: Human3.6M and the full ZJU-MoCap release need registration or a signed agreement and may not arrive in
time. People-Snapshot's site could not be opened on Oct 26; GaussianAvatar's README links a processed copy.

Every downloaded dataset gets a line in `results/DATA_LOG.md`: URL, date, version, license, local path, size, SHA-256.
Datasets, videos and licensed models never go into this public repository.

---

## 4. Tracks

Each track has a ready-to-paste prompt for Claude Code on the school GPU server in
[`server-prompts/`](server-prompts/). Run P0 first.

### T1. Multi-view reconstruction (EasyMocap) → Q1

1. Full `zju-ls-feng` run on GPU: all 23 cameras, all 800 frames, detection → triangulation → SMPL → overlay video.
   This becomes the all-view reference.
2. Replace the COCO-17 keypoint detector with a whole-body one that outputs feet, if EasyMocap supports it at the
   pinned commit; otherwise record why not.
3. Explain the reprojection tail: per-camera and per-joint error tables.
4. CMU Panoptic: choose one single-person sequence. Run EasyMocap with 3, 4, 5, 6 and all 31 HD cameras, in spread-out
   and clustered layouts. Compare each against the Panoptic skeletons on the joints both skeletons share (MPJPE in mm,
   after checking units and joint mapping).
5. Sync tolerance on real data: shift one camera by 1, 2, 4, 8 frames (`tools/make_variant.py --shift`).

Paper source for the sync idea: multinb (SIGGRAPH 2022) estimates a per-camera time offset from body-pose reprojection
and reaches sub-frame accuracy with 8 GoPros. Worth reproducing later for the team's rig.

### T2. Single-camera reconstruction → Q2

Candidates (checked Oct 27):
- [GVHMR](https://github.com/zju3dv/GVHMR) (ZJU, SIGGRAPH Asia 2024). Needs SMPL-X registration.
- [WHAM](https://github.com/yohanshin/WHAM) (CVPR 2024). Needs SMPL + SMPLify registration. Uses ViTPose and SLAM
  for the camera.

Plan: run one of them (GVHMR first; WHAM if GVHMR fails to install) on each REHAB24-6 camera separately and on one
Panoptic HD camera. Compare with (a) mocap / dome ground truth and (b) the T1 multi-view result on the same frames.
Report MPJPE and PA-MPJPE (after rigid alignment) on shared joints, per exercise, and list the worst joints.

### T3. Internet video → Q4 (stretch, after T1 and T2)

Download a few YouTube clips of one rehab exercise (squat). Run T2's method per clip and inspect the result. If time
remains, read the iMoCap code and try combining two clips. Qualitative only.

### T4. Gaussian Splatting → Q3

1. Warm-up, static scene: record a 30–60 s phone video walking around an object in the lab, run nerfstudio
   `splatfacto` (gsplat backend, COLMAP for cameras). This checks the CUDA toolchain on the server.
2. Human avatar from one video: [GaussianAvatar](https://github.com/aipixel/GaussianAvatar) (CVPR 2024) on one NeuMan
   sequence, then on one REHAB24-6 camera. [GoMAvatar](https://github.com/wenj/GoMAvatar) (CVPR 2024) is the
   alternative; it has ZJU-MoCap and YouTube examples.
3. Record training time, GPU memory, render FPS and, where held-out frames exist, PSNR / SSIM / LPIPS.

Skipped for now: ASH (needs a person-specific template mesh and many cameras), Neural Body / multinb training
(about 14–30 h on 4 GPUs according to the papers).

---

## 5. How results are measured and reported

- **Ground truth first.** Accuracy numbers (MPJPE, PA-MPJPE) only against true ground truth (REHAB24-6 mocap, Panoptic
  dome). Agreement with an all-view EasyMocap result is reported as "agreement", not accuracy.
- **Same joints, same units.** Skeletons differ (BODY25, SMPL 24, Panoptic 19, REHAB24-6 26). Compare only a named
  shared subset, after checking axes, units (m vs. cm) and frame alignment. Write the mapping down in the result file.
- **Every number with its conditions:** dataset, sequence, frames, cameras, method commit, GPU, date.
- **One result file per run** in `results/YYYY-MM-DD-<track>-<name>.md`: what was run, exact commands, environment,
  numbers, pictures or video paths, what failed, what was not tested.
- **No claim without output.** "Runs" means the output file exists and was opened; "works" needs a number or a video.

---

## 6. Timeline to early November

Meetings are on Tuesdays at 12:30.

| Days | Work | Show |
|---|---|---|
| Oct 27–28 | SMPL on the laptop subset (done Oct 27). P0 server setup. Download REHAB24-6 and one Panoptic sequence | First SMPL overlay; server check output |
| Oct 28–30 | T1 full run + Panoptic camera-count; T2 on REHAB24-6 | First MPJPE tables |
| Oct 30 – Nov 2 | T4 static scene, then GaussianAvatar on NeuMan; T3 if time remains | Avatar video, timing |
| Nov 2–3 | P4: collect results into one summary and slides | **Nov 3 meeting:** results + which question looks paper-worthy |

If a track is blocked for more than half a day (install failure, missing license, no GPU), write down the blocker
and move to the next track.

---

## 7. Not in scope here

- The team's rig hardware, camera settings and calibration (Hardware, Software, Seishin).
- Motion ↔ language models (TM2T, T2M-GPT, MotionGPT) and rehab scoring models (ExerciseLLM, CNN-SE). They consume
  our output later.
- Training new networks from scratch, and recording people outside the team (needs Prof. Cao's IRB/consent check).
