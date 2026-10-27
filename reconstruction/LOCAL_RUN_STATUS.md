# Local EasyMocap result — 2026-09-29

**Fresh real-video detection and triangulation are verified on this Windows
laptop. SMPL fitting and the body-mesh overlay remain blocked by the missing
licensed neutral model. The full assignment is not complete.**

## Actual run

- Project: `D:\Research Fall 2026 - Shengting Cao`; local branch `codex/local-easymocap`.
- Input: official `zju-ls-feng` example, original camera calibration and video.
- Cameras: 1, 7, 13, 19. Original frames 0,2,...,98: **50 time steps / 200 images**.
- Source video: 1024 x 1024, 25 fps. Output: 50 frames at 12.5 fps, **4 seconds**.
- Fresh YOLOv5 person detection and HRNet-W48 keypoints; bundled OpenPose results
  were not used. Triangulation used EasyMocap's iterative method.
- Reprojection median: **5.65 pixels**;
  p95: **19.69 pixels** over
  3792 confident joint/view comparisons.
- Mean confident joint count: **19.00/19**.
  COCO17 supplies 17 detected points; neck/pelvis are derived and six BODY25 foot
  landmarks are unavailable. Missing points remain confidence zero.

Reprojection measures consistency with detections, not independent ground-truth
accuracy. This is a short four-camera subset, not all 23 cameras or all 800 frames.
Timing includes three reused smoke-test frames; do not treat it as a clean speed
benchmark. Full-array checks found finite results; both MP4s decoded all 50 frames.

## Verification evidence

`local-runs/logs/real-video-50frames.log` contains the run log.
`local-runs/zju-ls-feng-cpu/` holds exact configurations, source metadata, detection
caches, fresh 2D arrays, 3D JSON, images, MP4s and validation receipts.
The isolated environment passed `pip check` and imports. A separate offscreen
OpenGL cube check passed. The runner rejects `--fit` without an existing model
before the upstream automatic model-download code can run.

YOLOv5's version diagnostic emits a harmless Git path error because the project
path contains spaces. The pinned source loaded and inferred successfully. The
first run exposed an OpenCV array-contiguity issue in our diagnostic wrapper;
it was fixed and the real-data smoke test rerun successfully.

## Remaining assignment work

| Item | Current state |
|---|---|
| Local hardware and install | Verified Windows/AMD CPU environment |
| Sample + detector weights | Acquired from official sources, hashes recorded |
| Real-video 2D detection + 3D triangulation | Verified short subset, videos saved |
| SMPL fit + body-overlay video | **Pending licensed neutral SMPL model** |
| Full ZJU-MoCap release sequence | Access/agreement and run still pending |
| Paper reading | Code walkthrough and notes supplied; personal study still pending |
| Independent reproduction | Guide prepared; Seishin has not run it yet |

## Exact next step

Obtain the neutral model through https://smpl.is.tue.mpg.de/ and provide its local
path (it may be named `SMPL_NEUTRAL.pkl` or
`basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl`). Then use the `--fit --smpl` command
in [SETUP_WINDOWS.md](SETUP_WINDOWS.md). It reuses the detection cache, optimizes
the body and renders camera 1. The SMPL stage is prepared but unverified.

See [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md) for the source trace and
[READING_SESSION_NOTES.md](READING_SESSION_NOTES.md) for reading evidence limits.
