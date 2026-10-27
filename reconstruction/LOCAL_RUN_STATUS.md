# Local EasyMocap result — 2026-09-29, SMPL added 2026-10-27

**Fresh real-video detection, triangulation and SMPL fitting with a body-mesh
overlay are verified on this Windows laptop, on a 4-camera, 50-frame subset of
`zju-ls-feng`. The full 23-camera, 800-frame run is not done yet.**

## SMPL fit — 2026-10-27

- Model: licensed SMPL v1.1.0 for Python, neutral
  (`basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl`), loaded from outside the repo.
  Archive SHA-256 `87C9E6CB1DDDAD79CF3B8B01760E240A9CD0C29D7FF14EDA30FB07E3BD430C27`.
- Same input, cameras and frames as the run below; detections were reused.
- Output: 50 `smpl/*.json` files (keys `Rh`, `Th`, `poses` (69), `shapes` (10)),
  50 `vis_render/*.jpg` frames on camera 1, and `vis_render.mp4`, which decodes
  all 50 frames at 12.5 fps.
- Final 3D keypoint loss in `refine_poses`: 1.0289 (EasyMocap's weighted loss
  value, not an error in millimetres). Total wall time 1 min 59 s on CPU,
  including reused stages.
- Timing caveat: this run reused the cached detections, so the new
  `validation.json` reports `seconds_per_frame_mean` 0.22 s. That is not a
  detection speed. The Sep 29 receipt (7.28 s per time step, fresh detection) is
  kept in `local-runs/archive-2026-09-29-receipts/`.
- Visual check of frames 0 and 25: the mesh follows the T-pose and the raised
  right arm and sits on the feet. Clothing extends beyond the mesh, as expected
  for a body model without clothes.
- `validation.json` reports `smpl_completed: true`, `videos_encoded: true`.
  The detection/triangulation numbers are unchanged from Sep 29.
- Log: `local-runs/logs/smpl-fit-2026-10-27-run2.log`. Receipts from Sep 29 were
  archived in `local-runs/archive-2026-09-29-receipts/` before this run.

The first attempt failed in `refine_poses` while loading EasyMocap's pose prior
`easymocap/multistage/gmm_08.pkl` (`UnpicklingError: the STRING opcode argument
must be quoted`). Git for Windows (`core.autocrlf=true`) had converted that
text-format pickle to CRLF line endings: 55 extra bytes against the repository
blob (839,182 vs 839,127). Rewriting the file from the git blob fixed it; see
`SETUP_WINDOWS.md`, "Problems found locally".

Not verified: accuracy against ground truth (this sample has none), other
cameras' overlays, and any frame beyond the 50 listed.

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
| SMPL fit + body-overlay video | Verified on the 50-frame subset (2026-10-27) |
| Full ZJU-MoCap release sequence | Access/agreement and run still pending |
| Paper reading | Code walkthrough and notes supplied; personal study still pending |
| Independent reproduction | Guide prepared; Seishin has not run it yet |

## Exact next step

Run all 23 cameras and all 800 frames on the GPU server, as track T1 in
[EXPLORATION.md](EXPLORATION.md).

See [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md) for the source trace and
[READING_SESSION_NOTES.md](READING_SESSION_NOTES.md) for reading evidence limits.
