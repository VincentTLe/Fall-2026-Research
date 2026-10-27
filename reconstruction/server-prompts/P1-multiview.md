You are running track T1 (multi-view reconstruction with EasyMocap) of Tan's research plan on the school GPU server.

Step 0. In `~/fall2026/Fall-2026-Research`, `git pull`, then read in full `reconstruction/server-prompts/RULES.md`,
`reconstruction/EXPLORATION.md` (section 4, T1), `reconstruction/LOCAL_RUN_STATUS.md`,
`reconstruction/CODE_WALKTHROUGH.md` and the newest `reconstruction/results/*-server-setup.md`. Use the environment
and GPU that file describes. Follow RULES.md for the whole session.

Context you need: on Tan's Windows laptop (CPU), `reconstruction/tools/local_cpu.py` ran the EasyMocap config
`config/mv1p/detect_triangulate_fitSMPL.yml` on 4 of 23 cameras and 50 frames of the `zju-ls-feng` sample:
reprojection median 5.65 px, p95 19.69 px, SMPL fit and overlay verified. That script hard-codes CPU and this one
dataset.

Step 1. A general runner. Write `reconstruction/tools/run_mv1p.py`, based on `local_cpu.py`, with arguments for the
data folder, output folder, camera list, frame range and step, device (`cuda`/`cpu`), SMPL model path, and which
cameras to render. Keep the existing receipts (`validation.json` with reprojection stats, valid-joint counts,
timing, `smpl_completed`, `videos_encoded`). Leave `local_cpu.py` unchanged. Check the new runner reproduces the
laptop numbers on the same 4 cameras / 50 frames before going further (same detector weights; small GPU/CPU
differences are fine, explain them).

Step 2. Full `zju-ls-feng`. Download it (URL and SHA-256 in `reconstruction/SETUP_WINDOWS.md`; verify the hash). Run all
23 cameras, all 800 frames, with SMPL. Render an overlay for at least 4 cameras spread around the subject. Report:
time per frame on GPU, reprojection median/p95 overall, **per camera and per joint** (find where the p95 tail comes
from), valid joints, and look at overlay frames from each rendered camera (name the frames you looked at). Save this
run as the all-view reference for the next step.

Step 3. Feet. The current HRNet COCO-17 model gives no foot points (six BODY25 foot slots stay zero). Check whether
EasyMocap at the pinned commit can use a detector with feet (for example OpenPose BODY25, MediaPipe, or a whole-body
HRNet/ViTPose model). If one works without large changes, run it on the 4-camera subset and compare. If not, write
down what blocks it. Do not spend more than about two hours here.

Step 4. CMU Panoptic camera-count experiment (EXPLORATION Q1). Use the panoptic-toolbox
(https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox). Pick **one single-person sequence** with HD video
and 3D body keypoints; say which and why, and ask Tan if no single-person sequence is available. Download only the HD
videos and the body keypoints for that sequence, and only a frame range you need (at least 300 frames).
- Write `reconstruction/tools/panoptic_to_easymocap.py`: Panoptic calibration → EasyMocap `intri.yml`/`extri.yml`
  (reuse `tools/emc_yml.py`). Test it: project the Panoptic 3D joints into two HD cameras with the converted
  calibration and check they land on the person (report pixel error against the 2D detections, and look at one frame).
  Mind the units: Panoptic is in centimetres.
- Run EasyMocap with: all 31 HD cameras; 6, 5, 4 and 3 cameras spread around the subject; 5 cameras clustered on one
  side. Reuse the 2D detections across these runs instead of detecting again (`tools/make_variant.py` symlinks
  `images/` and `annots/`; check that your runner actually reads them).
- Compare each run with the Panoptic `joints19` skeletons on a joint subset both skeletons define. Write the mapping
  as a table. Report MPJPE (mm) and PA-MPJPE, the share of frames/joints missing, and per-joint errors for the 5-camera
  spread layout.
- Note that the Panoptic skeletons come from the dense dome, not from markers. Call them "dome reference" in the text.

Step 5. Sync tolerance on real data. On the 5-camera spread layout, shift one camera by 1, 2, 4 and 8 frames
(`make_variant.py --shift`) and report MPJPE against the dome reference. Compare the trend with the synthetic sweep
(`results/2026-09-29-synthetic-rig-sweep.md`) in one short paragraph.

Step 6. Write `reconstruction/results/YYYY-MM-DD-t1-multiview.md` (sections from RULES.md), with the camera-count table
and the sync table first. Commit the result file and the two new tools on branch `server/t1-YYYYMMDD`. Do not push.
Print the summary for Tan.

If time runs short, the order of priority is: Step 1, Step 2, Step 4, Step 5, Step 3.
