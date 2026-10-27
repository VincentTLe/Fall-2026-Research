You are running track T2 (single-camera reconstruction), plus T3 if time remains, of Tan's research plan on the
school GPU server.

Step 0. In `~/fall2026/Fall-2026-Research`, `git pull`, then read in full `reconstruction/server-prompts/RULES.md`,
`reconstruction/EXPLORATION.md` (sections 2–5) and the newest `reconstruction/results/*-server-setup.md`. Follow
RULES.md for the whole session.

Step 1. Data: REHAB24-6 (https://zenodo.org/records/13305826, CC BY-NC 4.0, about 5.7 GB). Download `videos.zip`,
`3d_joints.zip`, `2d_joints.zip`, `Segmentation.csv` and `joints_names.txt` into `~/fall2026/data/rehab24-6/`, record
the Zenodo checksums, and log the download in `reconstruction/results/DATA_LOG.md` (create it if missing). Then
**inspect before using**: list the archive structure, video resolution/fps per camera, joint names and order, units
of the 3D joints, how 2D joints map to each camera, and whether any camera calibration is included. Write that down
first; the Zenodo page does not list a calibration file.

Step 2. Method. Install GVHMR (https://github.com/zju3dv/GVHMR) in its own environment under `~/fall2026/envs/`,
following its README, and record the commit. It needs SMPL-X (and SMPL) body models: if Tan has not given you
SMPL-X, stop and ask him to register at https://smpl-x.is.tue.mpg.de and copy the files to `~/fall2026/models/`. Run
its demo video first. If GVHMR cannot be installed after three different fixes, try WHAM
(https://github.com/yohanshin/WHAM) instead and say why.

Step 3. Evaluate on REHAB24-6. Choose a subset that covers all six exercises (for example one subject per exercise,
both cameras), and say which recordings. Run the method on each camera's video separately. Then compare with the
mocap 3D joints:
- write the joint mapping between the method's output (SMPL/SMPL-X joints) and the 26 REHAB24-6 joints as a table, and
  use only joints both define;
- align time (frame index and fps) and check it on one recording by plotting or printing one joint over time;
- report PA-MPJPE (after a rigid alignment per frame), and MPJPE after root (pelvis) alignment, in mm, per exercise
  and per camera; list the five worst joints;
- check units on one frame before trusting any number (the dataset page mentions "virtual centimeters").

Step 4. Two cameras (optional, if Step 3 is done). The dataset gives 2D projections of the mocap joints for each
camera. If no calibration file exists, estimate each camera's projection matrix from these 2D–3D pairs (DLT or PnP),
report the reprojection error of that estimate, then triangulate the method's or a detector's 2D keypoints from both
cameras and compare with the mocap joints. This gives one-camera vs. two-camera numbers on the same frames.

Step 5. T3, YouTube (only if Steps 1–3 are done). Ask Tan for 2–3 clip URLs of people doing squats, or ask whether
he wants you to pick some. Run the same method on each clip, render the result, and describe what works and what
fails. These are qualitative only: no numbers, and no frames with identifiable faces in git.

Step 6. Write `reconstruction/results/YYYY-MM-DD-t2-monocular.md` (sections from RULES.md), with the per-exercise
table first. Commit it and any small scripts on branch `server/t2-YYYYMMDD`. Do not push. Print the summary for Tan.
