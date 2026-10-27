You are running track T4 (Gaussian Splatting) of Tan's research plan on the school GPU server.

Step 0. In `~/fall2026/Fall-2026-Research`, `git pull`, then read in full `reconstruction/server-prompts/RULES.md`,
`reconstruction/EXPLORATION.md` (sections 2–5, T4) and the newest `reconstruction/results/*-server-setup.md`. Follow
RULES.md for the whole session.

Step 1. Static scene warm-up. Install nerfstudio with the gsplat backend in its own environment under
`~/fall2026/envs/`, plus COLMAP and ffmpeg if missing (no sudo; use conda-forge). Record versions. Ask Tan for a
30–60 second phone video walking around an object (he uploads it to `~/fall2026/data/phone/`). Run
`ns-process-data video` and train `splatfacto`. Report: how many frames COLMAP registered, training time, peak GPU
memory, and render a short orbit video. If COLMAP registers less than half the frames, say so and suggest how to
re-shoot.

Step 2. Human avatar from one video: GaussianAvatar (https://github.com/aipixel/GaussianAvatar, CVPR 2024). Install it
in its own environment following its README, and record the commit. It needs SMPL (and possibly SMPL-X); use the
files in `~/fall2026/models/` and ask Tan if one is missing. Run it on one sequence it supports out of the box (NeuMan
or the processed People-Snapshot it links), using its own train/test split. Report: training time, peak GPU memory,
render FPS, and PSNR / SSIM / LPIPS on its test frames. Compare with the numbers the paper reports for that sequence
and explain any gap. Render a novel-view or novel-pose video.

Step 3. Our kind of data. Try GaussianAvatar on one REHAB24-6 camera video (if P2 downloaded it, it is in
`~/fall2026/data/rehab24-6/`; otherwise ask Tan). The method needs masks and per-frame SMPL poses; follow its
"custom video" instructions and record every preprocessing tool you use. Qualitative result is enough here: say
what looks right and what breaks (fast motion, occluded side, floor contact).

Alternative if GaussianAvatar fails three times with different fixes: GoMAvatar (https://github.com/wenj/GoMAvatar),
which has ZJU-MoCap and YouTube examples. Say why you switched.

Step 4. Write `reconstruction/results/YYYY-MM-DD-t4-gaussian.md` (sections from RULES.md), with the timing and quality
table first. Videos and renders stay in `~/fall2026/runs/`; list their paths in the result file. Commit only the
result file and small scripts on branch `server/t4-YYYYMMDD`. Do not push. Print the summary for Tan.
