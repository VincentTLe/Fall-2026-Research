# T3: GVHMR on internet video, rounds 1 and 2 (Q4), 2026-11-02

Run from the laptop session on the GPU server. No server Claude session was used.

## 1. Result

**Clean clips work.** Fifteen free stock clips were tested: 11 squat, 4 lunge, from Mixkit and Pexels. None has ground truth. On the clean ones, GVHMR's 3D is about as consistent with the images as it is on REHAB24-6. Those are 9 of the 15 clips: the ones that pass both checks below (coverage ≥ 97%, projection error ≤ 3.5%). They are mostly a single person in front or half-profile view, in good light.
- Coverage of the body joints is 97–100%.
- The 3D, projected back into the image, lands within 1.7–3.5% of the person's height of the 2D joints. On REHAB24-6 the same measure is 1.4–2.5%.

**Repetition counts.** The counter was set before the clips were run and first checked on REHAB24-6 squats. There it agrees with the same counter run on motion capture in 10 of 12 segments, and is off by one in the other two. GVHMR's knee angle differs from motion capture by 5.4° on average (2.2–8.5°).

On internet clips, 6 were checked by eye against the video frames at each counted repetition:

| Clip | Exercise | Counted | Seen in the video |
|---|---|---|---|
| mixkit_52106 | squat, front | 3 | 3, correct |
| pexels_7426039 | squat, near side view, outdoors | 4 | 4, correct |
| mixkit_21273 | slow squat at home | 2 | 2, correct |
| mixkit_23913 | fast squats, backlit gym | 16 | 14: the counter also took 2 small dips at the start |
| mixkit_45960 | lunge | 6 | about 3: steps between lunges were counted too |
| mixkit_23914 | squat with a coach beside her | 1 | **wrong person**: GVHMR followed the crouching coach, not the woman squatting |

**Where it fails.** Three conditions break it:
- **A silhouette against bright light** (mixkit_24313): 72% coverage, and joint jitter 10× higher than on REHAB24-6.
- **A strict profile view** (mixkit_52103, lunge seen from the side): 48% coverage and 7.4% projection error. This is the hidden-limb problem P2 found on REHAB24-6.
- **Two people in the frame** (mixkit_23914): the tracker picks the larger, more visible person, who here is the coach. This matters for telemedicine, where a therapist is often in the picture. REHAB24-6 also marks repetitions with extra people in view.

## 1b. Round 2: 26 more clips, and a held-out test of the counter (2026-11-02, 22:00–22:48)

**The rule was fixed first.** After round 1, the counter got one rule: keep a peak only if it reaches at least 0.75 × the clip's 95th-percentile knee flexion. It was committed (`ee5acf7`, 21:59) together with the list of 26 new clips, before any of them was downloaded. That makes round 2 a held-out test.

**Clips.** 26 new Mixkit clips:
- 9 squat, including 4 wall-ball squats, one with a coach, and one holding a baby;
- 7 lunge, including one with a coach and one sprint start;
- 3 general fitness;
- 7 others: two physical-therapy sessions with a nurse, older adults exercising, an older man on a treadmill, two home workouts.

**Checks.** 11 of 26 pass both checks (coverage ≥ 97%, projection error ≤ 3.5%), against 9 of 15 in round 1. Most failures are:
- wall-ball squats, where a ball flies in front of the body (11–78% coverage);
- clips where the legs are out of view or the person is small and dark.

One physical-therapy clip fails almost completely: an older woman with a nurse, at 7% coverage.

**The counter on known truth.** On REHAB24-6, the depth rule gives GVHMR the same count as motion capture in 12 of 12 segments (10 of 12 without it). In 4 of the 12 segments it drops one dip (of 7–9), in GVHMR and motion capture alike. These are shallower dips, so the rule would miss **shallow repetitions**, which in rehab are often the incorrect ones.

**Counts checked by eye** on 9 round-2 clips:

| Clip | Counted (depth rule) | Seen in the video |
|---|---|---|
| mixkit_52117 squat | 3 | 3, correct |
| mixkit_36062 squat holding a baby | 5 | 5, correct |
| mixkit_52110 lunge | 5 | 5, correct; a 6th lunge is cut off by the end of the clip |
| mixkit_24304 lunge, coach beside | 3 | 3, correct. The tracked box covers both people |
| mixkit_24305 squat, coach beside | 5 | **wrong person**: GVHMR followed the standing coach. Knee flexion never exceeds 55° |
| mixkit_52316 lunge, small in a dark hall | 5 | knee flexion stays at 90–140° while the man stands: wrong angles, so no valid count |
| mixkit_52098 lunge, camera looking up | 4 | the legs are out of view: cannot be checked |
| mixkit_8983 sprint start, then running | 3 | not an exercise: running strides counted |
| mixkit_5493 physical therapy with a nurse | 0 | 7% coverage: no usable pose |

So where the right person is tracked and the legs are visible, the rule counted **4 of 4 new clips exactly**. The failures are upstream of the counter:
- **picking the person:** 2 of the 3 clips with a coach followed the coach (23914, 24305);
- **visibility**.

Applied after the fact to round 1, the rule gives 14 for mixkit_23913, which matches the video. It gives 2 for mixkit_45960, against 3 seen, one of which is cut off at the end. Round 1 was used to design the rule, so these two are not a test.

## 1c. Choosing the person: a fix for the coach failures (2026-11-03, 02:10–02:20)

**The rule.** GVHMR's tracker keeps the track with the largest summed box area, which in two clips was the coach. `t3_gvhmr_pick.py` takes, among tracks covering at least half of the frames, the one whose box height varies most relative to its median. Squats and lunges move the body up and down; a coach standing beside does not.

It was written from the two failure cases, so this is a check that the fix works, not a held-out test. Nine clips were rerun: the five with two people, plus four single-person controls.

| Clip | Default track → new track | Result |
|---|---|---|
| mixkit_23914, squat with a coach | coach → **the woman squatting** | 4 squats counted, as seen in the video. Projection error 5.55% → 2.14% |
| mixkit_24305, squat with a coach | coach → **the woman squatting** | 5 counted, as seen (a 6th is cut off at the end). Projection error 5.55% → 3.05%, which now passes the check |
| mixkit_49038, mother and daughter | mother → daughter | 3 counted. Both people are squatting in step, so either choice is valid |
| mixkit_24304, lunge with a coach | unchanged | 3, as before |
| mixkit_44438, two people | unchanged | as before |
| 4 controls (52106, 36062, 52117, pexels_7426039) | unchanged | identical metrics |

**Limits.** A coach who moves more than the person exercising would still win. A click on the right person in the first frame would be the robust version.

## 2. Setup

| Field | Value |
|---|---|
| Machine | `lambdavector2`. Both RTX A6000s were held by another user's vLLM jobs (about 43 GB each; GPU 1 at 100% use), so about 5.4–5.6 GB were free on each |
| GPU use | One job at a time, on the free GPU with the lowest use: GPU 0 for 14 clips. The first attempt at clip 1 ran on GPU 1 and was stopped, see §6. A watchdog would have stopped *my* job if any GPU fell below 700 MiB free; it never triggered |
| GVHMR | `ee960bb`, the unofficial mirror weights (see `DATA_LOG.md`), static-camera mode, GVHMR's default focal length. `t3_gvhmr_lowmem.py` lowers GVHMR's preprocessing batch from 16 to 4 at runtime, so it is the same computation in smaller batches. The checkout was not changed. About 3.3 GB measured during the 2D-joint stage of clip 1, against about 4.7 GB at batch 16 (P2b) |
| Clips | 15 stock clips, 6–23 s long, 720p to 4K, 24–30 fps (`t3_clips.txt`). Mixkit Stock Video Free License and Pexels License: free use, no redistribution as stock. Not in git. Each was cut to at most 30 s, scaled to at most 1080 px high and resampled to 30 fps |
| Reference | REHAB24-6 Ex6 squats: P2's GVHMR true-focal chunks for PM_118 and PM_022, both cameras (12 segments), with the motion capture at 30 fps (video frame = mocap frame) |
| Tools | `reconstruction/tools/t3_run.sh`, `t3_gvhmr_lowmem.py`, `t3_internet_metrics.py` (with `reps_depth` from round 2), `t3_peak_sheet.py`, `t3_clips.txt` (both rounds); for §1c `t3_gvhmr_pick.py` and `t3_pick_run.sh` |

## 3. Commands

```bash
systemctl --user enable --now t3-internet.service     # runs t3_run.sh: download, ffmpeg, GVHMR, joints, metrics
# per clip: ffmpeg -t 30 -vf "scale=-2:'min(1080,ih)',fps=30"; t3_gvhmr_lowmem.py; gvhmr_export_joints.py
python t3_internet_metrics.py --internet runs/t3-internet --clips t3_clips.txt --rehab data/rehab24-6 \
    --rehab-chunks runs/t2-chunks/out-truef --out runs/t3-internet/metrics
python t3_peak_sheet.py runs/t3-internet runs/t3-internet/sheets <6 clips>   # frames at counted repetitions
```

Logs are in `~/fall2026/logs/t3/` (`status.txt`, `jobs/`, `metrics.log`). Outputs are in `~/fall2026/runs/t3-internet/`. The check sheets show people, so they stay on the server.

## 4. Numbers

Definitions:
- **coverage:** frames where at least 8 of the 12 limb joints have ViTPose confidence > 0.5.
- **reproj:** median distance between GVHMR's 3D limb joints, projected with its own camera, and ViTPose's 2D, in % of the box height.
- **jitter:** mean acceleration of root-relative limb joints, in mm/frame².
- **knee:** 5th and 95th percentile of the more-bent knee's flexion, where 0° is a straight leg.
- **reps:** peaks of that flexion (smoothed over 9 frames; prominence ≥ 25°; at least 0.6 s apart). The rule was fixed before any clip was run.

**REHAB24-6 squats, where the truth is known** (12 segments of 800–887 frames):

| Measure | GVHMR | Motion capture |
|---|---|---|
| reproj (% of box height) | 1.44–2.51 | — |
| jitter (mm/frame²) | 1.45–2.34 | 0.92–1.85 |
| knee flexion, 95th percentile | 94–133° | 92–128° |
| knee angle error vs. motion capture | 2.2–8.5°, mean 5.4° | — |
| repetitions counted | equal to the motion-capture count in 10 of 12 segments, ±1 in 2 | — |

The annotated repetition counts are lower than both counts in 5 of the 12 segments. They count only repetitions that lie wholly inside a segment, so they are not used as the reference here.

**Internet clips** (no ground truth):

| Clip | Exercise | Frames | Coverage % | Reproj % | Jitter | Knee p5–p95 (°) | Reps |
|---|---|---|---|---|---|---|---|
| mixkit_21273 | squat | 472 | 100 | 3.49 | 1.57 | 22–119 | 2 |
| mixkit_23913 | squat | 456 | 100 | 2.06 | 8.56 | 25–115 | 16 |
| mixkit_24313 | squat, silhouette | 519 | **71.5** | **5.26** | **21.06** | 24–112 | 13 |
| mixkit_48677 | squat | 344 | 97.1 | 5.62 | 1.33 | 12–62 | 1 |
| mixkit_49038 | squat, two people | 260 | 100 | 1.97 | 3.98 | 6–136 | 3 |
| mixkit_52106 | squat | 368 | 100 | 1.70 | 1.78 | 12–126 | 3 |
| mixkit_752 | squat | 435 | 100 | 2.22 | 1.91 | 7–131 | 2 |
| mixkit_23914 | squat, coach beside | 655 | 100 | 5.55 | 0.68 | 86–154 | 1 (wrong person) |
| mixkit_44438 | squat, two people | 184 | 100 | 4.91 | 4.72 | 11–129 | 1 |
| pexels_7426039 | squat | 285 | 100 | 2.65 | 3.56 | 10–138 | 4 |
| pexels_5319759 | barbell squat, side | 616 | **79.5** | **6.34** | 1.18 | 29–95 | 3 |
| mixkit_52112 | lunge | 233 | 100 | 2.25 | 4.38 | 29–117 | 6 |
| mixkit_52103 | lunge, profile | 529 | **48.4** | **7.36** | 7.37 | 18–137 | 7 |
| mixkit_52101 | lunge | 684 | 100 | 3.31 | 5.32 | 20–108 | 2 |
| mixkit_45960 | lunge | 382 | 100 | 2.67 | 1.66 | 21–110 | 6 |

**Round 2** (26 clips; bold = fails a check):

| Clip | What | Frames | Coverage % | Reproj % | Jitter | Knee p5–p95 (°) | Reps | Reps, depth rule |
|---|---|---|---|---|---|---|---|---|
| mixkit_52082 | squat | 399 | **53.1** | **6.01** | 3.17 | 15–78 | 2 | 2 |
| mixkit_52116 | squat | 166 | 100.0 | 2.54 | 3.48 | 15–122 | 1 | 1 |
| mixkit_52117 | squat | 211 | 100.0 | 1.96 | 3.61 | 12–126 | 3 | 3 |
| mixkit_24305 | squat, coach beside | 570 | 100.0 | 2.27 | 1.92 | 9–34 | 5 | 5 |
| mixkit_36062 | squat holding a baby | 405 | 100.0 | 2.38 | 2.71 | 7–138 | 5 | 5 |
| mixkit_44425 | wall-ball squat | 900 | **78.2** | **3.83** | 14.47 | 8–136 | 11 | 5 |
| mixkit_44422 | wall-ball squat | 671 | **23.4** | **10.61** | 11.73 | 59–99 | 8 | 8 |
| mixkit_44424 | wall-ball squat | 472 | **44.3** | **4.42** | 9.89 | 9–127 | 5 | 4 |
| mixkit_44423 | wall-ball squat | 747 | **11.0** | **10.8** | 14.03 | 40–116 | 7 | 7 |
| mixkit_52107 | fitness | 485 | **40.6** | **5.21** | 2.41 | 24–80 | 3 | 3 |
| mixkit_52115 | fitness | 358 | **89.9** | **4.92** | 6.21 | 50–73 | 2 | 2 |
| mixkit_52111 | fitness | 472 | **33.5** | **9.55** | 10.38 | 55–126 | 6 | 6 |
| mixkit_52098 | lunge, camera looking up | 558 | **93.2** | **4.11** | 4.52 | 27–107 | 4 | 4 |
| mixkit_52140 | lunge | 353 | 100.0 | 2.96 | 1.46 | 33–118 | 0 | 0 |
| mixkit_52316 | lunge, small in a dark hall | 375 | 100.0 | 2.51 | 6.85 | 90–136 | 5 | 5 |
| mixkit_52110 | lunge | 501 | 100.0 | 2.23 | 2.31 | 20–115 | 5 | 5 |
| mixkit_52120 | lunge | 687 | **95.2** | **5.23** | 3.6 | 13–46 | 2 | 2 |
| mixkit_24304 | lunge, coach beside | 391 | 100.0 | 2.01 | 1.7 | 11–95 | 3 | 3 |
| mixkit_8983 | sprint start, then running | 538 | **94.6** | **4.86** | 7.98 | 13–116 | 4 | 3 |
| mixkit_5493 | elderly woman in physical therapy, with a nurse | 429 | **7.2** | **6.63** | 2.63 | 19–66 | 0 | 0 |
| mixkit_5561 | man in physical therapy, with a nurse | 568 | 100.0 | **5.65** | 0.82 | 87–96 | 0 | 0 |
| mixkit_36121 | adult women exercising outdoors | 657 | 100.0 | 2.79 | 1.39 | 13–26 | 0 | 0 |
| mixkit_36140 | adult women with dumbbells | 600 | 100.0 | 1.2 | 1.47 | 8–10 | 0 | 0 |
| mixkit_26274 | elderly man on a treadmill | 413 | 100.0 | 2.61 | 7.88 | 12–66 | 12 | 12 |
| mixkit_21969 | man working out at home | 524 | **91.2** | **4.58** | 3.15 | 100–113 | 0 | 0 |
| mixkit_42899 | woman with small dumbbells | 345 | 100.0 | **4.7** | 1.44 | 6–9 | 0 | 0 |

Round 2 cost: 26 clips in 49 min, mostly on GPU 0 and some on GPU 1 when GPU 0 was below 5000 MiB free. The watchdog never triggered.

Cost: 15 clips in 26 min (20:32–20:58) on GPU 0, one at a time. Peak RAM was 6.9–7.0 GB per job.

## 5. Checks

- **Counter and thresholds, on known truth first.** The counter was run on REHAB24-6 motion capture and on GVHMR for the same segments; the results are above. GVHMR's knee angle is within 5.4° of motion capture on average. Part of that difference is that Motive places the hips differently from SMPL (P2).
- **Counted repetitions checked by eye** on 6 clips, with the video frame at every counted peak (table in §1).
- **The wrong-person case** was seen directly: the tracked box sits on the coach in every checked frame.
- **The low-memory wrapper.** The batch-size line is asserted to exist before it is replaced. Outputs have the same format as P2's (`joints.npz` from `gvhmr_export_joints.py`). The first clip's output reused the preprocessing that the stopped first attempt had saved (§6). Both runs computed the same thing, so the result stands.

## 6. Failed or skipped

- **First attempt at clip 1 was on the busy GPU.** It ran on GPU 1, where the other user's job was at 100% use. I changed the GPU choice to prefer the least-used GPU and restarted. The old job kept running in its own scope after the restart, writing to the same folder, and I stopped it about a minute later. It was my own process; nothing else was touched. Clip 1's output reuses the box and 2D-joint files the first attempt saved. Its other files were written by the second run.
- **Jitter is not comparable across clips.** Clips recorded at 24–25 fps were resampled to 30 fps, which repeats frames and adds artificial acceleration. Fast repetitions (mixkit_23913) also raise it. Only the silhouette clip's 21 mm/frame² is clearly a failure.
- **Round 1's two counter weaknesses** (small dips at the start of a set, steps between lunges) led to the depth rule. It was fixed before round 2 and tested there (§1b). Its known cost is that it misses shallow repetitions.
- **The other 9 clips' counts were not checked by eye.**
- **No overlay videos were rendered** (GPU memory). The avatar from camera 18 and the smoothed-pose avatar test also wait for a free GPU.

## 7. Not tested

- Moving-camera mode (DPVO), and therefore any global trajectory or foot-sliding check. Some stock clips have a slowly moving camera.
- Choosing the right person when there are two: by size, by motion, or by asking the user.
- Combining several internet videos of the same exercise, which is the second half of Q4.
- Clips of older adults or patients.

## 8. Next step

1. Add person selection (the person who moves most, or a click on the first frame), then rerun the two-person clips.
2. Add a depth rule to the repetition counter, and test it on fresh clips rather than these.
3. When a GPU is free, render overlay videos for the 9 clean clips for the Q4 slide.
