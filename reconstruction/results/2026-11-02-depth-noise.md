# Would smoothing fix one-camera depth? GVHMR's depth error on REHAB24-6, 2026-11-02

This was a laptop-session analysis on the GPU server: CPU only, existing outputs, no server Claude session. It answers T4 §8's open question, whether GVHMR's poses can be fixed by smoothing them over time before building an avatar.

## 1. Result

No. GVHMR's error in how far the person is from the camera is almost all a constant offset plus a slow drift. Frame-to-frame jitter is a small part of it.

| What | RMS over 63,594 frames |
|---|---|
| Depth error of the mid-hip | 177 mm |
| Without the constant offset per 30-second segment | 72 mm |
| Of that, the slow drift (longer than 2 s) | 69 mm |
| Of that, the fast jitter | 17 mm |
| After smoothing GVHMR's depth over 0.5 s | still 72 mm |

The avatar's recording (PM_037, camera 17) behaves the same way: 105 mm before smoothing and 105 mm after.

So the avatar's 9.5 dB pose gap (T4) cannot be closed by temporal smoothing. Depth has to come from somewhere else: the second camera (T2: 43 mm with no alignment), or a per-recording depth or scale correction.

This also revises the reading in T4 §1. There, training moved each pose by about 10 cm "differently in every frame". Measured against motion capture, GVHMR's own depth error is mostly slow, not frame-to-frame.

## 2. Setup

| Field | Value |
|---|---|
| Data | REHAB24-6, P2's 12 recordings, both cameras: P2's GVHMR true-focal chunks (`runs/t2-chunks/out-truef`, 72 chunks, all frames of the repetition spans). Motion capture at 30 fps, where video frame = mocap frame |
| Cameras | the authors' new estimate (`runs/t2b/calib-new`), used to put the motion capture in each camera's frame |
| Root | mid-hip: SMPL joints 1/2 for GVHMR, Motive 16/21 for motion capture. Their fixed difference (P2: 9–13 cm) is part of the constant offset |
| Split | per chunk: offset = mean error; slow = Savitzky–Golay fit over 61 frames minus the offset; fast = the rest. The smoother tested is Savitzky–Golay over 15 frames on GVHMR's depth |
| Tool | `reconstruction/tools/t3_depth_noise.py` |

## 4. Numbers

| Set | Frames | Error | No offset | Slow | Fast | Smoothed | Motion capture's own fast part |
|---|---|---|---|---|---|---|---|
| Camera 17 | 31,797 | 202 | 74 | 70 | 21 | 73 | 7 |
| Camera 18 | 31,797 | 148 | 70 | 68 | 12 | 70 | 7 |
| Both | 63,594 | 177 | 72 | 69 | 17 | 72 | 7 |
| PM_037 camera 17 (the avatar) | 1,942 | 359 | 105 | 101 | 25 | 105 | 11 |
| PM_037 camera 18 | 1,942 | 240 | 90 | 89 | 14 | 90 | 9 |

All values are RMS in mm. The "fast" part of the motion capture itself (7 mm) shows how much real motion a 2-second trend removes. GVHMR's fast part is only about 10 mm above that.

## 5. Checks

- The total RMS (177 mm) is close to P2's unaligned errors (abs 160–197 mm), as expected, since depth dominates abs.
- Motion capture's own fast part is small (7–11 mm), so the 2-second split does not remove real motion as "jitter".

## 6. Failed or skipped

- Only depth of the root. Joint-level jitter was not split this way.
- The smoother is one fixed window. A longer one would start to remove the drift but also real movement; this was not tuned.
