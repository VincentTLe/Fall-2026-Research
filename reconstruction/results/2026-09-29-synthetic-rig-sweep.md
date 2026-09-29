# Synthetic rig sweep: camera layout, count, 2D noise and sync (2026-09-29)

**What this is:** EasyMocap's own multi-view triangulation (`apps/demo/mv1p.py`, commit `e6006fd`) run on synthetic 2D keypoints, where the true 3D joints are known exactly. **What it isn't:** a measurement on real video. No detector, no SMPL fitting, no real people. Use it to **compare options**, not as a prediction of our final accuracy.

Reproduce (about 1 minute on 4 CPU cores, no GPU or downloads):
```bash
EASYMOCAP=/path/to/EasyMocap bash reconstruction/tools/rig_sweep.sh /tmp/rig_sweep
```

## Setup

| Parameter | Value |
|---|---|
| Cameras | 1920×1080, 85° horizontal FOV, no lens distortion, 3 m from the subject, 1.6 m high, aimed at 0.9 m |
| Motion | squat 0.35 m deep with forward arm raise, 2 s per rep (1 s for the "fast" rows), turning 20°/s; 120 frames at 60 fps |
| 2D keypoints | true projection + Gaussian noise (σ = 3 px unless stated), confidence 0.9 |
| Occlusion | four body capsules (spine, both flanks, head) hide joints behind the body; limbs don't hide each other |
| Metric | MPJPE = mean 3D joint error vs. truth (mm). "Joints reconstructed" = share of true joints EasyMocap output |

## Results

| Variant | MPJPE (mm) | Root-rel. MPJPE (mm) | Joints reconstructed | Frames missing |
|---|---|---|---|---|
| 5 cams, even 360°, 3 px noise (**baseline**) | 8.9 | 15.0 | 100.0% | 0 |
| 5 cams, even 360°, no occlusion model | 7.9 | 10.8 | 100.0% | 0 |
| 5 cams, front 180° | 8.8 | 16.6 | 100.0% | 0 |
| 5 cams, clustered 90° | 9.8 | 15.7 | 100.0% | 0 |
| 4 cams, even 360° | 10.9 | 23.4 | 100.0% | 0 |
| 3 cams, even 360° | 11.7 | 17.1 | 98.5% | 0 |
| 6 cams, even 360° | 8.1 | 15.8 | 100.0% | 0 |
| 5 cams, even, 8 px noise | 23.7 | 40.1 | 100.0% | 0 |
| 5 cams, even, 15% of 2D joints randomly missing | 10.1 | 16.9 | 97.6% | 0 |
| 5 cams, heights 0.6 / 1.6 / 2.4 m mixed | 9.0 | 14.9 | 100.0% | 0 |
| 5 cams, even, cam 03 lags 1 frame | 9.0 | 15.0 | 100.0% | 0 |
| 5 cams, even, cam 03 lags 2 frames | 9.3 | 15.7 | 100.0% | 0 |
| 5 cams, even, cam 03 lags 4 frames | 10.3 | 17.4 | 100.0% | 0 |
| 5 cams, even, cam 03 lags 8 frames | 13.2 | 21.3 | 100.0% | 0 |
| 5 cams, fast squat (1 s), cam 03 lags 2 | 10.3 | 17.3 | 100.0% | 0 |
| 5 cams, fast squat (1 s), cam 03 lags 4 | 13.0 | 21.0 | 100.0% | 0 |

Root-relative MPJPE is larger than MPJPE because it also absorbs the error of the mid-hip joint, which side cameras can't see (13–22 mm on its own). For a calibrated rig, **MPJPE is the number to use**.

## What it suggests

1. **2D keypoint quality matters most.** Going from 3 px to 8 px of 2D error makes 3D error 2.7× worse (8.9 → 23.7 mm). No layout change came close to that. For the other teams, that means:
   - **Hardware:** make the subject fill the frame (distance / field of view) and light the space well.
   - **Software:** fast shutter so frames are sharp, and the highest resolution that's practical.
2. **Camera count:** 3 → 4 → 5 → 6 cameras gives 11.7 → 10.9 → 8.9 → 8.1 mm. Five is a reasonable point; with three, some joints (1.5%) are seen by fewer than two cameras and get dropped.
3. **Sync (60 fps):** one camera 1–2 frames late costs ≤ 0.4 mm for a 2 s squat. At 4 frames it costs 1.4 mm (4.1 mm for a 1 s squat), and at 8 frames 4.3 mm. **Proposed target for Danny/Katie: residual offset ≤ 1 frame (≈17 ms at 60 fps).** Up to 2 frames is tolerable for slow rehab-speed motion. Faster motion (walking arm swing, jumps) needs tighter sync, roughly in proportion to speed.
4. **Layout (spread, heights):** in this model, 360° vs front-180° vs mixed heights barely differ (8.8–9.0 mm), and a 90° cluster is ~10% worse. **Don't conclude "front-only is fine."** Three things the model leaves out all favour cameras all around the subject and at mixed heights:
   - limbs hiding each other
   - 2D detectors doing worse on back and side views
   - the subject turning fully around

   This has to be checked on real multi-view data (Week 2–3 plan in `README.md`, using the all-view reference).

## Next steps

- When hardware has a candidate layout, run it directly: `synth_rig.py OUT --positions "x,y,z; ..."` (meters, z up, subject at the origin), then the same `mv1p.py` + `compare_keypoints3d.py` steps.
- Repeat the camera-count and layout comparison on real multi-view video once a sample dataset and detector weights are available (`SETUP.md` §6a).
- Measure our real 2D keypoint error: compare detector output to a few hand-labelled frames. That tells us which noise row of this table we're actually in.
