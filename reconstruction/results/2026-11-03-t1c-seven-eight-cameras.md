# T1c: 7 and 8 cameras on CMU Panoptic, 2026-11-03

Follow-up to `2026-10-27-t1-multiview.md`, which tested 3–6 cameras and all 31. The team's rig has **8 GoPros**, so 7 and 8 were added. This run was done from the laptop session, because the server Claude account had hit its weekly limit.

## 1. Result

| Layout | MPJPE, 10 limb joints | Note |
|---|---|---|
| All 31 | 11.7 mm | T1 |
| **8 spread, mean of 3 sets** | **14.8 mm** | 14.0 / 15.8 / 14.7 |
| 7 spread, mean of 3 sets | 15.1 mm | 16.7 / 14.3 / 14.2 |
| 6 spread | 16.8 mm | T1 |
| 5 spread | 16.8 mm | T1 |
| **8 bunched on one side** (157° arc) | **18.4 mm** | worse than 5 spread |
| 5 bunched on one side | 24.6–26.0 mm | T1 |

No joints were missing in any 7- or 8-camera layout.

- **Eight spread cameras** are about 12% better than five or six (14.8 vs. 16.8 mm). That still leaves a gap to the full dome (11.7 mm).
- **Placement still matters more than count.** Eight cameras over half a circle (18.4 mm) are worse than five cameras around the person (16.8 mm).

For the team's rig: spread the 8 GoPros all around the person.

## 2. Setup

Everything is as in T1:
- `171204_pose1`, HD frames 650–1041 (392 frames);
- the cached HRNet 2D of all 31 cameras, so no re-detection;
- EasyMocap `e6006fd` iterative triangulation, no SMPL;
- scored against the dome's own 3D, on the same 10 limb joints, with `panoptic_to_easymocap.py eval --frames 392 --no-ankles`.

Runs were on CPU, with `nice -n 19`. Both GPUs were held by another user's job.

**Layouts.** These follow T1's rule over the 15 mid-height cameras (0.5–1.5 m above the hip). A has the smallest largest azimuth gap. B and C are the next best, each differing from A by at least 2 cameras.

| Name | Cameras (in azimuth order) | Largest gap |
|---|---|---|
| s7 (A) | 18 22 05 21 29 08 16 | 67° |
| s7_alt1 | 03 12 20 00 11 27 23 | 68° |
| s7_alt2 | 03 12 20 00 11 27 30 | 68° |
| s8 (A) | 03 12 20 00 29 08 16 30 | 50° |
| s8_alt1 | 03 12 05 21 29 08 16 30 | 55° |
| s8_alt2 | 03 22 05 21 29 08 16 30 | 55° |
| c8bunch | 11 08 27 16 23 30 18 03, the 8 consecutive mid-height cameras spanning the narrowest arc | span 157° |

The picks came from a short script, `reconstruction/tools/t1c_pick.py`, run on the laptop. The runs used `reconstruction/tools/t1c_run.sh` (run from `~/fall2026/logs/`), which loops T1's `make_variant.py → run_mv1p.py --use-annots --device cpu → eval`. The log is `~/fall2026/logs/2026-11-03-t1c.log`.

## 4. Numbers

| Layout | MPJPE (10 j.) | PA-MPJPE (10 j.) | MPJPE (12 j., with ankles) | Joints missing |
|---|---|---|---|---|
| s7 | 16.7 | 14.7 | 19.2 | 0.0% |
| s7_alt1 | 14.3 | 13.0 | 15.1 | 0.0% |
| s7_alt2 | 14.2 | 13.0 | 15.9 | 0.0% |
| s8 | 14.0 | 12.6 | 14.8 | 0.0% |
| s8_alt1 | 15.8 | 13.9 | 16.6 | 0.0% |
| s8_alt2 | 14.7 | 13.3 | 16.1 | 0.0% |
| c8bunch | 18.4 | 16.0 | 17.8 | 0.0% |

## 5. Checks

- Same code, 2D and frames as T1. The T1 rows are T1's own numbers.
- The ranking rule matches T1's description. The A sets have the smallest largest gap: 67° for 7 cameras, 50° for 8.

## 6. Failed or skipped

- No SMPL fit. T1b showed that EasyMocap's default fit makes every layout worse.
- One sequence, one slow-moving person, and the dome reference rather than motion capture, as in T1.
