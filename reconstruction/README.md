# Reconstruction workstream (Tan)

Goal for the semester: **synchronized multi-view GoPro video → calibrated, reproducible 3D human motion (SMPL)**, using EasyMocap, with setup docs that anyone on the team can follow.

| File | What it is |
|---|---|
| `README.md` (this) | Plan, timeline, interfaces with other teams, risks |
| [`SETUP.md`](SETUP.md) | Step-by-step EasyMocap setup + run guide (fill in "verified" notes as you go) |
| [`READING.md`](READING.md) | Reading list: ZJU core works + Prof. Cao's extra directions, with what to pull from each |
| [`tools/`](tools/) | Small helper scripts (ZJU-MoCap converter, view/sync ablation, 3D keypoint comparison) |

---

## 1. Where this workstream sits

```
 Hardware (Ahsan, Huy, Nghiem) ──► rig + 5 GoPros, camera IDs 01..05
 Software (Danny, Katie)       ──► synced videos: videos/01.mp4 ... 05.mp4 + measured offsets
 Calibration (Seishin)         ──► intri.yml + extri.yml (EasyMocap format)
                                              │
 Reconstruction (Tan)          ◄──────────────┘
   extract frames → 2D keypoints → triangulate → fit SMPL → keypoints3d/*.json, smpl/*.json
                                              │
                                              ▼
   downstream (later): avatars, XR rendering, motion↔language, rehab assessment
```

You are the **consumer of everyone else's output**. So besides running EasyMocap, your most useful job early on is to tell the other teams exactly what the pipeline needs (Section 4), and to back up hardware decisions with numbers (Section 3, Week 2–3 experiments).

---

## 2. Deliverables (definition of done)

1. **EasyMocap runs end-to-end on ZJU data**: multi-view video → `keypoints3d/` + `smpl/` + rendered overlay video, from a clean machine, following `SETUP.md` only.
2. **`SETUP.md` verified by a second person** (e.g., Seishin, who needs the same environment for calibration).
3. **5-view experiment report**: how much accuracy we lose going from 23 → 5 cameras, and which 5-camera layout works best. Hardware uses this to pick the layout.
4. **Sync tolerance number**: how many frames of offset between cameras the pipeline tolerates. Software uses this as their target.
5. **First end-to-end run on our own 5 GoPros** (existing cameras, before the rig is built) using Seishin's calibration.
6. Short notes on the core ZJU papers (see `READING.md`) and what to present to Prof. Cao.

---

## 3. Timeline (starting Tue Sep 29, 2026)

Meetings are Tuesdays at 12:30. Each week ends with something to show at the next meeting.

### Week 1: Sep 29 – Oct 4: Access, environment, reading
- [ ] **Today:** sign the ZJU-MoCap agreement and email it (see `SETUP.md` §1). Approval can take days to weeks, so start it first.
- [ ] Register on the SMPL / SMPLify (neutral model) / SMPL-X sites and download the models.
- [ ] Get GPU access (lab machine, or ask Prof. Cao which server). Record the GPU, CUDA driver and OS in `SETUP.md`.
- [ ] Create the conda env and install EasyMocap (`SETUP.md` §2–3). Write down every error you hit and how you fixed it.
- [ ] Read EasyMocap docs + **Neural Body** (introduces ZJU-MoCap) + **MVPose** (the multi-view matching/triangulation idea).
- **Show on Oct 6:** env installed, screenshot of `python -c "import easymocap"`, a paragraph on how the pipeline works.

### Week 2: Oct 5 – 11: Run the demo, understand the code
- [ ] Download the public sample `zju-ls-feng` (23 calibrated cameras, 800 frames; no agreement needed).
- [ ] Run the v0.1 pipeline (`apps/demo/mv1p.py`) → SMPL. Then run the v0.2 pipeline (`emc ... detect_triangulate_fitSMPL.yml`).
- [ ] Trace the code: `extract_video.py` → 2D keypoints (`annots/`) → triangulation → SMPL fitting. Write down which function does each step (for the team doc).
- [ ] Start the **camera-count experiment** on the sample data (`tools/make_variant.py` + `tools/compare_keypoints3d.py`, see `SETUP.md` §6):
  - reference = all 23 views
  - variants = several 5-view subsets (spread evenly around vs. front half vs. clustered)
- **Show on Oct 13:** rendered SMPL overlay video + first table of "5 views vs 23 views" error.

### Week 3: Oct 12 – 18: Full ZJU-MoCap + decisions for hardware
- [ ] If the agreement is approved: convert one LightStage sequence (e.g. 313 or 377) with `tools/zju_to_easymocap.py` and run the pipeline.
- [ ] Finish the camera-count experiment: also try 4 vs 5 vs 6 views, if hardware is still deciding how many cameras to buy.
- [ ] **Sync-tolerance experiment:** shift one view by 1, 2, 4 frames (`make_variant.py --shift`) and measure the error. Send the result to Danny/Katie.
- [ ] Meet Seishin: agree on the calibration file format (Section 4). Run `check_calib.py` on a calibration they produce.
- **Show on Oct 20:** recommendation for hardware (layout + count) and software (max sync offset).

### Week 4: Oct 19 – 25: First run on OUR cameras
- [ ] Get a short test recording of one person walking/squatting from the 5 existing GoPros (Danny/Katie) + calibration (Seishin).
- [ ] Run the full pipeline. Expect problems (lens distortion, sync, resolution mismatch). Log each one in `SETUP.md` → "Our data".
- [ ] Check reprojection overlays (`--vis_repro`) per camera: which camera is worst and why?

### Week 5: Oct 26 – Nov 1: Hardening + docs
- [ ] Turn the working commands into one script: `run_session.sh <session_dir>` (extract → keypoints → fit → render).
- [ ] Someone else follows `SETUP.md` on a fresh machine; fix every gap they hit.
- [ ] Decide the output format we keep per session (Section 5).

### November onward: Real rig + downstream
- [ ] Process the first captures from the built rig; add a quality check per session (mean reprojection error, % frames with a fitted SMPL).
- [ ] Pick **one** of Prof. Cao's directions to prototype on our data (suggestion in `READING.md`: rehab exercise assessment or motion→language, because they only need SMPL/joints we already produce).

---

## 4. Interface contract with the other teams (agree on this early)

Put this in front of Seishin, Danny and Katie in Week 1–3. Every item here is something that silently breaks reconstruction if it's wrong.

**Folder layout per capture session** (what EasyMocap reads):
```
<session>/
├── intri.yml          # from Seishin
├── extri.yml          # from Seishin
└── videos/
    ├── 01.mp4         # file name == camera ID == ID on the physical label
    ├── 02.mp4
    └── ... 05.mp4
```

**Calibration (Seishin):**
- EasyMocap format (OpenCV YAML): `intri.yml` has `names`, `K_<id>` (3×3), `dist_<id>` (1×5); `extri.yml` has `names`, `R_<id>` (3×1 Rodrigues), `Rot_<id>` (3×3), `T_<id>` (3×1). Camera names must match the video file names (`01`, `02`, …).
- **Units:** `T` in **meters** (set the chessboard `--grid` in meters, e.g. `0.1` for 10 cm squares).
- **Same resolution** for calibration videos and capture videos. If any setting changes (resolution, lens mode, zoom), re-calibrate the intrinsics.
- Extrinsics must be re-done whenever any camera is moved/bumped (tape marks on the floor help, but re-check with `check_calib.py`).
- Ask for: the reprojection error printed by `calib_intri.py` per camera, and the cube-check images from `check_calib.py`.

**Camera settings (Danny, Katie):** these are the important ones for reconstruction
- **Lens: Linear** (not Wide/SuperView/HyperView). EasyMocap/OpenCV uses a pinhole + 5-coefficient distortion model; GoPro's wide modes are fisheye-like and calibrate poorly with it.
- **Stabilization (HyperSmooth): OFF.** Stabilization crops/warps each frame differently, so the intrinsics change from frame to frame and the calibration becomes invalid.
- Fixed resolution + fps on all cameras (e.g. 1080p or 2.7K @ 60 fps). Shutter fixed and fast enough to avoid motion blur (e.g. 1/480 s or faster at 60 fps, lighting permitting). White balance and ISO fixed.
- **Sync:** report the measured residual offset per camera, in frames. At 60 fps, 1 frame = 16.7 ms. My sync-tolerance experiment (Week 3) will say how much is acceptable.
- Rename downloaded files to `01.mp4` … `05.mp4` (by camera ID), and trim them to a common start frame (or give me the per-camera offsets and I'll trim).

**Hardware (Ahsan, Huy, Nghiem):**
- Every camera should see the **whole body** of the subject everywhere in the capture area (feet included; feet are often cut off).
- Spread cameras around the subject (a wide angle between neighbouring cameras gives better triangulation than cameras clustered on one side). Use the Week 2–3 experiment result.
- The calibration board must be visible to **all** cameras at once for extrinsics (at least overlapping pairs). A large board (≥ A1/A0 size) helps at room scale.

---

## 5. What we store per session (proposal)

```
<session>/
├── videos/, intri.yml, extri.yml     # raw inputs (never modified)
├── images/, annots/                  # derived, can be regenerated
└── output/
    ├── keypoints3d/000000.json       # BODY25 3D joints, meters, world frame of extri
    ├── smpl/000000.json              # EasyMocap SMPL params: Rh, Th, poses, shapes
    └── render.mp4                    # overlay for quick visual QA
```
Plus a `session.yml`: date, subject ID (anonymous), camera settings, calibration used, sync offsets, notes. Rehab data may involve human subjects, so **check IRB/consent requirements with Prof. Cao before recording anyone outside the team.**

---

## 6. Risks and what to do about them

| Risk | Early sign | Mitigation |
|---|---|---|
| ZJU-MoCap approval is slow | No reply after ~1 week | Everything in Weeks 1–3 can use the public `zju-ls-feng` sample instead |
| Install hell (old torch/chumpy/pyrender) | Import errors | Follow the gotchas in `SETUP.md` §3; pin versions; keep a log |
| No display on GPU server | pyrender crashes | `PYOPENGL_PLATFORM=egl` (or `osmesa`) |
| GoPro lens/stabilization breaks calibration | Large reprojection error, SMPL "floats" | Linear lens + stabilization OFF (Section 4) |
| Cameras out of sync | Jittery 3D, limbs doubled in fast motion | Sync-tolerance experiment → target for software team |
| 5 cameras not enough for occlusions | Missing joints / flipped limbs | Layout from Week 2–3 experiment; keep the subject centered; use `--robust3d`/smoothing |

---

## 7. Your weekly rhythm (suggestion)

You're not on the lab schedule, and most of this work runs on a GPU machine, so it can be done remotely. Two things do need people:
- **Tuesday (9am–2pm, meeting 12:30):** Seishin, Danny and Katie are all in. Use it for handoffs: calibration files, test recordings, the interface contract.
- **When Week 4's test recording happens:** be there in person, so you can check the videos on-site before everyone leaves.

Each Tuesday bring: (1) what ran, (2) one number or picture, (3) what you're blocked on and who unblocks it.
