# EasyMocap setup and run guide

A working log that should become the team's reproducible guide.
**Rule:** every step you actually run gets a ✅ plus the date and machine. Every error goes under "Problems hit" with its fix. Steps marked *(unverified)* come from the EasyMocap source and docs and haven't been run by us yet.

Sources: [EasyMocap repo](https://github.com/zju3dv/EasyMocap) (`doc/installation.md`, `doc/quickstart.md`, `apps/calibration/Readme.md`, `doc/02_output.md`) and the [public docs](https://chingswy.github.io/easymocap-public-doc/quickstart/quickstart.html).

---

## Source check: September 29, 2026

The [collected source notes](research/SOURCE_NOTES.md) verify upstream paths and distinguish the legacy and configurable workflows. The source-collection update itself ran nothing; the install and triangulation were verified afterwards (§2, §6b). Download links below are acquisition routes, not downloaded assets.

The current public quickstart uses `street_dance`; the legacy repo guide uses `zju-ls-feng`. CPU fallback exists in the current HRNet component, but end-to-end CPU feasibility and speed remain untested.

## 0. Machine log

| Field | Value |
|---|---|
| Machine / server | |
| OS | |
| GPU + VRAM | |
| NVIDIA driver / CUDA | (`nvidia-smi`) |
| Conda version | |
| EasyMocap commit | (`git rev-parse HEAD`) |

---

## 1. Accounts and downloads (start these first, since they take time)

1. **ZJU-MoCap dataset:** sign the [agreement PDF](https://pengsida.net/project_page_assets/files/ZJU-MoCap_Agreement.pdf) and email it to Qing Shuai (s_q@zju.edu.cn), cc Xiaowei Zhou (xwzhou@zju.edu.cn), to request the download link. Ask Prof. Cao whether the lab already has access, or whether the request should come from them.
2. **SMPL models** (free registration):
   - SMPL male/female v1.0.0: https://smpl.is.tue.mpg.de
   - SMPL neutral: https://smplify.is.tue.mpg.de
   - (Optional) SMPL-X: https://smpl-x.is.tue.mpg.de, and MANO/SMPL+H: https://mano.is.tue.mpg.de
   - Follow the terms of the exact model download and the lab's permitted use. **Don't commit `.pkl` model files to this repo.**
3. **Publicly linked sample data** (download availability and terms not verified): `zju-ls-feng`, 23 cameras × 800 frames, [Dropbox](https://www.dropbox.com/s/24mb7r921b1g9a7/zju-ls-feng.zip?dl=0).

---

## 2. Environment

✅ **Verified 2026-09-29** on a Linux x86-64 cloud container (4 CPU cores, **no GPU**), EasyMocap commit `e6006fd3814d5f8ad45a7ce965beb9bcc90767f2`, Python 3.9.23 (via `uv`; conda works the same way). What was verified: the install below, the imports, and EasyMocap's multi-view **triangulation** (`apps/demo/mv1p.py`) on synthetic data (§6b). **Not yet verified:** 2D detection (needs detector weights), SMPL fitting (needs SMPL models), rendering, GPU speed, Windows.

The plain upstream steps fail in three places. The fixes are marked `# FIX` below:

```bash
conda create -n easymocap python=3.9 -y        # or: uv venv -p 3.9 easymocap
conda activate easymocap

# PyTorch: pick the build matching your driver (https://pytorch.org/get-started/previous-versions/).
# Verified: the default PyPI build (2.0.1+cu117), which also runs CPU-only.
pip install torch==2.0.1 torchvision==0.15.2

git clone https://github.com/zju3dv/EasyMocap.git
cd EasyMocap
echo "numpy<1.24" > constraints.txt             # chumpy uses np.bool / np.int, removed in 1.24

# FIX 1: chumpy's setup.py imports pip, so it fails under pip's default build isolation
#        ("ModuleNotFoundError: No module named 'pip'").
pip install --no-build-isolation "git+https://github.com/mattloper/chumpy.git"

# FIX 2: requirements.txt lists chumpy again (triggers FIX 1's error) and pins
#        mediapipe==0.10.0, which has no Linux/Python 3.9 wheel. Only the optional
#        mediapipe keypoint backend uses it.
grep -v chumpy requirements.txt | sed 's/^mediapipe==0.10.0$/mediapipe/' > requirements-fixed.txt
pip install -r requirements-fixed.txt -c constraints.txt

pip install pyrender -c constraints.txt         # rendering the SMPL mesh
python setup.py develop                         # installs the package + the `emc` command
pip check                                       # verified: "No broken requirements found."
```

Verified versions: torch 2.0.1, numpy 1.23.5, opencv-python 4.11.0.86, chumpy 0.71, mediapipe 1.0.1, pyrender 0.1.45, pytorch-lightning 1.5.0, ultralytics 8.4.165, setuptools 59.5.0.

Sanity check (verified output: `2.0.1+cu117 False` on the CPU container):
```bash
python -c "import torch, easymocap, chumpy; print(torch.__version__, torch.cuda.is_available())"
which emc
```

**Which machine?** Before installing, run `nvidia-smi` on the machine you plan to use. EasyMocap's detectors and SMPL fitting are written for NVIDIA CUDA. Triangulation runs fine on CPU, but full-video detection on CPU will be slow. An AMD GPU won't be used by this PyTorch build. If your laptop has no NVIDIA GPU, ask Prof. Cao for a lab GPU server (Linux) and use this guide there.

**Headless server (no monitor):** pyrender needs an offscreen GL backend:
```bash
export PYOPENGL_PLATFORM=egl      # or: osmesa (slower, CPU)
```
FIX 3: `ImportError: ('Unable to load EGL library' ...)` means the system EGL library is missing. On Ubuntu: `sudo apt install libegl1` (or `libosmesa6` for osmesa). This only matters for rendering (`--vis_smpl`); triangulation doesn't need it.

---

## 3. Model files

```
EasyMocap/data/smplx/
├── J_regressor_body25.npy        # already in the repo
├── smpl/
│   ├── SMPL_FEMALE.pkl           # renamed from basicModel_f_lbs_10_207_0_v1.0.0.pkl
│   ├── SMPL_MALE.pkl             # renamed from basicmodel_m_lbs_10_207_0_v1.0.0.pkl
│   └── SMPL_NEUTRAL.pkl          # renamed from basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
└── smplx/ (optional)  SMPLX_{FEMALE,MALE,NEUTRAL}.pkl
```

**2D keypoint detector.** Choose one:
- **OpenPose** (the v0.1 default, `--openpose <path>`): the best-documented path, but painful to build. Try it only if a pre-built copy exists on the lab machine.
- **YOLO + HRNet** (no OpenPose build): the v0.1 path is `extract_video.py --mode yolo-hrnet`, which needs `data/models/yolov4.weights` and `data/models/pose_hrnet_w48_384x288.pth`. The v0.2 `emc` pipeline also uses `pose_hrnet_w48_384x288.pth`, and downloads `yolov5m` via `torch.hub` on the first run, so it needs internet. The legacy `installation.md` has empty detector links. The current [MyHRNet source](https://github.com/zju3dv/EasyMocap/blob/master/myeasymocap/backbone/hrnet/myhrnet.py) supplies upstream weight locations; download success has not been checked.

**v0.2 path gotcha:** `config/mv1p/detect_triangulate_fitSMPL.yml` looks for the body model at different paths than v0.1. From the EasyMocap root, add symlinks so both pipelines use the same files:
```bash
mkdir -p models/pare/data/body_models/smpl
ln -s "$PWD/data/smplx/smpl/SMPL_NEUTRAL.pkl" models/pare/data/body_models/smpl/SMPL_NEUTRAL.pkl
ln -s "$PWD/data/smplx/J_regressor_body25.npy" models/J_regressor_body25.npy
```

---

## 4. Run the demo on `zju-ls-feng`

```bash
data=/path/to/zju-ls-feng      # contains intri.yml, extri.yml, videos/ (or images/)

# 4a. Frames + 2D keypoints
python scripts/preprocess/extract_video.py ${data} --openpose <openpose_path> --handface   # OpenPose
# or
python scripts/preprocess/extract_video.py ${data} --mode yolo-hrnet                       # no OpenPose

# 4b. v0.1: triangulate + fit SMPL, and render views 1/7/13/19
python apps/demo/mv1p.py ${data} --out ${data}/output/smpl \
    --vis_det --vis_repro --undis --sub_vis 1 7 13 19 --vis_smpl

# 4c. v0.2 (config-driven): detect → triangulate → fit SMPL
emc --data config/datasets/mvimage.yml \
    --exp config/mv1p/detect_triangulate_fitSMPL.yml \
    --root ${data} --subs_vis 01 07 13 19
```

Camera names must match between `intri.yml`, `extri.yml` and the `images/<name>/` folders. Check `ls ${data}/images` against the `names:` list in the yml before using `--sub_vis`.

Outputs (see `doc/02_output.md`):
- `keypoints3d/NNNNNN.json`: `[{"id": 0, "keypoints3d": [[x, y, z, conf] × 25]}]`, BODY25 joints; coordinate units follow calibration and must be confirmed (use meters for this team's convention).
- `smpl/NNNNNN.json`: `Rh` (global rotation), `Th` (translation), `poses` (first 3 set to 0), `shapes`. Note that EasyMocap's `Rh`/`Th` are not the same as SMPL's `global_orient`/`transl` (see `02_output.md`). This matters when exporting to other projects.

What to look at: `--vis_repro` images (the reprojected skeleton should sit on the person in every view) and the rendered SMPL overlay.

---

## 5. Full ZJU-MoCap (LightStage) sequences

The raw release is in the Neural Body format: one folder per sequence, with `annots.npy` (cameras `K, D, R, T` with **T in millimeters**, plus the per-frame image list) and camera image folders. Convert it to EasyMocap's layout:

```bash
python reconstruction/tools/zju_to_easymocap.py /data/zju_mocap/CoreView_313 /data/easymocap/313
# then run step 4a/4b/4c with data=/data/easymocap/313
```
The converter writes `intri.yml`/`extri.yml` (T converted to meters, cameras named `01`…`NN`) and symlinks the frames into `images/NN/000000.jpg`. It doesn't copy them.

---

## 6. Experiments: how many cameras, which layout, how much sync error

### 6a. On real multi-view data (needs a sample dataset + detector)

These experiments use the all-camera result as a **reference estimate**. Agreement with it is not independent ground-truth accuracy.

```bash
# Reference: all views
python apps/demo/mv1p.py ${data} --out ${data}/output/all --undis

# Variant: only 5 views (e.g. evenly spread around the subject)
python reconstruction/tools/make_variant.py ${data} /tmp/v5_even --views 01 05 10 14 19
python apps/demo/mv1p.py /tmp/v5_even --out /tmp/v5_even/output --undis

# Variant: 5 views, camera 05 lagging by 2 frames (sync test)
python reconstruction/tools/make_variant.py ${data} /tmp/v5_lag2 --views 01 05 10 14 19 --shift 05:2
python apps/demo/mv1p.py /tmp/v5_lag2 --out /tmp/v5_lag2/output --undis

# Compare 3D joints against the reference
python reconstruction/tools/compare_keypoints3d.py ${data}/output/all/keypoints3d /tmp/v5_even/output/keypoints3d
```
`make_variant.py` symlinks `images/` and `annots/`, so it's fast and uses no extra disk space. Run 2D detection once on the full dataset first, so that `annots/` exists.

Suggested table for the Oct 20 meeting:

| Variant | Views | MPJPE vs all (mm) | Root-rel. MPJPE (mm) | Frames missing |
|---|---|---|---|---|
| 5 even | 01 05 10 14 19 | | | |
| 5 front half | … | | | |
| 4 even | … | | | |
| 5 even, 1-frame lag | … | | | |
| 5 even, 2-frame lag | … | | | |

Look up each camera's position with `apps/calibration/vis_camera_by_open3d.py`, or compute `center = -Rᵀ T` from `extri.yml`, so "evenly spread" is based on real angles.

### 6b. Synthetic rig with exact ground truth (runs now, no downloads) ✅

`tools/synth_rig.py` builds an EasyMocap dataset for any camera layout, with a scripted squat + arm-raise motion. It writes cameras, 2D keypoints (projected + noise) and the true 3D joints. EasyMocap's own triangulation then runs on it, and we score against the truth. It needs only the §2 environment: no SMPL, no weights, no GPU.

```bash
# one layout (5 cams on a 3 m circle, 1.6 m high)
python reconstruction/tools/synth_rig.py /tmp/rig --cams 5 --radius 3 --height 1.6
cd EasyMocap && python apps/demo/mv1p.py /tmp/rig --out /tmp/rig/output --body body25
#   -> stops with "data/smplx/smpl does not exist" AFTER writing output/keypoints3d (expected without SMPL)
python ../reconstruction/tools/compare_keypoints3d.py /tmp/rig/gt/keypoints3d /tmp/rig/output/keypoints3d

# the hardware team's actual plan (meters, z up, subject at the origin)
python reconstruction/tools/synth_rig.py /tmp/plan --positions "3,0,1.6; 0,3,2.2; -3,0,1.6; 0,-3,2.2; 2.1,2.1,0.5"

# full comparison table (layouts, counts, noise, sync lag): ~5 min on 4 CPU cores
EASYMOCAP=$PWD/EasyMocap bash reconstruction/tools/rig_sweep.sh /tmp/rig_sweep
```
Results from 2026-09-29 and how to read them: [`results/2026-09-29-synthetic-rig-sweep.md`](results/2026-09-29-synthetic-rig-sweep.md).

---

## 7. Our GoPro data

Checklist before running (see the interface contract in `README.md` §4):
- [ ] Videos named `01.mp4`…`05.mp4`, trimmed to a common start
- [ ] Same resolution as the calibration videos
- [ ] Linear lens, stabilization off
- [ ] `check_calib.py ${data} --out ${data}/output --mode cube --write` looks right in every view

Log of runs:

| Date | Session | Result | Issues |
|---|---|---|---|
| | | | |

---

## Problems hit

| Date | Step | Error | Fix |
|---|---|---|---|
| 2026-09-29 | `pip install -r requirements.txt` | chumpy: `ModuleNotFoundError: No module named 'pip'` | `pip install --no-build-isolation git+https://github.com/mattloper/chumpy.git`, then drop chumpy from requirements (§2 FIX 1) |
| 2026-09-29 | `pip install -r requirements.txt` | `No matching distribution found for mediapipe==0.10.0` (Linux, Py 3.9) | unpin mediapipe, keep `numpy<1.24` as a constraint (§2 FIX 2) |
| 2026-09-29 | `import pyrender` with `PYOPENGL_PLATFORM=egl` | `Unable to load EGL library` | `apt install libegl1` (§2 FIX 3); not needed for triangulation |
| 2026-09-29 | `mv1p.py` without SMPL models | `AssertionError: Path data/smplx/smpl does not exist!` | expected until SMPL is downloaded (§3); `keypoints3d/` is already written |
