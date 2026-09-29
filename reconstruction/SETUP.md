# EasyMocap setup and run guide

A working log that should become the team's reproducible guide.
**Rule:** every step you actually run gets a ✅ plus the date and machine. Every error goes under "Problems hit" with its fix. Steps marked *(unverified)* come from the EasyMocap source and docs and haven't been run by us yet.

Sources: [EasyMocap repo](https://github.com/zju3dv/EasyMocap) (`doc/installation.md`, `doc/quickstart.md`, `apps/calibration/Readme.md`, `doc/02_output.md`) and the [public docs](https://chingswy.github.io/easymocap-public-doc/quickstart/quickstart.html).

---

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
   - Licenses are per person and non-commercial. **Don't commit `.pkl` model files to this repo.**
3. **Public sample data** (no agreement needed): `zju-ls-feng`, 23 cameras × 800 frames, [Dropbox](https://www.dropbox.com/s/24mb7r921b1g9a7/zju-ls-feng.zip?dl=0).

---

## 2. Environment *(unverified)*

EasyMocap's code base is from 2021–2023. Old Python is intentional because `chumpy` (needed to load SMPL `.pkl`) breaks on Python ≥ 3.11 and NumPy ≥ 1.24.

```bash
conda create -n easymocap python=3.9 -y
conda activate easymocap

# PyTorch: pick the build matching your driver (see https://pytorch.org/get-started/previous-versions/)
# e.g. for a CUDA 11.8-capable driver:
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/zju3dv/EasyMocap.git
cd EasyMocap
pip install "numpy<1.24"          # chumpy uses np.bool / np.int, removed in 1.24
pip install -r requirements.txt
pip install pyrender               # rendering the SMPL mesh
python setup.py develop            # installs the package + the `emc` command
```

Sanity check:
```bash
python -c "import torch, easymocap, chumpy; print(torch.__version__, torch.cuda.is_available())"
which emc
```

**Headless GPU server (no monitor):** pyrender needs an offscreen GL backend:
```bash
export PYOPENGL_PLATFORM=egl      # or: osmesa (slower, CPU)
```

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
- **YOLO + HRNet** (no OpenPose build): the v0.1 path is `extract_video.py --mode yolo-hrnet`, which needs `data/models/yolov4.weights` and `data/models/pose_hrnet_w48_384x288.pth`. The v0.2 `emc` pipeline also uses `pose_hrnet_w48_384x288.pth`, and downloads `yolov5m` via `torch.hub` on the first run, so it needs internet. The download links in the repo's `installation.md` are empty; get them from the public docs' quickstart page or HRNet's official release.

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
- `keypoints3d/NNNNNN.json`: `[{"id": 0, "keypoints3d": [[x, y, z, conf] × 25]}]`, BODY25 joints, in meters.
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

Both experiments use the all-camera result as a pseudo ground truth.

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
| | | | |
