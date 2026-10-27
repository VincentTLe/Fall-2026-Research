# GPU server setup (P0), 2026-10-27

## 1. Result

EasyMocap (commit `e6006fd`) is installed on the school server `lambdavector2` with PyTorch 2.0.1+cu117 on an RTX A6000,
and the GPU, the pose prior `gmm_08.pkl`, offscreen EGL rendering and Tan's SMPL neutral model all load. The synthetic
rig sweep (SETUP.md §6b) reproduces the 2026-09-29 table exactly: baseline 5 cams, even 360°, 3 px gives
**MPJPE 8.9 mm here vs. 8.9 mm on Sep 29**, and all 16 rows agree. With SMPL in place, `mv1p.py` now runs through
SMPL fitting on the synthetic rig (120/120 frames).

## How to start a session

```bash
cd ~/fall2026                                   # = /net/home/tle/fall2026 (network home)
conda activate ~/fall2026/envs/easymocap        # or call ~/fall2026/envs/easymocap/bin/python directly
export CUDA_VISIBLE_DEVICES=1                   # GPU 0 also drives the desktop display; check nvidia-smi first
export PYOPENGL_PLATFORM=egl                    # offscreen rendering, works without sudo
cd external/EasyMocap                           # mv1p.py / emc expect to run from here
```

Paths:
- `~/fall2026/{Fall-2026-Research,external,models,logs}` are on `/net/home` (MooseFS network storage).
- `~/fall2026/{envs,data,runs}` are **symlinks** to `/home/tle/fall2026/{envs,data,runs}` on the local NVMe disk
  (Tan's choice, 2026-10-27: about 19× faster writes, see Numbers). `/home/tle/fall2026` holds only those three folders.
  The logs are in `/net/home/tle/fall2026/logs`.
- SMPL: `~/fall2026/models/smpl/SMPL_NEUTRAL.pkl` (mode 600), linked into EasyMocap at `data/smplx/smpl/` and
  `models/pare/data/body_models/smpl/` (SETUP.md §3).
- If you use conda from `/home/tle/miniconda3`, note that `~/.condarc` has only the `defaults` channel. This env was
  built with `--override-channels -c conda-forge`.

## 2. Setup

| Field | Value |
|---|---|
| Host | `lambdavector2`, Ubuntu 24.04.4 LTS, kernel 6.8.0-137-generic |
| CPU / RAM | AMD Ryzen Threadripper PRO 5955WX, 16 cores / 32 threads; 251 GiB RAM |
| GPUs | 2× NVIDIA RTX A6000, 49140 MiB each; driver 595.84, CUDA 13.2 (driver API); no system `nvcc` |
| GPU use at check (14:41) | idle: GPU 0 had 69 MiB (Xorg, gnome-shell), GPU 1 had 15 MiB (Xorg); no compute processes |
| Scheduler / modules | none (no Slurm or PBS binaries, no `module` command); shared box, another user also logged in |
| Tools | conda 26.3.2 (`/home/tle/miniconda3`), uv, gcc 13.3.0, cmake 3.28.3, ffmpeg 6.1.1, git 2.43.0; no mamba/micromamba |
| Env | `~/fall2026/envs/easymocap`: Python 3.9.23 (conda-forge), torch 2.0.1+cu117 (CUDA 11.7, cuDNN 8.5.0), torchvision 0.15.2, numpy 1.23.5, opencv-python 4.11.0.86, chumpy 0.71, mediapipe 1.1.1, pyrender 0.1.45, pytorch-lightning 1.5.0, ultralytics 8.4.191, setuptools 59.5.0; 5.4 GB |
| EasyMocap | https://github.com/zju3dv/EasyMocap at `e6006fd3814d5f8ad45a7ce965beb9bcc90767f2`, in `~/fall2026/external/EasyMocap` |
| This repo | `fb01683` (main) when the session started |
| SMPL neutral | copied by Tan from the laptop (`basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl`), 247,186,228 bytes, SHA-256 `4924f235e63f7c5d5b690acedf736419c2edb846a2d69fc0956169615fa75688` (matches the laptop) |
| `gmm_08.pkl` | SHA-256 `e1374908aae055a2afa01a2cd9a169bc6cfec1ceb7aa590e201a47b383060491`, 0 CR bytes |
| Data | none downloaded; the sweep is synthetic (seed 0), so `DATA_LOG.md` was not created |

Why PyTorch 2.0.1+cu117: it is the build SETUP.md §2 verified. It bundles its own CUDA 11.7 runtime, which driver
595 supports, and cu117 includes Ampere (sm_86) kernels for the A6000. A newer build was not needed for anything in P0.

## 3. Commands

Full output is in `~/fall2026/logs/2026-10-27-p0-{server-check,easymocap-install,rig-sweep,smpl}.log`.

```bash
# Step 1
hostname; uname -a; head -4 /etc/os-release; nproc; lscpu | grep 'Model name'; free -h
nvidia-smi; nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv
nvcc --version                                   # not found
df -h $HOME /home/tle /tmp; df -h -x tmpfs ...; quota -s   # quota: not installed
mfsgetquota /net/home/tle /net/home /net/data; mfsdirinfo -h /net/home/tle
du -sh /home/tle /home/tle/*; du -sh /net/home/tle
dd if=/dev/zero of=<dir>/ddtest bs=1M count=1024 oflag=direct   # write speed, both disks (see Failed or skipped)
command -v conda mamba micromamba python3 gcc g++ cmake ffmpeg git uv; type module; command -v sinfo squeue srun qsub
curl -sI -m 15 -w '%{http_code}' <url>           # 7 sites, see Numbers
ldconfig -p | grep -E 'libEGL.so|libOSMesa'

# storage split (Tan's choice)
mkdir -p /home/tle/fall2026/{envs,data,runs}
ln -s /home/tle/fall2026/{envs,data,runs} ~/fall2026/

# Step 2
git clone https://github.com/zju3dv/EasyMocap.git external/EasyMocap
git -C external/EasyMocap checkout e6006fd3814d5f8ad45a7ce965beb9bcc90767f2
conda create -y -p /home/tle/fall2026/envs/easymocap python=3.9 --override-channels -c conda-forge
conda install -y -p /home/tle/fall2026/envs/easymocap --override-channels -c conda-forge pip "setuptools<70"   # new: env had no pip
pip install torch==2.0.1 torchvision==0.15.2
cd external/EasyMocap; echo "numpy<1.24" > constraints.txt
pip install --no-build-isolation "git+https://github.com/mattloper/chumpy.git"                    # FIX 1
grep -v chumpy requirements.txt | sed 's/^mediapipe==0.10.0$/mediapipe/' > requirements-fixed.txt # FIX 2
pip install -r requirements-fixed.txt -c constraints.txt
pip install pyrender -c constraints.txt
python setup.py develop
pip check                                          # No broken requirements found.
CUDA_VISIBLE_DEVICES=1 python -c "<torch/cuda check + 4096x4096 matmul vs. float64 CPU>"
python -c "pickle.load(open('easymocap/multistage/gmm_08.pkl','rb'), encoding='latin1')"
PYOPENGL_PLATFORM=egl python -c "<pyrender 64x64 offscreen box>"

# Step 3
PATH=~/fall2026/envs/easymocap/bin:$PATH CUDA_VISIBLE_DEVICES=1 \
  EASYMOCAP=~/fall2026/external/EasyMocap bash reconstruction/tools/rig_sweep.sh ~/fall2026/runs/rig_sweep

# Step 4
sha256sum ~/fall2026/models/smpl/SMPL_NEUTRAL.pkl
cd external/EasyMocap
mkdir -p data/smplx/smpl models/pare/data/body_models/smpl
ln -s ~/fall2026/models/smpl/SMPL_NEUTRAL.pkl data/smplx/smpl/SMPL_NEUTRAL.pkl
ln -s $PWD/data/smplx/smpl/SMPL_NEUTRAL.pkl models/pare/data/body_models/smpl/SMPL_NEUTRAL.pkl
ln -s $PWD/data/smplx/J_regressor_body25.npy models/J_regressor_body25.npy
CUDA_VISIBLE_DEVICES=1 python -c "from easymocap.smplmodel import load_model; load_model(gender='neutral', model_type='smpl', model_path='data/smplx', device='cuda') ..."
CUDA_VISIBLE_DEVICES=1 python apps/demo/mv1p.py ~/fall2026/runs/rig_sweep/even5 --out ~/fall2026/runs/smpl_check/even5 --body body25
```

## 4. Numbers

**Storage** (asked by Tan: should envs and data live on /home/tle instead?)

| Volume | What it is | Size / free | Quota | Seq. write, 1 GiB `oflag=direct` |
|---|---|---|---|---|
| `/net/home` (`~`, holds `~/fall2026`) | MooseFS network FS, shared with `/net/data` | 22 T, 2.1 T free (91% used) | none set (`mfsgetquota`); stores two copies, so `/net/home/tle` uses 351 GiB `size` but 701 GiB `realsize` | 118 MB/s |
| `/home/tle` (local) | root NVMe `/dev/nvme0n1p2`, shared with the OS and other local homes | 1.8 T, 308 G free (82% used) | none (`quota` not installed) | 2.2 GB/s |

`/home/tle` already uses 301 GB: `models` 161 G, `.cache` 57 G, `ollama_models` 28 G, `miniconda3` 13 G, other items
smaller. Decision: envs, data and runs go on local disk, behind symlinks, because they are rebuildable and I/O-heavy.
The repo, logs and licensed models stay on `/net/home`. Risk: if datasets plus extracted frames grow past about 150 GB,
the 308 G free on `/` gets tight. In that case move `data/` back to `/net/home`. Local disk is probably not backed up
(not checked).

**Network** (HEAD request, HTTP status, 14:4x): github.com 200, pypi.org 200, download.pytorch.org 200,
huggingface.co 200, zenodo.org 200, www.dropbox.com 200, domedb.perception.cs.cmu.edu 200.

**GPU check** (GPU 1): `torch.cuda.is_available()` True, device "NVIDIA RTX A6000". 4096×4096 float32 matmul on GPU
vs. float64 on CPU: max relative error 2.8e-6.

**Synthetic rig sweep**: EasyMocap `mv1p.py` triangulation, synthetic squat + arm raise, 120 frames at 60 fps, seed 0.
Run time 1 m 19 s (CPU-bound, 32 threads).

| Variant | MPJPE (mm), server | MPJPE (mm), 2026-09-29 | Root-rel. (mm), server / Sep 29 | Joints reconstructed |
|---|---|---|---|---|
| 5 cams, even 360°, 3 px (**baseline**) | **8.9** | **8.9** | 15.0 / 15.0 | 100.0% |
| 5 cams, no occlusion model | 7.9 | 7.9 | 10.8 / 10.8 | 100.0% |
| 5 cams, front 180° | 8.8 | 8.8 | 16.6 / 16.6 | 100.0% |
| 5 cams, clustered 90° | 9.8 | 9.8 | 15.7 / 15.7 | 100.0% |
| 4 cams, even | 10.9 | 10.9 | 23.4 / 23.4 | 100.0% |
| 3 cams, even | 11.7 | 11.7 | 17.1 / 17.1 | 98.5% |
| 6 cams, even | 8.1 | 8.1 | 15.8 / 15.8 | 100.0% |
| 5 cams, 8 px noise | 23.7 | 23.7 | 40.1 / 40.1 | 100.0% |
| 5 cams, 15% joints missing | 10.1 | 10.1 | 16.9 / 16.9 | 97.6% |
| 5 cams, mixed heights | 9.0 | 9.0 | 14.9 / 14.9 | 100.0% |
| cam 03 lags 1 / 2 / 4 / 8 frames | 9.0 / 9.3 / 10.3 / 13.2 | 9.0 / 9.3 / 10.3 / 13.2 | 15.0 / 15.7 / 17.4 / 21.3 (same) | 100.0% |
| fast squat, cam 03 lags 2 / 4 | 10.3 / 13.0 | 10.3 / 13.0 | 17.3 / 21.0 (same) | 100.0% |

No frames were missing in any row. The match is exact rather than close, which is what we should expect: the
synthetic data uses a fixed seed (`--seed 0` in `synth_rig.py`), and triangulation is deterministic NumPy on the CPU,
so the GPU and machine don't enter.

**SMPL**: neutral model through EasyMocap `load_model` on GPU 1: 6890 vertices, 13776 faces, J_regressor 24×6890,
T-pose height 1.717 m, BODY25 joints (1, 25, 3). `mv1p.py` on `rig_sweep/even5` with SMPL: exit 0 in 17 s,
120 `keypoints3d` + 120 `smpl` JSON files. Frame 60 has `poses` 72, `shapes` 10, `Th` (-0.097, -0.037, 0.85) m.

## 5. Checks

- Units on one frame (`even5`, frame 0): true pelvis (0, 0, 0.95) m vs. triangulated (-0.012, 0.005, 0.949) m,
  mean joint error 8.1 mm on that frame. Both are in meters, so the mm values are not off by a factor.
- `even5/output/keypoints3d` has 120 files. Its `mv1p.log` ends with the expected
  `Path data/smplx/smpl does not exist!` (the sweep ran before SMPL was linked).
- SMPL SHA-256 matches the laptop value Tan sent. The mesh height and vertex count are those of SMPL.
- `gmm_08.pkl` has no `\r` bytes (`grep -c $'\r'` = 0) and loads as covars (8, 69, 69), weights (8,), means (8, 69).
  The CRLF damage seen on Windows is absent here.
- EGL: `libEGL.so.1` is present. A pyrender offscreen render of a box gave a 64×64 image with 552 covered pixels.
- `pip check` reports no broken requirements. The `emc` entry point exists in the env.

## 6. Failed or skipped

- **First `pip` steps did nothing.** conda 26 / conda-forge Python 3.9 came without pip (`No module named pip`), so
  torch, chumpy, requirements and `setup.py develop` all failed on the first try. Fix: `conda install pip "setuptools<70"`
  into the env, then rerun SETUP §2 unchanged. requirements.txt then pinned setuptools to 59.5.0, as on Sep 29.
- **Disk write test touched `/home/tle`.** The `dd` speed test wrote a 1 GiB file into `/home/tle`. Tan interrupted
  that command and the file was left behind. Tan approved deleting it, and it was removed. No other file outside
  `~/fall2026` was written, apart from the new `/home/tle/fall2026/` folder.
- FIX 3 (`apt install libegl1`) was not needed. EGL is already installed.
- SETUP.md's `which emc` check: `emc` is in the env's `bin/` and is on PATH only when the env is activated.
- `torch.hub` YOLOv5 and HRNet weights were not downloaded (not part of P0).

## 7. Not tested

- 2D detection (YOLO/HRNet) on real video, and anything on `zju-ls-feng` (not copied to the server).
- SMPL fit **quality**. The synthetic run shows the fitting stage runs and writes files, not that the fits are good.
  The `--vis_smpl` overlay was not rendered.
- Backup policy of `/home/tle` (local disk), and whether other users plan large writes to `/`.
- GPU 0, multi-GPU use, and the v0.2 `emc` pipeline.

## 8. Next step

Copy `zju-ls-feng` (or download it from the Dropbox link in SETUP.md §1) into `~/fall2026/data`, download the HRNet and
YOLO weights, and start T1 step 1: the full 23-camera, 800-frame run on GPU 1 with an SMPL overlay video. At the same
time, start the REHAB24-6 download (5.7 GB) and log it in `DATA_LOG.md`.
