# Windows CPU setup and real-video check

This supplements `SETUP.md` for the local laptop. Use the result receipt in
`LOCAL_RUN_STATUS.md` to distinguish verified stages from remaining work.

## Machine and source versions

On 2026-09-29 the local machine reported Windows 11 Pro (10.0.26100), Ryzen 5
5600H (6 cores / 12 threads), about 15.4 GiB usable RAM, Radeon RX 5500M and AMD
integrated graphics. There is no present NVIDIA adapter. `nvidia-smi` exists but
reports insufficient permission even when run outside the sandbox. PnP/CIM
hardware enumeration, rather than that executable's presence, established the
available GPUs. This setup deliberately installs **CPU PyTorch**.

Project folder: `D:\Research Fall 2026 - Shengting Cao`.

| Component | Pin |
|---|---|
| EasyMocap | `e6006fd3814d5f8ad45a7ce965beb9bcc90767f2` |
| YOLOv5 | v7.0, `915bbf294bb74c859f0b41f1c23bc395014ea679` |
| Python | 3.9.23, conda-forge |
| PyTorch / torchvision | 2.0.1+cpu / 0.15.2+cpu |
| NumPy / SciPy / OpenCV | 1.23.5 / 1.10.1 / 4.11.0.86 |
| Chumpy | 0.70 |
| pyrender / pyglet / PyOpenGL | 0.1.45 / 1.5.27 / 3.1.0 |

This is a minimal environment for the configured detection/triangulation/SMPL
path. It does not install optional MediaPipe, Lightning, neural-renderer training
or CUDA components. It is not a tested installation of every EasyMocap feature.

## Recreate the isolated environment

Prerequisites: Git, Conda and ffmpeg on the second machine. The installed ffmpeg
here is `D:\Octave-10.3.0\mingw64\bin\ffmpeg.exe`; set `$ffmpeg` to a valid path
on another machine. Use a regular PowerShell terminal, then:

```powershell
$project = 'D:\Research Fall 2026 - Shengting Cao'
$env:CONDA_PKGS_DIRS = Join-Path $project '.cache\conda-pkgs'
conda create --prefix "$project\.envs\easymocap" --override-channels -c conda-forge python=3.9.23 pip=24.0 -y
$py = Join-Path $project '.envs\easymocap\python.exe'
$env:PATH = "$project\.envs\easymocap;$project\.envs\easymocap\Library\bin;$project\.envs\easymocap\Scripts;" + $env:PATH
$env:PIP_CACHE_DIR = Join-Path $project '.cache\pip'
& $py -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install -r "$project\reconstruction\requirements-windows-cpu.txt"
& $py -m pip install --no-build-isolation chumpy==0.70
git clone https://github.com/zju3dv/EasyMocap.git "$project\external\EasyMocap"
git -C "$project\external\EasyMocap" checkout e6006fd3814d5f8ad45a7ce965beb9bcc90767f2
git clone --branch v7.0 --depth 1 https://github.com/ultralytics/yolov5.git "$project\external\yolov5"
& $py -m pip install --no-deps --no-build-isolation -e "$project\external\EasyMocap"
& $py -m pip check
```

Do not recreate these folders on the already configured laptop. In each fresh
terminal, only set `$project`, `$py`, and the environment PATH as above. Calling
the explicit Python executable avoids accidentally using the machine's Python
3.12 or Anaconda Python 3.13. Windows package extraction may take several minutes.

## Download official sample and detector weights

```powershell
New-Item -ItemType Directory -Force "$project\downloads", "$project\data\zju-ls-feng" | Out-Null
Invoke-WebRequest 'https://www.dropbox.com/s/24mb7r921b1g9a7/zju-ls-feng.zip?dl=1' -OutFile "$project\downloads\zju-ls-feng.zip"
Expand-Archive -LiteralPath "$project\downloads\zju-ls-feng.zip" -DestinationPath "$project\data\zju-ls-feng"
Invoke-WebRequest 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt' -OutFile "$project\downloads\yolov5m.pt"
& $py -m gdown 11ezQ6a_MxIRtj26WqhH3V3-xPI3XqYAw -O "$project\downloads\pose_hrnet_w48_384x288.pth"
Get-FileHash "$project\downloads\zju-ls-feng.zip", "$project\downloads\yolov5m.pt", "$project\downloads\pose_hrnet_w48_384x288.pth"
```

The HRNet ID comes from `myeasymocap/backbone/hrnet/myhrnet.py` in the pinned
upstream commit. This session downloaded its public Drive endpoint directly;
the gdown command above is the upstream-compatible reproduction method.

| File | Bytes | SHA256 recorded in this session |
|---|---:|---|
| zju-ls-feng.zip | 115474684 | `6a89a83106890edb37457d025e4a3664296013bfa798cb54ba72fa91b0bc16ea` |
| yolov5m.pt | 42806829 | `61d933360ba5a7733a36764996c800287d973889d875227f5beedd2473a97a56` |
| pose_hrnet_w48_384x288.pth | 255061287 | `95e0fec3194826d5e3f806ea89be68bbb84517b114c3a32b3058c56610b5ef61` |

The archive root contains `videos/`, `openpose/`, `intri.yml`, `extri.yml`. It
contains 23 videos. Camera 1 was checked as 1024-by-1024, 800 frames, 25 fps; the
runner records metadata for every selected camera. Keep camera names `1`, `7`,
`13`, `19` without zero-padding. This differs from some quickstart examples.

## Run fresh detection and triangulation

```powershell
$ffmpeg = 'D:\Octave-10.3.0\mingw64\bin\ffmpeg.exe'
$env:PYTHONIOENCODING = 'utf-8'
& $py -u "$project\reconstruction\tools\local_cpu.py" --root $project --frames 6 --step 2 --ffmpeg $ffmpeg
# After checking the 3-frame smoke test:
& $py -u "$project\reconstruction\tools\local_cpu.py" --root $project --frames 100 --step 2 --ffmpeg $ffmpeg
```

The second command samples original frames 0,2,...,98 across cameras 1,7,13,19:
50 time steps, 200 images, four seconds of original footage. The videos play at
12.5 fps to preserve time. The runner extracts images from video and executes
YOLOv5 plus HRNet; it does not use the bundled `openpose/` detections.

Outputs are under `local-runs/zju-ls-feng-cpu/`:

- `source_videos.json`, `cfg_data.yml`, `cfg_exp.yml`: exact input/configuration.
- `detections/*.npz`: observed 2D points, boxes and corresponding 3D points.
- `triangulated/*.json`: 3D joints named by original source frame.
- `vis_keypoints2d.mp4`: fresh image detections.
- `vis_repro.mp4`: reconstructed 3D skeleton reprojected onto original images.
- `validation.json`: confidence coverage, reprojection diagnostics and timing.

Caches are reused for identical image/model inputs. Start a new output directory
or archive the existing run before changing calibration, weights or input images.
The runner's current default output is fixed to this sample; do not mix datasets
there. It retains extracted images and overlay frames for inspection.

## Continue with the licensed SMPL model

Obtain the neutral SMPL model through the [official SMPL portal](https://smpl.is.tue.mpg.de/),
as described in [EasyMocap's model instructions](https://chingswy.github.io/easymocap-public-doc/install/install_smpl.html).
The neutral filename may be `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` or
`SMPL_NEUTRAL.pkl`, depending on the authorized distribution. Supply its actual
path. Keep the license and files private; do not commit the model to GitHub.

```powershell
$model = 'D:\path\to\your\licensed\SMPL_NEUTRAL.pkl'
& $py -u "$project\reconstruction\tools\local_cpu.py" --root $project --frames 100 --step 2 --ffmpeg $ffmpeg --fit --smpl $model
```

The command reuses detections, fits SMPL and renders onto camera 1. Verify
`smpl/*.json`, `vis_render/*.jpg` and `vis_render.mp4`. `validation.json` must
report `smpl_completed: true` and `videos_encoded: true`; also inspect the video
for alignment, limb swaps, float/drift and missing frames. Command preparation
or a skeleton video alone does not establish this milestone.

## Independent reproduction checklist for Seishin

Use a separate environment and the same commits/hashes. Record OS, CPU/GPU,
Python version, install log, `pip check`, exact invocation and output counts.
Run the small sample first. Report the first traceback, not just “it failed.”
Then compare valid joint counts and reprojection distributions, inspect actual
overlay frames, and run the licensed SMPL stage. This independent check remains
pending until Seishin actually performs it.

The public EasyMocap demonstration archive is separate from obtaining the complete
ZJU-MoCap release through its agreement process. Keep those tasks distinct.

## Problems found locally

- The first diagnostic call to OpenCV rejected a non-contiguous XYZ slice. The
  runner now passes `np.ascontiguousarray`; the real-video smoke test was rerun.
- YOLOv5 v7.0 prints `fatal: cannot change to 'D:\Research'` while trying to
  describe its Git version. Its upstream diagnostic constructs an unquoted shell
  path. Model loading and inference continue successfully; the actual source SHA
  is recorded above independently. No global Git safety settings were changed.
- Chumpy's legacy setup needs pip available during build; install it with
  `--no-build-isolation`. Python 3.9 and NumPy 1.23.5 preserve the legacy APIs it
  uses. `pip check` and explicit imports passed locally.
- A separate 128-by-128 offscreen cube render passed (nonzero depth). This checks
  the OpenGL backend only, and is not evidence that a body mesh has been fitted.
