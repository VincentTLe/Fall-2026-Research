# Source notes and corrections

Verified 2026-09-29. All commands below are source-derived examples, not executed results.

## EasyMocap has multiple generations of instructions

The [public quickstart](https://chingswy.github.io/easymocap-public-doc/quickstart/quickstart.html) currently uses `01_triangulate/street_dance.zip` and points readers to v0.3 development documentation. Its example command is:

```bash
data=data/examples/street_dance
emc --data config/datasets/mvimage.yml --exp config/mv1p/detect_triangulate_fitSMPL.yml --root "${data}" --subs_vis 07 01 05 03
```

The page puts output in `output/detect_triangulate_fitSMPL`. Its runtime claim applies to that documented example/hardware context; no runtime for Tan's machine was established.

The [legacy repository quickstart](https://github.com/zju3dv/EasyMocap/blob/master/doc/quickstart.md) uses `zju-ls-feng`: 800 frames, 23 synchronized/calibrated cameras. It uses `scripts/preprocess/extract_video.py` followed by `apps/demo/mv1p.py`. Both paths were confirmed in the inspected repository tree. Do not replace `scripts/` with `apps/` indiscriminately: the separate frame extractor is `apps/preprocess/extract_image.py`.

The legacy and configurable workflows use different defaults. Camera labels must be read from each dataset, not copied from an example command.

## Body-model conventions

The [output documentation](https://github.com/zju3dv/EasyMocap/blob/master/doc/02_output.md) describes per-person JSON with `Rh`, `Th`, `poses` and `shapes`; joint output includes confidence. It treats global rotation/translation outside the canonical body model. The document explicitly distinguishes this from the original SMPL/SMPL-X convention.

Consequences for exporters: verify transformed vertices/joints numerically, do not merely rename parameters, and preserve confidence and person IDs. Absolute scale must be established by calibration; a JSON XYZ field does not itself establish meters.

## Current installation evidence

The [legacy installation guide](https://github.com/zju3dv/EasyMocap/blob/master/doc/installation.md) lists Python >=3.6 and an old PyTorch 1.4.0 / torchvision 0.5.0 setup. Python 3.9 and torch 2.0.1 in our SETUP guide are an untested candidate combination, not an upstream mandate.

The inspected [requirements](https://github.com/zju3dv/EasyMocap/blob/master/requirements.txt) include `mediapipe==0.10.0`, `setuptools==59.5.0`, `tensorboard==2.8.0`, `pytorch-lightning==1.5.0`, and unpinned packages. Installing them can change a previously selected environment. Resolve in an isolated environment, then capture package versions and `pip check` output.

The [HRNet implementation](https://github.com/zju3dv/EasyMocap/blob/master/myeasymocap/backbone/hrnet/myhrnet.py) selects CPU when CUDA is unavailable. This verifies a fallback in that component only. It does not verify a complete CPU pipeline, speed, rendering support, or support for the laptop's AMD GPU.

## Configurable pipeline assets

The inspected [configuration](https://github.com/zju3dv/EasyMocap/blob/master/config/mv1p/detect_triangulate_fitSMPL.yml) uses:

| Asset | Expected path / model |
|---|---|
| Object detector | `yolov5m` |
| HRNet checkpoint | `data/models/pose_hrnet_w48_384x288.pth` |
| Body model | `models/pare/data/body_models/smpl/SMPL_NEUTRAL.pkl` |
| Joint regressor | `models/J_regressor_body25.npy` |

The legacy model directory is `data/smplx/`. Model-loading errors can therefore mean the wrong workflow's paths were followed. Acquire body models through their official registration/licensing flow.

## multinb is an additional renderer

The [multinb documentation](https://chingswy.github.io/easymocap-public-doc/works/multinb.html) gives these stages:

```bash
python3 apps/preprocess/extract_image.py "${data}"
python3 apps/postprocess/write_vertices.py "${data}/output-smpl-3d/smpl" "${data}/output-smpl-3d/vertices" --cfg_model "${data}/output-smpl-3d/cfg_model.yml" --mode vertices
pip install -r requirements_neuralbody.txt
python3 apps/neuralbody/demo.py --mode soccer1_6 "${data}" --gpus 0,1,2,3
```

Use only with the matching example/configuration and prepared model outputs. The page's reduced-ray example is a separate option; no neural renderer was installed or trained in this collection task.

## ExerciseLLM released-code details

[Generator](https://github.com/jessicaxtang/ExerciseLLM/blob/main/generate_REHAB246.py) and
[feature functions](https://github.com/jessicaxtang/ExerciseLLM/blob/main/utils/features_REHAB246.py):

- Actual generator filename: `generate_REHAB246.py`. The README's `generate_REHAB24-6.py` command does not match the inspected root.
- Generator reads `annotations.csv` inside `dataset/REHAB24-6/2d_joints_segmented/`.
- Feature functions explicitly describe two coordinate channels and include 2D reference vectors. The paper's 3D description must not silently replace the released-code contract.
- Joint names and indices differ from EasyMocap BODY25. Mapping requires semantic review; equal array length does not prove compatibility.

These are static source checks. Sample arrays, extracted feature values and LLM calls remain untested.

## Verification boundary

No GPU benchmark, dataset download, body-model acquisition, calibration, reconstruction, training or clinical validation was performed. Original helper scripts were preserved and not re-tested in this documentation task.
