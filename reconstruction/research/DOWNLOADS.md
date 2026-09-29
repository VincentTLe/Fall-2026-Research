# Data and model acquisition checklist

Collected 2026-09-29. **This file records official access routes; the binaries have not been downloaded or verified.**
A link in an upstream README is not evidence that a download currently completes.
Keep licensed models, large data and participant recordings outside this public research repository.

| Item | Official source / route | What remains |
|---|---|---|
| Legacy EasyMocap sample: `zju-ls-feng` | [Upstream quickstart](https://github.com/zju3dv/EasyMocap/blob/master/doc/quickstart.md) → [Dropbox sample](https://www.dropbox.com/s/24mb7r921b1g9a7/zju-ls-feng.zip?dl=0) | Download availability, archive contents and reuse terms not checked. Upstream documents 23 views and 800 frames. |
| Current quickstart example: `street_dance` | [Public quickstart](https://chingswy.github.io/easymocap-public-doc/quickstart/quickstart.html), follow its example-data link to `01_triangulate/street_dance.zip` | Download and inspect the matching calibration/annotations before running. |
| Full ZJU-MoCap | [Neural Body INSTALL](https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md#zju-mocap-dataset) links an access form and agreement/email route. [EasyMocap](https://github.com/zju3dv/EasyMocap#zju-mocap) publishes an agreement/email route with a different contact. | Confirm whether the lab already has access. No form or email was submitted and no agreement signed. Do not promise an approval date. |
| SMPL male/female v1.0.0 | [SMPL](https://smpl.is.tue.mpg.de/) | Registration and license review; use the files required by the selected config. |
| SMPL neutral | [SMPLify](https://smplify.is.tue.mpg.de/) | Registration/download. See the [legacy layout](https://github.com/zju3dv/EasyMocap/blob/master/doc/installation.md). |
| Optional SMPL-X / SMPL+H / MANO | [SMPL-X](https://smpl-x.is.tue.mpg.de/), [MANO/SMPL+H](https://mano.is.tue.mpg.de/) | Only acquire what the chosen workflow needs. Licenses and file formats differ. |
| HRNet weights | [MyHRNet source](https://github.com/zju3dv/EasyMocap/blob/master/myeasymocap/backbone/hrnet/myhrnet.py) names `pose_hrnet_w48_384x288.pth` and the upstream mirrors | Put at the config's `data/models/` path. Auto-download is not proof of validity; verify the downloaded file. |
| GaussianAvatar assets | [README: models/data](https://github.com/aipixel/gaussianavatar#download-models-and-data) | Upstream links assets, processed data and checkpoints; separate SMPL/SMPL-X registration is required. |
| GoMAvatar inputs/checkpoints | [README](https://github.com/wenj/GoMAvatar) | ZJU-MoCap or PeopleSnapshot preprocessing, model files, checkpoints and masks/cameras/poses. |
| ASH subject assets | [README](https://github.com/kv2000/ASH) → [author assets](https://gvv-assets.mpi-inf.mpg.de/ASH/) | Subject metadata, clothed-character dependencies and raw RGB/mask video. Review the stated noncommercial terms. |
| Motion-language assets | [TM2T](https://github.com/EricGuo5513/TM2T), [MotionGPT](https://github.com/OpenMotionLab/MotionGPT), [T2M-GPT](https://github.com/mael-zys/t2m-gpt) | Use the selected method's checkpoint and preprocessing/statistics together. Training data are separate from demo weights. |
| HumanML3D source data | [HumanML3D acquisition instructions](https://github.com/EricGuo5513/HumanML3D#how-to-obtain-the-data) | The repository explains AMASS redistribution restrictions and reconstruction scripts; obtain the relevant model/data permissions. |
| ExerciseLLM inputs | [README](https://github.com/jessicaxtang/exercisellm) → [UI-PRMD](https://webpages.uidaho.edu/ui-prmd/) and [REHAB24-6](https://zenodo.org/records/13305826) | Check the released array shapes and annotations. See the paper/code discrepancy in READING.md. |

For each eventual download, log the source URL, date, dataset/model version, license, local path, byte size and SHA-256 checksum. Record camera IDs, units, skeleton ordering and frame rate separately. This gives the next person enough information to reproduce the input without committing the data itself.
