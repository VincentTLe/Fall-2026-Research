# Verified reading list for Tan

Collected 2026-09-29 from the professor's supplied links and supporting primary sources.
This is a source collection, not a report that EasyMocap or any downstream model has been run.

Start with [the Claude handoff](research/CLAUDE_HANDOFF.md). The [source manifest](research/sources.json)
records all 14 distinct supplied URLs, access method, and inspected GitHub file hashes.
[Download requirements](research/DOWNLOADS.md) distinguish available links from acquired assets.
Bibliographic records are in [references.bib](research/references.bib).

## 1. Reconstruction first

| Resource | What it establishes | Use for the five-GoPro project |
|---|---|---|
| [EasyMocap repository](https://github.com/zju3dv/EasyMocap) | RGB motion capture toolbox; detection, triangulation, body-model fitting and rendering are separate stages. The legacy example has 23 calibrated, synchronized views. | First reproduce a small supplied sequence, inspect reprojection, then try the team's calibrated capture. |
| [Public quickstart](https://chingswy.github.io/easymocap-public-doc/quickstart/quickstart.html) | The current page points to v0.3 development docs and uses `street_dance` with the `emc` configuration pipeline. | Do not mix its example paths and camera names with the older `zju-ls-feng` instructions. See [source notes](research/SOURCE_NOTES.md). |
| [Neural Body](https://github.com/zju3dv/neuralbody) — Peng et al., CVPR 2021 | Sparse calibrated images and posed body geometry support novel-view rendering. The repository documents ZJU-MoCap and updated EasyMocap SMPL fits. | Supporting reading for dataset layout and pose conventions; renderer training is a later task. |
| [MVPose paper](https://arxiv.org/abs/1901.04111) | Multi-person multi-view pose estimation; linked by EasyMocap's README. | Additional background for multiple people, not a prerequisite for the first single-person run. |

Neural Body and MVPose are supporting sources selected for reconstruction. The professor's explicit ZJU links are EasyMocap, its quickstart, and multinb; do not describe a larger guessed list as assigned reading.

### The supplied multinb page

**Novel View Synthesis of Human Interactions from Sparse Multi-view Videos** — Qing Shuai, Chen Geng, Qi Fang, Sida Peng, Wenhao Shen, Xiaowei Zhou and Hujun Bao; SIGGRAPH 2022.

The [project documentation](https://chingswy.github.io/easymocap-public-doc/works/multinb.html) describes calibrated multi-person video, novel views and instance masks. Its outdoor example uses eight GoPros. The documented workflow extracts frames, exports vertices from fitted SMPL parameters, then trains the neural renderer. The page recommends four RTX 3090s and also shows a reduced-ray single-GPU command. These are renderer examples, not minimum requirements for all EasyMocap operations.

For our project, this is a relevant later rendering target. Five cameras do not reproduce the documented eight-camera setting; evaluate coverage and occlusion rather than assuming equivalent results. Code entry points and commands are summarized in [source notes](research/SOURCE_NOTES.md).

## 2. Avatar and rendering directions

| Work and source | Input → output | Dataset / implementation evidence | Connection to our capture (proposed adaptation) |
|---|---|---|---|
| **GaussianAvatar**, Hu et al., CVPR 2024. [Repository](https://github.com/aipixel/gaussianavatar), [paper](https://arxiv.org/abs/2312.02134) | Monocular images, masks, cameras and body poses → animatable Gaussian avatar. | README includes PeopleSnapshot and custom-video workflows; custom data uses `images/`, `masks/`, `cameras.npz`, `poses_optimized.npz`. CUDA rasterization components are compiled. | Retain images and masks alongside poses. An EasyMocap JSON file needs convention and format conversion; it is not the complete input. |
| **GoMAvatar**, Wen et al., CVPR 2024. [Repository](https://github.com/wenj/GoMAvatar), [paper](https://arxiv.org/abs/2404.07991) | Monocular capture with body/camera information → mesh-attached Gaussian avatar and novel-view/pose rendering. | ZJU-MoCap and PeopleSnapshot preparation scripts are released. README tests CUDA 11.6, PyTorch 1.13.0 and PyTorch3D 0.7.0. | The ZJU preparation path makes this a sensible first avatar candidate after reconstruction. Our captures still need equivalent masks, calibration and pose metadata. |
| **ASH**, Pang et al., CVPR 2024. [Repository](https://github.com/kv2000/ASH), [project](https://vcai.mpi-inf.mpg.de/projects/ash/) | Multi-view RGB/mask video and a clothed character model with skeletal motion → animatable Gaussian rendering. | Uses DeepCharacters Pytorch, subject metadata and cached character geometry; README tests Python 3.9, PyTorch 1.12.1, CUDA 11.3. | Larger integration effort: generic SMPL or BODY25 output does not replace its clothed-character assets. Rendering speed alone does not establish an XR application. |

Use additional views as held-out image evidence where possible. All-view EasyMocap fits are a useful reference, but are not independent motion-capture ground truth.

## 3. Motion and language directions

| Work | Verified scope and sources | What our data would need |
|---|---|---|
| **TM2T: Stochastic and Tokenized Modeling for the Reciprocal Generation of 3D Human Motions and Texts**, Guo, Zuo, Wang and Cheng; ECCV 2022 | Discrete motion tokens support both text-to-motion and motion-to-text. [Paper](https://arxiv.org/abs/2207.01696); [code](https://github.com/EricGuo5513/TM2T) uses HumanML3D and KIT-ML. | Match the checkpoint's skeleton, feature representation and normalization; add paired captions for supervised adaptation. |
| **MotionGPT: Human Motion as a Foreign Language**, Jiang et al.; NeurIPS 2023 | Unified motion/language tasks. [Paper](https://arxiv.org/abs/2306.14795); [code](https://github.com/OpenMotionLab/MotionGPT) documents generated arrays shaped `(frames,22,3)`. | Input features and generated joint arrays are different representations. Use the selected checkpoint's preprocessing. |
| **T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations**, Zhang et al.; CVPR 2023 | VQ motion representation and autoregressive text-conditioned generation. [Paper](https://arxiv.org/abs/2301.06052); [code](https://github.com/mael-zys/t2m-gpt) includes HumanML3D/KIT-ML preparation. README reports training on one V100 with 32 GB memory; this is not a verified minimum. | Start with released inference assets, then decide whether custom motion/text pairs are needed. |

The [HumanML3D repository](https://github.com/EricGuo5513/HumanML3D) specifies 22 SMPL-structure joints, 20 fps and separate joint-position/feature-vector data. KIT-ML uses 21 joints. Consequently, taking the first 22 entries of BODY25 is not a valid conversion.

Proposed export validation: recover world-space body joints with EasyMocap's own model, map semantic joints, standardize axes and units, resample using timestamps, run the target feature pipeline, and visualize the recovered motion. This conversion has not been implemented here.

## 4. Previously inaccessible rehabilitation sources

### IEEE 11303747: RehabKD

**RehabKD With LLM-KG-LDACX: Orchestrating AI Models for Rehabilitation Knowledge Discovery** — Wei Zhang, Zhizhong Xing and Shaochun Chen. *IEEE Access* 13, 215003–215030; published December 18, 2025. [Publisher](https://ieeexplore.ieee.org/document/11303747); [DOI](https://doi.org/10.1109/ACCESS.2025.3645851).

- **Input:** research literature text; the methods describe 699 retained Web of Science records.
- **Output:** topic/network analyses and knowledge graphs.
- **Idea:** combine language-model extraction, topic modeling and co-occurrence analysis.
- **Relevance:** helps organize rehabilitation literature. It is not a joint-sequence exercise-quality estimator. Keep the original supplied link, but correct its category.
- **Access:** publisher abstract and full-text sections read in the browser; a plain web fetch returned no useful article text.

### PubMed 40327204: CNN-SE assessment

**A deep learning model with interpretable squeeze-and-excitation for automated rehabilitation exercise assessment** — Md Johir Raihan, Md Atiqur Rahman Ahad and Abdullah-Al Nahid. *Medical & Biological Engineering & Computing* 63(10), 2871–2887, 2025; online May 6. [PubMed](https://pubmed.ncbi.nlm.nih.gov/40327204/); [DOI](https://doi.org/10.1007/s11517-025-03372-4).

The abstract describes a CNN with squeeze-and-excitation, grey-wolf parameter optimization and SHAP explanations, evaluated on KIMORE and UI-PRMD. It reports mean absolute deviation 0.127 and 0.014 respectively. These are author-reported dataset results, not comparable percentages or measurements on our capture. It is a candidate assessment baseline once skeleton inputs, score normalization and subject splits are verified.

**Evidence limit:** metadata and abstract read; full-text implementation, exact tensor format and runnable official code were not verified. Do not fabricate those details.

### ExerciseLLM: assessment and feedback

**Rehabilitation Exercise Quality Assessment and Feedback Generation Using Large Language Models with Prompt Engineering** — Jessica Tang, Ali Abedi, Tracey J. F. Colella and Shehroz S. Khan; arXiv 2505.18412, 2025. The [repository](https://github.com/jessicaxtang/exercisellm) identifies ARIAL@IJCAI2025 and links a Springer chapter. [Abstract](https://arxiv.org/abs/2505.18412); [full text](https://arxiv.org/html/2505.18412v1).

The paper feeds joint sequences or exercise-specific features, exercise identity and prompts to GPT-4o for correctness assessment and text feedback. It evaluates UI-PRMD and REHAB24-6. Reported limitations include overconfidence, invented thresholds and no quantitative ground-truth feedback evaluation. Treat generated explanations as research outputs requiring evaluation.

**Important paper/code mismatch:** the paper describes UI-PRMD as 22 joints with XYZ and REHAB24-6 as 26 joints with XYZ from wearable sensors. The released README and `generate_REHAB246.py` instead use `2d_joints_segmented`; `utils/features_REHAB246.py` documents `(frames,joints,2)` and constructs 2D reference vectors. Preserve this discrepancy explicitly.

**Integration inference:** map joint semantics, decide whether to reproduce the released 2D path or adapt features to 3D, segment repetitions, retain side/exercise labels, and compare features on known samples. Do not feed raw BODY25 arrays directly into these functions.

## 5. Cross-dataset source

**Recovering Complete Actions for Cross-dataset Skeleton Action Recognition** — Hanchao Liu, Yujiang Li, Tai-Jiang Mu and Shi-Min Hu; arXiv 2410.23641, submitted October 31, 2024. [Abstract](https://arxiv.org/abs/2410.23641); [full text](https://arxiv.org/html/2410.23641v1).

- **Input/output:** labeled skeleton sequences → augmented sequences used to train an action recognizer.
- **Idea:** recover fuller actions and resample them to reduce temporal mismatch between datasets.
- **Benchmarks:** principal 18-action setting spans NTU60-RGBD, PKU-MMD and ETRI-Activity3D; additional comparisons involve Kinetics.
- **Preprocessing:** the principal experiment resizes sequences to 64 frames and removes camera rotation and trajectory motion.
- **Project relevance (inference):** useful after we have an action-recognition baseline and labeled data. Preserve complete repetitions and their boundaries now. It is not an automatic conversion between body-model formats.
- **Evidence limit:** paper read; no runnable official implementation or publication venue beyond the arXiv record was verified.

## 6. Suggested sequence

1. Reproduce EasyMocap on a short supplied calibrated sequence and record the exact commit/environment.
2. Validate our five-camera calibration and synchronization through reprojection and capture logs.
3. Choose one downstream interface. GoMAvatar has an explicit ZJU preparation path; ExerciseLLM has accessible feature code but the dimensionality mismatch must be resolved first.
4. Run camera-count and synchronization experiments against an all-view reference; report agreement, missing joints and reprojection error, not absolute accuracy without independent ground truth.

These priorities are proposed for this project; they are not claims made by the papers.
