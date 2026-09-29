# Reading list

For each paper, write 5 lines in the table under it. Don't summarize the whole paper. Answer:
1. **Input** (how many cameras? monocular? what annotations?)
2. **Output** (keypoints, SMPL, mesh, rendering, text?)
3. **Key idea** in one sentence
4. **Data used** (and whether it's ZJU-MoCap)
5. **What we take from it** for our 5-GoPro setup

> Replace or extend Section A with the exact list Prof. Cao gave. The entries below are the ZJU papers that EasyMocap is built on or that produced ZJU-MoCap.

---

## A. Core ZJU works (read first, Weeks 1–3)

| Priority | Paper | Why it matters to us |
|---|---|---|
| 1 | **EasyMocap docs + code**: `doc/quickstart.md`, `apps/calibration/Readme.md`, `doc/02_output.md` | The tool we run. Understand the steps: detect → triangulate → fit SMPL |
| 1 | **Neural Body** (Peng et al., CVPR 2021) | Introduced the ZJU-MoCap LightStage dataset; explains its capture setup (≈21–23 synced cameras around the subject) |
| 2 | **MVPose**: *Fast and Robust Multi-Person 3D Pose Estimation from Multiple Views* (Dong et al., CVPR 2019) | The multi-view matching + triangulation behind `mvmp`, for when we capture more than one person |
| 2 | **Novel View Synthesis of Human Interactions from Sparse Multi-view Videos** (Shuai et al., SIGGRAPH 2022), [doc page](https://chingswy.github.io/easymocap-public-doc/works/multinb.html) | The "multinb" work Prof. Cao linked. Sparse multi-view (few cameras, like ours), multiple people, built on EasyMocap. Code config: `config/neuralbody_index_mnb.yml` |
| 3 | **Mirrored Human**: *Reconstructing 3D Human Pose by Watching Humans in the Mirror* (Fang et al., CVPR 2021) | Shows how EasyMocap fits SMPL with few views; useful background |
| 3 | **Animatable NeRF** (Peng et al., ICCV 2021) | Next step after SMPL: animatable avatars from multi-view video |

Notes:

| Paper | Input | Output | Key idea | Data | What we take |
|---|---|---|---|---|---|
| | | | | | |

---

## B. Prof. Cao's extra directions (after the pipeline runs)

What they all have in common: **they consume SMPL parameters or 3D joints.** That's exactly what our pipeline produces, so our capture system becomes a data source for each of them.

### Monocular video → 3D avatar
- **GaussianAvatar** (CVPR 2024), https://github.com/aipixel/gaussianavatar: an animatable 3D-Gaussian avatar from a single video; needs SMPL fits per frame.
- **GoMAvatar** (CVPR 2024), https://github.com/wenj/GoMAvatar: Gaussians-on-Mesh, efficient and real-time; its README preprocesses **ZJU-MoCap** (`scripts/prepare_zju-mocap`).
- *Link to us:* train on one of our GoPro views and use the other views plus our multi-view SMPL as ground truth for evaluation.

### Real-time XR rendering
- **GoMAvatar** (above) and **ASH**: *Animatable Gaussian Splats for Efficient and Photoreal Human Rendering* (CVPR 2024), https://github.com/kv2000/ASH. Trained from multi-view video + skeletal motion.
- *Link to us:* this is the most demanding direction for capture quality (needs many views and good sync). A 5-camera rig is on the sparse side, so check what view counts they use.

### Motion ↔ language
- **TM2T** (ECCV 2022), https://github.com/EricGuo5513/TM2T: motion → text and text → motion using motion tokens.
- **T2M-GPT** (CVPR 2023), https://github.com/mael-zys/t2m-gpt: VQ-VAE motion tokens + GPT for text → motion.
- **MotionGPT** (NeurIPS 2023), https://github.com/OpenMotionLab/MotionGPT: one model for both directions, motion treated as a language.
- *Link to us:* these use the **HumanML3D** motion representation (derived from SMPL joints, 20 fps). To feed our captures in, convert EasyMocap SMPL → 22 SMPL joints → the HumanML3D feature script. Watch the coordinate frame (up axis) and EasyMocap's `Rh/Th` convention (`SETUP.md` §4).

### Rehab exercise quality assessment + explainable rehab
- **ExerciseLLM**: *Rehabilitation Exercise Quality Assessment and Feedback Generation Using LLMs with Prompt Engineering* (ARIAL@IJCAI 2025), [arXiv 2505.18412](https://arxiv.org/abs/2505.18412), https://github.com/jessicaxtang/exercisellm. Uses skeleton data from **UI-PRMD** (Kinect joints) and **REHAB24-6** (2D joints), computes exercise-specific features, and prompts an LLM for a score + feedback.
- IEEE doc [11303747](https://ieeexplore.ieee.org/document/11303747): *(read and fill in: title, input, dataset)*
- PubMed [40327204](https://pubmed.ncbi.nlm.nih.gov/40327204/): *(read and fill in)*
- *Link to us:* **the easiest direction to prototype.** It needs only joint positions, which our pipeline already outputs (`keypoints3d/`), and the ExerciseLLM code is small. A first demo: record team members doing a squat correctly and incorrectly, run EasyMocap, convert to ExerciseLLM's feature format, and see if the LLM's feedback makes sense.

### Cross data
- [arXiv 2410.23641](https://arxiv.org/abs/2410.23641): *(read and fill in: which datasets it bridges and in what body format; that decides what format we should export)*

---

## C. Suggested order

1. Week 1: EasyMocap docs → Neural Body (sections on the dataset and setup) → MVPose
2. Week 2–3: multinb (sparse views, closest to our rig) → skim Mirrored Human
3. Nov: one downstream direction. **Recommendation: rehab (ExerciseLLM)**, then motion↔language.
