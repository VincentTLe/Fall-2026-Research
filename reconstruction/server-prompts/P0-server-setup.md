You are setting up the school GPU server for Tan's reconstruction research. This is the first session on this
server.

Step 0. Work in `~/fall2026` (create it if missing). Clone https://github.com/VincentTLe/Fall-2026-Research into
`~/fall2026/Fall-2026-Research` if it is not there, then read, in full, `reconstruction/server-prompts/RULES.md`,
`reconstruction/EXPLORATION.md` and `reconstruction/SETUP.md`. Follow RULES.md for the whole session.

Step 1. Server check. Record, with the commands you used:
- hostname, OS and kernel, CPU cores, RAM;
- `nvidia-smi` (all GPUs, memory, current users/processes), driver and CUDA versions, `nvcc --version` if present;
- free disk space and quota in `$HOME` (and any scratch/data volume the server offers);
- available tools: conda / mamba / micromamba, python3, gcc/g++, cmake, ffmpeg, git, `module avail` if environment
  modules exist, and whether a job scheduler (Slurm etc.) is in use;
- network access (a HEAD request is enough) to: github.com, pypi.org, download.pytorch.org, huggingface.co,
  zenodo.org, www.dropbox.com, domedb.perception.cs.cmu.edu.

If no GPU is free or there is no outbound network, stop and report to Tan before installing anything.

Step 2. EasyMocap on GPU. Follow `reconstruction/SETUP.md` §2 (the Linux install with its three fixes), with these
changes:
- clone EasyMocap into `~/fall2026/external/EasyMocap` and check out commit
  `e6006fd3814d5f8ad45a7ce965beb9bcc90767f2`;
- create the environment under `~/fall2026/envs/easymocap` (Python 3.9);
- install a CUDA build of PyTorch that matches the driver (record which), and confirm
  `torch.cuda.is_available()` and a small matrix multiply on the GPU;
- after install, confirm that `easymocap/multistage/gmm_08.pkl` loads with
  `pickle.load(f, encoding='latin1')` (on Windows this file was corrupted by CRLF conversion; check it here too).

Step 3. Smoke test with no downloads: SETUP.md §6b, the synthetic rig sweep
(`EASYMOCAP=~/fall2026/external/EasyMocap bash reconstruction/tools/rig_sweep.sh ~/fall2026/runs/rig_sweep`).
Compare the baseline row with `reconstruction/results/2026-09-29-synthetic-rig-sweep.md` (5 cams, even 360°, 3 px:
MPJPE 8.9 mm). Report the numbers side by side. A small difference is fine if you explain it; a large one means
the install differs, so find out why before moving on.

Step 4. Licensed model. Ask Tan to copy his SMPL neutral model from the laptop, for example:
`scp "D:\Research Fall 2026 - Shengting Cao\downloads\smpl_v1.1.0_extracted\SMPL_python_v.1.1.0\smpl\models\basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl" <user>@<server>:~/fall2026/models/smpl/SMPL_NEUTRAL.pkl`.
Wait for him. Then link it where EasyMocap expects it (`data/smplx/smpl/SMPL_NEUTRAL.pkl` inside the EasyMocap
clone, see SETUP.md §3), load it once in Python, and record its SHA-256 in the result file (not the file itself).

Step 5. Write `reconstruction/results/YYYY-MM-DD-server-setup.md` (sections from RULES.md). Include a short
"how to start a session" block: which env to activate, which GPU to use, paths. Commit it on branch
`server/setup-YYYYMMDD`. Do not push. Print the summary for Tan.

Not in this session: downloading datasets, running P1–P3.
