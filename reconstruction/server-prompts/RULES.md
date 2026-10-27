# Rules for every server session

Read this whole file before doing anything.

## Context

- You work for Tan Le, an undergraduate at Knox College, on the reconstruction part of Prof. Shengting Cao's
  computer-vision project (multi-camera human capture for telemedicine/telepresence).
- Repository: https://github.com/VincentTLe/Fall-2026-Research (public). Plan: `reconstruction/EXPLORATION.md`.
  Earlier verified work: `reconstruction/LOCAL_RUN_STATUS.md`, `reconstruction/SETUP.md`.
- This is a shared school GPU server. Tan reads your output in a terminal and copies results back to his laptop.

## Folder layout on the server

```
~/fall2026/
├── Fall-2026-Research/   # git clone of the repo
├── envs/                 # conda/micromamba environments
├── external/             # third-party code (EasyMocap, GVHMR, GaussianAvatar, ...)
├── data/                 # downloaded datasets (never committed)
├── models/               # licensed SMPL / SMPL-X files, copied by Tan (never committed)
├── runs/                 # all outputs
└── logs/                 # one log per command: logs/YYYY-MM-DD-<step>.log
```

## Evidence

- Record the environment once per session: hostname, OS, GPU model and memory, driver/CUDA, Python, PyTorch, and the
  commit of every third-party repo you use.
- Run commands with their output saved to `logs/`. Quote the command and the relevant output lines in the result file.
- "Runs" means the output files exist and you opened or counted them. "Works" needs a number or a picture/video you
  checked. Never claim a step succeeded without that output.
- Every number comes with its conditions: dataset, sequence, frames, cameras, method and commit, GPU.
- Accuracy (MPJPE, PA-MPJPE) only against true ground truth. Agreement with another reconstruction is called agreement.
- Before comparing skeletons, write down the joint mapping, axes and units (m vs. cm) and check them on one frame.
- If a result looks too good or too bad, check the units and the joint mapping before reporting it.
- Log every dataset or model download in `reconstruction/results/DATA_LOG.md` (create it if missing): URL, date,
  version, license, local path, size, SHA-256 or file count.

## Licenses and privacy

- SMPL and SMPL-X files are licensed to Tan personally. Never commit, upload, print or copy them elsewhere.
- Datasets are for non-commercial research. Never commit dataset files, videos, frames or meshes to git.
- Do not put frames showing identifiable faces from YouTube or dataset videos into the repository.

## Shared server etiquette

- Check `nvidia-smi` first. Use one free GPU (`CUDA_VISIBLE_DEVICES`) unless Tan says otherwise. Never kill other
  users' processes. If there is a job scheduler (Slurm etc.), use it.
- No `sudo`. Install into `~/fall2026/envs` with conda/micromamba or pip. Check free disk space before downloads over
  5 GB.

## Git

- Work on a branch named `server/<track>-YYYYMMDD`. Never commit to or push `main`.
- Commit only: result markdown files in `reconstruction/results/`, small scripts in `reconstruction/tools/` (with a
  test or a check), and doc fixes. Commit messages are short sentences, like the existing history.
- Do not push. At the end, show `git log --oneline` and `git diff --stat main...HEAD` and let Tan decide.

## Stop and ask Tan when

- A step needs `sudo`, a license, a registration, or a file only Tan has.
- A download is over 20 GB, or the disk has less than 2× the space needed.
- No GPU is free.
- One step has failed three times with three different fixes.
- You are about to delete anything outside `~/fall2026/runs` or `~/fall2026/logs`.
- The plan in `EXPLORATION.md` turns out to be wrong for what you found (say what you found).

## Result file

One file per prompt: `reconstruction/results/YYYY-MM-DD-<track>.md` with these sections:

1. **Result** in two or three sentences, with the main numbers.
2. **Setup:** machine, environment, commits, data (with checksums or file counts).
3. **Commands** actually run, in order.
4. **Numbers** in tables, each with its conditions.
5. **Checks:** what you looked at to trust the numbers (frames viewed, unit checks, sanity tests).
6. **Failed or skipped**, and why.
7. **Not tested.**
8. **Next step** you would take.

End the session by printing that file's path, the branch name and a five-line summary for Tan.
