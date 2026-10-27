# Prompts for Claude Code on the GPU server

Each file here is a prompt to paste into a fresh Claude Code session on the school GPU server. The server's Claude
has not seen any earlier conversation, so every prompt points it to [`RULES.md`](RULES.md) and
[`../EXPLORATION.md`](../EXPLORATION.md) first.

| Prompt | Track | Needs before starting |
|---|---|---|
| [`P0-server-setup.md`](P0-server-setup.md) | Server check, EasyMocap on GPU, smoke test | SSH access to the server |
| [`P1-multiview.md`](P1-multiview.md) | T1: full `zju-ls-feng` run, CMU Panoptic camera-count, sync | P0 done; SMPL model copied to the server |
| [`P2-monocular.md`](P2-monocular.md) | T2 (+ T3): one-camera methods on REHAB24-6, YouTube clips | P0 done; SMPL-X registration for GVHMR |
| [`P3-gaussian.md`](P3-gaussian.md) | T4: Gaussian Splatting, static scene then human avatar | P0 done; a 30–60 s phone video of an object |
| [`P4-summary.md`](P4-summary.md) | Collect all results for the Nov 3 meeting | P1–P3 result files |

## How to use

1. SSH to the server, `mkdir -p ~/fall2026 && cd ~/fall2026`, start `claude`.
2. Paste the whole content of one prompt file. One prompt per session.
3. When it asks you something (a license file, a sudo step, which sequence), answer it there.
4. At the end it prints a short summary and the branch name. Push that branch when you agree with the result, then
   on the laptop run `git fetch` and tell the laptop Claude which result file to read.

Licensed model files (SMPL, SMPL-X) are copied by you, by hand, to `~/fall2026/models/` on the server. They never go
into git.
