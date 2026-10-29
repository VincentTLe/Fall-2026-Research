You are collecting the results of Tan's reconstruction experiments for the Tuesday meeting with Prof. Cao.

Step 0. Read the inputs. Follow RULES.md.
- In `~/fall2026/Fall-2026-Research`, run `git fetch --all`.
- Read in full: `reconstruction/server-prompts/RULES.md`, `reconstruction/EXPLORATION.md`, and every file in `reconstruction/results/`, on `main` and on all `server/*` branches.
- Also read `~/fall2026/logs/HANDOFF-*.md` to see what is still running or unfinished.
- Prior work to read (author copies in `~/fall2026/data/papers/`; never commit them):
  - arXiv 2609.35726, by the REHAB24-6 group: one vs. two cameras for exercise assessment;
  - the Information Systems 2025 REHAB24-6 paper: 3D pose benchmark, Table 5.

Step 1. Write up Q1–Q4 (EXPLORATION.md section 2).
- For each question, write what the results show, with the numbers and their conditions, and what they do not show yet.
- Use only numbers that appear in a result file, and cite the file for each. Do not add new claims.
- If two files disagree, show both and say which one to trust and why.
- If a number comes from a branch not yet merged into main, say so next to it.
- For Q2, separate what the prior work already showed from what our results add, and give each new claim's caveats next to it (unofficial GVHMR weights, grade-B reproduction, 5 or 10 people).

Step 2. List what is still missing or failed, and the smallest next experiment for each.

Step 3. Write a slide outline (markdown, 6–8 slides) for a 10-minute update. Include:
- what was tried;
- the camera-count table;
- the one-camera vs. multi-camera numbers;
- one slide on prior work and what is new;
- one avatar picture path (no faces in git);
- the problems;
- which question looks most worth a paper, and why.

Tan will present this: use plain words, and explain each metric in one line (PA-MPJPE, Ψ, PSNR/SSIM/LPIPS).

Step 4. Write both into `reconstruction/results/YYYY-MM-DD-summary.md`. Commit on branch `server/summary-YYYYMMDD`. Do not push. Print the summary for Tan.
