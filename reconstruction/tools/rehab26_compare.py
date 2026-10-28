"""Paired comparison of Psi between methods on the same REHAB26 sequences.

Inputs are rehab26_psi_run.py outputs. Sequences = (video_id, orientation, cam); only those present in every
method are used. Psi per sequence = mean over seeds. Reports per exercise: mean Psi per method, paired
difference vs. MPP 2D and MPP 3D with a bootstrap 95% CI over persons (resampling the 5 persons), and the same
per Table 3 viewpoint (only viewpoints where all methods have all 5 persons).

Usage: python -I rehab26_compare.py --method NAME=CSV[:dims] ... [--md out.md]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PAPER_EX = {1: "Ex1 arm abduction", 7: "Ex2 shoulder extension", 6: "Ex3 squat"}
TABLE3_VP = [-90, -60, -40, -30, -20, -15, -10, 0, 10, 15, 20, 30, 40, 60, 90]

ap = argparse.ArgumentParser()
ap.add_argument("--method", nargs="+", required=True)
ap.add_argument("--md")
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()
frames = []
for m in a.method:
    name, spec = m.split("=", 1)
    path, dims = (spec.rsplit(":", 1) + [None])[:2] if spec.endswith((":2d", ":3d")) else (spec, None)
    d = pd.read_csv(path)
    if dims:
        d = d[d.dims == dims]
    d = d.groupby(["exercise", "video_id", "orientation", "cam", "viewpoint"]).psi.mean().rename(name)
    frames.append(d)
df = pd.concat(frames, axis=1, join="inner").reset_index()
df["person"] = df.video_id.str.split("_").str[1]
names = [m.split("=", 1)[0] for m in a.method]
rng = np.random.default_rng(0)
out = [f"Paired sequences: {len(df)} (method columns: {', '.join(names)})", ""]


def boot_ci(d):
    """d: per-sequence differences with a 'person' column -> CI of the mean, resampling persons."""
    persons = d.person.unique()
    by = {p: d[d.person == p].diff_.values for p in persons}
    stats = []
    for _ in range(a.boot):
        s = rng.choice(persons, len(persons))
        stats.append(np.mean(np.concatenate([by[p] for p in s])))
    return np.percentile(stats, [2.5, 97.5])


out.append("| Exercise | n seq | " + " | ".join(f"Ψ {n}" for n in names) + " | " +
           " | ".join(f"{n} − {ref}" for ref in names[:2] for n in names[2:]) + " |")
out.append("|---|---|" + "---|" * (len(names) + 2 * (len(names) - 2)))
for ex in (1, 7, 6):
    g = df[df.exercise == ex]
    if not len(g):
        continue
    cells = [f"{g[n].mean():.3f}" for n in names]
    for ref in names[:2]:
        for n in names[2:]:
            d = g.assign(diff_=g[n] - g[ref])
            lo, hi = boot_ci(d) if d.person.nunique() > 1 else (np.nan, np.nan)
            cells.append(f"{d.diff_.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    out.append(f"| {PAPER_EX[ex]} | {len(g)} | " + " | ".join(cells) + " |")
out += ["", "Per Table 3 viewpoint (only viewpoints with all 5 persons in every method), mean Ψ:", ""]
out.append("| Exercise | Viewpoint | n seq | " + " | ".join(names) + " |")
out.append("|---|---|---|" + "---|" * len(names))
for ex in (1, 7, 6):
    for vp in TABLE3_VP:
        g = df[(df.exercise == ex) & (df.viewpoint == vp)]
        if g.person.nunique() < 5:
            continue
        out.append(f"| {PAPER_EX[ex]} | {vp:+d} | {len(g)} | " + " | ".join(f"{g[n].mean():.2f}" for n in names) + " |")
txt = "\n".join(out)
print(txt)
if a.md:
    Path(a.md).write_text(txt + "\n")
