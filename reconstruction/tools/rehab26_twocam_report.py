"""T5b report: grade two-camera MediaPipe rows against Table 3 and compare with T5 single-camera rows.

Usage: python -I rehab26_twocam_report.py <t5b_psi_dir> <t5_psi_dir> <spec_dir>
Viewpoint of a two-camera sequence = cam1 viewing angle (-o). Single-camera rows use the same (recording,
orientation) blocks, cameras 1 and 3, averaged per block ("1-cam mean"). Ψ per block = mean over seeds 0-9;
CIs: 2000 bootstrap resamples of the 5 persons.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

two, one, spec = map(Path, sys.argv[1:4])
assert hashlib.sha256((spec / "twocam_targets.csv").read_bytes()).hexdigest().startswith("affe820f"), "targets changed"
EXN = {1: "Ex1 arm abduction", 7: "Ex2 shoulder extension", 6: "Ex3 squat"}
P2D = {1: 1, 2: 7, 3: 6}
rng = np.random.default_rng(0)


def grade(rel):
    return "A" if rel <= .02 else "B" if rel <= .2 else "C" if rel <= .4 else "D" if rel <= .6 else "F"


def block(path, dims=None, cams=None):
    d = pd.read_csv(path)
    if dims:
        d = d[d.dims == dims]
    if cams:
        d = d[d.cam.isin(cams)]
    d = d.groupby(["exercise", "video_id", "orientation", "cam", "seed"]).psi.mean().reset_index()
    per_seed = d.groupby(["exercise", "video_id", "orientation", "seed"]).psi.mean().reset_index()   # mean over cams
    return per_seed


def ci(x, persons):
    ps = np.unique(persons); by = {p: x[persons == p] for p in ps}
    s = [np.mean(np.concatenate([by[p] for p in rng.choice(ps, len(ps))])) for _ in range(2000)]
    return np.percentile(s, [2.5, 97.5])


# 1. grading against Table 3 multi-camera rows (viewpoint means per seed, then mean over seeds)
tgt = pd.read_csv(spec / "twocam_targets.csv")
print("## Grading against Table 3 (column = cam1 viewpoint)\n")
print("| Ex | Row | " + " | ".join(c[3:] for c in tgt.columns[2:]) + " |\n|---|---|" + "---|" * 7)
allg = {}
for meth, f in (("Ortho", "mpp_ortho.csv"), ("Merged", "mpp_merged.csv")):
    d = pd.read_csv(two / f)
    vp = d.groupby(["exercise", "viewpoint", "seed"]).psi.mean().groupby(["exercise", "viewpoint"]).agg(["mean", "min", "max"])
    gl = []
    for t in tgt[tgt.method == meth].itertuples():
        ex = P2D[t.exercise]; paper, ours, gr = [], [], []
        for c in tgt.columns[2:]:
            v = int(c[3:]); p = tgt.loc[t.Index, c]; m = vp.loc[(ex, v)]
            paper.append(f"{p:.2f}"); ours.append(f"{m['mean']:.2f} ({m['min']:.2f}–{m['max']:.2f})")
            g = grade(abs(m["mean"] - p) / p); gr.append(g); gl.append((g, abs(m["mean"] - p) / p, t.exercise, v, m["mean"], p))
        print(f"| {t.exercise} | {meth} paper | " + " | ".join(paper) + " |")
        print(f"| {t.exercise} | {meth} ours (seed range) | " + " | ".join(ours) + " |")
        print(f"| {t.exercise} | grade | " + " | ".join(gr) + " |")
    allg[meth] = gl
for meth, gl in allg.items():
    cnt = pd.Series([g[0] for g in gl]).value_counts().reindex(list("ABCDF"), fill_value=0).to_dict()
    w = max(gl, key=lambda g: g[1])
    print(f"\n{meth}: grades {cnt}; worst Ex{w[2]} {w[3]:+d}: ours {w[4]:.3f} vs {w[5]:.2f} ({100*w[1]:.0f}%); "
          f"mean |dev| {np.mean([abs(g[4]-g[5]) for g in gl]):.3f}")

# 2. comparison with single-camera rows, all 11 orientations
rows = {
    "MPP 2D, 1 cam (mean of cam1, cam3)": block(one / "mpp_default.csv", "2d", [1, 3]),
    "MPP 3D, 1 cam": block(one / "mpp_default.csv", "3d", [1, 3]),
    "ViTPose 2D, 1 cam": block(one / "all_vitpose2d.csv", None, [1, 3]),
    "GVHMR, 1 cam": block(one / "all_gvhmr_incam.csv", None, [1, 3]),
    "MPP 2D Ortho": block(two / "mpp_ortho.csv"),
    "MPP 2D Ortho, fixed cams": block(two / "mpp_ortho_fixed.csv"),
    "MPP 3D Merged": block(two / "mpp_merged.csv"),
    "ViTPose 2D Ortho": block(two / "vit_ortho.csv"),
    "GVHMR Merged (equal weights)": block(two / "gvhmr_merged.csv"),
    "GVHMR per-feature camera": block(two / "gvhmr_ortho.csv"),
}
seqs = {k: v.groupby(["exercise", "video_id", "orientation"]).psi.mean() for k, v in rows.items()}
print("\n## Ψ per exercise (mean over the 5 persons x 11 orientations = 55 blocks)\n")
print("| Row | " + " | ".join(EXN[e] for e in (1, 7, 6)) + " |\n|---|---|---|---|")
for k, s in seqs.items():
    print(f"| {k} | " + " | ".join(f"{s.loc[e].mean():.3f}" if e in s.index.get_level_values(0) else "–" for e in (1, 7, 6)) + " |")
print("\n## Paired differences (95% CI over persons)\n")
pairs = [("MPP 2D Ortho", "MPP 2D, 1 cam (mean of cam1, cam3)"), ("ViTPose 2D Ortho", "ViTPose 2D, 1 cam"),
         ("GVHMR Merged (equal weights)", "GVHMR, 1 cam"), ("GVHMR Merged (equal weights)", "MPP 2D Ortho"),
         ("GVHMR per-feature camera", "MPP 2D Ortho"), ("ViTPose 2D Ortho", "MPP 2D Ortho"),
         ("MPP 3D Merged", "MPP 3D, 1 cam")]
print("| A − B | " + " | ".join(EXN[e] for e in (1, 7, 6)) + " |\n|---|---|---|---|")
for A, B in pairs:
    cells = []
    for e in (1, 7, 6):
        if e not in seqs[A].index.get_level_values(0) or e not in seqs[B].index.get_level_values(0):
            cells.append("–"); continue
        j = pd.concat([seqs[A].loc[e].rename("a"), seqs[B].loc[e].rename("b")], axis=1, join="inner").reset_index()
        x = (j.a - j.b).values; pers = j.video_id.str.split("_").str[1].values
        lo, hi = ci(x, pers); cells.append(f"{x.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print(f"| {A} − {B} | " + " | ".join(cells) + " |")
