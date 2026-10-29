"""Compare per-sequence Psi (rehab26_psi_run.py output) with the frozen Table 3 targets.

Viewpoint score = mean over the sequences at that viewing angle (paper Sec. 5.1), computed per seed;
reported as mean over seeds with the seed min-max. Grades per SPEC.md: A<=2%, B<=20%, C<=40%, D<=60%, F.

Usage: python -I rehab26_grade.py <psi.csv> <spec_dir> [--label name] [--md out.md]
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_SHA = "41e965ce471ff086a6371646d93ddd751e1ccbe246172783fb6682beddea738e"
PAPER_TO_DATA = {1: 1, 2: 7, 3: 6}


def grade(rel):
    return "A" if rel <= .02 else "B" if rel <= .2 else "C" if rel <= .4 else "D" if rel <= .6 else "F"


def table(psi_csv, spec_dir):
    tpath = Path(spec_dir) / "table3_targets.csv"
    assert hashlib.sha256(tpath.read_bytes()).hexdigest() == EXPECTED_SHA, "targets changed"
    tgt = pd.read_csv(tpath)
    df = pd.read_csv(psi_csv)
    per_seed = df.groupby(["exercise", "dims", "viewpoint", "seed"]).psi.mean().reset_index()
    agg = per_seed.groupby(["exercise", "dims", "viewpoint"]).psi.agg(["mean", "min", "max"]).reset_index()
    nseq = df[df.seed == df.seed.min()].groupby(["exercise", "dims", "viewpoint"]).size().rename("nseq").reset_index()
    agg = agg.merge(nseq)
    rows = []
    for t in tgt.itertuples():
        ex = PAPER_TO_DATA[t.exercise]
        for c in tgt.columns[3:]:
            vp = int(c[3:])
            ours = agg[(agg.exercise == ex) & (agg.dims == t.dims.lower()) & (agg.viewpoint == vp)]
            theirs = tgt.loc[t.Index, c]
            if len(ours) == 0:
                rows.append(dict(ex=t.exercise, dims=t.dims, vp=vp, paper=theirs, ours=np.nan, lo=np.nan, hi=np.nan,
                                 nseq=0, rel=np.nan, grade="F"))
                continue
            o = ours.iloc[0]
            rel = abs(o["mean"] - theirs) / abs(theirs)
            rows.append(dict(ex=t.exercise, dims=t.dims, vp=vp, paper=theirs, ours=o["mean"], lo=o["min"], hi=o["max"],
                             nseq=int(o.nseq), rel=rel, grade=grade(rel)))
    return pd.DataFrame(rows)


def summary(g):
    out = {}
    order = "ABCDF"
    out["grades"] = g.grade.value_counts().reindex(list(order), fill_value=0).to_dict()
    out["median_grade"] = order[int(np.median([order.index(x) for x in g.grade]))]
    w = g.loc[g.rel.idxmax()]
    out["worst"] = f"Ex{w.ex} {w.dims} {w.vp:+d}: ours {w.ours:.3f} vs {w.paper:.2f} ({100*w.rel:.0f}%)"
    out["mean_abs_dev"] = float((g.ours - g.paper).abs().mean())
    out["mean_signed_dev"] = float((g.ours - g.paper).mean())
    sign = {}
    for ex in (1, 2, 3):
        p = g[(g.ex == ex)].pivot_table(index="vp", columns="dims", values=["paper", "ours"])
        sign[ex] = dict(paper=float((p["paper"]["2D"] - p["paper"]["3D"]).mean()),
                        ours=float((p["ours"]["2D"] - p["ours"]["3D"]).mean()))
    out["2D_minus_3D"] = sign
    out["sign_match_all"] = all(np.sign(v["paper"]) == np.sign(v["ours"]) for v in sign.values())
    out["corr_over_cells"] = float(np.corrcoef(g.ours, g.paper)[0, 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("psi_csv"); ap.add_argument("spec_dir")
    ap.add_argument("--md"); ap.add_argument("--label", default="")
    a = ap.parse_args()
    g = table(a.psi_csv, a.spec_dir)
    s = summary(g)
    print(a.label, s)
    if a.md:
        lines = [f"### {a.label}", "", "| Ex | Dims | " + " | ".join(f"{v:+d}" for v in sorted(g.vp.unique())) + " |",
                 "|---|---|" + "---|" * g.vp.nunique()]
        for (ex, d), gg in g.groupby(["ex", "dims"]):
            gg = gg.sort_values("vp")
            lines.append(f"| {ex} | {d} paper | " + " | ".join(f"{x:.2f}" for x in gg.paper) + " |")
            lines.append(f"| {ex} | {d} ours | " + " | ".join(f"{x:.2f}" for x in gg.ours) + " |")
            lines.append(f"| {ex} | {d} seeds | " + " | ".join(f"{l:.2f}–{h:.2f}" for l, h in zip(gg.lo, gg.hi)) + " |")
            lines.append(f"| {ex} | {d} grade | " + " | ".join(gg.grade) + " |")
        lines += ["", f"Summary: {s}", ""]
        Path(a.md).write_text("\n".join(lines))
    g.to_csv(Path(a.psi_csv).with_suffix(".graded.csv"), index=False)


if __name__ == "__main__":
    main()
