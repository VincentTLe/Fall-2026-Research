"""Markdown tables for the T2 result file from rehab_eval.py / rehab_twocam.py outputs of two GVHMR runs
(default focal = image diagonal, and the focal recovered by DLT). Frame-weighted means over clean
repetitions; "excl." drops repetitions that overlap an outlier chunk.

Outlier chunk rule (no 3D ground truth used): in rehab_eval's per-chunk check, the median 2D distance
between ViTPose and the projected mocap joints on fast-moving joints is > 40 px at offset 0. A
single-camera repetition is dropped if it overlaps such a chunk of that camera; a two-camera
repetition if it overlaps one in either camera.

  python rehab_report.py --default ~/fall2026/runs/t2-results --truef ~/fall2026/runs/t2-results-truef \
      --seg ~/fall2026/data/rehab24-6/Segmentation.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rehab_eval import EXTRA, HIPJ, MAIN

OUTLIER_PX = 40.0


def outlier_chunks(checks):
    return [(c["vid"], c["cam"], c["chunk"][0], c["chunk"][1], c["offset_px"]["0"]) for c in checks
            if c["offset_px"]["0"] > OUTLIER_PX]


def rep_span(seg, vid, rep):
    r = seg[(seg.video_id == vid) & (seg.repetition_number == rep)].iloc[0]
    return int(r.first_frame), int(r.last_frame)


def overlaps(seg, out, vid, rep, cams):
    a, b = rep_span(seg, vid, rep)
    return any(v == vid and c in cams and a <= e and s <= b for v, c, s, e, _ in out)


def wmean(g, col, wcol="frames"):
    return float(np.average(g[col], weights=g[wcol]))


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.1f}" if isinstance(v, float) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def mono_table(rows, keys):
    out = []
    for k, g in rows.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        out.append({**dict(zip(keys, k)), "reps": len(g), "frames": int(g.frames.sum()),
                    **{m: round(wmean(g, m), 1) for m in ("pa", "root", "cent", "abs")}})
    return pd.DataFrame(out)


def main(a):
    seg = pd.read_csv(a.seg, sep=";")
    D = json.load(open(Path(a.default) / "reps.json"))
    Tf = json.load(open(Path(a.truef) / "reps.json"))
    out = outlier_chunks(D["checks"])
    print("## Outlier chunks (fast-joint 2D median > 40 px at offset 0)")
    for v, c, s, e, px in out:
        print(f"- {v} camera {c}, frames {s}-{e}: {px:.0f} px")
    d = pd.DataFrame(D["rows"])
    t = pd.DataFrame(Tf["rows"])
    key = ["vid", "cam", "rep"]
    assert len(d) == len(t) and (d[key].values == t[key].values).all() and (d.frames.values == t.frames.values).all()
    for df in (d, t):
        df["outlier"] = [overlaps(seg, out, r.vid, r.rep, (r.cam,)) for r in df.itertuples()]
    dc, tc = d[d.clean], t[t.clean]
    print("\n## Main: clean repetitions, per exercise and camera (default focal | true focal)")
    m = mono_table(dc, ["exercise", "cam"]).merge(mono_table(tc, ["exercise", "cam"]), on=["exercise", "cam", "reps", "frames"],
                                                     suffixes=("", " (true f)"))
    print(md(m[["exercise", "cam", "reps", "frames", "pa", "pa (true f)", "root", "root (true f)", "cent", "cent (true f)", "abs", "abs (true f)"]]))
    for name, sub in (("all clean", lambda x: x), ("clean, excl. outlier chunks", lambda x: x[~x.outlier])):
        print(f"\n## Per camera, {name}")
        m = mono_table(sub(dc), ["cam"]).merge(mono_table(sub(tc), ["cam"]), on=["cam", "reps", "frames"], suffixes=("", " (true f)"))
        print(md(m))
    print("\n## By camera and view (true focal, clean)")
    print(md(mono_table(tc, ["cam", "view"])))
    print("\n## By lights (true focal, clean), per camera")
    print(md(mono_table(tc, ["cam", "lights_on"])))
    print("\n## Per recording and camera (true focal, clean)")
    g = mono_table(tc, ["exercise", "vid", "cam"])
    g["subject"] = [int(tc[tc.vid == v].person.iloc[0]) for v in g.vid]
    g["lights"] = [int(tc[tc.vid == v].lights_on.iloc[0]) for v in g.vid]
    print(md(g[["exercise", "vid", "subject", "lights", "cam", "reps", "frames", "pa", "root", "cent", "abs"]]))
    names = [n for n, _, _ in MAIN + HIPJ + EXTRA]
    print("\n## Per-joint PA error (mm), true focal, clean (PA fitted on the 12 metric joints)")
    rows = []
    for cam in ("17", "18"):
        g = tc[tc.cam == cam]
        pa = {n: np.average([x[n] for x in g.joint_pa], weights=g.frames) for n in names}
        bias = {n: np.average([np.linalg.norm(x[n]) for x in g.joint_mean_resid], weights=g.frames) for n in names}
        rows.append({"cam": cam, **{n: round(float(pa[n]), 1) for n in names}})
        rows.append({"cam": cam + " per-rep |mean resid|", **{n: round(float(bias[n]), 1) for n in names}})
        worst = sorted([n for n, _, _ in MAIN], key=lambda n: -pa[n])[:5]
        print(f"- camera {cam}, five worst metric joints: " + ", ".join(f"{n} {pa[n]:.1f}" for n in worst))
    print(md(pd.DataFrame(rows)))

    # Two cameras.
    D2 = pd.DataFrame(json.load(open(Path(a.default) / "twocam.json")))
    T2 = pd.DataFrame(json.load(open(Path(a.truef) / "twocam.json")))
    assert (D2[["vid", "rep"]].values == T2[["vid", "rep"]].values).all()
    for df in (D2, T2):
        df["outlier"] = [overlaps(seg, out, r.vid, r.rep, ("17", "18")) for r in df.itertuples()]
    cols = ["tri_abs", "tri_pa", "tri_root", "tri_cent", "emc_abs", "emc_pa", "emc_root", "emc_cent"]
    mono = [f"mono{c}_{m}" for c in ("17", "18") for m in ("abs", "pa", "root", "cent")]

    def tw(df, keys=None):
        out_ = []
        groups = df.groupby(keys) if keys else [("all", df)]
        for k, g in groups:
            out_.append({"group": k, "reps": len(g), "frames": int(g.frames_scored.sum()),
                         **{c: round(wmean(g, c, "frames_scored"), 1) for c in cols + mono}})
        return pd.DataFrame(out_)
    for name, sub in (("all clean", lambda x: x[x.clean]), ("clean, excl. outlier chunks", lambda x: x[x.clean & ~x.outlier])):
        print(f"\n## One vs. two cameras, same frames, 10 joints, {name}: default focal / true focal for GVHMR")
        a_, b_ = tw(sub(D2)), tw(sub(T2))
        assert (a_[cols].values == b_[cols].values).all()  # two-camera numbers do not depend on GVHMR's focal
        r = a_[["reps", "frames"] + cols].iloc[0].to_dict()
        for c in mono:
            r[c] = f"{a_[c].iloc[0]:.1f} / {b_[c].iloc[0]:.1f}"
        for k, v in r.items():
            print(f"- {k}: {v}")
    print("\n## One vs. two cameras per exercise (true focal, clean, all chunks)")
    print(md(tw(T2[T2.clean], "exercise")[["group", "reps", "frames", "tri_abs", "tri_pa", "emc_abs", "emc_pa",
                                            "mono17_pa", "mono18_pa", "mono17_cent", "mono18_cent", "mono17_abs", "mono18_abs"]]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--default", required=True)
    ap.add_argument("--truef", required=True)
    ap.add_argument("--seg", required=True)
    main(ap.parse_args())
