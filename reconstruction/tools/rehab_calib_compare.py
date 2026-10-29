"""Which REHAB24-6 calibration matches the real video? Project the mocap joints with the authors' "old"
and "new" cameras (rehab_authors_calib.py output) and compare with two independent 2D detectors:
GVHMR's ViTPose-H (COCO-17, from the chunk runs) and EasyMocap's HRNet-W48 (BODY25 layout, from the
two-camera runs). Joints: shoulders, elbows, wrists, knees, ankles (hips left out: Motive defines them
differently). Frames inside repetitions, detector confidence > 0.5, projection inside the image.

Per recording, camera, calibration and detector: median |error| and the split of the 2D residual into
a constant part (|mean residual vector| per joint) and a varying part (RMS of residual minus its mean).

  python rehab_calib_compare.py --data ~/fall2026/data/rehab24-6 --chunks ~/fall2026/runs/t2-chunks/out \
      --emc ~/fall2026/runs/t2-emc --emc-data ~/fall2026/data/rehab24-6-emc --calib ~/fall2026/runs/t2b \
      --rec PM_032 ... --out compare.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rehab_eval import load_chunks

# name: (REHAB index, COCO-17 index, BODY25 index)
JOINTS = {"L_shoulder": (7, 5, 5), "R_shoulder": (12, 6, 2), "L_elbow": (8, 7, 6), "R_elbow": (13, 8, 3),
          "L_wrist": (9, 9, 7), "R_wrist": (14, 10, 4), "L_knee": (17, 13, 13), "R_knee": (22, 14, 10),
          "L_ankle": (18, 15, 14), "R_ankle": (23, 16, 11)}
VIEW = {"17": 0, "18": 1}   # order of views in EasyMocap's detections (--views 17 18)


def project(P, X):
    h = X @ P[:, :3].T + P[:, 3]
    return h[..., :2] / h[..., 2:]


def load_hrnet(emc_root, emc_data, vid, cam, n):
    """HRNet 2D from EasyMocap's detections/<i>.npz, indexed by mocap/video frame; NaN where missing."""
    out = np.full((n, 25, 3), np.nan)
    start = json.load(open(Path(emc_data) / vid / "span.json"))["start"]
    for f in sorted((Path(emc_root) / vid / "detections").glob("*.npz")):
        k = np.load(f)["keypoints2d"]
        i = start + int(f.stem)
        if i < n and k.shape[0] > VIEW[cam]:
            out[i] = k[VIEW[cam]]
    return out


def main(a):
    seg = pd.read_csv(Path(a.data) / "Segmentation.csv", sep=";")
    rows = []
    for vid in a.rec:
        s = seg[seg.video_id == vid]
        ex = int(s.exercise_id.iloc[0])
        X = np.load(Path(a.data) / f"3d_joints/Ex{ex}/{vid}-30fps.npy")[..., :3]
        frames = np.concatenate([np.arange(r.first_frame, min(r.last_frame + 1, len(X))) for r in s.itertuples()])
        for cam in ("17", "18"):
            wh = (1920, 1080) if cam == "17" else (1080, 1920)
            G, _ = load_chunks(a.chunks, vid, cam, len(X))
            det = {"ViTPose": (G["vitpose"], 1), "HRNet": (load_hrnet(a.emc, a.emc_data, vid, cam, len(X)), 2)}
            for which in ("old", "new"):
                P = np.load(Path(a.calib) / f"calib_{which}_c{cam}.npz")["P"]
                for dname, (kp, col) in det.items():
                    for jn, idx in JOINTS.items():
                        g = project(P, X[frames, idx[0]])
                        d = kp[frames, idx[col]]
                        ok = (~np.isnan(d).any(1) & (d[:, 2] > 0.5) & (g[:, 0] >= 0) & (g[:, 0] < wh[0])
                              & (g[:, 1] >= 0) & (g[:, 1] < wh[1]))
                        if ok.sum() < 20:
                            continue
                        r = d[ok, :2] - g[ok]
                        mean = r.mean(0)
                        rows.append({"vid": vid, "exercise": ex, "cam": cam, "calib": which, "detector": dname, "joint": jn,
                                     "n": int(ok.sum()), "median_px": float(np.median(np.linalg.norm(r, axis=1))),
                                     "const_px": float(np.linalg.norm(mean)), "mean_dx": float(mean[0]), "mean_dy": float(mean[1]),
                                     "vary_px": float(np.sqrt(((r - mean) ** 2).sum(1).mean()))})
    df = pd.DataFrame(rows)
    df.to_json(a.out, orient="records", indent=1)
    pd.set_option("display.width", 220)
    w = lambda g: pd.Series({"n": int(g.n.sum()), **{c: round(float(np.average(g[c], weights=g.n)), 1)
                                                     for c in ("median_px", "const_px", "vary_px")}})
    print("== per camera and detector (joint observations pooled, n-weighted means of per-recording values)")
    t = df.groupby(["cam", "detector", "calib"]).apply(w).unstack("calib")
    print(t.to_string())
    print("\n== median px per camera, detector and joint: old / new")
    for (cam, dname), g in df.groupby(["cam", "detector"]):
        o = g[g.calib == "old"].groupby("joint").apply(lambda x: np.average(x.median_px, weights=x.n))
        n = g[g.calib == "new"].groupby("joint").apply(lambda x: np.average(x.median_px, weights=x.n))
        print(f"cam {cam} {dname:7s}", {j: f"{o[j]:.1f}/{n[j]:.1f}" for j in JOINTS if j in o})
    print("\n== per recording: constant / varying part (px), all 10 joints n-weighted, ViTPose and HRNet; old -> new")
    for (vid, cam), g in df.groupby(["vid", "cam"]):
        parts = []
        for dname in ("ViTPose", "HRNet"):
            q = {c: g[(g.detector == dname) & (g.calib == c)] for c in ("old", "new")}
            parts.append(f"{dname} const {np.average(q['old'].const_px, weights=q['old'].n):5.1f} -> "
                         f"{np.average(q['new'].const_px, weights=q['new'].n):5.1f}, vary "
                         f"{np.average(q['old'].vary_px, weights=q['old'].n):5.1f} -> {np.average(q['new'].vary_px, weights=q['new'].n):5.1f}")
        print(f"{vid} c{cam} Ex{g.exercise.iloc[0]}: " + " | ".join(parts))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("data", "chunks", "emc", "emc-data", "calib", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--rec", nargs="+", required=True)
    main(ap.parse_args())
