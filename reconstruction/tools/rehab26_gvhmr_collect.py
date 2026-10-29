"""Assemble GVHMR chunk outputs (mp33.npz) into per-video MediaPipe-ordered arrays, with checks.

Writes <out>/<vid>_cam<c>_<kind>.npy (T_video, 33, 3), NaN outside finished chunks, for kind in incam, glob
(glob converted to y-down by a 180 deg rotation about x: (x, -y, -z)), and vitpose2d (GVHMR's ViTPose COCO-17 2D
mapped to MediaPipe indices, x/1080, y/1920, confidence; no heels/toes, so squat features are NaN). Also writes checks.csv, one row per chunk:
  bone lengths (median, m) for upper arm, forearm, thigh, shin; median y of ankles minus y of shoulders in incam
  (positive = feet lower in the image, y down); best frame lag of GVHMR ViTPose right-wrist track against
  MediaPipe 2D right wrist (pixels) within +-15 frames, and the correlation at lag 0.

Usage: python -I rehab26_gvhmr_collect.py <data_root> <gvhmr_out_dir> <jobs.txt> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root, gout, jobs, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
J = pd.read_csv(jobs, sep=" ", names=["video", "name", "start", "end"])
J["vid"] = J.name.str.extract(r"^(ex\d_[A-Za-z]+)_cam")[0]
J["cam"] = J.name.str.extract(r"_cam(\d)-")[0].astype(int)
COCO_TO_MP = {0: 0, 5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16, 11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28}
BONES = {"upper_arm": (12, 14), "forearm": (14, 16), "thigh": (24, 26), "shin": (26, 28)}
rows = []
for (vid, cam), g in J.groupby(["vid", "cam"]):
    ex = vid.split("_")[0]
    mp2 = np.load(root / "extracted" / "mediapipe" / ex / f"{vid}_cam{cam}_2d.npy", allow_pickle=False)
    T = mp2.shape[0]
    arr = {k: np.full((T, 33, 3), np.nan, np.float32) for k in ("incam", "glob", "vitpose2d")}
    n_done = 0
    for r in g.itertuples():
        f = gout / r.name / "mp33.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=False)
        n = r.end - r.start + 1
        assert z["incam"].shape[0] == n, (r.name, z["incam"].shape, n)
        arr["incam"][r.start:r.end + 1] = z["incam"]
        gl = z["glob"].copy(); gl[..., 1:] *= -1
        arr["glob"][r.start:r.end + 1] = gl
        vp = z["vitpose"]                     # COCO-17 pixels + conf -> MediaPipe order, normalised x/1080, y/1920
        for c, m in COCO_TO_MP.items():
            arr["vitpose2d"][r.start:r.end + 1, m] = \
                np.c_[vp[:, c, 0] / 1080, vp[:, c, 1] / 1920, vp[:, c, 2]]
        n_done += 1
        inc = z["incam"]
        row = dict(name=r.name, vid=vid, cam=cam, frames=n)
        for b, (a, c) in BONES.items():
            row[b] = float(np.median(np.linalg.norm(inc[:, a] - inc[:, c], axis=-1)))
        row["feet_below_shoulders_m"] = float(np.median(inc[:, [27, 28], 1].mean(1) - inc[:, [11, 12], 1].mean(1)))
        # frame lag: COCO-17 right wrist = 10; MediaPipe right wrist = 16
        vw = z["vitpose"][:, 10, :2]
        mw = mp2[:, 16, :2] * [1080, 1920]
        sv = np.linalg.norm(np.diff(vw, axis=0), axis=-1)          # speed at clip frame i (i = 0..n-2)
        sm = np.linalg.norm(np.diff(mw, axis=0), axis=-1)          # speed at video frame t
        best, bc, c0 = 0, -9.0, np.nan
        for L in range(-15, 16):                                   # clip frame i <-> skeleton row start + i + L
            idx = r.start + np.arange(len(sv)) + L
            ok = (idx >= 0) & (idx < len(sm))
            x, y = sv[ok], sm[idx[ok]]
            cc = float(np.corrcoef(x, y)[0, 1])
            if L == 0:
                c0 = cc
            if cc > bc:
                best, bc = L, cc
        row.update(lag=best, corr_best=bc, corr_lag0=c0,
                   wrist_px_err_median=float(np.median(np.linalg.norm(vw - mw[r.start:r.end + 1], axis=-1))))
        rows.append(row)
    if n_done:
        for k, a in arr.items():
            np.save(out / f"{vid}_cam{cam}_{k}.npy", a)
ck = pd.DataFrame(rows)
ck.to_csv(out / "checks.csv", index=False)
print(f"chunks collected {len(ck)} of {len(J)}; videos written {ck.groupby(['vid', 'cam']).ngroups if len(ck) else 0}")
if len(ck):
    print(ck[list(BONES) + ["feet_below_shoulders_m", "corr_lag0", "wrist_px_err_median"]].describe().round(3).to_string())
    print("lag counts:", ck.lag.value_counts().sort_index().to_dict())
