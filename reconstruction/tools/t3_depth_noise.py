"""How much of GVHMR's one-camera depth error could temporal smoothing remove?

REHAB24-6, P2's GVHMR true-focal chunks against motion capture, camera frame from the authors' new
cameras. Root = mid-hip (SMPL 1/2 vs. Motive 16/21; their fixed offset ends up in the bias). Per chunk,
the depth error e(t) = z_gvhmr - z_mocap is split into
  bias  its mean over the chunk (constant offset)
  slow  a Savitzky-Golay fit over 61 frames (2 s), minus the bias
  fast  the rest: frame-to-frame jitter, the only part smoothing can remove
and GVHMR's root depth is also smoothed directly (Savitzky-Golay, 15 frames = 0.5 s) to see what an
actual smoother gains. RMS values in mm, frame-weighted over chunks.

  python t3_depth_noise.py --data DATA/rehab24-6 --chunks RUNS/t2-chunks/out-truef --calib RUNS/t2b/calib-new
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--calib", required=True)
    a = ap.parse_args()
    seg = list(csv.DictReader(open(Path(a.data) / "Segmentation.csv"), delimiter=";"))
    ex_of = {r["video_id"]: r["exercise_id"] for r in seg}
    acc = {}
    for d in sorted(Path(a.chunks).glob("PM_*-c1[78]-f*")):
        rec, cam, s, e = d.name.split("-")
        s, e = int(s[1:]), int(e)
        z = np.load(d / "joints.npz")
        root_g = z["incam"][:, [1, 2]].mean(1)
        c = np.load(Path(a.calib) / f"calib_{cam}.npz")
        mo = np.load(Path(a.data) / "3d_joints" / f"Ex{ex_of[rec]}" / f"{rec}-30fps.npy")[s:e + 1, :, :3]
        n = min(len(mo), len(root_g))
        root_m = (mo[:n][:, [16, 21]].mean(1) - c["C"]) @ c["R"].T
        zg, zm = root_g[:n, 2], root_m[:, 2]
        if n < 61:
            continue
        err = zg - zm
        bias = err.mean()
        trend = savgol_filter(err, 61, 2)
        zg_s = savgol_filter(zg, 15, 2)
        for key in (cam, "all", f"{rec}-{cam}" if rec == "PM_037" else None):
            if key is None:
                continue
            t = acc.setdefault(key, {"n": 0, "total": 0.0, "nobias": 0.0, "slow": 0.0, "fast": 0.0, "smoothed": 0.0, "zm_fast": 0.0})
            t["n"] += n
            t["total"] += np.sum(err ** 2)
            t["nobias"] += np.sum((err - bias) ** 2)
            t["slow"] += np.sum((trend - bias) ** 2)
            t["fast"] += np.sum((err - trend) ** 2)
            t["smoothed"] += np.sum((zg_s - zm - (zg_s - zm).mean()) ** 2)
            t["zm_fast"] += np.sum((zm - savgol_filter(zm, 61, 2)) ** 2)
    print("| Set | frames | depth error RMS | without the constant bias | slow part (> 2 s) | fast part (jitter) | after smoothing GVHMR (0.5 s) | mocap's own fast part |")
    print("|---|---|---|---|---|---|---|---|")
    for k, t in acc.items():
        f = lambda v: f"{1000 * np.sqrt(v / t['n']):.0f}"
        print(f"| {k} | {t['n']} | {f(t['total'])} | {f(t['nobias'])} | {f(t['slow'])} | {f(t['fast'])} | {f(t['smoothed'])} | {f(t['zm_fast'])} |")


if __name__ == "__main__":
    main()
