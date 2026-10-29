"""Time-alignment check for REHAB24-6 + GVHMR chunk runs.

For each recording and camera, over the frames covered by chunks, take each COCO joint's image
trajectory from GVHMR's ViTPose and from the dataset's projected mocap 2D joints, remove each
joint's mean (the projection has a constant offset of ~15-20 px), and find the lag in -5..5
frames that minimizes the median residual on joints that project inside the image with
confidence > 0.5. Lag L means video frame i matches mocap frame i + L. Also prints one joint
over time for one recording (--trace).

  python rehab_sync_check.py --data ~/fall2026/data/rehab24-6 --chunks ~/fall2026/runs/t2-chunks/out \
      --rec PM_032 ... [--trace PM_118 17 16]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rehab_eval import COCO2REHAB, load_chunks


def lag_curve(kp, conf, gt2, frames, wh, lags=range(-5, 6)):
    coco, rh = list(COCO2REHAB), list(COCO2REHAB.values())
    res = {}
    for L in lags:
        f = frames[(frames + L >= 0) & (frames + L < len(gt2))]
        d, g, c = kp[f][:, coco, :2], gt2[f + L][:, rh], conf[f][:, coco]
        inside = (g[..., 0] >= 0) & (g[..., 0] < wh[0]) & (g[..., 1] >= 0) & (g[..., 1] < wh[1]) & (c > 0.5)
        r = []
        for j in range(len(coco)):
            m = inside[:, j]
            if m.sum() < 50:
                continue
            dj, gj = d[m, j] - d[m, j].mean(0), g[m, j] - g[m, j].mean(0)
            r.append(np.linalg.norm(dj - gj, axis=1))
        res[L] = float(np.median(np.concatenate(r)))
    return min(res, key=res.get), res


def main(a):
    seg = pd.read_csv(Path(a.data) / "Segmentation.csv", sep=";")
    for vid in a.rec:
        ex = int(seg[seg.video_id == vid].exercise_id.iloc[0])
        for cam in ("17", "18"):
            gt2 = np.load(Path(a.data) / f"2d_joints/Ex{ex}/{vid}-c{cam}-30fps.npy")
            G, _ = load_chunks(a.chunks, vid, cam, len(gt2))
            frames = np.where(~np.isnan(G["vitpose"][:, 0, 0]))[0]
            wh = (1920, 1080) if cam == "17" else (1080, 1920)
            L, res = lag_curve(G["vitpose"], G["vitpose"][..., 2], gt2, frames, wh)
            print(f"{vid} c{cam} frames {len(frames)} best lag {L:+d}  residual px at -2..+2: "
                  + " ".join(f"{res[k]:.1f}" for k in range(-2, 3)))
    if a.trace:
        vid, cam, coco_j = a.trace[0], a.trace[1], int(a.trace[2])
        ex = int(seg[seg.video_id == vid].exercise_id.iloc[0])
        gt2 = np.load(Path(a.data) / f"2d_joints/Ex{ex}/{vid}-c{cam}-30fps.npy")
        G, _ = load_chunks(a.chunks, vid, cam, len(gt2))
        r = seg[seg.video_id == vid].iloc[2]
        print(f"trace {vid} c{cam} COCO joint {coco_j} (REHAB {COCO2REHAB[coco_j]}), image y in px, repetition {r.repetition_number}:")
        for f in range(int(r.first_frame), int(r.last_frame) + 1, 6):
            print(f"  frame {f}: ViTPose {G['vitpose'][f, coco_j, 1]:7.1f}   mocap 2D {gt2[f, COCO2REHAB[coco_j], 1]:7.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--rec", nargs="+", required=True)
    ap.add_argument("--trace", nargs=3, metavar=("REC", "CAM", "COCO_JOINT"))
    main(ap.parse_args())
