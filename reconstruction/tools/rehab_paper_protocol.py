"""GVHMR on the REHAB24-6 benchmark protocol of "Pose Estimation Analysis and Fine-Tuning on the REHAB24-6
Rehabilitation Dataset" (Information Systems 2025), Sec. 4.2.1-4.2.3 and 6.1-6.4, as far as the paper
specifies it. Choices the paper leaves open are marked OPEN and exposed as options.

- Joints: the paper's 12 (shoulders, elbows, wrists, hips, knees, ankles), hips included.
- Frames: inside annotated repetitions (Segmentation.csv). mocap_erroneous repetitions: OPEN (not mentioned);
  kept by default, --drop-erroneous removes them.
- Invalid frames: frames without a prediction, and frames where the IoU of the 2D boxes around the GT and
  predicted leg + shoulder joints is below a "low threshold" (value OPEN; --iou, default 0.1). GT 2D = the
  dataset's 2D joints (projected with the authors' old camera); predicted 2D = GVHMR's own 2D input (ViTPose).
- BestFit: Procrustes (scale, rotation, translation) per frame (granularity OPEN; per frame = standard PA-MPJPE).
  Also reported: "bestfit_rep", one Procrustes per repetition over all its valid frames (a stricter reading).
- BBox (my reading of Sec. 4.2.3; OPEN in detail): GT rotated into the camera frame with the old camera, no
  rotation fitted. Prediction (GVHMR camera frame) scaled by the least-squares scale; x/y placed so that the
  offset between predicted and GT 3D box centres, relative to the GT 3D box size, equals the offset between
  the predicted and GT 2D box centres relative to the GT 2D box size; depth shifted by the mean depth
  difference. Reported: 3D error, 2D (x/y) part and depth (z) part, per joint, in cm.
- Aggregation: frame-weighted mean over all frames (MPJPE "across all poses and video sequences"); the
  video-weighted mean (each recording x camera counted once) is reported too, since "across all videos" is OPEN.

  python rehab_paper_protocol.py --data ~/fall2026/data/rehab24-6 --chunks <root with {vid}-c{cam}-f* dirs> \
      --calib-old ~/fall2026/runs/t2b/calib-old --rec PM_032 ... --out rows.json
  python rehab_paper_protocol.py --summary rows.json
  python rehab_paper_protocol.py --selftest
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rehab_eval import load_chunks, umeyama

# name: (SMPL-24 index, REHAB24-6 index, COCO-17 index)
J12 = {"L_shoulder": (16, 7, 5), "R_shoulder": (17, 12, 6), "L_elbow": (18, 8, 7), "R_elbow": (19, 13, 8),
       "L_wrist": (20, 9, 9), "R_wrist": (21, 14, 10), "L_hip": (1, 16, 11), "R_hip": (2, 21, 12),
       "L_knee": (4, 17, 13), "R_knee": (5, 22, 14), "L_ankle": (7, 18, 15), "R_ankle": (8, 23, 16)}
BOX_JOINTS = ["L_shoulder", "R_shoulder", "L_hip", "R_hip", "L_knee", "R_knee", "L_ankle", "R_ankle"]  # legs + shoulders
VIEW18 = {"front": "profile", "half-profile": "half-profile", "profile": "front"}


def box(p):
    return np.r_[p.min(0), p.max(0)]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def bestfit_err(p, g):
    s, R, t = umeyama(p, g)
    return np.linalg.norm(s * p @ R.T + t - g, axis=1)


def bbox_err(p, g, p2, g2):
    """p: predicted joints, camera frame (J,3); g: GT joints, camera frame (J,3); p2, g2: 2D joints (J,2)."""
    pc, gc = p - p.mean(0), g - g.mean(0)
    s = (pc * gc).sum() / (pc ** 2).sum()                     # least-squares scale, no rotation
    q = s * p
    b2p, b2g = box(p2), box(g2)
    size2 = b2g[2:] - b2g[:2]
    off2 = ((b2p[:2] + b2p[2:]) / 2 - (b2g[:2] + b2g[2:]) / 2) / size2
    b3q, b3g = box(q[:, :2]), box(g[:, :2])
    size3 = b3g[2:] - b3g[:2]
    target = (b3g[:2] + b3g[2:]) / 2 + off2 * size3            # where the predicted 3D box centre should be
    q[:, :2] += target - (b3q[:2] + b3q[2:]) / 2
    q[:, 2] += (g[:, 2] - q[:, 2]).mean()                      # depth shift
    d = q - g
    return np.linalg.norm(d, axis=1), np.linalg.norm(d[:, :2], axis=1), np.abs(d[:, 2])


def evaluate(a):
    seg = pd.read_csv(Path(a.data) / "Segmentation.csv", sep=";")
    names = list(J12)
    ia = [J12[n][0] for n in names]
    ib = [J12[n][1] for n in names]
    ic = [J12[n][2] for n in names]
    bj = [names.index(n) for n in BOX_JOINTS]
    rows = []
    for vid in a.rec:
        s = seg[seg.video_id == vid]
        ex = int(s.exercise_id.iloc[0])
        X = np.load(Path(a.data) / f"3d_joints/Ex{ex}/{vid}-30fps.npy")[..., :3]
        for cam in ("17", "18"):
            g2all = np.load(Path(a.data) / f"2d_joints/Ex{ex}/{vid}-c{cam}-30fps.npy")
            cal = np.load(Path(a.calib_old) / f"calib_c{cam}.npz")
            Rw, C = cal["R"], cal["C"]
            G, _ = load_chunks(a.chunks, vid, cam, len(X))
            for r in s.itertuples():
                f = np.arange(r.first_frame, min(r.last_frame + 1, len(X)))
                has = ~np.isnan(G["incam"][f, 0, 0])
                acc = {"frames": len(f), "detected": int(has.sum())}
                ious = np.zeros(len(f))
                bf, b3, b2, bz = [], [], [], []
                for k, fi in enumerate(f):
                    if not has[k]:
                        continue
                    p2 = G["vitpose"][fi][ic, :2]
                    g2 = g2all[fi][ib]
                    ious[k] = iou(box(p2[bj]), box(g2[bj]))
                    p = G["incam"][fi][ia]
                    g = X[fi][ib]
                    bf.append(bestfit_err(p, g))
                    e3, e2, ez = bbox_err(p, (g - C) @ Rw.T, p2, g2)
                    b3.append(e3), b2.append(e2), bz.append(ez)
                fv = f[has]
                bf, b3, b2, bz = (np.array(x) * 100 for x in (bf, b3, b2, bz))   # m -> cm
                iv = ious[has]
                for thr in (0.05, 0.1, 0.3):
                    ok = iv >= thr
                    acc[f"valid@{thr}"] = int(ok.sum())
                    for nm, e in (("bestfit", bf), ("bbox3d", b3), ("bbox2d", b2), ("bboxz", bz)):
                        acc[f"{nm}_sum@{thr}"] = float(e[ok].mean(1).sum()) if ok.any() else 0.0
                    if ok.any():   # one similarity transform for the whole repetition
                        P = G["incam"][fv[ok]][:, ia].reshape(-1, 3)
                        Q = X[fv[ok]][:, ib].reshape(-1, 3)
                        sc, Rr, tt = umeyama(P, Q)
                        er = np.linalg.norm(sc * P @ Rr.T + tt - Q, axis=1).reshape(-1, 12) * 100
                        acc[f"bestfit_rep_sum@{thr}"] = float(er.mean(1).sum())
                    else:
                        acc[f"bestfit_rep_sum@{thr}"] = 0.0
                    if thr == 0.1:
                        acc["joint_bestfit_sum"] = dict(zip(names, (bf[ok].sum(0) if ok.any() else np.zeros(12)).tolist()))
                rows.append({"vid": vid, "cam": cam, "exercise": ex, "rep": int(r.repetition_number),
                             "view": r.cam17_orientation if cam == "17" else VIEW18[r.cam17_orientation],
                             "lights_on": int(r.lights_on), "erroneous": int(r.mocap_erroneous),
                             "extra_person": int(getattr(r, f"extra_person_in_cam{cam}")), **acc})
    json.dump(rows, open(a.out, "w"))
    print(f"{len(rows)} repetition x camera rows -> {a.out}")


def summarize(path, thr=0.1, drop_err=False):
    d = pd.DataFrame(json.load(open(path)))
    if drop_err:
        d = d[d.erroneous == 0]
    v = f"valid@{thr}"
    mets = ("bestfit", "bestfit_rep", "bbox3d", "bbox2d", "bboxz")

    def fw(g):  # frame-weighted
        n = g[v].sum()
        return pd.Series({"reps": len(g), "frames": int(g.frames.sum()), "valid_%": round(100 * n / g.frames.sum(), 2),
                          **{m: round(g[f"{m}_sum@{thr}"].sum() / n, 2) for m in mets}})

    def vw(g):  # video-weighted: each recording x camera once
        per = g.groupby(["vid", "cam"]).apply(lambda x: pd.Series({m: x[f"{m}_sum@{thr}"].sum() / x[v].sum() for m in mets}))
        return pd.Series({m: round(per[m].mean(), 2) for m in mets})
    return d, fw, vw, v


def report(path):
    for drop in (False, True):
        for thr in (0.1, 0.05, 0.3):
            d, fw, vw, v = summarize(path, thr, drop)
            tag = f"IoU >= {thr}, mocap_erroneous {'dropped' if drop else 'kept'}"
            print(f"\n== ALL (Table 5 style), {tag}: frame-weighted | video-weighted")
            print(pd.concat([fw(d), vw(d).add_prefix("vw_")]).to_string())
            if thr == 0.1 and not drop:
                print("\n== per camera (frame-weighted)")
                print(d.groupby("cam").apply(fw).to_string())
                print("\n== per exercise, camera and view (Table 6 style, frame-weighted)")
                print(d.groupby(["exercise", "cam", "view"]).apply(fw).to_string())
                print("\n== per exercise (frame-weighted, both cameras)")
                print(d.groupby("exercise").apply(fw).to_string())
                n = d[v].sum()
                js = pd.DataFrame(list(d.joint_bestfit_sum)).sum() / n
                print("\n== BestFit per joint (cm)", js.round(2).to_dict())


def selftest():
    rng = np.random.default_rng(0)
    g = rng.normal(size=(12, 3)) * 0.4 + [0, 0, 4]
    a = 0.5
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    p = 0.8 * g @ R.T + [1, 2, 3]
    assert bestfit_err(p, g).max() < 1e-9
    # BBox: a pure scale + depth shift of the GT, with identical 2D boxes, is recovered exactly.
    p = 0.8 * g + [0, 0, 1.5]
    g2 = g[:, :2] / g[:, 2:] * 1000
    e3, e2, ez = bbox_err(p, g, g2, g2)
    assert e3.max() < 1e-9, e3.max()
    # A 2D box shifted right by 10% of its width moves the prediction right by 10% of the 3D box width.
    w2 = np.ptp(g2[:, 0])
    _, e2, ez = bbox_err(g.copy(), g, g2 + [0.1 * w2, 0], g2)
    assert np.allclose(e2, 0.1 * np.ptp(g[:, 0])) and ez.max() < 1e-9
    assert abs(iou(np.array([0, 0, 2, 2.]), np.array([1, 0, 3, 2.])) - 1 / 3) < 1e-12
    print("selftest ok: BestFit exact, BBox recovers scale+depth exactly, 2D box shift maps to 3D, IoU 1/3")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--chunks")
    ap.add_argument("--calib-old")
    ap.add_argument("--rec", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--summary")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.summary:
        report(a.summary)
    else:
        evaluate(a)
