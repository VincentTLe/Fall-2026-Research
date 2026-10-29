"""Two-camera check on REHAB24-6. The dataset's 2D joints are exact projections of the mocap
joints through one fixed 3x4 matrix per camera (DLT residual ~1e-12 px over all 65 recordings;
calib_c17/18.npz, made by the claim check). With those matrices, triangulate 2D keypoints
(GVHMR's ViTPose, COCO-17, from the chunk runs) from cameras 17 and 18 and compare with mocap,
per repetition, on the MAIN joints that COCO has (10: no toes). On the same frames and joints,
GVHMR from each single camera is scored too, so one- and two-camera numbers are comparable.

  python rehab_twocam.py --selftest
  python rehab_twocam.py --data ... --chunks ... --calib ... --rec PM_032 ... --out twocam.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from rehab_eval import COCO2REHAB, MAIN, load_chunks, pa_errors


def dlt(X, x):
    """3x4 P from 3D points X (N, 3) and pixels x (N, 2), with Hartley normalization."""
    def norm(p):
        m, s = p.mean(0), np.sqrt(p.shape[1]) / np.linalg.norm(p - p.mean(0), axis=1).mean()
        T = np.eye(p.shape[1] + 1)
        T[:-1, :-1] *= s
        T[:-1, -1] = -s * m
        return T
    T3, T2 = norm(X), norm(x)
    Xn = (np.c_[X, np.ones(len(X))] @ T3.T)
    xn = (np.c_[x, np.ones(len(x))] @ T2.T)
    A = []
    for Xi, (u, v, w) in zip(Xn, xn):
        A.append(np.r_[np.zeros(4), -w * Xi, v * Xi])
        A.append(np.r_[w * Xi, np.zeros(4), -u * Xi])
    P = np.linalg.svd(np.asarray(A), full_matrices=False)[2][-1].reshape(3, 4)  # thin SVD: U would be 2N x 2N
    return np.linalg.inv(T2) @ P @ T3


def project(P, X):
    h = np.c_[X, np.ones(len(X))] @ P.T
    return h[:, :2] / h[:, 2:]


def triangulate(Ps, xs, ws):
    """Weighted linear triangulation of one point from several views."""
    A = []
    for P, (u, v), w in zip(Ps, xs, ws):
        A.append(w * (u * P[2] - P[0]))
        A.append(w * (v * P[2] - P[1]))
    X = np.linalg.svd(np.asarray(A))[2][-1]
    return X[:3] / X[3]


# COCO-17 index -> BODY25 index (EasyMocap's triangulated keypoints3d).
COCO2BODY25 = {5: 5, 6: 2, 7: 6, 8: 3, 9: 7, 10: 4, 11: 12, 12: 9, 13: 13, 14: 10, 15: 14, 16: 11}


def scores(t, g, hp, hg, prefix):
    """abs / PA / mid-hip root / centroid errors (mm) of world-frame joints t vs. mocap g (F, J, 3)."""
    mm = 1000.0
    return {f"{prefix}_abs": float(np.linalg.norm(t - g, axis=2).mean() * mm),
            f"{prefix}_pa": float(pa_errors(t, g)[0].mean() * mm),
            f"{prefix}_root": float(np.linalg.norm((t - hp) - (g - hg), axis=2).mean() * mm),
            f"{prefix}_cent": float(np.linalg.norm((t - t.mean(1, keepdims=True)) - (g - g.mean(1, keepdims=True)), axis=2).mean() * mm)}


def evaluate_rec(data, chunk_root, calib_dir, vid, emc_root=None, emc_data=None, min_conf=0.3):
    import pandas as pd
    seg = pd.read_csv(Path(data) / "Segmentation.csv", sep=";")
    seg = seg[seg.video_id == vid]
    ex = int(seg.exercise_id.iloc[0])
    gt3 = np.load(Path(data) / f"3d_joints/Ex{ex}/{vid}-30fps.npy")[..., :3]
    coco, rh = list(COCO2REHAB), list(COCO2REHAB.values())
    name_of = {b: nm for nm, _, b in MAIN}
    keep = [k for k, b in enumerate(rh) if b in name_of]          # 10 metric joints with a COCO match
    smpl_of = {b: a for _, a, b in MAIN}
    hips = [rh.index(16), rh.index(21)]
    cams = {}
    for cam in ("17", "18"):
        cal = np.load(Path(calib_dir) / f"calib_c{cam}.npz")
        G, _ = load_chunks(chunk_root, vid, cam, len(gt3))
        cams[cam] = (cal["P"], cal["R"], G)
    # EasyMocap HRNet triangulation (optional): BODY25 per frame, indexed by video/mocap frame.
    emc = np.full((len(gt3), 25, 4), np.nan)
    if emc_root is not None:
        start = json.load(open(Path(emc_data) / vid / "span.json"))["start"]
        for fjs in sorted((Path(emc_root) / vid / "triangulated").glob("*.json")):
            k = np.array(json.load(open(fjs))[0]["keypoints3d"])
            if start + int(fjs.stem) < len(gt3):
                emc[start + int(fjs.stem)] = k
    b25 = [COCO2BODY25[c] for c in coco]
    rows = []
    for _, r in seg.iterrows():
        f = np.arange(r.first_frame, min(r.last_frame + 1, len(gt3)))
        f = f[~np.isnan(cams["17"][2]["vitpose"][f, 0, 0]) & ~np.isnan(cams["18"][2]["vitpose"][f, 0, 0])]
        if len(f) < 10:
            continue
        tri = np.full((len(f), len(coco), 3), np.nan)
        for i, fi in enumerate(f):
            for j, c in enumerate(coco):
                obs = [(cams[cm][0], cams[cm][2]["vitpose"][fi, c, :2], cams[cm][2]["vitpose"][fi, c, 2]) for cm in cams]
                if all(w > min_conf for _, _, w in obs):
                    tri[i, j] = triangulate(*zip(*obs))
        ok = ~np.isnan(tri).any(axis=(1, 2))
        if emc_root is not None:
            e = emc[f][:, b25]
            ok &= ~np.isnan(e).any(axis=(1, 2)) & (e[..., 3] > 0).all(axis=1)
        if ok.sum() < 10:
            continue
        fo = f[ok]
        g = gt3[fo][:, rh]
        hg = g[:, hips].mean(1, keepdims=True)
        g = g[:, keep]
        row = {"vid": vid, "exercise": ex, "rep": int(r.repetition_number), "lights_on": int(r.lights_on),
               "view17": r.cam17_orientation,
               "clean": bool(int(r.mocap_erroneous) == 0 and r.extra_person_in_cam17 <= 1 and r.extra_person_in_cam18 <= 1),
               "frames": int(len(f)), "frames_scored": int(ok.sum())}
        t = tri[ok]
        row.update(scores(t[:, keep], g, t[:, hips].mean(1, keepdims=True), hg, "tri"))
        row["tri_joint_abs"] = {name_of[rh[k]]: float(np.linalg.norm(t[:, k] - g[:, i], axis=1).mean() * 1000) for i, k in enumerate(keep)}
        if emc_root is not None:
            e = emc[fo][:, b25, :3]
            row.update(scores(e[:, keep], g, e[:, hips].mean(1, keepdims=True), hg, "emc"))
            row["emc_joint_abs"] = {name_of[rh[k]]: float(np.linalg.norm(e[:, k] - g[:, i], axis=1).mean() * 1000) for i, k in enumerate(keep)}
        # GVHMR single camera, same frames, same 10 joints, rotated into world axes with the
        # recovered camera rotation (abs adds the camera centre: GVHMR's own depth/scale).
        sm = [smpl_of[rh[k]] for k in keep]
        for cam, (P, Rwc, G) in cams.items():
            C = -np.linalg.solve(P[:, :3], P[:, 3])
            pw = G["incam"][fo] @ Rwc + C
            sc = scores(pw[:, sm], g, pw[:, [1, 2]].mean(1, keepdims=True), hg, f"mono{cam}")
            sc[f"mono{cam}_pa"] = float(pa_errors(G["incam"][fo][:, sm], g)[0].mean() * 1000)  # same, PA is frame-free
            row.update(sc)
        rows.append(row)
    return rows


def selftest():
    rng = np.random.default_rng(1)
    K = np.array([[1400, 0, 960], [0, 1400, 540], [0, 0, 1.0]])
    a = 0.6
    R = np.array([[np.cos(a), 0, -np.sin(a)], [0, -1, 0], [np.sin(a), 0, np.cos(a)]])
    R = R * np.sign(np.linalg.det(R))  # keep a proper rotation
    P1 = K @ np.c_[R, [0.2, 1.0, 4.0]]
    P2 = K @ np.c_[np.eye(3) @ np.diag([1, -1, -1]), [0.0, 1.0, 4.5]]
    X = rng.uniform([-0.8, 0, -0.8], [0.8, 2, 0.8], size=(500, 3))
    x1, x2 = project(P1, X), project(P2, X)
    P1e = dlt(X, x1)
    assert np.abs(project(P1e, X) - x1).max() < 1e-6
    Xt = np.array([triangulate([P1, P2], [a_, b_], [1, 1]) for a_, b_ in zip(x1, x2)])
    assert np.abs(Xt - X).max() < 1e-6
    n1 = x1 + rng.normal(scale=2, size=x1.shape)
    e = np.linalg.norm(project(dlt(X, n1), X) - x1, axis=1).mean()
    assert e < 1.0, e
    print("selftest ok: DLT exact, triangulation exact, 2 px noise -> mean reproj vs truth", round(e, 2), "px")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data")
    ap.add_argument("--chunks")
    ap.add_argument("--calib")
    ap.add_argument("--rec", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--emc", help="EasyMocap output root: <emc>/<rec>/triangulated/")
    ap.add_argument("--emc-data", help="EasyMocap data root with <rec>/span.json")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        import pandas as pd
        rows = [x for v in a.rec for x in evaluate_rec(a.data, a.chunks, a.calib, v, a.emc, a.emc_data)]
        json.dump(rows, open(a.out, "w"), indent=1)
        df = pd.DataFrame(rows)
        c = df[df.clean]
        cols = [c for c in df.columns if "joint" not in c and c.split("_")[-1] in ("abs", "pa", "root", "cent")]
        w = lambda g: pd.Series({"reps": len(g), "frames": int(g.frames_scored.sum()),
                                 **{k: round(float(np.average(g[k], weights=g.frames_scored)), 1) for k in cols}})
        pd.set_option("display.width", 250)
        print("== clean in both cameras, per exercise"); print(c.groupby("exercise").apply(w).to_string())
        print("== clean, all"); print(w(c).to_string())
        print("== clean, lights"); print(c.groupby("lights_on").apply(w).to_string())
