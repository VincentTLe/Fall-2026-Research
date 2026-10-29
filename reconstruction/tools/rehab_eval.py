"""Compare GVHMR (SMPL 24 joints) with REHAB24-6 mocap joints (26-joint OptiTrack skeleton),
per repetition, from GVHMR runs on chunks of the repetition spans (see gvhmr_run.py).

Metrics on the 12 limb joints in MAIN, in mm:
  PA-MPJPE   per-frame similarity alignment (rotation, translation, scale; Umeyama).
  MPJPE-root GVHMR's camera-frame joints rotated into the mocap world with the camera rotation
             recovered by DLT (calib_c17/18.npz), then per-frame root translation; no fitted
             rotation or scale. Root = midpoint of the two hip joints in each skeleton. Motive
             places its hip joints (LeftUpLeg/RightUpLeg) level with "Hips", about 9-13 cm higher
             than SMPL (thigh 0.52 m vs. 0.39 m on PM_101), so this root carries a near-constant
             definition offset into every joint.
  MPJPE-cent same rotation, but the per-frame translation aligns the centroid of the 12 joints,
             which does not depend on either skeleton's pelvis definition.
  MPJPE-abs  no alignment at all: GVHMR's camera-frame joints mapped to the world with the full
             recovered camera (R, centre). Includes GVHMR's depth/scale error.
Hips, neck and head are listed per joint only.

Time: chunk frame k is video frame start + k and is compared with mocap frame start + k (offset 0:
videos have the same frame count as the mocap or one fewer, and the independent check found sync
within +-1 frame). The offset is still estimated per chunk, as a check only, by matching GVHMR's
ViTPose 2D keypoints with the dataset's projected 2D joints on fast-moving joints that project
inside the image (offsets -3..3). Repetitions in which a projected LeftHand/RightHand leaves the
image of that camera are flagged hand_out.

Usage:
  python rehab_eval.py --selftest
  python rehab_eval.py --data ~/fall2026/data/rehab24-6 --chunks ~/fall2026/runs/t2-chunks/out \
      --calib ~/fall2026/runs/t2-gvhmr --rec PM_032 ... --out reps.json
  python rehab_eval.py --summary reps.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

REHAB = ["Hips", "Spine", "Spine1", "Neck", "Head", "Head_end",
         "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHand_end",
         "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHand_end",
         "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "LeftToeBase_end",
         "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "RightToeBase_end"]

# (name, SMPL-24 index, REHAB24-6 index). MAIN is used for all metrics. HIPJ defines the root.
# HIPJ and EXTRA appear only in the per-joint table, because Motive places hips, neck and head
# differently from SMPL (see the docstring).
MAIN = [("L_knee", 4, 17), ("R_knee", 5, 22),
        ("L_ankle", 7, 18), ("R_ankle", 8, 23), ("L_toe", 10, 19), ("R_toe", 11, 24),
        ("L_shoulder", 16, 7), ("R_shoulder", 17, 12), ("L_elbow", 18, 8), ("R_elbow", 19, 13),
        ("L_wrist", 20, 9), ("R_wrist", 21, 14)]
HIPJ = [("L_hip", 1, 16), ("R_hip", 2, 21)]
EXTRA = [("neck", 12, 3), ("head", 15, 4)]

# COCO-17 index -> REHAB24-6 index, for the 2D time-offset check.
COCO2REHAB = {5: 7, 6: 12, 7: 8, 8: 13, 9: 9, 10: 14, 11: 16, 12: 21, 13: 17, 14: 22, 15: 18, 16: 23}


def umeyama(X, Y, scale=True):
    """Return s, R, t minimizing |s R X + t - Y| for point sets X, Y of shape (N, 3)."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (Xc ** 2).sum() if scale else 1.0
    return s, R, my - s * R @ mx


def pa_errors(pred, gt):
    """Per-frame similarity alignment. Returns errors (F, J) in input units and scales (F,)."""
    err, scales = np.empty(pred.shape[:2]), np.empty(len(pred))
    for f in range(len(pred)):
        s, R, t = umeyama(pred[f], gt[f])
        err[f] = np.linalg.norm(s * pred[f] @ R.T + t - gt[f], axis=1)
        scales[f] = s
    return err, scales


def root_errors(pred, gt, root_p=None, root_g=None):
    """One rotation per sequence, per-frame root translation. Roots (F, 3) default to the
    centroid of the given joints. Returns errors (F, J) and R."""
    p = pred - (pred.mean(1) if root_p is None else root_p)[:, None]
    g = gt - (gt.mean(1) if root_g is None else root_g)[:, None]
    _, R, _ = umeyama(p.reshape(-1, 3), g.reshape(-1, 3), scale=False)
    return np.linalg.norm(p @ R.T - g, axis=2), R


def best_offset(kp2d, conf, gt2d, offsets=range(-3, 4), min_conf=0.5, fast_pct=90, wh=None):
    """Median 2D distance (px) between detected and projected joints for each offset.
    kp2d, conf: video frames (Fv, J, 2), (Fv, J). gt2d: mocap frames (Fm, J, 2).
    Only joint observations moving faster than the fast_pct percentile of projected-joint
    speed are used: still joints look the same at every offset and hide a lag."""
    speed = np.zeros(gt2d.shape[:2])
    speed[1:-1] = np.linalg.norm(gt2d[2:] - gt2d[:-2], axis=2) / 2
    thr = np.percentile(speed, fast_pct)
    if wh is not None:  # ignore joints projected outside the image: the detector cannot see them
        out = (gt2d[..., 0] < 0) | (gt2d[..., 0] >= wh[0]) | (gt2d[..., 1] < 0) | (gt2d[..., 1] >= wh[1])
        speed[out] = 0
    res = {}
    for o in offsets:
        i = np.arange(len(kp2d))
        ok = (i + o >= 0) & (i + o < len(gt2d))
        d = np.linalg.norm(kp2d[ok] - gt2d[i[ok] + o], axis=2)
        use = (conf[ok] > min_conf) & (speed[i[ok] + o] > thr)
        res[o] = float(np.median(d[use]))
    return min(res, key=res.get), res


VIEW18 = {"front": "profile", "half-profile": "half-profile", "profile": "front"}


def load_chunks(chunk_root, vid, cam, n_video):
    """Stitch chunk outputs into arrays indexed by video frame (NaN where no chunk)."""
    out = {"incam": np.full((n_video, 24, 3), np.nan), "vitpose": np.full((n_video, 17, 3), np.nan),
           "bbx": np.full((n_video, 4), np.nan)}
    chunks = []
    for d in sorted(Path(chunk_root).glob(f"{vid}-c{cam}-f*")):
        s, e = (int(x[1:]) for x in d.name.split("-")[-2:])
        J = np.load(d / "joints.npz")
        assert len(J["incam"]) == e - s + 1, (d, len(J["incam"]))
        out["incam"][s:e + 1], out["vitpose"][s:e + 1], out["bbx"][s:e + 1] = J["incam"], J["vitpose"], J["bbx_xyxy"]
        chunks.append((s, e))
    return out, chunks


def evaluate_rec(data, chunk_root, calib_dir, vid, cam):
    import pandas as pd
    seg = pd.read_csv(Path(data) / "Segmentation.csv", sep=";")
    seg = seg[seg.video_id == vid]
    ex = int(seg.exercise_id.iloc[0])
    gt3 = np.load(Path(data) / f"3d_joints/Ex{ex}/{vid}-30fps.npy")[..., :3]
    gt2 = np.load(Path(data) / f"2d_joints/Ex{ex}/{vid}-c{cam}-30fps.npy")
    cal = np.load(Path(calib_dir) / f"calib_c{cam}.npz")
    Rwc, C = cal["R"], cal["C"]  # world->camera rotation, camera centre (m)
    G, chunks = load_chunks(chunk_root, vid, cam, len(gt3))
    coco, rh = list(COCO2REHAB), list(COCO2REHAB.values())
    wh = (1920, 1080) if cam == "17" else (1080, 1920)

    checks = []
    for s, e in chunks:
        o, tab = best_offset(G["vitpose"][s:e + 1, coco, :2], G["vitpose"][s:e + 1, coco, 2], gt2[s:, rh], wh=wh)
        mid, b = gt2[s:e + 1, [16, 21]].mean(1), G["bbx"][s:e + 1]
        inside = ((mid[:, 0] > b[:, 0]) & (mid[:, 0] < b[:, 2]) & (mid[:, 1] > b[:, 1]) & (mid[:, 1] < b[:, 3])).mean()
        checks.append({"chunk": [s, e], "offset": o, "offset_px": tab, "bbox_contains_gt_hip": float(inside)})
    hand = gt2[:, [9, 14]]
    hand_out = (hand[..., 0] < 0) | (hand[..., 0] >= wh[0]) | (hand[..., 1] < 0) | (hand[..., 1] >= wh[1])

    allj = MAIN + HIPJ + EXTRA
    ia, ib = [a for _, a, _ in allj], [b for _, _, b in allj]
    n = len(MAIN)
    rows = []
    for _, r in seg.iterrows():
        fv = np.arange(r.first_frame, r.last_frame + 1)  # video frames
        fv = fv[(fv < len(gt3))]
        fv = fv[~np.isnan(G["incam"][fv, 0, 0])]
        fm = fv  # offset 0, see docstring
        if len(fv) < 10:
            continue
        pc = G["incam"][fv][:, ia]          # camera frame
        pw = pc @ Rwc                       # rotated into world axes (R^T x per point)
        g = gt3[fm][:, ib]
        pa = np.empty((len(fv), len(allj)))
        resid = np.empty((len(fv), len(allj), 3))
        for k in range(len(fv)):
            sc, Rk, t = umeyama(pc[k, :n], g[k, :n])
            resid[k] = sc * pc[k] @ Rk.T + t - g[k]
        pa = np.linalg.norm(resid, axis=2)
        hp, hg = pw[:, n:n + 2].mean(1, keepdims=True), g[:, n:n + 2].mean(1, keepdims=True)
        root = np.linalg.norm((pw[:, :n] - hp) - (g[:, :n] - hg), axis=2)
        cent = np.linalg.norm((pw[:, :n] - pw[:, :n].mean(1, keepdims=True)) - (g[:, :n] - g[:, :n].mean(1, keepdims=True)), axis=2)
        absw = np.linalg.norm(pw[:, :n] + C - g[:, :n], axis=2)
        clean = int(r.mocap_erroneous) == 0 and int(r[f"extra_person_in_cam{cam}"]) <= 1
        mm = 1000.0
        rows.append({
            "vid": vid, "cam": cam, "exercise": ex, "rep": int(r.repetition_number), "person": int(r.person_id),
            "lights_on": int(r.lights_on), "view": r.cam17_orientation if cam == "17" else VIEW18[r.cam17_orientation],
            "mocap_erroneous": int(r.mocap_erroneous), "extra_person": int(r[f"extra_person_in_cam{cam}"]),
            "clean": bool(clean), "hand_out": bool(hand_out[fm].any()), "frames": int(len(fv)),
            "pa": float(pa[:, :n].mean() * mm), "root": float(root.mean() * mm), "cent": float(cent.mean() * mm),
            "abs": float(absw.mean() * mm),
            "joint_pa": {nm: float(pa[:, k].mean() * mm) for k, (nm, _, _) in enumerate(allj)},
            "joint_root": {nm: float(root[:, k].mean() * mm) for k, (nm, _, _) in enumerate(MAIN)},
            "joint_mean_resid": {nm: (resid[:, k].mean(0) * mm).tolist() for k, (nm, _, _) in enumerate(allj)},
        })
    return rows, checks


def summarize(rows):
    """Frame-weighted means over repetitions, grouped as in the result file."""
    import pandas as pd
    df = pd.DataFrame(rows)
    w = lambda g, c: float(np.average(g[c], weights=g.frames))
    def table(d, keys):
        out = []
        for k, g in d.groupby(keys):
            out.append(dict(zip(keys, k if isinstance(k, tuple) else (k,)), reps=len(g), frames=int(g.frames.sum()),
                            **{m: round(w(g, m), 1) for m in ("pa", "root", "cent", "abs")}))
        return pd.DataFrame(out)
    c = df[df.clean]
    pd.set_option("display.width", 200)
    print("== clean repetitions, per exercise and camera"); print(table(c, ["exercise", "cam"]).to_string(index=False))
    print("== clean, per camera"); print(table(c, ["cam"]).to_string(index=False))
    print("== clean, lights"); print(table(c, ["exercise", "lights_on"]).to_string(index=False))
    print("== clean, by camera and view"); print(table(c, ["cam", "view"]).to_string(index=False))
    print("== clean, by camera and hand_out (a projected hand leaves the image)"); print(table(c, ["cam", "hand_out"]).to_string(index=False))
    print("== excluded repetitions (extra person > 1 or mocap error), per camera"); print(table(df[~df.clean], ["cam"]).to_string(index=False))
    jp = pd.DataFrame([{**r["joint_pa"], "frames": r["frames"], "cam": r["cam"]} for r in rows if r["clean"]])
    names = [nm for nm, _, _ in MAIN + HIPJ + EXTRA]
    print("== clean, per-joint PA error (mm), joints of all 16 mapped (PA fitted on the 12 metric joints)")
    for cam in ("17", "18", None):
        g = jp if cam is None else jp[jp.cam == cam]
        print(" cam", cam or "both", {nm: round(float(np.average(g[nm], weights=g.frames)), 1) for nm in names})
    # Constant part of each joint's error: frame-weighted mean residual vector over clean reps.
    mr = {nm: np.linalg.norm(np.average([r["joint_mean_resid"][nm] for r in rows if r["clean"]], axis=0,
                                        weights=[r["frames"] for r in rows if r["clean"]])) for nm in names}
    print("== |mean residual| per joint (mm), the part of the PA error that is constant over clean reps")
    print({nm: round(float(v), 1) for nm, v in mr.items()})


def selftest():
    rng = np.random.default_rng(0)
    F, J = 200, len(MAIN)
    gt = rng.normal(size=(F, J, 3)) * 0.3 + np.array([0, 1, 0])
    a = 0.7
    R0 = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    R0 = R0 @ np.diag([1, -1, -1])  # y-down camera frame
    assert np.isclose(np.linalg.det(R0), 1)
    pred = (gt @ R0.T) * 1.1 + rng.normal(size=(F, 1, 3))  # scaled, rotated, moving root
    pa, sc = pa_errors(pred, gt)
    assert pa.max() < 1e-9 and np.allclose(sc, 1 / 1.1), (pa.max(), sc[:3])
    # Root metric has no scale: a 10% scale error must show up, a pure rotation must not,
    # with the centroid root and with an explicit root (joints 0, 1 as "hips").
    ro, R = root_errors(pred / 1.1, gt)
    assert ro.max() < 1e-9 and np.allclose(R, R0.T), ro.max()
    ro, _ = root_errors(pred / 1.1, gt, (pred / 1.1)[:, :2].mean(1), gt[:, :2].mean(1))
    assert ro.max() < 1e-9, ro.max()
    # A root defined 9 cm higher in GT than in pred spreads 90 mm into every joint.
    ro, _ = root_errors(pred / 1.1, gt, (pred / 1.1)[:, :2].mean(1), gt[:, :2].mean(1) + [0, 0.09, 0])
    assert np.allclose(ro, 0.09), ro.mean()
    ro, _ = root_errors(pred, gt)
    assert ro.mean() > 0.01
    # Noise of 10 mm per axis gives about 10*sqrt(3)*0.92 mm PA error (fewer dof left).
    noisy = pred / 1.1 + rng.normal(scale=0.01, size=pred.shape)
    e = pa_errors(noisy, gt)[0].mean()
    assert 0.010 < e < 0.017, e
    # Offset recovery: detections lag the projected joints by 2 frames.
    t = np.arange(300)
    g2 = np.stack([np.stack([100 * np.sin(t / 7 + j), 50 * np.cos(t / 5 + j)], 1) for j in range(12)], 1)
    det = g2[2:] + rng.normal(scale=1, size=g2[2:].shape)
    off, tab = best_offset(det, np.ones(det.shape[:2]), g2)
    assert off == 2, tab
    print("selftest ok: PA exact", pa.max(), "| noisy 10 mm/axis ->", round(e * 1000, 1), "mm | offset", off)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data")
    ap.add_argument("--chunks")
    ap.add_argument("--calib")
    ap.add_argument("--rec", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--summary")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.summary:
        summarize(json.load(open(a.summary))["rows"])
    else:
        rows, checks = [], []
        for vid in a.rec:
            for cam in ("17", "18"):
                r, c = evaluate_rec(a.data, a.chunks, a.calib, vid, cam)
                rows += r
                checks += [{"vid": vid, "cam": cam, **x} for x in c]
                for x in c:
                    print(f"{vid} c{cam} chunk {x['chunk']} offset {x['offset']} "
                          f"(px at -1/0/+1: {x['offset_px'][-1]:.1f}/{x['offset_px'][0]:.1f}/{x['offset_px'][1]:.1f}) "
                          f"inbox {x['bbox_contains_gt_hip']:.3f}")
        json.dump({"rows": rows, "checks": checks}, open(a.out, "w"), indent=1)
        summarize(rows)
