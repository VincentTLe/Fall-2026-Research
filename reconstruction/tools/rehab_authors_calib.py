"""Convert the REHAB24-6 camera parameters sent by the dataset authors (camera{17,18}_{old,new}.json, not
in git) to our convention, and check them against the dataset's 2D joints.

Convention (found by testing all axis/sign/transpose combinations; reproduces the dataset 2D to ~1e-12 px):
  X' = M X           M swaps y and z of the 3d_joints world (y up)
  x_cam = -R (X' - T)   R as given (world -> camera, camera looks along -z), T = camera centre in X'
  u = f_x x/z + c_x,  v = f_y y/z + c_y   (no distortion)
So in the 3d_joints world: P = K [A | -A C] with A = -R M and C = M T. Camera 18's R is not orthonormal
(singular values ~0.9956, 0.9956, 0.9904), so A is kept as given for P; for a proper rotation, P is
RQ-decomposed into an upper-triangular K' (fx != fy, small skew) and a rotation R' (as our DLT did).

Writes <out>/calib_<which>_c<cam>.npz with P (3x4, scaled to unit norm), K' (3x3), R' (world->camera
rotation), C (camera centre, m), plus the raw K and A.

  python rehab_authors_calib.py --authors ~/fall2026/data/rehab24-6/authors_calib \
      --data ~/fall2026/data/rehab24-6 --dlt ~/fall2026/runs/t2-gvhmr --out ~/fall2026/runs/t2b
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import scipy.linalg as sl

M = np.array([[1.0, 0, 0], [0, 0, 1], [0, 1, 0]])


def load(authors, cam, which):
    j = json.load(open(Path(authors) / f"camera{cam}_{which}.json"))
    R = np.array(j["extrinsics"]["R"], float).reshape(3, 3)
    T = np.array(j["extrinsics"]["T"], float).reshape(3)
    f = np.array(j["intrinsics_wo_distortion"]["f"], float).reshape(-1)
    c = np.array(j["intrinsics_wo_distortion"]["c"], float).reshape(-1)
    K = np.array([[f[0], 0, c[0]], [0, f[-1], c[1]], [0, 0, 1]])
    A, C = -R @ M, M @ T
    P = K @ np.c_[A, -A @ C]
    Kq, Rq = sl.rq(P[:, :3])
    S = np.diag(np.sign(np.diag(Kq)))
    Kq, Rq = Kq @ S, S @ Rq
    if np.linalg.det(Rq) < 0:
        Rq = -Rq
        Kq = -Kq
    scale = Kq[2, 2]
    return {"P": P / np.linalg.norm(P), "K": Kq / scale, "R": Rq, "C": C, "K_raw": K, "A": A,
            "R_singular_values": np.linalg.svd(R)[1]}


def project(P, X):
    h = X @ P[:, :3].T + P[:, 3]
    return h[:, :2] / h[:, 2:]


def rot_angle_deg(R1, R2):
    return float(np.degrees(np.arccos(np.clip((np.trace(R1 @ R2.T) - 1) / 2, -1, 1))))


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = sorted((p.split("/")[-2], os.path.basename(p)[:-10]) for p in glob.glob(f"{a.data}/3d_joints/Ex*/*-30fps.npy"))
    for cam in ("17", "18"):
        cams = {w: load(a.authors, cam, w) for w in ("old", "new")}
        for w, c in cams.items():
            np.savez(out / f"calib_{w}_c{cam}.npz", **c)
            print(f"camera {cam} {w}: K' fx {c['K'][0, 0]:.2f} fy {c['K'][1, 1]:.2f} skew {c['K'][0, 1]:.3f} "
                  f"c ({c['K'][0, 2]:.2f}, {c['K'][1, 2]:.2f}); centre {np.round(c['C'], 4).tolist()} m; "
                  f"R singular values {np.round(c['R_singular_values'], 4).tolist()}")
        err = []
        for ex, v in recs:
            X = np.load(f"{a.data}/3d_joints/{ex}/{v}-30fps.npy")[..., :3].reshape(-1, 3)
            x = np.load(f"{a.data}/2d_joints/{ex}/{v}-c{cam}-30fps.npy").reshape(-1, 2)
            err.append(np.abs(project(cams["old"]["P"], X) - x).max())
        print(f"camera {cam} old vs dataset 2D, all {len(recs)} recordings: max |error| {max(err):.2e} px")
        if a.dlt:
            d = np.load(Path(a.dlt) / f"calib_c{cam}.npz")
            X = np.load(f"{a.data}/3d_joints/Ex1/PM_032-30fps.npy")[..., :3].reshape(-1, 3)
            print(f"camera {cam} old vs our DLT: centre diff {np.linalg.norm(cams['old']['C'] - d['C']) * 1000:.4f} mm, "
                  f"rotation diff {rot_angle_deg(cams['old']['R'], d['R']):.5f} deg, "
                  f"projection diff max {np.abs(project(cams['old']['P'], X) - project(d['P'], X)).max():.2e} px")
        o, n = cams["old"], cams["new"]
        X = np.concatenate([np.load(f"{a.data}/3d_joints/{ex}/{v}-30fps.npy")[::30, :, :3].reshape(-1, 3) for ex, v in recs])
        dpx = np.linalg.norm(project(o["P"], X) - project(n["P"], X), axis=1)
        print(f"camera {cam} new vs old: centre moves {np.linalg.norm(n['C'] - o['C']) * 100:.1f} cm, rotation "
              f"{rot_angle_deg(n['R'], o['R']):.2f} deg, f {n['K_raw'][0, 0]:.2f} vs {o['K_raw'][0, 0]:.2f}, "
              f"c {n['K_raw'][:2, 2].tolist()} vs {o['K_raw'][:2, 2].tolist()}, new R singular values "
              f"{np.round(n['R_singular_values'], 4).tolist()}; joint projections move median {np.median(dpx):.1f} px, "
              f"p95 {np.percentile(dpx, 95):.1f} px (every 30th frame of all recordings)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--authors", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dlt")
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
