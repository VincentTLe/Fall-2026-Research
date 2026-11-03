"""T4b run 1: T2c's two-camera EasyMocap SMPL fit -> SMPL vertices in one camera's frame, for gvhmr_to_gaussianavatar.py.

EasyMocap's SMPL parameters (Rh, Th, poses, shapes) live in the world frame of the camera files it was run with (T2c:
the authors' new cameras, runs/t2b/emc-cams-new). The vertices are computed with EasyMocap's own SMPLModel.vertices,
the same call T2c's rehab_emc_smpl.py uses for scoring, then moved into camera <cam>'s frame: X_c = Rot_cam X + T_cam.
Output npz (one row per frame_map row, same order):
  verts (N, 6890, 3) camera frame, metres; init_global_orient (N, 3); init_body_pose (N, 63) (fit initialisation only)
  K (3, 3) of <cam> from intri.yml.

Run with the EasyMocap env:
  ~/fall2026/envs/easymocap/bin/python twocam_smpl_to_camera.py --fit runs/t2c/fit/PM_037 --cams runs/t2c/data/PM_037 \
      --frame-map RUN_DIR/all/frame_map.csv --start 230 --cam 17 --smpl models/smpl/SMPL_NEUTRAL.pkl --out RUN_DIR/twocam_c17.npz
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path.home() / "fall2026/external/EasyMocap"))
from easymocap.bodymodel.smpl import SMPLModel  # noqa: E402


def read_cam(cams, cam):
    ex = cv2.FileStorage(str(Path(cams) / "extri.yml"), cv2.FILE_STORAGE_READ)
    ix = cv2.FileStorage(str(Path(cams) / "intri.yml"), cv2.FILE_STORAGE_READ)
    R, T, K = ex.getNode(f"Rot_{cam}").mat(), ex.getNode(f"T_{cam}").mat().reshape(3), ix.getNode(f"K_{cam}").mat()
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-5)
    return R, T, K


def main():
    ap = argparse.ArgumentParser()
    for k in ("fit", "cams", "frame-map", "cam", "smpl", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--start", type=int, required=True, help="global video frame of fit frame 000000")
    a = ap.parse_args()
    torch.set_num_threads(8)
    R, T, K = read_cam(a.cams, a.cam)
    rows = list(csv.DictReader(open(a.frame_map)))
    params = {k: [] for k in ("Rh", "Th", "poses", "shapes")}
    for r in rows:
        p = json.load(open(Path(a.fit) / "smpl" / f"{int(r['global']) - a.start:06d}.json"))
        assert len(p) == 1 and p[0]["id"] == 0
        for k in params:
            params[k].append(np.array(p[0][k], dtype=np.float32).reshape(-1))
    params = {k: np.stack(v) for k, v in params.items()}
    model = SMPLModel(model_path=a.smpl, model_type="smpl", device="cpu", NUM_SHAPES=10)
    with torch.no_grad():
        vw = model.vertices(params, return_tensor=True).numpy()  # (N, 6890, 3) world
    vc = vw @ R.T + T
    # initial guess for the SMPL fit. EasyMocap's SMPL 'poses' has 69 values = joints 1..23 (no root); the root
    # rotation is Rh, applied about the model origin, then Th. Root in camera frame: R * R(Rh); body joints 1..21.
    assert params["poses"].shape[1] == 69, params["poses"].shape
    go = [cv2.Rodrigues(R @ cv2.Rodrigues(rh.astype(np.float64))[0])[0].reshape(3) for rh in params["Rh"]]
    shp = params["shapes"]
    np.savez(a.out, verts=vc.astype(np.float32), init_global_orient=np.array(go, np.float32),
             init_body_pose=params["poses"][:, :63].astype(np.float32), K=K)
    z = vc.mean(1)[:, 2]
    print(f"{len(rows)} frames -> {a.out}; camera {a.cam} K fx {K[0,0]:.2f} fy {K[1,1]:.2f} c ({K[0,2]:.1f}, {K[1,2]:.1f}); "
          f"body depth {z.min():.2f}-{z.max():.2f} m (median {np.median(z):.2f}); shapes vary by max {np.abs(shp - shp[0]).max():.4f}")


if __name__ == "__main__":
    main()
