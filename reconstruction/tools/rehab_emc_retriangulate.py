"""Re-triangulate EasyMocap's cached 2-view HRNet detections with another camera calibration, using
EasyMocap's own code path for `--triangulate-mode naive` (read_camera -> Undistort.points ->
batch_triangulate, min_view 2). Run in the EasyMocap env.

  # write yml for a calibration npz (K', R', C from rehab_authors_calib.py) and re-triangulate
  python rehab_emc_retriangulate.py --calib ~/fall2026/runs/t2b/calib-new --cams-out ~/fall2026/runs/t2b/emc-cams-new \
      --emc ~/fall2026/runs/t2-emc --out ~/fall2026/runs/t2b/emc-new --rec PM_032 ...
  # check: with the cameras EasyMocap actually used, the output must equal runs/t2-emc/<rec>/triangulated
  python rehab_emc_retriangulate.py --cams ~/fall2026/data/rehab24-6-emc/PM_032 --emc ... --out ... --rec PM_032 --check
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

EASY = Path.home() / "fall2026/external/EasyMocap"
sys.path.insert(0, str(EASY))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from easymocap.mytools.camera_utils import Undistort, read_camera  # noqa: E402
from myeasymocap.operations.triangulate import batch_triangulate  # noqa: E402

from emc_yml import write_cameras  # noqa: E402

VIEWS = ["17", "18"]


def write_yml(calib_dir, out):
    cams = {}
    for v in VIEWS:
        c = np.load(Path(calib_dir) / f"calib_c{v}.npz")
        cams[v] = {"K": c["K"], "dist": np.zeros(5), "R": c["R"], "T": -c["R"] @ c["C"]}
    write_cameras(out, cams)


def main(a):
    if a.calib:
        write_yml(a.calib, a.cams_out)
    cam_dir = a.cams or a.cams_out
    worst = 0.0
    for vid in a.rec:
        cams = read_camera(str(Path(cam_dir.replace("{rec}", vid)) / "intri.yml"),
                           str(Path(cam_dir.replace("{rec}", vid)) / "extri.yml"), VIEWS)
        Pall = np.stack([cams[v]["P"] for v in VIEWS])
        out = Path(a.out) / vid / "triangulated"
        out.mkdir(parents=True, exist_ok=True)
        files = sorted((Path(a.emc) / vid / "detections").glob("*.npz"))
        for f in files:
            k2d = np.load(f)["keypoints2d"]  # (views, 25, 3), views in VIEWS order
            k2d = np.stack([np.c_[Undistort.points(k2d[i][:, :2], cams[v]["K"], cams[v]["dist"]), k2d[i][:, 2:]]
                            if k2d[i].shape[0] > 0 else k2d[i] for i, v in enumerate(VIEWS)])
            k3d = batch_triangulate(k2d, Pall)
            json.dump([{"id": 0, "keypoints3d": k3d.tolist()}], open(out / f"{f.stem}.json", "w"))
            if a.check:
                ref = np.array(json.load(open(Path(a.emc) / vid / "triangulated" / f"{f.stem}.json"))[0]["keypoints3d"])
                worst = max(worst, float(np.abs(ref - k3d).max()))
        print(f"{vid}: {len(files)} frames re-triangulated" + (f", max |diff| to EasyMocap so far {worst:.2e}" if a.check else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", help="folder with calib_c17.npz / calib_c18.npz (K', R', C)")
    ap.add_argument("--cams-out", help="where to write intri.yml / extri.yml for --calib")
    ap.add_argument("--cams", help="existing yml folder (may contain {rec})")
    ap.add_argument("--emc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rec", nargs="+", required=True)
    ap.add_argument("--check", action="store_true")
    main(ap.parse_args())
