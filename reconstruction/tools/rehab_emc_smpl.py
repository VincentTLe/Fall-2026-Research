"""Two cameras + SMPL on REHAB24-6 with EasyMocap mv1p, reusing the cached HRNet 2D (no re-detection).

prep:   build <out>/<rec>/ as an EasyMocap data folder: images/ -> the T2 frames (symlink), intri/extri.yml
        from a camera folder (e.g. the authors' new cameras), annots/<cam>/<frame>.json written from T2's
        <emc>/<rec>/detections/<frame>.npz (BODY25 2D + bbox), in the same format run_mv1p --write-annots uses.
joints: SMPL-24 joints (SMPL J_regressor on the fitted vertices, as for GVHMR) from <fit>/<rec>/smpl/*.json,
        saved as <fit>/<rec>/smpl_joints.npz: joints (n_video, 24, 3) indexed by mocap/video frame, NaN elsewhere.

  python rehab_emc_smpl.py prep --emc ~/fall2026/runs/t2-emc --emc-data ~/fall2026/data/rehab24-6-emc \
      --cams ~/fall2026/runs/t2b/emc-cams-new --out ~/fall2026/runs/t2c/data --rec PM_032
  python rehab_emc_smpl.py joints --fit ~/fall2026/runs/t2c/fit --emc-data ~/fall2026/data/rehab24-6-emc \
      --smpl ~/fall2026/models/smpl/SMPL_NEUTRAL.pkl --data ~/fall2026/data/rehab24-6 --rec PM_032
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

VIEWS = ["17", "18"]


def prep(a):
    for vid in a.rec:
        src, dst = Path(a.emc_data) / vid, Path(a.out) / vid
        dst.mkdir(parents=True, exist_ok=True)
        if not (dst / "images").exists():
            os.symlink(src / "images", dst / "images")
        for y in ("intri.yml", "extri.yml"):
            shutil.copy(Path(a.cams) / y, dst / y)
        shutil.copy(src / "span.json", dst / "span.json")
        n = 0
        for f in sorted((Path(a.emc) / vid / "detections").glob("*.npz")):
            d = np.load(f)
            for nv, view in enumerate(VIEWS):
                p = dst / "annots" / view / f"{f.stem}.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({"filename": f"{view}/{f.stem}.jpg", "annots": [
                    {"personID": 0, "bbox": d["bbox"][nv].tolist(), "keypoints": d["keypoints2d"][nv].tolist()}]}))
            n += 1
        print(f"{vid}: {n} frames of cached 2D -> {dst / 'annots'}")


def joints(a):
    import torch
    sys.path.insert(0, str(Path.home() / "fall2026/external/EasyMocap"))
    from easymocap.bodymodel.smpl import SMPLModel
    model = SMPLModel(model_path=str(a.smpl), model_type="smpl", device="cpu", NUM_SHAPES=10)
    Jr = model.J_regressor  # (24, 6890)
    for vid in a.rec:
        span = json.load(open(Path(a.emc_data) / vid / "span.json"))
        start, n_video = span["start"], len(np.load(Path(a.data) / f"3d_joints/Ex{span['exercise']}/{vid}-30fps.npy", mmap_mode="r"))
        files = sorted((Path(a.fit) / vid / "smpl").glob("*.json"))
        out = np.full((n_video, 24, 3), np.nan)
        for i in range(0, len(files), 500):
            batch = [json.load(open(f))[0] for f in files[i:i + 500]]
            params = {k: np.concatenate([np.array(b[k], dtype=np.float32).reshape(1, -1) for b in batch])
                      for k in ("Rh", "Th", "poses", "shapes")}
            with torch.no_grad():
                v = model.vertices(params, return_tensor=True)
                j = torch.einsum("jv,fvc->fjc", Jr, v).numpy()
            for f, jj in zip(files[i:i + 500], j):
                if start + int(f.stem) < n_video:
                    out[start + int(f.stem)] = jj
        np.savez(Path(a.fit) / vid / "smpl_joints.npz", joints=out)
        print(f"{vid}: {len(files)} SMPL frames -> smpl_joints.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    for k in ("emc", "emc-data", "cams", "out"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--rec", nargs="+", required=True)
    j = sub.add_parser("joints")
    for k in ("fit", "emc-data", "smpl"):
        j.add_argument(f"--{k}", required=True)
    j.add_argument("--data", required=True)
    j.add_argument("--rec", nargs="+", required=True)
    a = ap.parse_args()
    prep(a) if a.cmd == "prep" else joints(a)
