"""Two cameras + SMPL on REHAB24-6 from GVHMR's ViTPose 2D (both views), with a view-agreement filter.

Per recording:
1. ViTPose COCO-17 of cameras 17 and 18 (GVHMR chunk outputs, repetition spans only), mapped to BODY25 (neck and
   mid-hip = mean of the pair, minimum confidence; no feet). Confidence < MIN_CONF -> joint unused.
2. View-agreement filter, fixed before any scoring: triangulate each joint from the two views (EasyMocap
   batch_triangulate, cameras from intri/extri.yml), reproject; if the error in either view exceeds THR_PX, the
   joint is dropped in that frame (confidence 0 in both views). Then triangulate again.
3. SMPL fit with the at_final stages of EasyMocap's config/mv1p/detect_triangulate_fitSMPL.yml (init, shape, RT,
   pose refinement with smoothing and GMM prior), run per contiguous block of frames where both cameras have 2D
   (the mv1p dataset reader needs images, which exist only for 12 recordings, so the stages are driven directly).
4. SMPL-24 joints (J_regressor on the fitted vertices) -> <out>/<rec>/smpl_joints.npz, indexed by video frame,
   plus filter statistics in <out>/<rec>/stats.json.

  python rehab_vitpose_smpl.py --data ~/fall2026/data/rehab24-6 --chunks ~/fall2026/runs/t2b/chunks/out \
      --cams ~/fall2026/runs/t2b/emc-cams-new --smpl ~/fall2026/models/smpl/SMPL_NEUTRAL.pkl \
      --out ~/fall2026/runs/t2d/fit --rec PM_032 ...
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

EASY = Path.home() / "fall2026/external/EasyMocap"
sys.path.insert(0, str(EASY))
sys.path.insert(0, str(Path(__file__).resolve().parent))

THR_PX = 40.0     # view-agreement threshold (reprojection, px), fixed 2026-11-02 before scoring
MIN_CONF = 0.3    # same as rehab_twocam.py
MIN_BLOCK = 10    # frames; shorter blocks are not fitted
VIEWS = ["17", "18"]
COCO2BODY25 = {0: 0, 6: 2, 8: 3, 10: 4, 5: 5, 7: 6, 9: 7, 12: 9, 14: 10, 16: 11, 11: 12, 13: 13, 15: 14,
               2: 15, 1: 16, 4: 17, 3: 18}


def to_body25(k):
    """k: (F, 17, 3) COCO with NaN for missing frames -> (F, 25, 3)."""
    out = np.zeros(k.shape[:1] + (25, 3))
    for c, b in COCO2BODY25.items():
        out[:, b] = k[:, c]
    for b, (i, j) in ((1, (5, 6)), (8, (11, 12))):
        out[:, b, :2] = (k[:, i, :2] + k[:, j, :2]) / 2
        out[:, b, 2] = np.minimum(k[:, i, 2], k[:, j, 2])
    out[out[..., 2] < MIN_CONF] = 0
    return out


def triangulate_all(k2d, Pall):
    from myeasymocap.operations.triangulate import batch_triangulate
    return np.stack([batch_triangulate(k2d[:, f], Pall) for f in range(k2d.shape[1])])


def reproj_err(k3d, Pall, k2d):
    h = k3d[None, :, :, :3] @ Pall[:, None, :, :3].transpose(0, 1, 3, 2) + Pall[:, None, None, :, 3]
    return np.linalg.norm(h[..., :2] / h[..., 2:] - k2d[..., :2], axis=-1)   # (views, F, J)


def blocks(ok):
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return []
    cuts = np.where(np.diff(idx) > 1)[0]
    return [idx[s:e + 1] for s, e in zip(np.r_[0, cuts + 1], np.r_[cuts, len(idx) - 1])]


def main(a):
    import torch
    torch.set_num_threads(a.threads)
    from easymocap.config import Config
    from easymocap.config.yacs import CfgNode
    from easymocap.mytools.camera_utils import read_camera
    from myeasymocap.stages.basestage import MultiStage
    from rehab_eval import load_chunks
    cams = read_camera(str(Path(a.cams) / "intri.yml"), str(Path(a.cams) / "extri.yml"), VIEWS)
    Pall = np.stack([cams[v]["P"] for v in VIEWS])
    cfg = Config.load(str(EASY / "config/mv1p/detect_triangulate_fitSMPL.yml"))
    final = cfg.args.at_final
    final.load_body_model.args.model_path = str(a.smpl)
    final.load_body_model.args.regressor_path = str(EASY / "data/smplx/J_regressor_body25.npy")
    for k in ("write", "render_ground", "make_video"):
        del final[k]
    import pandas as pd
    seg = pd.read_csv(Path(a.data) / "Segmentation.csv", sep=";")
    for vid in a.rec:
        out = Path(a.out) / vid
        if (out / "smpl_joints.npz").exists():
            print(vid, "done already"); continue
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        ex = int(seg[seg.video_id == vid].exercise_id.iloc[0])
        n = len(np.load(Path(a.data) / f"3d_joints/Ex{ex}/{vid}-30fps.npy", mmap_mode="r"))
        k2d = np.stack([to_body25(load_chunks(a.chunks, vid, v, n)[0]["vitpose"]) for v in VIEWS])  # (2, n, 25, 3)
        both = ~np.isnan(k2d[..., :2]).any(axis=(0, 2, 3))
        k2d = np.nan_to_num(k2d)
        stats = {"frames_both_views": int(both.sum()), "thr_px": THR_PX, "min_conf": MIN_CONF}
        fr = np.where(both)[0]
        k3d0 = triangulate_all(k2d[:, fr], Pall)
        err = reproj_err(k3d0, Pall, k2d[:, fr])
        seen = (k3d0[..., 3] > 0)
        bad = seen & (err.max(0) > THR_PX)
        stats.update(joints_seen=int(seen[:, :19].sum()), joints_dropped=int(bad[:, :19].sum()),
                     frames_with_drop=int(bad[:, :19].any(1).sum()))
        sub = k2d[:, fr]
        sub[:, bad] = 0
        k2d[:, fr] = sub
        k3d = np.zeros((n, 25, 4))
        k3d[fr] = triangulate_all(k2d[:, fr], Pall)
        J = np.full((n, 24, 3), np.nan)
        stats["blocks"] = []
        for b in blocks(both):
            if len(b) < MIN_BLOCK:
                stats["blocks"].append([int(b[0]), int(b[-1]), "skipped"]); continue
            model = MultiStage(output=str(out), at_step=CfgNode({}), at_final=final, timer=False)
            infos = [{"keypoints3d": k3d[f], "keypoints": k2d[:, f]} for f in b]
            ret = model.at_final(infos)
            p = ret["params"]
            with torch.no_grad():
                bm = model.model_finals["load_body_model"].smplmodel
                pp = {k: np.asarray(p[k]) for k in ("Rh", "Th", "poses", "shapes")}
                pp["shapes"] = np.repeat(pp["shapes"][:1], len(b), 0) if len(pp["shapes"]) == 1 else pp["shapes"]
                verts = bm.vertices(pp, return_tensor=True)
                J[b] = torch.einsum("jv,fvc->fjc", bm.J_regressor, verts).cpu().numpy()
            stats["blocks"].append([int(b[0]), int(b[-1]), "fitted"])
        stats["seconds"] = round(time.time() - t0, 1)
        np.savez(out / "smpl_joints.npz", joints=J, keypoints3d=k3d.astype(np.float32))
        json.dump(stats, open(out / "stats.json", "w"), indent=1)
        print(vid, json.dumps({k: v for k, v in stats.items() if k != "blocks"}), len(stats["blocks"]), "blocks", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("data", "chunks", "cams", "smpl", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--rec", nargs="+", required=True)
    main(ap.parse_args())
