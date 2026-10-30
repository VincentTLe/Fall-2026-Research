"""Convert GVHMR (SMPL-X, camera frame) results + images + masks into GaussianAvatar's SMPL dataset layout.

Used for T4 Step 3 (REHAB24-6 PM_037 camera 17). Run with the GVHMR environment from the GVHMR repo (it uses
GVHMR's own body-model chain, as in tools/demo/demo.py and reconstruction/tools/gvhmr_export_joints.py):

    cd ~/fall2026/external/GVHMR
    PYTHONPATH=$PWD ~/fall2026/envs/gvhmr/bin/python <this file> RUN_DIR --chunks DIR --calib calib_c17.npz \
        --seg Segmentation.csv --video PM_037 [--test-reps 3] [--check]

RUN_DIR/all must hold images/<frame>.png (full frames), masks/<frame>.png (0/255) and frame_map.csv
(global,clip,local: clip-local index = row of that clip's hmr4d_results.pt).

Steps
1. SMPL-X "supermotion" forward on smpl_params_incam (camera frame, metres), then GVHMR's smplx2smpl_sparse.pt
   -> SMPL-topology vertices (the mesh GVHMR itself renders).
2. Fit neutral SMPL to those vertices: one shared beta (10), per-frame global_orient (3) + body_pose (69) + transl (3),
   initialised from the SMPL-X body pose (21 joints; SMPL's two hand joints start at 0). Loss = mean squared vertex
   distance. GaussianAvatar's tested path is SMPL (all its shipped data is 72-dim SMPL), hence the fit.
3. Crop every frame to the union of the mask boxes (+ margin, even size); K's principal point is shifted.
4. Split by repetition: the last --test-reps repetitions of the video (Segmentation.csv) are test, the rest train.
5. Write RUN_DIR/{train,test}/{images,masks,smpl_parms.pth,cam_parms.npz} with extrinsic = identity
   (poses are already in the camera frame, as in GaussianAvatar's People-Snapshot and NeuMan data).
--check: project the fitted SMPL mesh with K onto every frame, report silhouette IoU against the masks, and save
overlays for 6 frames in RUN_DIR/check/. The script also re-checks its own outputs (row counts, K shift).
"""
import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
import smplx
import torch

from hmr4d.utils.smplx_utils import make_smplx


def load_gvhmr(chunks, rows):
    """Camera-frame SMPL-X params for the sampled rows, in frame_map order."""
    cache, out = {}, {k: [] for k in ("global_orient", "body_pose", "betas", "transl")}
    K = None
    for r in rows:
        if r["clip"] not in cache:
            cache[r["clip"]] = torch.load(Path(chunks) / r["clip"] / "hmr4d_results.pt", map_location="cpu")
        res = cache[r["clip"]]
        i = int(r["local"])
        for k in out:
            out[k].append(res["smpl_params_incam"][k][i])
        Ki = res["K_fullimg"][i].numpy()
        K = Ki if K is None else K
        assert np.abs(Ki - K).max() < 1e-3, "K changes between frames"
    return {k: torch.stack(v).float() for k, v in out.items()}, K


def smplx_to_smpl_verts(p):
    bm = make_smplx("supermotion")
    sx2s = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt")
    with torch.no_grad():
        v = bm(**p).vertices
        return torch.stack([torch.matmul(sx2s, vi) for vi in v])  # (N, 6890, 3)


def fit_smpl(target, p, steps=600):
    smpl = smplx.SMPL("inputs/checkpoints/body_models/smpl", gender="neutral", batch_size=len(target))
    n = len(target)
    beta = torch.zeros(1, 10, requires_grad=True)
    go = p["global_orient"].clone().requires_grad_(True)
    bp = torch.cat([p["body_pose"], torch.zeros(n, 6)], 1).clone().requires_grad_(True)
    with torch.no_grad():
        v0 = smpl(betas=beta.expand(n, -1), global_orient=go, body_pose=bp, transl=torch.zeros(n, 3)).vertices
    tr = (target.mean(1) - v0.mean(1)).clone().requires_grad_(True)
    stages = [([go, tr], 100, 0.01), ([beta, go, bp, tr], steps, 0.01)]
    hist = []
    for params, n_steps, lr in stages:
        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps)
        for s in range(n_steps):
            v = smpl(betas=beta.expand(n, -1), global_orient=go, body_pose=bp, transl=tr).vertices
            loss = ((v - target) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if s % 100 == 0 or s == n_steps - 1:
                hist.append((len(hist), float(loss.sqrt()) * 1000))
                print(f"  fit step {s:4d}/{n_steps} rms {float(loss.sqrt())*1000:.2f} mm", flush=True)
    with torch.no_grad():
        v = smpl(betas=beta.expand(n, -1), global_orient=go, body_pose=bp, transl=tr).vertices
        err = (v - target).norm(dim=-1)  # (N, 6890) metres
    return dict(beta=beta.detach(), global_orient=go.detach(), body_pose=bp.detach(), transl=tr.detach(),
                verts=v, err=err, faces=smpl.faces.astype(np.int64))


def silhouette(verts, faces, K, h, w):
    """Filled-triangle silhouette of a mesh in pixel coordinates (camera frame, z forward)."""
    uv = (verts @ K.T)
    uv = uv[:, :2] / uv[:, 2:3]
    m = np.zeros((h, w), np.uint8)
    # one fillConvexPoly per triangle: a single fillPoly over many overlapping triangles fills by parity and
    # leaves holes wherever triangles overlap (first version of this check gave IoU ~0.55 for that reason)
    for tri in np.round(uv[faces]).astype(np.int32):
        cv2.fillConvexPoly(m, tri, 1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("--chunks", required=True); ap.add_argument("--calib", required=True)
    ap.add_argument("--seg", required=True); ap.add_argument("--video", required=True)
    ap.add_argument("--test-reps", type=int, default=3); ap.add_argument("--margin", type=int, default=40)
    ap.add_argument("--steps", type=int, default=600); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(16)
    run = Path(a.run_dir)
    rows = list(csv.DictReader(open(run / "all/frame_map.csv")))
    names = [f"{int(r['global']):05d}" for r in rows]
    assert names == sorted(names), "frame_map must be in time order"

    p, K_gvhmr = load_gvhmr(a.chunks, rows)
    K = np.load(a.calib)["K"].astype(np.float64)
    print("K (calibration)\n", K.round(2), "\nK used by GVHMR\n", K_gvhmr.round(2))
    assert np.abs(K_gvhmr - K).max() < 1.0, "GVHMR K differs from the calibration; use the true-focal rerun"
    target = smplx_to_smpl_verts(p)
    print(f"fitting SMPL to {len(target)} frames of SMPL-topology vertices")
    f = fit_smpl(target, p, a.steps)
    e = f["err"].numpy() * 1000
    stats = dict(rms_mm=float(np.sqrt((e ** 2).mean())), mean_mm=float(e.mean()), p99_mm=float(np.percentile(e, 99)),
                 max_mm=float(e.max()), worst_frame=names[int(e.mean(1).argmax())], beta=f["beta"][0].tolist())
    print("fit:", json.dumps(stats))

    # crop window = union of mask boxes + margin, even size, inside the image
    h, w = cv2.imread(str(run / "all/masks" / f"{names[0]}.png"), 0).shape
    x0, y0, x1, y1 = w, h, 0, 0
    for n in names:
        ys, xs = np.nonzero(cv2.imread(str(run / "all/masks" / f"{n}.png"), 0) > 127)
        x0, y0, x1, y1 = min(x0, xs.min()), min(y0, ys.min()), max(x1, xs.max()), max(y1, ys.max())
    x0, y0 = max(0, x0 - a.margin), max(0, y0 - a.margin)
    x1, y1 = min(w, x1 + a.margin + 1), min(h, y1 + a.margin + 1)
    x1 -= (x1 - x0) % 2; y1 -= (y1 - y0) % 2
    Kc = K.copy(); Kc[0, 2] -= x0; Kc[1, 2] -= y0
    print(f"crop x {x0}:{x1} y {y0}:{y1} -> {x1-x0}x{y1-y0}")

    # split by repetition
    reps = [r for r in csv.DictReader(open(a.seg), delimiter=";") if r["video_id"] == a.video]
    reps.sort(key=lambda r: int(r["first_frame"]))
    test_start = int(reps[-a.test_reps]["first_frame"])
    split = ["test" if int(r["global"]) >= test_start else "train" for r in rows]
    print(f"{len(reps)} repetitions; test = last {a.test_reps} (frames >= {test_start}): "
          f"{split.count('train')} train / {split.count('test')} test")

    pose72 = torch.cat([f["global_orient"], f["body_pose"]], 1)
    for sp in ("train", "test"):
        idx = [i for i, s in enumerate(split) if s == sp]
        d = run / sp
        (d / "images").mkdir(parents=True, exist_ok=True); (d / "masks").mkdir(parents=True, exist_ok=True)
        for i in idx:
            for sub in ("images", "masks"):
                im = cv2.imread(str(run / "all" / sub / f"{names[i]}.png"), cv2.IMREAD_UNCHANGED)
                cv2.imwrite(str(d / sub / f"{names[i]}.png"), im[y0:y1, x0:x1])
        torch.save({"beta": f["beta"].clone(), "body_pose": pose72[idx].clone(), "trans": f["transl"][idx].clone()},
                   d / "smpl_parms.pth")
        np.savez(d / "cam_parms.npz", intrinsic=Kc, extrinsic=np.eye(4))
        chk = torch.load(d / "smpl_parms.pth")
        assert chk["body_pose"].shape == (len(idx), 72) and len(os.listdir(d / "images")) == len(idx)
    json.dump(dict(stats, crop=[int(x0), int(y0), int(x1), int(y1)], K_crop=Kc.tolist(), test_start=test_start,
                   n_train=split.count("train"), n_test=split.count("test")), open(run / "convert_summary.json", "w"), indent=1)

    if a.check:
        (run / "check").mkdir(exist_ok=True)
        V, F = f["verts"].numpy(), f["faces"]
        ious = []
        for i, n in enumerate(names):
            sil = silhouette(V[i], F, K, h, w)
            m = cv2.imread(str(run / "all/masks" / f"{n}.png"), 0) > 127
            ious.append(float((sil & m).sum() / max(1, (sil | m).sum())))
        ious = np.array(ious)
        print(f"silhouette IoU (fitted SMPL vs SAM mask), {len(ious)} frames: mean {ious.mean():.3f} "
              f"median {np.median(ious):.3f} min {ious.min():.3f} at {names[int(ious.argmin())]}")
        pick = sorted({0, len(names) // 4, len(names) // 2, 3 * len(names) // 4, len(names) - 1, int(ious.argmin())})
        for i in pick:
            img = cv2.imread(str(run / "all/images" / f"{names[i]}.png"))
            sil = silhouette(V[i], F, K, h, w)
            cnt, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            m = (cv2.imread(str(run / "all/masks" / f"{names[i]}.png"), 0) > 127).astype(np.uint8)
            cm, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            uv = V[i] @ K.T; uv = (uv[:, :2] / uv[:, 2:3]).round().astype(int)
            for (u, v_) in uv[::7]:
                cv2.circle(img, (int(u), int(v_)), 1, (0, 255, 0), -1)
            cv2.drawContours(img, cnt, -1, (0, 255, 0), 2); cv2.drawContours(img, cm, -1, (0, 0, 255), 2)
            cv2.putText(img, f"{names[i]} IoU {ious[i]:.3f}", (x0 + 10, y0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            cv2.imwrite(str(run / "check" / f"overlay_{names[i]}.jpg"), img[y0:y1, x0:x1])
        json.dump(dict(iou_mean=float(ious.mean()), iou_median=float(np.median(ious)), iou_min=float(ious.min()),
                       iou_min_frame=names[int(ious.argmin())], per_frame=dict(zip(names, ious.round(4).tolist()))),
                  open(run / "check" / "iou.json", "w"), indent=1)


if __name__ == "__main__":
    main()
