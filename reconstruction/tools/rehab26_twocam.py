"""Two-camera (cam1 + cam3, orthogonal) separability on REHAB26 without calibration (T5b).

Strategies (rules frozen in runs/t5b/spec/SPEC.md before scoring):
  ortho        per sequence, frontal camera = smaller |viewing angle| (cam1: -o, cam3: 90-o; tie -> cam1);
               frontal-plane and transverse features (F2-F7, F9, F10) from it, sagittal (F1, F8, F11-F15) from the other
  ortho_fixed  frontal-plane/transverse features always cam1, sagittal always cam3
  merged       per camera: normalise_3d (mid-hip to origin, hips on +x), scale to unit median body length
               (mid-hip->mid-shoulder + mean thigh + mean shin), then Eq. 4: sum(W^2 J) / sum(W^2);
               W = visibility (channel 3) if present and --weights vis, else 1
Ψ per sequence with rehab_psi.sequence_psi, T5 defaults, same seeding scheme as rehab26_psi_run.py (cam tag "13").

Usage: python -I rehab26_twocam.py --data D --strategy ortho --dims 2d --src mpp|npy:<dir>:<suffix> --out o.csv
"""
import argparse
import hashlib
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rehab_psi as rp  # noqa: E402

FRONTAL = {"F2", "F3", "F4", "F5", "F6", "F7", "F9", "F10"}    # T5 planes xy and xz
VIEW = {1: lambda o: -o, 3: lambda o: 90 - o}


def load(a, vid, cam):
    ex = vid.split("_")[0]
    if a.src == "mpp":
        return np.load(Path(a.data) / "extracted" / "mediapipe" / ex / f"{vid}_cam{cam}_{a.dims}.npy", allow_pickle=False)
    _, d, suf = a.src.split(":", 2)
    return np.load(Path(d) / f"{vid}_cam{cam}{suf}.npy", allow_pickle=False)


def body_len(P):
    m = lambda i, j: 0.5 * (P[:, i] + P[:, j])
    n = lambda v: np.linalg.norm(v, axis=-1)
    L = (n(m(rp.L_SH, rp.R_SH) - m(rp.L_HIP, rp.R_HIP))
         + 0.5 * (n(P[:, rp.L_HIP] - P[:, rp.L_KN]) + n(P[:, rp.R_HIP] - P[:, rp.R_KN]))
         + 0.5 * (n(P[:, rp.L_KN] - P[:, rp.L_AN]) + n(P[:, rp.R_KN] - P[:, rp.R_AN])))
    return float(np.nanmedian(L))


def fuse(J1, J3, s, e, weights):
    """Eq. 4 on frames s..e (inclusive) of two (T, 33, >=3) skeletons; returns (T, 33, 3) with NaN outside s..e."""
    out = np.full((len(J1), 33, 3), np.nan)
    num = 0.0; den = 0.0
    for J in (J1, J3):
        P = rp.normalise_3d(J[s:e + 1, :, :3].astype(float))
        P = P / body_len(P)
        W = J[s:e + 1, :, 3] if (weights == "vis" and J.shape[-1] > 3) else np.ones(P.shape[:2])
        W2 = (W ** 2)[..., None]
        num = num + W2 * P; den = den + W2
    out[s:e + 1] = num / np.maximum(den, 1e-12)
    return out


def job(task):
    a, cfg, vid, ex, g = task
    J = {c: load(a, vid, c) for c in (1, 3)}
    Fc = {} if a.strategy == "merged" else {c: rp.features(J[c], a.dims, ex, cfg) for c in (1, 3)}
    main = cfg.dtw_feat(ex)
    used = {main} | {k for ks in rp.ERROR_FEATURES[ex].values() for k in ks}
    rows = []
    for o, b in g.groupby("person_orientation"):
        if a.strategy == "merged":
            s, e = int(b.first_frame.min()), int(b.last_frame.max())
            F = rp.features(fuse(J[1], J[3], s, e, a.weights), "3d", ex, cfg)
        else:
            if a.strategy == "ortho":
                fr = 1 if abs(VIEW[1](o)) <= abs(VIEW[3](o)) else 3
            else:
                fr = 1
            la = 4 - fr
            F = {k: Fc[fr if k in FRONTAL else la][k] for k in Fc[1]}
        F = {k: v for k, v in F.items() if k in used}
        ok, nok, skip = [], [], False
        for r in b.itertuples():
            f = {k: v[r.first_frame:r.last_frame + 1] for k, v in F.items()}
            if any(np.isnan(v).any() for v in f.values()):
                skip = True; break
            if r.correctness_score == 1:
                ok.append(f)
            else:
                nok.append((f, rp.ERROR_FEATURES[ex][r.error_description]))
        if skip:
            continue
        for sd in a.seeds:
            h = int(hashlib.sha256(f"{sd}|{vid}|{o}|13".encode()).hexdigest()[:8], 16)
            rows.append(dict(exercise=ex, video_id=vid, orientation=o, cam=13, viewpoint=VIEW[1](o), dims=a.dims,
                             seed=sd, psi=rp.sequence_psi(ok, nok, main, cfg, np.random.default_rng(h))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--src", default="mpp")
    ap.add_argument("--strategy", choices=["ortho", "ortho_fixed", "merged"], required=True)
    ap.add_argument("--dims", default="2d"); ap.add_argument("--weights", default="vis", choices=["vis", "equal"])
    ap.add_argument("--seeds", default="0-9"); ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-")); a.seeds = list(range(lo, hi + 1))
    cfg = rp.Config(aspect=(1080.0, 1920.0))
    seg = pd.read_csv(Path(a.data) / "raw" / "Segmentation.csv", sep=";")
    tasks = [(a, cfg, v, int(g.exercise_id.iloc[0]), g) for v, g in seg.groupby("video_id")]
    with Pool(a.procs) as p:
        rows = [r for res in p.imap_unordered(job, tasks) for r in res]
    df = pd.DataFrame(rows).sort_values(["exercise", "video_id", "orientation", "seed"])
    df.to_csv(a.out, index=False)
    print("wrote", a.out, len(df), "rows;", a.strategy, a.src, a.dims, a.weights)


if __name__ == "__main__":
    main()
