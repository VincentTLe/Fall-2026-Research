"""Per-sequence separability Psi on REHAB26-ViewAngles skeletons, for several random-reference seeds.

A sequence = (recording, person_orientation, camera). Viewing angle per Segmentation.txt:
cam1: -o, cam2: 45 - o, cam3: 90 - o.

Usage:
  python -I rehab26_psi_run.py --data <root> --source mediapipe --out psi.csv [--seeds 0-9] [--cfg key=val ...]
  --source mediapipe reads <root>/extracted/mediapipe/ex*/<vid>_cam<c>_<2d|3d>.npy
  --source npy:<dir>:<suffix> reads <dir>/<vid>_cam<c><suffix>.npy (T,33,3), e.g. GVHMR mapped to MediaPipe order,
           frames not covered by a repetition may be NaN.
"""
import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rehab_psi as rp  # noqa: E402

VIEW = {1: lambda o: -o, 2: lambda o: 45 - o, 3: lambda o: 90 - o}


def parse_cfg(items):
    kw = {}
    for it in items:
        k, v = it.split("=", 1)
        if k == "smooth":
            kw[k] = int(v)
        elif k == "aspect":
            kw[k] = tuple(float(x) for x in v.split(","))
        elif k == "dtw_feature":
            kw[k] = tuple(tuple(p.split(":")) for p in v.split(","))
        elif k in ("include_additional", "plane3d"):
            kw[k] = v.lower() in ("1", "true", "yes")
        else:
            kw[k] = v
    return rp.Config(**kw)


def load(args, vid, ex, cam, dims):
    if args.source == "mediapipe":
        return np.load(Path(args.data) / "extracted" / "mediapipe" / f"ex{ex}" / f"{vid}_cam{cam}_{dims}.npy",
                       allow_pickle=False)
    _, d, suffix = args.source.split(":", 2)
    return np.load(Path(d) / f"{vid}_cam{cam}{suffix}.npy", allow_pickle=False)


def job(task):
    args, cfg, vid, ex, cam, dims, seeds, seg_rows = task
    try:
        J = load(args, vid, ex, cam, dims)
    except FileNotFoundError:
        if args.skip_incomplete:
            return []
        raise
    F = rp.features(J, dims, ex, cfg)
    main = cfg.dtw_feat(ex)
    used = {main} | {k for ks in rp.ERROR_FEATURES[ex].values() for k in ks}
    F = {k: v for k, v in F.items() if k in used}     # only the exercise's features (others may be NaN)
    out = []
    for o, g in seg_rows.groupby("person_orientation"):
        ok, nok = [], []
        if args.skip_incomplete and any(np.isnan(v[r.first_frame:r.last_frame + 1]).any()
                                        for r in g.itertuples() for v in F.values()):
            continue                     # block not (fully) reconstructed yet
        if args.shuffle_labels:   # null model: permute correct/incorrect (with error codes) within the block
            g = g.copy()
            hs = int(hashlib.sha256(f"shuffle|{vid}|{o}".encode()).hexdigest()[:8], 16)
            perm = np.random.default_rng(hs).permutation(len(g))
            for c in ("correctness_score", "error_description", "additional_errors"):
                g[c] = g[c].values[perm]
        for r in g.itertuples():
            f = {k: v[r.first_frame:r.last_frame + 1] for k, v in F.items()}
            if any(np.isnan(v).any() for v in f.values()):
                raise ValueError(f"NaN in {vid} cam{cam} o={o} rep {r.repetition_number}")
            if r.correctness_score == 1:
                ok.append(f)
            else:
                codes = [r.error_description]
                if cfg.include_additional and isinstance(r.additional_errors, str):
                    codes += [c.strip() for c in r.additional_errors.split(",") if c.strip()]
                ks = sorted({k for c in codes for k in rp.ERROR_FEATURES[ex].get(c, [])})
                if not ks:
                    raise ValueError(f"no features for {codes} in ex{ex}")
                nok.append((f, ks))
        for s in seeds:
            h = int(hashlib.sha256(f"{s}|{vid}|{o}|{cam}".encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(h)     # same reference for 2D and 3D of one sequence and seed
            psi = rp.sequence_psi(ok, nok, main, cfg, rng)
            out.append(dict(exercise=ex, video_id=vid, orientation=o, cam=cam, viewpoint=VIEW[cam](o), dims=dims,
                            seed=s, psi=psi, n_ok=len(ok), n_nok=len(nok)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--source", default="mediapipe")
    ap.add_argument("--dims", default="2d,3d")
    ap.add_argument("--cams", default="1,2,3")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--videos", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--cfg", nargs="*", default=[])
    ap.add_argument("--shuffle-labels", action="store_true", help="null model: permute labels within each block")
    ap.add_argument("--skip-incomplete", action="store_true", help="skip orientation blocks with NaN (unfinished chunks)")
    args = ap.parse_args()
    cfg = parse_cfg(args.cfg)
    a, b = (args.seeds.split("-") + [None])[:2]
    seeds = list(range(int(a), int(b) + 1)) if b else [int(a)]
    seg = pd.read_csv(Path(args.data) / "raw" / "Segmentation.csv", sep=";")
    vids = [v for v in seg.video_id.unique() if not args.videos or v in args.videos.split(",")]
    tasks = [(args, cfg, v, int(seg[seg.video_id == v].exercise_id.iloc[0]), int(c), d, seeds, seg[seg.video_id == v])
             for v in vids for c in args.cams.split(",") for d in args.dims.split(",")]
    with Pool(args.procs) as p:
        rows = [r for res in p.imap_unordered(job, tasks) for r in res]
    df = pd.DataFrame(rows).sort_values(["exercise", "video_id", "orientation", "cam", "dims", "seed"])
    df.to_csv(args.out, index=False)
    Path(args.out).with_suffix(".cfg.json").write_text(json.dumps(
        dict(config=str(cfg), source=args.source, seeds=seeds, n_rows=len(df)), indent=1))
    print("wrote", args.out, len(df), "rows;", cfg)


if __name__ == "__main__":
    main()
