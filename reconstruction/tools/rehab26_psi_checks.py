"""Sanity checks for the T5 separability pipeline on REHAB26 MediaPipe skeletons.

1. Feature ranges on frontal sequences (anatomical plausibility, 2D pixel aspect and 3D).
2. Fault injection: (a) shuffled correct/incorrect labels within each sequence, (b) each error scored on the
   features of a different error of the same exercise. Both should collapse Psi compared with the true labels.

Usage: python -I rehab26_psi_checks.py <data_root>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rehab_psi as rp  # noqa: E402

root = Path(sys.argv[1])
seg = pd.read_csv(root / "raw" / "Segmentation.csv", sep=";")
cfg = rp.Config(aspect=(1080.0, 1920.0))


def feats(vid, cam, dims):
    ex = int(vid[2])
    J = np.load(root / "extracted" / "mediapipe" / f"ex{ex}" / f"{vid}_cam{cam}_{dims}.npy", allow_pickle=False)
    return rp.features(J, dims, ex, cfg)


print("1. feature ranges over correct reps, frontal view (cam1 o=0), all persons; 5th/50th/95th pct in degrees")
for ex, keys in ((1, ["F1", "F2", "F3", "F4", "F5"]), (7, ["F1", "F2", "F5", "F6", "F7"]), (6, ["F6", "F9", "F11", "F13", "F15"])):
    for dims in ("2d", "3d"):
        vals = {k: [] for k in keys}
        for vid in sorted(seg[seg.exercise_id == ex].video_id.unique()):
            F = feats(vid, 1, dims)
            for r in seg[(seg.video_id == vid) & (seg.person_orientation == 0) & (seg.correctness_score == 1)].itertuples():
                for k in keys:
                    vals[k].append(F[k][r.first_frame:r.last_frame + 1])
        print(f"  ex{ex} {dims}: " + "  ".join(
            f"{k} {np.percentile(np.concatenate(v), 5):.0f}/{np.percentile(np.concatenate(v), 50):.0f}/{np.percentile(np.concatenate(v), 95):.0f}"
            for k, v in vals.items()))

print("2. fault injection (seed 0, all sequences of 2 persons per exercise, cams 1-3)")
rng0 = np.random.default_rng(123)
res = {"true": [], "shuffled_labels": [], "wrong_features": []}
for ex in (1, 7, 6):
    codes = list(rp.ERROR_FEATURES[ex])
    for vid in sorted(seg[seg.exercise_id == ex].video_id.unique())[:2]:
        for cam in (1, 2, 3):
            for dims in ("2d", "3d"):
                F = feats(vid, cam, dims)
                main = cfg.dtw_feat(ex)
                for o, g in seg[seg.video_id == vid].groupby("person_orientation"):
                    reps = [({k: v[r.first_frame:r.last_frame + 1] for k, v in F.items()}, r) for r in g.itertuples()]
                    ok = [f for f, r in reps if r.correctness_score == 1]
                    nok = [(f, rp.ERROR_FEATURES[ex][r.error_description]) for f, r in reps if r.correctness_score == 5]
                    res["true"].append(rp.sequence_psi(ok, nok, main, cfg, np.random.default_rng(0)))
                    # (a) shuffle labels: same counts, random assignment; "erroneous" keep a random error code
                    perm = rng0.permutation(len(reps))
                    n_ok = len(ok)
                    ok_s = [reps[i][0] for i in perm[:n_ok]]
                    nok_s = [(reps[i][0], rp.ERROR_FEATURES[ex][codes[rng0.integers(len(codes))]]) for i in perm[n_ok:]]
                    res["shuffled_labels"].append(rp.sequence_psi(ok_s, nok_s, main, cfg, np.random.default_rng(0)))
                    # (b) wrong features: error i scored on the features of error i+1 (cyclic)
                    nok_w = [(f, rp.ERROR_FEATURES[ex][codes[(codes.index(r.error_description) + 1) % len(codes)]])
                             for f, r in reps if r.correctness_score == 5]
                    res["wrong_features"].append(rp.sequence_psi(ok, nok_w, main, cfg, np.random.default_rng(0)))
for k, v in res.items():
    print(f"  {k:16s} n={len(v)} mean Psi {np.mean(v):.3f}  median {np.median(v):.3f}")
