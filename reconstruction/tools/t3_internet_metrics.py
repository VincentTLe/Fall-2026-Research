"""T3: checks for GVHMR on internet clips, which have no ground truth, calibrated on REHAB24-6 squats,
which do (motion capture and annotated repetitions).

Per sequence (SMPL-24 joints from gvhmr_export_joints.py, camera frame, 30 fps):
  coverage   share of frames where at least 8 of the 12 limb joints have ViTPose confidence > 0.5
  reproj     median distance between GVHMR's 3D limb joints projected with its own K and ViTPose's 2D
             joints (confidence > 0.5), in % of the tracked box height: does the 3D agree with the image?
  jitter     mean acceleration of root-relative limb joints, mm/frame^2 (smoothness)
  knee       5th/95th percentile of the more-bent knee's flexion (180 deg = straight leg)
  reps       peaks of that flexion (Savitzky-Golay 9 frames; prominence 25 deg; at least 0.6 s apart)
  reps_depth the same peaks, kept only if at least 0.75 x the clip's 95th-percentile flexion. Rule fixed
             on 2026-11-02 after round 1 (two failure cases: small dips before a set, steps between lunges),
             before any round-2 clip was downloaded; round 2 is its test.
REHAB24-6 rows add the same numbers from motion capture, the knee-angle error of GVHMR against it, and
the annotated repetition count, so the counter and the thresholds can be judged where the truth is known.

  python t3_internet_metrics.py --internet RUNS/t3-internet --clips clips.txt --rehab DATA/rehab24-6 \
      --rehab-chunks RUNS/t2-chunks/out-truef --out RUNS/t3-internet/metrics
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, savgol_filter

COCO = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]          # L/R shoulder, elbow, wrist, hip, knee, ankle
SMPL = [16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]           # the same joints in SMPL-24
MOCAP = [7, 12, 8, 13, 9, 14, 16, 21, 17, 22, 18, 23]       # the same joints in REHAB24-6's 26
SMPL_LEG = ((1, 4, 7), (2, 5, 8))                           # (hip, knee, ankle) left, right
MOCAP_LEG = ((16, 17, 18), (21, 22, 23))
FPS = 30


def knee_flexion(j, legs):
    """(F, 2) knee flexion in degrees, 0 = straight leg."""
    out = []
    for h, k, a in legs:
        u, v = j[:, h] - j[:, k], j[:, a] - j[:, k]
        c = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)
        out.append(180.0 - np.degrees(np.arccos(np.clip(c, -1, 1))))
    return np.stack(out, 1)


def reps(flex, depth=False):
    f = savgol_filter(flex, 9, 2) if len(flex) >= 9 else flex
    peaks, _ = find_peaks(f, prominence=25, distance=int(0.6 * FPS))
    if depth:
        peaks = peaks[f[peaks] >= 0.75 * np.percentile(f, 95)]
    return int(len(peaks))


def jitter(j, root):
    """j: (F, 12, 3) limb joints in metres; root: indices of the two hips in j."""
    r = j - j[:, root].mean(1, keepdims=True)
    acc = r[2:] - 2 * r[1:-1] + r[:-2]
    return float(np.linalg.norm(acc, axis=-1).mean() * 1000)


def gvhmr_metrics(z):
    inc, vit, bbx, K = z["incam"], z["vitpose"], z["bbx_xyxy"], z["K"]
    conf = vit[:, COCO, 2] > 0.5
    p = inc[:, SMPL] @ K.T
    uv = p[..., :2] / np.maximum(p[..., 2:3], 1e-6)
    h = np.maximum(bbx[:, 3] - bbx[:, 1], 1)[:, None]
    e = np.linalg.norm(uv - vit[:, COCO, :2], axis=-1) / h * 100
    flex = knee_flexion(inc, SMPL_LEG).max(1)
    return {
        "frames": int(len(inc)),
        "coverage_pct": round(float((conf.sum(1) >= 8).mean() * 100), 1),
        "reproj_pct_box": round(float(np.median(e[conf])), 2) if conf.any() else None,
        "jitter_mm": round(jitter(inc[:, SMPL], [6, 7]), 2),
        "knee_flex_p5_p95": [round(float(np.percentile(flex, q)), 1) for q in (5, 95)],
        "reps": reps(flex),
        "reps_depth": reps(flex, depth=True),
        "_flex": flex,
        "_inc": inc,
    }


def internet(args):
    rows = []
    for line in Path(args.clips).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        src, cid, kind = line.split()[:3]
        d = Path(args.internet) / "out" / f"{src}_{cid}"
        if not (d / "joints.npz").is_file():
            rows.append({"name": f"{src}_{cid}", "kind": kind, "status": "missing"})
            continue
        m = gvhmr_metrics(np.load(d / "joints.npz"))
        rows.append({"name": f"{src}_{cid}", "kind": kind, "status": "ok",
                     **{k: v for k, v in m.items() if not k.startswith("_")}})
    return rows


def rehab(args):
    seg = list(csv.DictReader(open(Path(args.rehab) / "Segmentation.csv"), delimiter=";"))
    rows = []
    for d in sorted(Path(args.rehab_chunks).glob("PM_*-c1[78]-f*")):
        rec, cam, s, e = d.name.split("-")
        s, e = int(s[1:]), int(e)
        ex = next(r["exercise_id"] for r in seg if r["video_id"] == rec)
        if ex != "6" or not (d / "joints.npz").is_file():
            continue
        m = gvhmr_metrics(np.load(d / "joints.npz"))
        mo = np.load(Path(args.rehab) / "3d_joints" / "Ex6" / f"{rec}-30fps.npy")[s:e + 1, :, :3]
        n = min(len(mo), m["frames"])
        mflex = knee_flexion(mo[:n], MOCAP_LEG).max(1)
        ann = sum(1 for r in seg if r["video_id"] == rec and int(r["first_frame"]) >= s and int(r["last_frame"]) <= e)
        rows.append({
            "name": d.name, "kind": "squat (REHAB24-6)", "status": "ok",
            **{k: v for k, v in m.items() if not k.startswith("_")},
            "mocap_jitter_mm": round(jitter(mo[:n, MOCAP], [6, 7]), 2),
            "mocap_knee_flex_p5_p95": [round(float(np.percentile(mflex, q)), 1) for q in (5, 95)],
            "knee_err_deg": round(float(np.abs(m["_flex"][:n] - mflex).mean()), 1),
            "mocap_reps": reps(mflex),
            "mocap_reps_depth": reps(mflex, depth=True),
            "annotated_reps": ann,
        })
    return rows


def table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--internet", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--rehab", required=True)
    ap.add_argument("--rehab-chunks", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reh, net = rehab(args), internet(args)
    json.dump({"rehab": reh, "internet": net}, open(out / "metrics.json", "w"), indent=1)
    cols = ["name", "kind", "frames", "coverage_pct", "reproj_pct_box", "jitter_mm", "knee_flex_p5_p95", "reps", "reps_depth"]
    md = ["## REHAB24-6 squats (ground truth available)", "",
          table(reh, cols + ["mocap_jitter_mm", "mocap_knee_flex_p5_p95", "knee_err_deg", "mocap_reps", "mocap_reps_depth", "annotated_reps"]),
          "", "## Internet clips (no ground truth)", "", table(net, cols + ["status"]), ""]
    (out / "metrics.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
