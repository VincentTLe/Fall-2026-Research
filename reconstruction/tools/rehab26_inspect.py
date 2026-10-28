"""Inspect REHAB26-ViewAngles: skeleton shapes vs. segmentation ranges and orientation labels.

Usage: python -I rehab26_inspect.py <data_root>   (data_root holds raw/ and extracted/mediapipe/)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
seg = pd.read_csv(root / "raw" / "Segmentation.csv", sep=";")
print("rows", len(seg), "columns", list(seg.columns))
print(seg.groupby(["exercise_id"]).size().to_string())
print("orientations", sorted(seg.person_orientation.unique()))
print("persons", sorted(seg.person.unique()))
print("correctness", seg.correctness_score.value_counts().to_dict())
g = seg.groupby(["video_id", "person_orientation"])
counts = g.correctness_score.agg(lambda s: (int((s == 1).sum()), int((s == 5).sum())))
print("(ok,nok) per recording-orientation:", counts.value_counts().to_dict())
print("orientations per video:", seg.groupby("video_id").person_orientation.nunique().value_counts().to_dict())
print("error codes:", seg.error_description.value_counts(dropna=False).to_dict())
print("additional_errors non-empty:", int(seg.additional_errors.notna().sum()))
seg["len"] = seg.last_frame - seg.first_frame + 1
print("rep length frames:", seg.len.describe().round(1).to_dict())
# per-file shape and range check
bad = 0
for f in sorted((root / "extracted" / "mediapipe").rglob("*.npy")):
    a = np.load(f, mmap_mode="r", allow_pickle=False)
    vid = "_".join(f.stem.split("_")[:2])
    s = seg[seg.video_id == vid]
    mx = s.last_frame.max()
    nan = float(np.isnan(np.asarray(a[:, :, :])).mean())
    flag = "" if mx < a.shape[0] else "  <-- segmentation beyond skeleton"
    bad += bool(flag)
    print(f.relative_to(root / "extracted"), a.shape, a.dtype, "maxframe", mx, f"nan={nan:.4f}", flag)
print("files with ranges beyond skeleton:", bad)
# value ranges on one file each 2d/3d
for kind in ("2d", "3d"):
    f = sorted((root / "extracted" / "mediapipe").rglob(f"*_{kind}.npy"))[0]
    a = np.load(f, allow_pickle=False)
    print(kind, f.name, "per-channel min", np.nanmin(a, axis=(0, 1)).round(3), "max", np.nanmax(a, axis=(0, 1)).round(3))
