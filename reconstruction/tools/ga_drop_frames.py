"""Drop frames from a GaussianAvatar split whose input pose misses the person (T4b run 1).

Rule (no mocap used): projected-SMPL vs SAM-mask silhouette IoU below --min-iou, from RUN_DIR/check/iou.json written by
gvhmr_to_gaussianavatar.py --check. Removes images/<f>.png, masks/<f>.png and the matching smpl_parms.pth rows
(rows are in sorted file-name order, as GaussianAvatar reads them). Dropped files move to RUN_DIR/dropped/<split>/.
  python ga_drop_frames.py RUN_DIR --split train --min-iou 0.6
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("run_dir"); ap.add_argument("--split", default="train"); ap.add_argument("--min-iou", type=float, default=0.6)
a = ap.parse_args()
run = Path(a.run_dir); d = run / a.split
iou = json.load(open(run / "check/iou.json"))["per_frame"]
names = sorted(f[:-4] for f in os.listdir(d / "images"))
p = torch.load(d / "smpl_parms.pth")
assert len(p["body_pose"]) == len(names) == len(p["trans"])
keep = [i for i, n in enumerate(names) if iou[n] >= a.min_iou]
drop = [n for i, n in enumerate(names) if iou[n] < a.min_iou]
for n in drop:
    for sub in ("images", "masks"):
        (run / "dropped" / a.split / sub).mkdir(parents=True, exist_ok=True)
        shutil.move(str(d / sub / f"{n}.png"), str(run / "dropped" / a.split / sub / f"{n}.png"))
torch.save({"beta": p["beta"], "body_pose": p["body_pose"][keep].clone(), "trans": p["trans"][keep].clone()}, d / "smpl_parms.pth")
json.dump({"min_iou": a.min_iou, "dropped": {n: iou[n] for n in drop}, "kept": len(keep)},
          open(run / f"dropped_{a.split}.json", "w"), indent=1)
assert len(os.listdir(d / "images")) == len(keep) == len(torch.load(d / "smpl_parms.pth")["body_pose"])
print(f"{a.split}: dropped {len(drop)} frames {[(n, iou[n]) for n in drop]}; kept {len(keep)}")
