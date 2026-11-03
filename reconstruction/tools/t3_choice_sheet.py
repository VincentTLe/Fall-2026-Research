"""Side-by-side check of the two person choices: default track (red) and t3_gvhmr_pick.py (blue),
on three frames per clip (25/50/75%). Writes <out>/<name>.jpg; contains people, never commit.

  python t3_choice_sheet.py RUNS/t3-internet OUT name [name ...]
"""
import sys
from pathlib import Path

import cv2
import numpy as np

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
for name in sys.argv[3:]:
    bd = np.load(root / "out" / name / "joints.npz")["bbx_xyxy"]
    bp = np.load(root / "out-pick" / name / "joints.npz")["bbx_xyxy"]
    cap = cv2.VideoCapture(str(root / "clips" / f"{name}.mp4"))
    n = min(len(bd), len(bp))
    tiles = []
    for f in (int(n * 0.25), int(n * 0.5), int(n * 0.75)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        if not ok:
            continue
        for b, col, w in ((bd[f], (40, 40, 230), 8), (bp[f], (230, 120, 30), 4)):
            x0, y0, x1, y1 = b.astype(int)
            cv2.rectangle(im, (x0, y0), (x1, y1), col, w)
        tiles.append(cv2.resize(im, (int(im.shape[1] * 260 / im.shape[0]), 260)))
    if tiles:
        cv2.imwrite(str(out / f"{name}.jpg"), np.concatenate(tiles, 1), [cv2.IMWRITE_JPEG_QUALITY, 78])
        print(name, "ok")
