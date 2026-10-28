"""Plan GVHMR chunks for REHAB26-ViewAngles: repetition spans only, <= MAXF frames, whole repetitions per chunk.

For each (recording, orientation, camera) the reps are sorted; consecutive reps are grouped greedily while the span
from the group's first frame to its last frame stays <= MAXF. Jobs are ordered so that Table 3 viewpoints are
covered for all persons first. Output lines: <video> <name> <start> <end> (inclusive frames).

Usage: python -I rehab26_gvhmr_plan.py <data_root> <jobs.txt> [maxf]
"""
import sys
from pathlib import Path

import pandas as pd

root, out = Path(sys.argv[1]), Path(sys.argv[2])
MAXF = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
seg = pd.read_csv(root / "raw" / "Segmentation.csv", sep=";")
VIEW = {1: lambda o: -o, 2: lambda o: 45 - o, 3: lambda o: 90 - o}
PRIORITY = [0, -90, 90, -30, 30, -60, 60, -10, 10, -40, 40, -20, 20, -15, 15]
jobs = []
for (vid, o), g in seg.groupby(["video_id", "person_orientation"]):
    g = g.sort_values("first_frame")
    groups, cur = [], []
    for r in g.itertuples():
        if cur and r.last_frame - cur[0].first_frame + 1 > MAXF:
            groups.append(cur); cur = []
        cur.append(r)
    groups.append(cur)
    ex = vid.split("_")[0]
    for cam in (1, 2, 3):
        vp = VIEW[cam](o)
        pr = PRIORITY.index(vp) if vp in PRIORITY else len(PRIORITY) + abs(vp)
        for grp in groups:
            s, e = grp[0].first_frame, max(r.last_frame for r in grp)
            assert e - s + 1 <= MAXF
            jobs.append((pr, vid, cam, o, s, e,
                         str(root / "videos-extracted" / "videos" / ex / f"{vid}_cam{cam}.mp4"),
                         f"{vid}_cam{cam}-o{o:02d}-f{s:05d}-{e:05d}"))
jobs.sort()
out.write_text("".join(f"{j[6]} {j[7]} {j[4]} {j[5]}\n" for j in jobs))
tot = sum(j[5] - j[4] + 1 for j in jobs)
ntab = sum(1 for j in jobs if j[0] < len(PRIORITY))
print(f"{len(jobs)} chunks, {tot} frames; {ntab} chunks at Table 3 viewpoints; "
      f"rep frames {int((seg.last_frame - seg.first_frame + 1).sum()) * 3}")
