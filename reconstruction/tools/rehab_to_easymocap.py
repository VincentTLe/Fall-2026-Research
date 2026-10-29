"""Make an EasyMocap data folder for one REHAB24-6 recording: intri.yml/extri.yml for cameras
"17" and "18" from the DLT calibration (calib_c17/18.npz: K, R world->camera, centre C), and
the repetition span of both videos as images/<cam>/%06d.jpg (image 0 = video/mocap frame
`start`, written to span.json).

Check: the mocap joints projected through the written yml files must equal the dataset's 2D
joints (printed as max pixel difference over the span).

  python rehab_to_easymocap.py --data ~/fall2026/data/rehab24-6 --calib ~/fall2026/runs/t2-gvhmr \
      --rec PM_032 --out ~/fall2026/data/rehab24-6-emc/PM_032
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from emc_yml import read_cameras, write_cameras

VIDEO = {"17": "{v}-Camera17-30fps.mp4", "18": "{v}-Camera18-30fps-transposed.mp4"}


def main(a):
    seg = pd.read_csv(Path(a.data) / "Segmentation.csv", sep=";")
    seg = seg[seg.video_id == a.rec]
    ex = int(seg.exercise_id.iloc[0])
    start, end = int(seg.first_frame.min()), int(seg.last_frame.max())
    cams = {}
    for cam in VIDEO:
        c = np.load(Path(a.calib) / f"calib_c{cam}.npz")
        cams[cam] = {"K": c["K"], "dist": np.zeros(5), "R": c["R"], "T": -c["R"] @ c["C"]}
    out = Path(a.out)
    write_cameras(out, cams)
    gt3 = np.load(Path(a.data) / f"3d_joints/Ex{ex}/{a.rec}-30fps.npy")[start:end + 1, :, :3]
    for cam, cc in read_cameras(out).items():
        X = gt3.reshape(-1, 3) @ np.asarray(cc["R"]).T + np.asarray(cc["T"]).reshape(1, 3)
        x = X @ np.asarray(cc["K"]).T
        x = x[:, :2] / x[:, 2:]
        gt2 = np.load(Path(a.data) / f"2d_joints/Ex{ex}/{a.rec}-c{cam}-30fps.npy")[start:end + 1].reshape(-1, 2)
        print(f"{a.rec} cam {cam}: yml projection vs dataset 2D, max {np.abs(x - gt2).max():.3f} px")
    for cam, name in VIDEO.items():
        d = out / "images" / cam
        d.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(Path(a.data) / f"videos/Ex{ex}" / name.format(v=a.rec)),
                        "-vf", f"trim=start_frame={start}:end_frame={end + 1},setpts=PTS-STARTPTS",
                        "-q:v", "2", "-start_number", "0", str(d / "%06d.jpg")], check=True)
        print(f"{a.rec} cam {cam}: {len(list(d.glob('*.jpg')))} images (span {start}-{end}, {end - start + 1} frames)")
    json.dump({"rec": a.rec, "exercise": ex, "start": start, "end": end, "fps": 30}, open(out / "span.json", "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--rec", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
