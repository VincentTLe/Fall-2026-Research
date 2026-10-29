"""Check how REHAB26 skeleton rows line up with video frames when the video has more frames.

Motion energy of the video (mean |frame difference| on 64x114 gray frames) is cross-correlated with
MediaPipe 2D joint speed in sliding windows; a constant best lag means trimmed frames, a drifting lag
means dropped frames.

Usage: python -I rehab26_frame_sync.py <data_root> <video_stem> [<video_stem> ...]
"""
import subprocess
import sys
from pathlib import Path

import numpy as np


def video_energy(path):
    w, h = 64, 114
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={w}:{h},format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    f = np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32)
    return np.r_[0, np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))]


def z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def best_lag(v, s, start, win, lags):
    """lag L means skeleton row r corresponds to video frame r + L."""
    a = z(s[start:start + win])
    sc = []
    for L in lags:
        b = v[start + L:start + L + win]
        sc.append(np.dot(a, z(b)) / win if len(b) == win else -9)
    i = int(np.argmax(sc))
    return lags[i], sc[i]


root = Path(sys.argv[1])
for stem in sys.argv[2:]:
    ex = stem.split("_")[0]
    v = video_energy(root / "videos-extracted" / "videos" / ex / f"{stem}.mp4")
    J = np.load(root / "extracted" / "mediapipe" / ex / f"{stem}_2d.npy", allow_pickle=False)[..., :2] * [1080, 1920]
    s = np.r_[0, np.linalg.norm(np.diff(J[:, :25], axis=0), axis=-1).mean(1)]
    print(f"{stem}: video {len(v)} frames, skeleton {len(s)} rows, diff {len(v) - len(s)}")
    lags = list(range(-50, len(v) - len(s) + 51))
    for start in np.linspace(0, len(s) - 1201, 8).astype(int):
        L, c = best_lag(v, s, start, 1200, lags)
        print(f"  rows {start:6d}-{start + 1199:6d}: best lag {L:+5d}  corr {c:.2f}")
