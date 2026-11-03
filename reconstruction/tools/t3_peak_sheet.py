"""Check sheet per clip: knee-flexion trace with the counted repetitions, and the video frames at those
peaks, so a person can see whether each counted repetition is a real squat or lunge bottom.
Writes <out>/<name>.jpg (contains people: never commit).

  python t3_peak_sheet.py RUNS/t3-internet OUT name [name ...]
"""
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, savgol_filter

sys.path.insert(0, str(Path(__file__).parent))
from t3_internet_metrics import SMPL_LEG, knee_flexion  # noqa: E402

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
for name in sys.argv[3:]:
    z = np.load(root / "out" / name / "joints.npz")
    flex = knee_flexion(z["incam"], SMPL_LEG).max(1)
    f = savgol_filter(flex, 9, 2)
    peaks, _ = find_peaks(f, prominence=25, distance=18)
    cap = cv2.VideoCapture(str(root / "clips" / f"{name}.mp4"))
    frames = []
    for p in peaks[:8]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(p))
        ok, im = cap.read()
        if ok:
            x0, y0, x1, y1 = z["bbx_xyxy"][p].astype(int)
            im = cv2.rectangle(im, (x0, y0), (x1, y1), (0, 200, 255), 4)
            h = 300
            frames.append(cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h)))
    fig, ax = plt.subplots(figsize=(10, 2.2), dpi=100)
    ax.plot(f, color="#2a78d6")
    ax.plot(peaks, f[peaks], "o", color="#eb6834")
    ax.set_title(f"{name}: {len(peaks)} counted, knee flexion (deg)", fontsize=10, loc="left")
    fig.tight_layout()
    fig.canvas.draw()
    plot = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[..., :3], cv2.COLOR_RGB2BGR)
    plt.close(fig)
    if frames:
        strip = np.concatenate(frames, 1)
        w = plot.shape[1]
        strip = cv2.resize(strip, (w, int(strip.shape[0] * w / strip.shape[1])))
        plot = np.concatenate([plot, strip], 0)
    cv2.imwrite(str(out / f"{name}.jpg"), plot, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(name, len(peaks), list(map(int, peaks[:12])))
