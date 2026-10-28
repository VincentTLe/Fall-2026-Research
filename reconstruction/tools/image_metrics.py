"""PSNR / SSIM / LPIPS between two folders of same-named images, the way GaussianAvatar's eval.py computes them.

GaussianAvatar (eval.py, commit d981c62) feeds [0, 1] images to torchmetrics' LPIPS with the default
normalize=False, which expects [-1, 1]. This script reports that value ("lpips_as_eval_py") for comparison with the
paper, and also LPIPS on correctly scaled inputs ("lpips_normalized").

    python image_metrics.py RENDER_DIR GT_DIR [--device cuda] [--json out.json]
    python image_metrics.py --self-check

Metrics are per image (batch of 1) and then averaged, as in eval.py.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def load(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None]  # 1x3xHxW in [0, 1]


class Metrics:
    def __init__(self, device):
        self.device = device
        self.psnr = PeakSignalNoiseRatio(data_range=1).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1).to(device)
        self.lpips_raw = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
        self.lpips_norm = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)

    @torch.no_grad()
    def __call__(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        return {
            "psnr": self.psnr(x, y).item(),
            "ssim": self.ssim(x, y).item(),
            "lpips_as_eval_py": self.lpips_raw(x, y).item(),
            "lpips_normalized": self.lpips_norm(x, y).item(),
        }


def compare_dirs(render_dir, gt_dir, device):
    names = sorted(f for f in os.listdir(gt_dir) if f.lower().endswith((".png", ".jpg")))
    missing = [n for n in names if not os.path.exists(os.path.join(render_dir, n))]
    if missing or not names:
        sys.exit(f"{len(names)} GT images, {len(missing)} without a render (first: {missing[:3]})")
    m = Metrics(device)
    rows = []
    for n in names:
        x, y = load(os.path.join(render_dir, n)), load(os.path.join(gt_dir, n))
        if x.shape != y.shape:
            sys.exit(f"{n}: render {tuple(x.shape)} vs GT {tuple(y.shape)}")
        rows.append(m(x, y))
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return {"n_images": len(rows), "size_hw": list(y.shape[2:]), "mean": mean, "per_image": dict(zip(names, rows))}


def self_check(device):
    torch.manual_seed(0)
    m = Metrics(device)
    a = torch.rand(1, 3, 64, 64)
    same = m(a, a)
    assert same["psnr"] > 80 and abs(same["ssim"] - 1) < 1e-4, same
    assert same["lpips_as_eval_py"] < 1e-6 and same["lpips_normalized"] < 1e-6, same
    # PSNR of a constant 0.1 offset must be 20 dB: 10*log10(1 / 0.01)
    b = (a * 0.5 + 0.2)
    off = m(b + 0.1, b)
    assert abs(off["psnr"] - 20.0) < 1e-3, off
    # normalize=True on [0,1] must equal normalize=False on the same images mapped to [-1,1]
    c = torch.rand(1, 3, 64, 64)
    l1 = m.lpips_norm(a.to(device), c.to(device)).item()
    l2 = m.lpips_raw((a * 2 - 1).to(device), (c * 2 - 1).to(device)).item()
    assert abs(l1 - l2) < 1e-6, (l1, l2)
    print("self-check ok:", {k: round(v, 6) for k, v in off.items()}, "lpips norm/raw-mapped", round(l1, 6), round(l2, 6))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("render_dir", nargs="?")
    p.add_argument("gt_dir", nargs="?")
    p.add_argument("--device", default="cpu")
    p.add_argument("--json")
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        self_check(a.device)
        sys.exit(0)
    out = compare_dirs(a.render_dir, a.gt_dir, a.device)
    print(json.dumps({"n_images": out["n_images"], "size_hw": out["size_hw"], "mean": out["mean"]}, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
