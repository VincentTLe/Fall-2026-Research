"""Draw the summary figures shown in the top-level README.

Every number is copied from a result file in reconstruction/results/ (cited next to it).
No dataset images are used, so the figures are safe to commit.

    python reconstruction/tools/make_summary_figures.py reconstruction/results/figures
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def style(ax, grid_axis="y"):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def fig_cameras(out):
    # 2026-10-27-t1-multiview.md, camera-count table: MPJPE, 10 limb joints, CMU Panoptic 171204_pose1.
    spread = {3: [24.6, 22.7, 21.8], 4: [20.0, 18.3, 18.6], 5: [16.2, 16.3, 18.0], 6: [14.8, 18.7, 16.8]}
    lost = {3: [True, True, False]}          # subsets that dropped 23-28% of the joints
    clustered5 = [26.0, 24.6]                # 5 cameras bunched on one side (87-104 degree arc)
    all31 = 11.7
    xs = [0, 1, 2, 3, 4.3]
    labels = ["3", "4", "5", "6", "all 31"]

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    style(ax)
    means = []
    for x, n in zip(xs[:4], (3, 4, 5, 6)):
        vals = spread[n]
        means.append(sum(vals) / 3)
        for i, v in enumerate(vals):
            hollow = lost.get(n, [False] * 3)[i]
            ax.plot(x - 0.32 + 0.1 * i, v, "o", ms=6, mfc=SURFACE if hollow else "#86b6ef",
                    mec=BLUE if hollow else SURFACE, mew=1.5 if hollow else 1.0, zorder=3)
    ax.plot(xs[:4] + [xs[4]], means + [all31], "-", color=BLUE, lw=2, zorder=2)
    ax.plot(xs[:4] + [xs[4]], means + [all31], "o", ms=9, color=BLUE, mec=SURFACE, mew=2, zorder=4)
    for x, m in zip(xs[:4], means):
        ax.annotate(f"{m:.1f}", (x, m), xytext=(14, 0), textcoords="offset points", va="center",
                    color=INK, fontsize=10)
    ax.annotate(f"{all31:.1f} mm", (xs[4], all31), xytext=(0, -16), textcoords="offset points",
                ha="center", color=INK, fontsize=10)
    for i, v in enumerate(clustered5):
        ax.plot(2.32 + 0.12 * i, v, "D", ms=7, color=ORANGE, mec=SURFACE, mew=1.5, zorder=4)
    ax.annotate("5 cameras bunched\non one side: 25-26 mm", (2.44, 25.3), xytext=(12, 0),
                textcoords="offset points", va="center", color=INK2, fontsize=9.5)
    ax.text(-0.45, 13.0, "hollow = this camera set\nlost 23-28% of the joints", color=INK2, fontsize=9.5)
    ax.set_xticks(xs, labels)
    ax.set_xlim(-0.5, 4.8)
    ax.set_ylim(0, 30)
    ax.set_xlabel("Number of cameras, spread around the person")
    ax.set_ylabel("3D joint error (mm), lower is better")
    ax.set_title("Q1. More cameras help, but 5 spread cameras get most of the way",
                 loc="left", fontsize=12, color=INK, pad=12)
    ax.text(0, -0.25, "CMU Panoptic, 392 frames, 10 limb joints, EasyMocap triangulation vs. the dome's own 3D. "
            "Small dots: 3 camera\nsubsets per count; big dots: their mean. Source: results/2026-10-27-t1-multiview.md",
            transform=ax.transAxes, fontsize=8.5, color=MUTED, va="top")
    fig.tight_layout()
    fig.savefig(out / "q1_camera_count.png", bbox_inches="tight")
    plt.close(fig)


def fig_one_vs_two(out):
    # 2026-10-29-t2b-calibration-benchmark.md, section C, "New" column: the same 183 repetitions and 10 joints as
    # 2026-10-27-t2-monocular.md, with the authors' newer camera estimate (old cameras: 47.5 / 57.2 mm abs).
    names = ["Two cameras (ViTPose 2D)", "Two cameras (EasyMocap HRNet)", "One camera, front-ish (GVHMR)",
             "One camera, side (GVHMR)"]
    colors = [BLUE, BLUE, ORANGE, ORANGE]
    pa = [37.0, 43.1, 46.8, 69.6]
    abs_ = [42.9, 53.3, 196.6, 159.6]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), dpi=150, sharey=True)
    for ax, vals, title, xmax in (
            (axes[0], pa, "Body pose only\n(after the best rotation, shift and scale)", 80),
            (axes[1], abs_, "Where the body is in the room\n(no alignment at all)", 210)):
        style(ax, "x")
        y = list(range(len(names)))[::-1]
        ax.barh(y, vals, height=0.55, color=colors, edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, vals):
            ax.annotate(f"{v:.1f}", (v, yi), xytext=(5, 0), textcoords="offset points", va="center",
                        fontsize=10, color=INK)
        ax.set_xlim(0, xmax)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK2)
        ax.set_xlabel("joint error (mm), lower is better")
    axes[0].set_yticks(list(range(len(names)))[::-1], names)
    fig.suptitle("Q2. Two cameras beat one, most of all for where the body is", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.text(0.01, -0.06, "REHAB24-6 (rehab exercises, motion-capture ground truth), 12 recordings, the same 183 "
             "repetitions for every row, authors' newer cameras.\nBlue = two cameras, orange = one camera. GVHMR used "
             "unofficial weights. Sources: results/2026-10-27-t2-monocular.md, 2026-10-29-t2b-calibration-benchmark.md",
             fontsize=8.5, color=MUTED, va="top")
    fig.tight_layout()
    fig.savefig(out / "q2_one_vs_two_cameras.png", bbox_inches="tight")
    plt.close(fig)


def fig_benchmark(out):
    # 2026-10-29-t2b-calibration-benchmark.md, section D: 3D MPJPE (cm), BestFit, the paper's 12 joints, all 65
    # recordings, both cameras, all repetitions. Paper rows: Information Systems 2025, Table 5.
    rows = [("GVHMR (ours), per-frame fit", 6.00, BLUE), ("GVHMR (ours), one fit per repetition", 7.16, BLUE),
            ("MediaPipe Pose full", 8.4, ORANGE), ("MediaPipe Pose heavy", 8.7, ORANGE),
            ("MHFormer + HRNet", 9.0, ORANGE), ("MotionBERT-lite + HRNet", 9.2, ORANGE),
            ("MotionBERT-lite + perfect 2D (not realistic)", 7.7, MUTED)]
    fig, ax = plt.subplots(figsize=(8.6, 3.9), dpi=150)
    style(ax, "x")
    y = list(range(len(rows)))[::-1]
    ax.barh(y, [r[1] for r in rows], height=0.55, color=[r[2] for r in rows], edgecolor=SURFACE, linewidth=2)
    for yi, (_, v, c) in zip(y, rows):
        ax.annotate(f"{v:.2f}" if c == BLUE else f"{v:.1f}", (v, yi), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=10, color=INK)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 10.5)
    ax.set_xlabel("3D joint error (cm) after the best rotation, shift and scale, lower is better")
    ax.set_title("Q2c. On the REHAB24-6 authors' own benchmark, GVHMR has lower error than every method they tested",
                 loc="left", fontsize=12, color=INK, pad=12)
    fig.text(0.01, -0.03, "All 65 recordings, both cameras, 1,072 repetitions, the paper's 12 joints. Blue = GVHMR "
             "(unofficial weights), orange = the paper's Table 5.\nThe paper does not say whether its fit is per frame "
             "or per repetition, so both are shown. Source: results/2026-10-29-t2b-calibration-benchmark.md",
             fontsize=8.5, color=MUTED, va="top")
    fig.tight_layout()
    fig.savefig(out / "q2c_rehab24_benchmark.png", bbox_inches="tight")
    plt.close(fig)


def fig_assessment(out):
    # 2026-10-28-t5-rehab-assessment.md, section 4.3 (all 495 sequences) and the null-model floor.
    ex = ["Arm abduction", "Shoulder extension", "Squat"]
    mpp2d = [0.490, 0.489, 0.502]
    mpp3d = [0.460, 0.443, 0.450]
    gvhmr = [0.602, 0.579, 0.543]
    diff = ["+0.11 [+0.09, +0.13]", "+0.09 [+0.06, +0.12]", "+0.04 [+0.01, +0.07]\nnot robust (see file)"]
    fig, ax = plt.subplots(figsize=(8.6, 3.5), dpi=150)
    style(ax, "x")
    ax.axvspan(0, 0.11, color="#f0efec", zorder=0)
    ax.text(0.055, 2.55, "chance\nlevel", ha="center", va="center", fontsize=8.5, color=MUTED)
    y = [2, 1, 0]
    for yi, a, b, c in zip(y, mpp3d, mpp2d, gvhmr):
        ax.plot([min(a, b, c), max(a, b, c)], [yi, yi], color=GRID, lw=2, zorder=1)
    ax.plot(mpp3d, y, "o", ms=9, color=AQUA, mec=SURFACE, mew=2, label="MediaPipe 3D (paper's 3D)", zorder=3)
    ax.plot(mpp2d, y, "o", ms=9, color=ORANGE, mec=SURFACE, mew=2, label="MediaPipe 2D (paper's best)",
            zorder=3)
    ax.plot(gvhmr, y, "o", ms=10, color=BLUE, mec=SURFACE, mew=2, label="GVHMR 3D (ours)", zorder=4)
    for yi, g, d in zip(y, gvhmr, diff):
        ax.annotate(f"{g:.2f}", (g, yi), xytext=(0, 10), textcoords="offset points", ha="center",
                    fontsize=9.5, color=INK)
        ax.annotate(d, (0.70, yi), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9,
                    color=INK2, annotation_clip=False)
    ax.text(0.70, 2.55, "  GVHMR - MediaPipe 2D,\n  95% CI over people", fontsize=8.5, color=MUTED, va="center")
    ax.set_yticks(y, ex)
    ax.set_xlim(0, 0.70)
    ax.set_ylim(-0.6, 2.9)
    ax.set_xlabel("Separability Ψ of correct vs. incorrect repetitions, higher is better")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False,
              fontsize=9.5,
              handletextpad=0.3, columnspacing=1.2)
    ax.set_title("Q2b. With GVHMR, 3D separates good from bad arm reps better than 2D", loc="left",
                 fontsize=12, color=INK, pad=30)
    fig.text(0.01, -0.03, "REHAB26-ViewAngles, all 495 sequences (5 people x 3 cameras x 11 orientations x 3 "
             "exercises). Same metric code for every method.\nThe paper (arXiv 2609.35726) found 2D > 3D with "
             "MediaPipe. Caveats in the result file. Source: results/2026-10-28-t5-rehab-assessment.md",
             fontsize=8.5, color=MUTED, va="top")
    fig.tight_layout()
    fig.savefig(out / "q2b_assessment_separability.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "reconstruction/results/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig_cameras(out)
    fig_one_vs_two(out)
    fig_benchmark(out)
    fig_assessment(out)
    print("wrote", sorted(p.name for p in out.glob("*.png")))
