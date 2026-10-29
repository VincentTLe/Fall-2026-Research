"""Separability metric of Janosova et al. (arXiv 2609.35726, Sec. 4.1) for REHAB26-ViewAngles / REHAB24-6.

One shared implementation used for every T5 number (MediaPipe 2D/3D, GVHMR, mocap).
Skeleton input: (T, 33, >=3) MediaPipe-ordered joints. Other skeletons are mapped to these indices first
(see JOINT_MAP_NOTES in the result file); only the joints in USED_JOINTS are needed.

Paper-silent choices are parameters of Config; the defaults were fixed before any comparison with Table 3.
Run `python -I rehab_psi.py --selftest` for the known-answer tests.
"""
from __future__ import annotations

import dataclasses
import sys

import numpy as np
from scipy import stats

# MediaPipe Pose indices
NOSE, L_EAR, R_EAR = 0, 7, 8
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 23, 24, 25, 26, 27, 28
L_HEEL, R_HEEL, L_TOE, R_TOE = 29, 30, 31, 32
USED_JOINTS = [NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN,
               L_HEEL, R_HEEL, L_TOE, R_TOE]

# Error code -> relevant features (Table 1 of the paper, mapped to the dataset's Error_types.csv)
ERROR_FEATURES = {
    1: {"ERR_ARM_FORWARD": ["F1"], "ERR_ELEVATED_SHOULDER": ["F2"], "ERR_HEAD_TO_SIDE": ["F3"],
        "ERR_BENT_ARM": ["F4"], "ERR_LOW_ABDUCTION": ["F5"]},
    7: {"ERR_TORSO_FORWARD": ["F6"], "ERR_ELEVATED_SHOULDER": ["F2"], "ERR_TORSO_ROTATION": ["F7"],
        "ERR_BENT_ARM": ["F4", "F8"], "ERR_ARM_TO_SIDE": ["F5"]},
    6: {"ERR_KNEES_INSIDE": ["F9", "F10"], "ERR_TORSO_TO_SIDE": ["F6"], "ERR_SHALLOW_SQUAT": ["F15"],
        "ERR_KNEES_FORWARD": ["F11", "F12"], "ERR_LIFTED_HEELS": ["F13", "F14"]},
}
PAPER_EX = {1: "Ex1 arm abduction", 7: "Ex2 shoulder extension", 6: "Ex3 squat"}

# 3D measurement plane per feature (frontal xy, sagittal yz, transverse xz) after hip-to-x-axis normalisation
PLANE = {"F1": "yz", "F2": "xy", "F3": "xy", "F4": "xy", "F5": "xy", "F6": "xy", "F7": "xz", "F8": "yz",
         "F9": "xy", "F10": "xy", "F11": "yz", "F12": "yz", "F13": "yz", "F14": "yz", "F15": "yz"}


@dataclasses.dataclass(frozen=True)
class Config:
    dtw_feature: tuple = (("1", "F5"), ("7", "F1"), ("6", "F15"))  # exercise -> alignment feature
    tau: str = "mean"            # "mean": t*s/sqrt(n); "pred": t*s*sqrt(1+1/n)
    floor: str = "q1_tau"        # "q1_tau": tau >= Q1 of tau over frames; "none"
    include_additional: bool = False   # also score features of additional_errors
    f6_ex7_plane: str = "xy"     # Table 1 says lateral; "yz" treats Ex2 torso-forward as sagittal
    plane3d: bool = True         # False: full 3D angle instead of plane projection
    aspect: tuple = (1.0, 1.0)   # 2D: multiply normalised x, y by (width, height)
    smooth: int = 0              # >0: Savitzky-Golay (order 2) window in frames on joint tracks before features

    def dtw_feat(self, ex: int) -> str:
        return dict(self.dtw_feature)[str(ex)]


def _angle(u, v):
    """Angle in degrees between vector arrays (..., d)."""
    nu = np.linalg.norm(u, axis=-1)
    nv = np.linalg.norm(v, axis=-1)
    c = np.sum(u * v, axis=-1) / np.maximum(nu * nv, 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def normalise_3d(P):
    """Rotate about the vertical (y) axis so that the hip line (L->R) lies along +x; root at mid-hip."""
    P = P - 0.5 * (P[:, [L_HIP]] + P[:, [R_HIP]])
    h = P[:, R_HIP] - P[:, L_HIP]
    a = np.arctan2(h[:, 2], h[:, 0])           # angle of hip line in xz plane
    c, s = np.cos(-a), np.sin(-a)
    x, z = P[..., 0], P[..., 2]
    out = P.copy()
    out[..., 0] = c[:, None] * x - s[:, None] * z
    out[..., 2] = s[:, None] * x + c[:, None] * z
    return out


def features(J, dims: str, ex: int, cfg: Config) -> dict:
    """All features F1..F15 for a (T, 33, C) skeleton. dims '2d' uses x,y (image), '3d' uses x,y,z (y down)."""
    if cfg.smooth:
        from scipy.signal import savgol_filter
        J = J.copy()
        J[..., :3] = savgol_filter(J[..., :3], int(cfg.smooth), 2, axis=0)
    if dims == "2d":
        P = J[..., :2] * np.asarray(cfg.aspect)
        proj = {k: P for k in PLANE}
        vert = np.array([0.0, -1.0])
    else:
        P3 = normalise_3d(J[..., :3])
        idx = {"xy": [0, 1], "yz": [2, 1], "xz": [0, 2]}
        proj = {}
        for k, pl in PLANE.items():
            if k == "F6" and ex == 7:
                pl = cfg.f6_ex7_plane
            proj[k] = P3 if not cfg.plane3d else P3[..., idx[pl]]
        vert = None
    out = {}

    def V(k):  # vertical axis in the projection used for feature k
        if vert is not None:
            return vert
        if not cfg.plane3d:
            return np.array([0.0, -1.0, 0.0])
        return np.array([0.0, -1.0]) if PLANE[k] in ("xy", "yz") else np.array([0.0, 1.0])

    def mid(P, a, b):
        return 0.5 * (P[:, a] + P[:, b])

    P = proj["F1"]; out["F1"] = _angle(P[:, R_WR] - P[:, R_SH], P[:, R_HIP] - P[:, R_SH])
    P = proj["F2"]; out["F2"] = _angle(P[:, L_SH] - P[:, R_SH], P[:, R_HIP] - P[:, R_SH])
    P = proj["F3"]; out["F3"] = _angle(P[:, NOSE] - mid(P, L_SH, R_SH), np.broadcast_to(V("F3"), P[:, NOSE].shape))
    P = proj["F4"]; out["F4"] = _angle(P[:, R_SH] - P[:, R_EL], P[:, R_WR] - P[:, R_EL])
    P = proj["F5"]; out["F5"] = _angle(P[:, R_WR] - P[:, R_SH], P[:, L_SH] - P[:, R_SH])
    P = proj["F6"]; out["F6"] = _angle(mid(P, L_SH, R_SH) - mid(P, L_HIP, R_HIP), mid(P, L_KN, R_KN) - mid(P, L_HIP, R_HIP))
    P = proj["F7"]; out["F7"] = _angle(P[:, R_SH] - P[:, L_SH], P[:, R_HIP] - P[:, L_HIP])
    P = proj["F8"]; out["F8"] = _angle(P[:, R_SH] - P[:, R_EL], P[:, R_WR] - P[:, R_EL])
    for k, (h, kn, t) in {"F9": (L_HIP, L_KN, L_TOE), "F10": (R_HIP, R_KN, R_TOE)}.items():
        P = proj[k]; out[k] = _angle(P[:, h] - P[:, kn], P[:, t] - P[:, kn])
    for k, (kn, t) in {"F11": (L_KN, L_TOE), "F12": (R_KN, R_TOE)}.items():
        P = proj[k]; out[k] = _angle(P[:, kn] - P[:, t], np.broadcast_to(V(k), P[:, t].shape))
    for k, (h, he, t) in {"F13": (L_HIP, L_HEEL, L_TOE), "F14": (R_HIP, R_HEEL, R_TOE)}.items():
        P = proj[k]; out[k] = _angle(P[:, h] - P[:, he], P[:, t] - P[:, he])
    P = proj["F15"]
    out["F15"] = 0.5 * (_angle(P[:, L_HIP] - P[:, L_KN], P[:, L_AN] - P[:, L_KN])
                        + _angle(P[:, R_HIP] - P[:, R_KN], P[:, R_AN] - P[:, R_KN]))
    return out


def dtw_path(a, b):
    """Plain DTW, |a_i - b_j| cost, steps (1,0),(0,1),(1,1). Returns list of (i, j)."""
    n, m = len(a), len(b)
    cost = np.abs(a[:, None] - b[None, :])
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        Di, Dp, ci = D[i], D[i - 1], cost[i - 1]
        for j in range(1, m + 1):
            Di[j] = ci[j - 1] + min(Dp[j], Di[j - 1], Dp[j - 1])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    return path[::-1]


def warp_to_ref(ref_main, rep_main, rep_feats: dict) -> dict:
    """Map every feature of `rep` onto the reference timeline (mean of rep frames matched to each ref frame)."""
    path = np.asarray(dtw_path(ref_main, rep_main))
    n = len(ref_main)
    cnt = np.bincount(path[:, 0], minlength=n)
    return {k: np.bincount(path[:, 0], weights=v[path[:, 1]], minlength=n) / cnt for k, v in rep_feats.items()}


def build_model(warped_ok: list, cfg: Config):
    """warped_ok: correct reps already mapped onto the reference timeline (reference included)."""
    model = {}
    for k in warped_ok[0]:
        X = np.stack([w[k] for w in warped_ok])          # (n_ok, n_frames)
        n = X.shape[0]
        mu = X.mean(0)
        s = X.std(0, ddof=1)
        t = stats.t.ppf(0.975, n - 1)
        tau = t * s / np.sqrt(n) if cfg.tau == "mean" else t * s * np.sqrt(1 + 1 / n)
        if cfg.floor == "q1_tau":
            tau = np.maximum(tau, np.quantile(tau, 0.25))
        model[k] = (mu, np.maximum(tau, 1e-9))
    return model


def rep_error(warped: dict, model, keys) -> dict:
    out = {}
    for k in keys:
        mu, tau = model[k]
        d = np.abs(warped[k] - mu)
        e = np.where(d <= tau, 0.0, np.minimum(1.0, (d - tau) / tau))   # Eq. 1
        out[k] = float(e.mean())                                         # Eq. 2
    return out


def sequence_psi(ok_feats: list, nok: list, main: str, cfg: Config, rng, ref_idx=None) -> float:
    """nok: list of (feats, [relevant feature keys]). Returns mean Psi over (nok rep, relevant feature) pairs."""
    if ref_idx is None:
        ref_idx = int(rng.integers(len(ok_feats)))
    ref = ok_feats[ref_idx]
    w_ok = [warp_to_ref(ref[main], r[main], r) for r in ok_feats]
    model = build_model(w_ok, cfg)
    keys = sorted({k for _, ks in nok for k in ks})
    base = {k: np.mean([rep_error(w, model, [k])[k] for w in w_ok]) for k in keys}
    psis = []
    for f, ks in nok:
        e = rep_error(warp_to_ref(ref[main], f[main], {k: f[k] for k in ks}), model, ks)
        psis += [e[k] - base[k] for k in ks]                             # Eq. 3
    return float(np.mean(psis))


# ---------------------------------------------------------------- self-test
def _selftest():
    rng = np.random.default_rng(0)
    # 1. angles
    assert abs(_angle(np.array([1.0, 0]), np.array([0, 1.0])) - 90) < 1e-9
    assert abs(_angle(np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])) - 180) < 1e-9
    # 2. DTW recovers a pure time stretch
    a = np.sin(np.linspace(0, np.pi, 50)); b = np.sin(np.linspace(0, np.pi, 100))
    p = np.asarray(dtw_path(a, b))
    assert p[0].tolist() == [0, 0] and p[-1].tolist() == [49, 99]
    assert np.abs(a[p[:, 0]] - b[p[:, 1]]).max() < 0.05
    # 3. Eq.1 known answers via a hand-built model
    mu, tau = np.zeros(4), np.ones(4)
    model = {"F": (mu, tau)}
    rep = {"F": np.array([0.5, 1.5, 3.0, -1.0])}   # errors 0, .5, 1 (capped), 0
    assert abs(rep_error(rep, model, ["F"])["F"] - 0.375) < 1e-12
    m2 = {"F": (np.zeros(3), np.full(3, 2.0))}                         # tau = 2: errors 0, (3-2)/2, min(1, 4/2)
    assert abs(rep_error({"F": np.array([1.0, 3.0, 6.0])}, m2, ["F"])["F"] - 0.5) < 1e-12
    # warp: a rep identical to ref maps onto itself
    w = warp_to_ref(a, a, {"F": a * 2})
    assert np.allclose(w["F"], a * 2)
    # 4. 3D normalisation puts the hip line on +x and keeps lengths
    J = rng.normal(size=(5, 33, 3)); Jn = normalise_3d(J)
    h = Jn[:, R_HIP] - Jn[:, L_HIP]
    assert np.allclose(h[:, 2], 0) and np.all(h[:, 0] > 0)
    assert np.allclose(np.linalg.norm(Jn[:, 3] - Jn[:, 5], axis=-1), np.linalg.norm(J[:, 3] - J[:, 5], axis=-1))
    # 5. Psi: identical reps -> Psi of a clean copy 0; a strongly perturbed rep -> Psi near 1
    cfg = Config()
    base = np.sin(np.linspace(0, np.pi, 60)) * 90
    ok = [{"M": base + rng.normal(0, 1, 60), "F": 10 + rng.normal(0, 1, 60)} for _ in range(5)]
    bad = {"M": base.copy(), "F": 10 + 40 + rng.normal(0, 1, 60)}
    good = {"M": base.copy(), "F": 10 + rng.normal(0, 1, 60)}
    pb = sequence_psi(ok, [(bad, ["F"])], "M", cfg, np.random.default_rng(1))
    pg = sequence_psi(ok, [(good, ["F"])], "M", cfg, np.random.default_rng(1))
    assert pb > 0.8, pb
    assert abs(pg) < 0.3, pg
    # 6. tolerance: tau = t_{0.975,n-1} * s / sqrt(n) per frame, floored at Q1 of tau over frames
    X = np.array([[0.0, 0, 0, 0], [1, 2, 4, 8], [2, 4, 8, 16]])        # 3 reps x 4 frames
    m = build_model([{"F": x} for x in X], Config(floor="none"))["F"]
    s_ = np.std(X, axis=0, ddof=1); want = stats.t.ppf(0.975, 2) * s_ / np.sqrt(3)
    assert np.allclose(m[0], X.mean(0)) and np.allclose(m[1], want), (m, want)
    mf = build_model([{"F": x} for x in X], Config())["F"][1]
    assert np.allclose(mf, np.maximum(want, np.quantile(want, 0.25))) and mf[0] > want[0]
    # 7. warp: rep = reference stretched x2; its time index maps to ~2i (+0.5 mean of the matched pair)
    a2 = np.sin(np.linspace(0, np.pi / 2, 40)); b2 = np.sin(np.linspace(0, np.pi / 2, 80))
    w2 = warp_to_ref(a2, b2, {"t": np.arange(80.0)})["t"]
    assert np.median(np.abs(w2 - (2 * np.arange(40) + 0.5))) <= 1.0, w2[:6]
    # 8. features on a hand-built pose (MediaPipe world axes: x right, y down, z away from camera).
    #    Hips on the x-axis and centred, so normalise_3d is the identity. Expected angles worked out by hand.
    J = np.zeros((1, 33, 3))
    pose = {L_HIP: (-.2, 0, 0), R_HIP: (.2, 0, 0), L_SH: (-.2, -.5, 0), R_SH: (.2, -.5, 0), NOSE: (0, -.7, 0),
            R_EL: (.5, -.5, -.3), R_WR: (.5, -.8, -.6), L_EL: (-.2, -.2, 0), L_WR: (-.2, .1, 0),
            L_KN: (-.2, .4, 0), L_AN: (-.2, .8, 0), L_HEEL: (-.2, .85, .05), L_TOE: (-.2, .85, -.15),
            R_KN: (.2, .4, -.2), R_AN: (.2, .8, 0), R_HEEL: (.2, .85, .05), R_TOE: (.3, .85, -.15)}
    for i, v in pose.items():
        J[0, i] = v
    exp3 = {"F1": 90 + np.degrees(np.arctan(.3 / .6)),   # yz: wrist (z -.6, y -.3) vs hip straight down
            "F2": 90.0, "F3": 0.0,                       # shoulder line vs shoulder->hip; head straight up
            "F4": 90.0,                                  # xy: elbow->shoulder (-.3, 0), elbow->wrist (0, -.3)
            "F5": 135.0,                                 # xy: shoulder->wrist (.3, -.3) vs shoulder->L shoulder
            "F6": 180.0, "F7": 0.0,
            "F8": 135.0,                                 # yz: elbow->shoulder (z .3, 0), elbow->wrist (z -.3, y -.3)
            "F9": 180.0, "F10": 180 - np.degrees(np.arctan(.1 / .45)),
            "F11": np.degrees(np.arctan(.15 / .45)), "F12": np.degrees(np.arctan(.05 / .45)),
            "F15": 0.5 * (180.0 + np.degrees(np.arccos(-.6)))}   # right knee: (z .2,y -.4).(z .2,y .4)/.2 = -.6
    F3 = features(J, "3d", 1, Config())
    for k, v in exp3.items():
        assert abs(F3[k][0] - v) < 1e-6, ("3d", k, F3[k][0], v)
    Ff = features(J, "3d", 1, Config(plane3d=False))
    assert abs(Ff["F4"][0] - 120.0) < 1e-6, Ff["F4"]    # full 3D elbow angle: cos = -.09 / .18
    F2d = features(J, "2d", 1, Config(aspect=(1.0, 1.0)))
    for k in ("F2", "F4", "F5", "F9", "F10"):            # these are frontal anyway: same in the image plane
        assert abs(F2d[k][0] - exp3[k]) < 1e-6, ("2d", k, F2d[k][0])
    assert abs(F2d["F8"][0] - 90.0) < 1e-6               # 2D has one plane: F8 equals F4
    F2a = features(J, "2d", 1, Config(aspect=(2.0, 1.0)))
    assert abs(F2a["F5"][0] - (180 - np.degrees(np.arctan(.3 / .6)))) < 1e-6   # x doubled: (.6, -.3) vs (-.8, 0)
    print(f"selftest ok  psi(bad)={pb:.3f} psi(clean)={pg:.3f}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
