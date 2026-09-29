"""Generate a synthetic EasyMocap dataset for a proposed camera layout, with exact 3D ground truth.

Why: lets us (1) test the EasyMocap install and our file formats without any licensed
data or model downloads, and (2) compare camera layouts / 2D noise / sync offsets
against *true* 3D joints instead of an all-camera pseudo reference.

What it writes into OUT:
  intri.yml, extri.yml           cameras (pinhole, zero distortion, T in meters)
  images/<cam>/000000.jpg ...    black placeholder frames (EasyMocap needs the files to exist)
  annots/<cam>/000000.json ...   projected BODY25 keypoints + Gaussian pixel noise
  gt/keypoints3d/000000.json     true 3D joints, same format as EasyMocap output

Limitations (read before quoting numbers): the motion is a scripted squat + arm raise,
occlusion is modelled with four capsules (spine, both flanks, head) that hide joints
behind the body; arms and legs never hide each other, and 2D error is i.i.d. Gaussian. Real detector error is larger and
correlated. Use it to *compare* layouts, not to predict absolute accuracy.

Examples:
  # 5 cameras evenly around the subject, 3 m away, 1.6 m high
  python synth_rig.py /tmp/rig_even --cams 5 --radius 3 --height 1.6
  # 5 cameras only covering the front 180 degrees
  python synth_rig.py /tmp/rig_front --cams 5 --arc 180
  # explicit positions from the hardware team's plan (x,y,z in meters, z up)
  python synth_rig.py /tmp/rig_plan --positions "3,0,1.6; 0,3,2.2; -3,0,1.6; 0,-3,2.2; 2.1,2.1,0.5"

Then run EasyMocap's triangulation and score it against gt/:
  python apps/demo/mv1p.py /tmp/rig_even --out /tmp/rig_even/output --body body25
  python compare_keypoints3d.py /tmp/rig_even/gt/keypoints3d /tmp/rig_even/output/keypoints3d
"""
import argparse
import json
import os

import cv2
import numpy as np

from emc_yml import write_cameras

# BODY25 rest pose (meters), subject facing +x, +y to their left, z up, feet at z=0.
REST = np.array([
    [0.10, 0.00, 1.60], [0.00, 0.00, 1.45],                          # Nose, Neck
    [0.00, -0.18, 1.43], [0.00, -0.20, 1.15], [0.02, -0.21, 0.88],   # R shoulder/elbow/wrist
    [0.00, 0.18, 1.43], [0.00, 0.20, 1.15], [0.02, 0.21, 0.88],      # L shoulder/elbow/wrist
    [0.00, 0.00, 0.95],                                              # MidHip
    [0.00, -0.10, 0.95], [0.02, -0.10, 0.52], [0.00, -0.10, 0.08],   # R hip/knee/ankle
    [0.00, 0.10, 0.95], [0.02, 0.10, 0.52], [0.00, 0.10, 0.08],      # L hip/knee/ankle
    [0.09, -0.03, 1.65], [0.09, 0.03, 1.65],                         # R/L eye
    [0.02, -0.07, 1.62], [0.02, 0.07, 1.62],                         # R/L ear
    [0.18, 0.12, 0.02], [0.16, 0.16, 0.02], [-0.05, 0.10, 0.03],     # L big toe, small toe, heel
    [0.18, -0.12, 0.02], [0.16, -0.16, 0.02], [-0.05, -0.10, 0.03],  # R big toe, small toe, heel
])
UPPER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 18]
KNEES = [10, 13]
ARMS = [(2, [3, 4]), (5, [6, 7])]  # shoulder -> joints rotated with it


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def pose_at(t, period, depth, turn_speed):
    """Squat with forward arm raise, slowly turning in place. t in seconds."""
    phase = 0.5 * (1 - np.cos(2 * np.pi * t / period))  # 0 standing -> 1 bottom
    p = REST.copy()
    drop = depth * phase
    p[UPPER, 2] -= drop
    p[KNEES, 2] -= drop / 2
    p[KNEES, 0] += 0.8 * drop
    p[UPPER, 0] -= 0.3 * drop  # hips go back
    for shoulder, chain in ARMS:
        R = rot_y(-np.pi / 2 * phase)  # arms swing forward up to horizontal
        p[chain] = (p[chain] - p[shoulder]) @ R.T + p[shoulder]
    return p @ rot_z(turn_speed * t).T


def seg_dist(p, a, b):
    """Distance from points p (N,3) to segment a-b."""
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=-1)


# Occluding body parts as capsules: (joint a, joint b, radius m). Joint -1 = top of head.
CAPSULES = [(1, 8, 0.08),    # spine: neck - mid-hip
            (2, 9, 0.07),    # right flank: shoulder - hip
            (5, 12, 0.07),   # left flank
            (1, -1, 0.06)]   # head: neck - head top


def body_occluded(C, X, margin=0.06, samples=48):
    """True for joints whose line of sight from camera center C passes through a body capsule.

    The sight line stops `margin` meters before the joint, and a capsule never hides its
    own end joints, so joints on the body surface stay visible from the front.
    """
    head_top = (X[17] + X[18]) / 2 + np.array([0, 0, 0.08])
    ray = X - C
    length = np.linalg.norm(ray, axis=1)
    ts = np.linspace(0.02, 1., samples)[None, :] * ((length - margin) / length)[:, None]  # (J, S)
    pts = (C[None, None] + ts[..., None] * ray[:, None]).reshape(-1, 3)
    occ = np.zeros(len(X), bool)
    for ja, jb, r in CAPSULES:
        a, b = X[ja], (head_top if jb < 0 else X[jb])
        hit = (seg_dist(pts, a, b).reshape(len(X), samples) < r).any(axis=1)
        hit[[ja, jb] if jb >= 0 else [ja]] = False
        occ |= hit
    return occ


def look_at(center, target, up=np.array([0, 0, 1.])):
    """World->camera rotation for an OpenCV camera (x right, y down, z forward)."""
    z = target - center
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:  # looking straight down/up
        x = np.array([1., 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out')
    parser.add_argument('--cams', type=int, default=5)
    parser.add_argument('--radius', type=float, default=3.0, help='camera distance from center (m)')
    parser.add_argument('--height', type=float, default=1.6, help='camera height (m)')
    parser.add_argument('--arc', type=float, default=360., help='angular span covered by cameras (deg)')
    parser.add_argument('--positions', default=None, metavar='"X,Y,Z; X,Y,Z; ..."',
                        help='explicit camera positions in meters, separated by ";" '
                             '(overrides --cams/--radius/--height/--arc)')
    parser.add_argument('--target-height', type=float, default=0.9, help='height cameras aim at (m)')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--img-height', type=int, default=1080)
    parser.add_argument('--hfov', type=float, default=85., help='horizontal field of view (deg); measure yours')
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--fps', type=float, default=60.)
    parser.add_argument('--period', type=float, default=2.0, help='seconds per squat (smaller = faster motion)')
    parser.add_argument('--depth', type=float, default=0.35, help='squat depth (m)')
    parser.add_argument('--turn', type=float, default=20., help='turning speed (deg/s)')
    parser.add_argument('--noise', type=float, default=3.0, help='2D keypoint noise std (px)')
    parser.add_argument('--dropout', type=float, default=0.0, help='probability a 2D joint is missing')
    parser.add_argument('--no-occlusion', action='store_true', help='disable the body occlusion model')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    if args.positions:
        centers = [np.array([float(v) for v in p.split(',')]) for p in args.positions.split(';') if p.strip()]
    else:
        n = args.cams
        step = np.deg2rad(args.arc) / (n if args.arc >= 360 else max(n - 1, 1))
        start = 0. if args.arc >= 360 else -np.deg2rad(args.arc) / 2
        centers = [np.array([args.radius * np.cos(start + i * step), args.radius * np.sin(start + i * step),
                             args.height]) for i in range(n)]

    W, H = args.width, args.img_height
    f = W / 2 / np.tan(np.deg2rad(args.hfov) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.]])
    target = np.array([0, 0, args.target_height])
    cameras = {}
    for i, C in enumerate(centers):
        R = look_at(C, target)
        cameras['{:02d}'.format(i + 1)] = {'K': K, 'dist': np.zeros(5), 'R': R, 'T': -R @ C}
    write_cameras(args.out, cameras)

    blank = cv2.imencode('.jpg', np.zeros((H, W, 3), np.uint8))[1].tobytes()
    gt_dir = os.path.join(args.out, 'gt', 'keypoints3d')
    os.makedirs(gt_dir, exist_ok=True)
    visible = {name: 0 for name in cameras}
    centers_by_name = dict(zip(cameras, centers))
    for nf in range(args.frames):
        X = pose_at(nf / args.fps, args.period, args.depth, np.deg2rad(args.turn))
        fname = '{:06d}'.format(nf)
        with open(os.path.join(gt_dir, fname + '.json'), 'w') as fp:
            json.dump([{'id': 0, 'keypoints3d': np.c_[X, np.ones(len(X))].round(6).tolist()}], fp)
        for name, cam in cameras.items():
            Xc = X @ cam['R'].T + cam['T']
            uv = Xc @ K.T
            uv = uv[:, :2] / uv[:, 2:]
            uv += rng.normal(0, args.noise, uv.shape)
            conf = np.full(len(X), 0.9)
            inside = (Xc[:, 2] > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            if not args.no_occlusion:
                inside &= ~body_occluded(centers_by_name[name], X)
            conf[~inside] = 0.
            conf[rng.random(len(X)) < args.dropout] = 0.
            visible[name] += int(inside.sum())
            valid = uv[conf > 0]
            if len(valid):
                l, t = valid.min(0) - 20
                r, b = valid.max(0) + 20
            else:
                l = t = r = b = 0.
            annot = {'filename': 'images/{}/{}.jpg'.format(name, fname), 'height': H, 'width': W,
                     'annots': [{'personID': 0, 'bbox': [float(l), float(t), float(r), float(b), 1.0],
                                 'keypoints': np.c_[uv, conf].round(3).tolist(),
                                 'area': float((r - l) * (b - t))}]}
            for sub in ('images', 'annots'):
                os.makedirs(os.path.join(args.out, sub, name), exist_ok=True)
            with open(os.path.join(args.out, 'annots', name, fname + '.json'), 'w') as fp:
                json.dump(annot, fp)
            with open(os.path.join(args.out, 'images', name, fname + '.jpg'), 'wb') as fp:
                fp.write(blank)

    total = args.frames * len(REST)
    print('wrote {} cameras x {} frames to {}'.format(len(cameras), args.frames, args.out))
    for name, C in zip(cameras, centers):
        print('  cam {} at ({:5.2f}, {:5.2f}, {:4.2f}) m: {:5.1f}% of joints visible (in frame, not hidden by body)'.format(
            name, *C, 100. * visible[name] / total))


if __name__ == '__main__':
    main()
