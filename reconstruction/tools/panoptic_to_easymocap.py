"""CMU Panoptic (HD cameras, joints19 skeletons) -> EasyMocap layout, plus checks and scoring.

Panoptic stores calibration and skeletons in centimetres; EasyMocap uses metres. The world axes are
kept as they are (Panoptic: y points down), so EasyMocap's triangulation comes out in the same frame
as the dome skeletons and only the unit changes.

  convert  SEQ OUT --start 650 --end 1050 [--views 00 03 ...]
           SEQ holds calibration_<seq>.json, hdImgs/00_XX/%06d.jpg (frame `start` saved as 000000)
           and hdPose3d_stage1_coco19/. Writes OUT/intri.yml, OUT/extri.yml, OUT/images/XX/
           (symlinks) and OUT/dome/keypoints3d/%06d.json: the dome skeleton in BODY25 slots, metres.
  check    OUT --views 03 21 [--frame 100] [--image check.jpg]
           Projects the dome joints through the converted calibration and compares them with the 2D
           detections in OUT/annots (written by run_mv1p.py --write-annots).
  eval     OUT/dome/keypoints3d RUN/triangulated [--offset N] [--json out.json]
           MPJPE and PA-MPJPE (mm) on the shared limb joints, missing frames/joints, per-joint error.
  layout   OUT   prints each camera's azimuth and height around the subject.

The dome skeletons are reconstructed by the dense dome itself, not by markers: call them the
"dome reference", not ground truth from mocap.
"""
import argparse
import glob
import json
import os

import numpy as np

from emc_yml import read_cameras, write_cameras

# Panoptic joints19 index -> BODY25 index (names as in the panoptic-toolbox README).
J19_TO_BODY25 = {0: 1,   # Neck -> Neck (BODY25 neck is the shoulder midpoint in EasyMocap's COCO17 path)
                 1: 0,   # Nose
                 2: 8,   # BodyCenter (center of hips) -> MidHip (hip midpoint)
                 3: 5, 4: 6, 5: 7,      # lShoulder, lElbow, lWrist
                 6: 12, 7: 13, 8: 14,   # lHip, lKnee, lAnkle
                 9: 2, 10: 3, 11: 4,    # rShoulder, rElbow, rWrist
                 12: 9, 13: 10, 14: 11,  # rHip, rKnee, rAnkle
                 15: 16, 16: 18, 17: 15, 18: 17}  # lEye, lEar, rEye, rEar
BODY25 = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist',
          'MidHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye',
          'REar', 'LEar', 'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel']
# Joints detected directly by both skeletons (no synthesized neck / hip centre, no face points).
LIMBS = [2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14]
MAPPED = sorted(J19_TO_BODY25.values())


def read_dome(path):
    """First body of a body3DScene json as (19, 4) in cm, or None."""
    with open(path) as f:
        bodies = json.load(f)['bodies']
    if not bodies:
        return None
    return np.array(bodies[0]['joints19'], dtype=np.float64).reshape(19, 4)


def dome_to_body25(j19, min_conf=0.1):
    """(19, 4) cm -> (25, 4) m in BODY25 slots; joints below min_conf get confidence 0."""
    out = np.zeros((25, 4))
    for src, dst in J19_TO_BODY25.items():
        out[dst, :3] = j19[src, :3] / 100.
        out[dst, 3] = j19[src, 3] if j19[src, 3] >= min_conf else 0.
    return out


def load_k3d(folder):
    out = {}
    for path in sorted(glob.glob(os.path.join(folder, '*.json'))):
        with open(path) as f:
            people = json.load(f)
        if isinstance(people, dict):
            people = [people]
        if people:
            out[int(os.path.basename(path)[:-5])] = np.array(people[0]['keypoints3d'], dtype=np.float64)
    return out


def convert(args):
    seq = os.path.basename(os.path.normpath(args.seq))
    with open(os.path.join(args.seq, 'calibration_{}.json'.format(seq))) as f:
        calib = json.load(f)
    cameras = {}
    for cam in calib['cameras']:
        if cam['type'] != 'hd':
            continue
        name = '{:02d}'.format(cam['node'])
        if args.views and name not in args.views:
            continue
        cameras[name] = {'K': np.array(cam['K']), 'dist': np.array(cam['distCoef']).reshape(-1),
                         'R': np.array(cam['R']), 'T': np.array(cam['t']).reshape(3) / 100.}  # cm -> m
    write_cameras(args.out, cameras)
    nframes = args.end - args.start
    for name in cameras:
        src_dir = os.path.join(os.path.abspath(args.seq), 'hdImgs', '00_' + name)
        dst_dir = os.path.join(args.out, 'images', name)
        os.makedirs(dst_dir, exist_ok=True)
        for i in range(nframes):
            src = os.path.join(src_dir, '{:06d}.jpg'.format(i))
            if not os.path.isfile(src):
                raise SystemExit('missing ' + src)
            dst = os.path.join(dst_dir, '{:06d}.jpg'.format(i))
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(src, dst)
    dome_dir = os.path.join(args.out, 'dome', 'keypoints3d')
    os.makedirs(dome_dir, exist_ok=True)
    missing = 0
    for i in range(nframes):
        path = os.path.join(args.seq, 'hdPose3d_stage1_coco19', 'body3DScene_{:08d}.json'.format(args.start + i))
        j19 = read_dome(path) if os.path.isfile(path) else None
        if j19 is None:
            missing += 1
            continue
        with open(os.path.join(dome_dir, '{:06d}.json'.format(i)), 'w') as f:
            json.dump([{'id': 0, 'keypoints3d': dome_to_body25(j19).tolist()}], f)
    print('wrote {} cameras x {} frames (source frames {}..{}) to {}; dome frames missing: {}'.format(
        len(cameras), nframes, args.start, args.end - 1, args.out, missing))


def check(args):
    import cv2
    cameras = read_cameras(args.out)
    dome = load_k3d(os.path.join(args.out, 'dome', 'keypoints3d'))
    names = list(cameras.keys())
    for name in args.views:
        cam = cameras[name]
        errs = {j: [] for j in MAPPED}
        for frame, k3d in dome.items():
            annot = os.path.join(args.out, 'annots', name, '{:06d}.json'.format(frame))
            if not os.path.isfile(annot):
                continue
            with open(annot) as f:
                k2d = np.array(json.load(f)['annots'][0]['keypoints'])
            proj, _ = cv2.projectPoints(np.ascontiguousarray(k3d[:, :3]), cv2.Rodrigues(cam['R'])[0],
                                        cam['T'], cam['K'], cam['dist'])
            err = np.linalg.norm(proj[:, 0] - k2d[:, :2], axis=1)
            for j in MAPPED:
                if k3d[j, 3] > 0 and k2d[j, 2] > 0.3:
                    errs[j].append(err[j])
        allv = np.concatenate([errs[j] for j in LIMBS])
        print('camera {}: dome joints projected vs detections, limb joints: median {:.1f} px, p95 {:.1f} px, n={}'
              .format(name, np.median(allv), np.percentile(allv, 95), len(allv)))
        for j in MAPPED:
            if errs[j]:
                print('   {:<10} median {:6.1f} px  n={}'.format(BODY25[j], np.median(errs[j]), len(errs[j])))
        if args.image:
            img = cv2.imread(os.path.join(args.out, 'images', name, '{:06d}.jpg'.format(args.frame)))
            k3d = dome[args.frame]
            proj, _ = cv2.projectPoints(np.ascontiguousarray(k3d[:, :3]), cv2.Rodrigues(cam['R'])[0],
                                        cam['T'], cam['K'], cam['dist'])
            with open(os.path.join(args.out, 'annots', name, '{:06d}.json'.format(args.frame))) as f:
                k2d = np.array(json.load(f)['annots'][0]['keypoints'])
            for j in MAPPED:
                x, y = proj[j, 0]
                cv2.circle(img, (int(x), int(y)), 7, (0, 255, 0), 2)          # green: dome reference
                if k2d[j, 2] > 0.3:
                    cv2.drawMarker(img, (int(k2d[j, 0]), int(k2d[j, 1])), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
            path = args.image.replace('.jpg', '_{}.jpg'.format(name))
            cv2.imwrite(path, img)
            print('   wrote', path, '(green circle = dome joint, red cross = HRNet detection)')
    print('cameras in file:', names)


def procrustes(X, Y):
    """Align Y to X with rotation, translation and scale; return aligned Y."""
    mx, my = X.mean(0), Y.mean(0)
    X0, Y0 = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Y0.T @ X0)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (Y0 ** 2).sum()
    return s * Y0 @ R + mx


def evaluate(args):
    ref, test = load_k3d(args.ref), load_k3d(args.test)
    # Many Panoptic HD cameras frame the subject without feet; --no-ankles scores the rest.
    subset = [j for j in LIMBS if j not in (11, 14)] if args.no_ankles else LIMBS
    errs, pa_errs, per_joint = [], [], {j: [] for j in MAPPED}
    frames_ref = frames_missing = joints_ref = joints_missing = 0
    for r_idx, kr in ref.items():
        t_idx = r_idx - args.offset
        if args.frames and not (0 <= t_idx < args.frames):
            continue
        valid_ref = kr[subset, 3] > 0
        frames_ref += 1
        joints_ref += int(valid_ref.sum())
        kt = test.get(t_idx)
        if kt is None:
            frames_missing += 1
            joints_missing += int(valid_ref.sum())
            continue
        for j in MAPPED:
            if kr[j, 3] > 0 and kt[j, 3] > 0:
                per_joint[j].append(np.linalg.norm(kr[j, :3] - kt[j, :3]) * 1000.)
        both = valid_ref & (kt[subset, 3] > 0)
        joints_missing += int((valid_ref & ~both).sum())
        idx = np.array(subset)[both]
        if len(idx) == 0:
            continue
        d = np.linalg.norm(kr[idx, :3] - kt[idx, :3], axis=1) * 1000.
        errs.extend(d)
        if len(idx) >= 4:
            aligned = procrustes(kr[idx, :3], kt[idx, :3])
            pa_errs.extend(np.linalg.norm(kr[idx, :3] - aligned, axis=1) * 1000.)
    res = dict(ref=args.ref, test=args.test, offset=args.offset, joints=[BODY25[j] for j in subset],
               mpjpe_mm=float(np.mean(errs)), mpjpe_median_mm=float(np.median(errs)),
               pa_mpjpe_mm=float(np.mean(pa_errs)), frames_ref=frames_ref,
               frames_missing_pct=100. * frames_missing / max(frames_ref, 1),
               joints_missing_pct=100. * joints_missing / max(joints_ref, 1),
               per_joint_mm={BODY25[j]: (float(np.mean(e)) if e else None) for j, e in per_joint.items()},
               per_joint_count={BODY25[j]: len(e) for j, e in per_joint.items()})
    print('MPJPE {:.1f} mm (median {:.1f})  PA-MPJPE {:.1f} mm  frames missing {:.1f}%  joints missing {:.1f}%'
          .format(res['mpjpe_mm'], res['mpjpe_median_mm'], res['pa_mpjpe_mm'], res['frames_missing_pct'],
                  res['joints_missing_pct']))
    for name, e in res['per_joint_mm'].items():
        if e is not None:
            print('   {:<10} {:7.1f} mm  n={}'.format(name, e, res['per_joint_count'][name]))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(res, f, indent=2)


def layout(args):
    cameras = read_cameras(args.out)
    dome = load_k3d(os.path.join(args.out, 'dome', 'keypoints3d'))
    root = np.mean([k[8, :3] for k in dome.values()], axis=0)
    print('subject centre (m):', root.round(3))
    for name, cam in cameras.items():
        centre = (-cam['R'].T @ np.asarray(cam['T']).reshape(3)) - root
        az = np.degrees(np.arctan2(centre[2], centre[0])) % 360   # around the vertical (y) axis
        print('{} azimuth {:6.1f} deg  height {:5.2f} m  distance {:4.2f} m'.format(
            name, az, -centre[1], np.linalg.norm(centre[[0, 2]])))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('convert')
    p.add_argument('seq')
    p.add_argument('out')
    p.add_argument('--start', type=int, required=True, help='source HD frame saved as 000000')
    p.add_argument('--end', type=int, required=True, help='source HD frame end, exclusive')
    p.add_argument('--views', nargs='+', default=None)
    p.set_defaults(func=convert)
    p = sub.add_parser('check')
    p.add_argument('out')
    p.add_argument('--views', nargs='+', required=True)
    p.add_argument('--frame', type=int, default=0)
    p.add_argument('--image', default=None)
    p.set_defaults(func=check)
    p = sub.add_parser('eval')
    p.add_argument('ref')
    p.add_argument('test')
    p.add_argument('--offset', type=int, default=0, help='TEST frame i is REF frame i + offset')
    p.add_argument('--frames', type=int, default=None, help='only TEST frames 0..N-1')
    p.add_argument('--no-ankles', action='store_true', help='score 10 limb joints, without ankles')
    p.add_argument('--json', default=None)
    p.set_defaults(func=evaluate)
    p = sub.add_parser('layout')
    p.add_argument('out')
    p.set_defaults(func=layout)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
