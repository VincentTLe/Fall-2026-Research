"""Compare two EasyMocap keypoints3d/ folders (BODY25, meters) frame by frame.

Typical use: REF = true joints from synth_rig.py (gt/keypoints3d), or on real data the
                   all-camera result (a reference estimate, not independent ground truth);
             TEST = result with a camera subset / shifted camera.

Reports MPJPE and root-relative MPJPE (mid-hip aligned) in millimeters,
plus per-joint error, the share of reference joints TEST reconstructed, and how many
REF frames TEST is missing. MPJPE is the main number for a calibrated rig; root-relative
MPJPE also absorbs the error of the mid-hip joint itself, so it can be the larger one.

Usage:
  python compare_keypoints3d.py REF/keypoints3d TEST/keypoints3d [--offset N] [--pid 0] [--csv out.csv]
"""
import argparse
import csv
import json
import os

import numpy as np

BODY25 = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist',
          'MidHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye',
          'REar', 'LEar', 'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel']
ROOT = BODY25.index('MidHip')


def load(folder, pid):
    out = {}
    for name in sorted(os.listdir(folder)):
        if not name.endswith('.json'):
            continue
        with open(os.path.join(folder, name)) as f:
            people = json.load(f)
        if isinstance(people, dict):
            people = [people]
        match = [p for p in people if p.get('id', 0) == pid] or people[:1]
        if match:
            out[int(os.path.splitext(name)[0])] = np.array(match[0]['keypoints3d'], dtype=np.float64)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('ref')
    parser.add_argument('test')
    parser.add_argument('--offset', type=int, default=0,
                        help='TEST frame i corresponds to REF frame i + offset (printed by make_variant.py)')
    parser.add_argument('--pid', type=int, default=0, help='person id to compare')
    parser.add_argument('--csv', default=None, help='write per-joint errors to this CSV')
    args = parser.parse_args()

    ref, test = load(args.ref, args.pid), load(args.test, args.pid)
    errs, rel_errs = [], []
    per_joint = [[] for _ in BODY25]
    matched = 0
    ref_joints = both_joints = 0
    for t_idx, kt in test.items():
        kr = ref.get(t_idx + args.offset)
        if kr is None:
            continue
        matched += 1
        n = min(len(kr), len(kt), len(BODY25))
        valid = (kr[:n, 3] > 0) & (kt[:n, 3] > 0)
        ref_joints += int((kr[:n, 3] > 0).sum())
        both_joints += int(valid.sum())
        diff = np.linalg.norm(kr[:n, :3] - kt[:n, :3], axis=1) * 1000.
        for j in np.where(valid)[0]:
            per_joint[j].append(diff[j])
        errs.extend(diff[valid])
        if valid[ROOT]:
            rel = np.linalg.norm((kr[:n, :3] - kr[ROOT, :3]) - (kt[:n, :3] - kt[ROOT, :3]), axis=1) * 1000.
            rel_errs.extend(rel[valid])

    # REF frames inside TEST's frame span that TEST has no result for (e.g. fitting failed)
    span = range(args.offset, args.offset + max(test) + 1) if test else range(0)
    missing = sum(1 for f in ref if f in span and (f - args.offset) not in test)
    if not errs:
        raise SystemExit('no overlapping valid joints between {} and {}'.format(args.ref, args.test))

    print('frames: ref={} test={} matched={} missing_in_test={}'.format(len(ref), len(test), matched, missing))
    print('joints reconstructed: {:.1f}% of reference joints in matched frames'.format(
        100. * both_joints / max(ref_joints, 1)))
    print('MPJPE           mean {:7.1f} mm   median {:7.1f} mm'.format(np.mean(errs), np.median(errs)))
    if rel_errs:
        print('root-rel MPJPE  mean {:7.1f} mm   median {:7.1f} mm'.format(np.mean(rel_errs), np.median(rel_errs)))
    print('\n{:<10} {:>9} {:>7}'.format('joint', 'mean(mm)', 'count'))
    rows = []
    for name, e in zip(BODY25, per_joint):
        mean = float(np.mean(e)) if e else float('nan')
        rows.append((name, mean, len(e)))
        print('{:<10} {:>9.1f} {:>7}'.format(name, mean, len(e)))

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['joint', 'mean_mm', 'count'])
            w.writerows(rows)


if __name__ == '__main__':
    main()
