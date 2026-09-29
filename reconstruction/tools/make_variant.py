"""Make a camera-subset and/or de-synchronized copy of an EasyMocap dataset (via symlinks).

Used for two experiments:
  * camera count / layout:  --views 01 05 10 14 19
  * sync tolerance:         --shift 05:2   (camera 05 shows the frame 2 steps later
                                            than the others, i.e. it is 2 frames out of sync)

Symlinks images/<cam>/* and annots/<cam>/* (if present) and writes a subset
intri.yml / extri.yml. Frames are renumbered 000000.. in the variant; the script
prints the --offset to pass to compare_keypoints3d.py.

Usage:
  python make_variant.py SRC DST --views 01 05 10 14 19 [--shift 05:2 ...]
"""
import argparse
import os

from emc_yml import read_cameras, write_cameras


def list_frames(folder):
    if not os.path.isdir(folder):
        return None
    return sorted(f for f in os.listdir(folder) if not f.startswith('.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('src', help='EasyMocap dataset with intri.yml, extri.yml, images/ (annots/ optional)')
    parser.add_argument('dst', help='output folder (created)')
    parser.add_argument('--views', nargs='+', default=None, help='camera names to keep (default: all)')
    parser.add_argument('--shift', nargs='+', default=[], metavar='CAM:K',
                        help='offset camera CAM by K frames (K may be negative)')
    args = parser.parse_args()

    cameras = read_cameras(args.src)
    views = args.views or list(cameras.keys())
    missing = [v for v in views if v not in cameras]
    if missing:
        parser.error('cameras {} not in intri.yml (have: {})'.format(missing, list(cameras.keys())))

    shifts = {v: 0 for v in views}
    for item in args.shift:
        cam, k = item.split(':')
        if cam not in shifts:
            parser.error('--shift camera {} is not in --views'.format(cam))
        shifts[cam] = int(k)
    base = max(0, -min(shifts.values()))

    write_cameras(args.dst, {v: cameras[v] for v in views})

    num_out = None
    subdirs = ['images', 'annots']
    listing = {}
    for sub in subdirs:
        for v in views:
            frames = list_frames(os.path.join(args.src, sub, v))
            if frames is None:
                if sub == 'images':
                    parser.error('missing {}'.format(os.path.join(args.src, sub, v)))
                continue
            listing[(sub, v)] = frames
            avail = len(frames) - base - shifts[v]
            num_out = avail if num_out is None else min(num_out, avail)

    if not num_out or num_out <= 0:
        parser.error('no overlapping frames left after shifting')

    for (sub, v), frames in listing.items():
        out_dir = os.path.join(args.dst, sub, v)
        os.makedirs(out_dir, exist_ok=True)
        for i in range(num_out):
            name = frames[i + base + shifts[v]]
            ext = os.path.splitext(name)[1]
            dst = os.path.join(out_dir, '{:06d}{}'.format(i, ext))
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(os.path.join(args.src, sub, v, name)), dst)

    has_annots = any(sub == 'annots' for sub, _ in listing)
    print('variant: {} views x {} frames -> {}'.format(len(views), num_out, args.dst))
    print('shifts: {}'.format({k: s for k, s in shifts.items() if s}))
    if not has_annots:
        print('note: no annots/ found; run 2D detection on the variant (or on SRC first)')
    print('compare with: compare_keypoints3d.py REF TEST --offset {}'.format(base))


if __name__ == '__main__':
    main()
