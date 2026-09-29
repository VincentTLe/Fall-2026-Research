"""Convert a raw ZJU-MoCap LightStage sequence (Neural Body format) to EasyMocap layout.

Input  <seq>/annots.npy   dict with
         'cams': {'K': [N x 3x3], 'D': [N x 5x1], 'R': [N x 3x3], 'T': [N x 3x1] in millimeters}
         'ims':  [ {'ims': [relative image path for view 0..N-1]} per frame ]
Output <out>/intri.yml, <out>/extri.yml   (T in meters, cameras named 01..NN)
       <out>/images/NN/000000.jpg ...      (symlinks by default)

Usage:
  python zju_to_easymocap.py /data/zju_mocap/CoreView_313 /data/easymocap/313
  python zju_to_easymocap.py SEQ OUT --views 1 5 9 13 17 --start 0 --end 300 --copy
"""
import argparse
import os
import shutil

import numpy as np

from emc_yml import write_cameras


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('seq', help='raw ZJU-MoCap sequence folder (contains annots.npy)')
    parser.add_argument('out', help='output folder in EasyMocap layout')
    parser.add_argument('--views', type=int, nargs='+', default=None,
                        help='1-based camera indices to keep (default: all)')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--copy', action='store_true', help='copy images instead of symlinking')
    args = parser.parse_args()

    annots = np.load(os.path.join(args.seq, 'annots.npy'), allow_pickle=True).item()
    cams = annots['cams']
    num_views = len(cams['K'])
    views = args.views or list(range(1, num_views + 1))
    for v in views:
        if not 1 <= v <= num_views:
            parser.error('view {} out of range 1..{}'.format(v, num_views))

    cameras = {}
    for v in views:
        i = v - 1
        cameras['{:02d}'.format(v)] = {
            'K': np.array(cams['K'][i]),
            'dist': np.array(cams['D'][i]).reshape(-1),
            'R': np.array(cams['R'][i]),
            'T': np.array(cams['T'][i]).reshape(3) / 1000.,  # mm -> m
        }
    write_cameras(args.out, cameras)

    frames = annots['ims']
    end = len(frames) if args.end < 0 else min(args.end, len(frames))
    link = shutil.copyfile if args.copy else os.symlink
    count = 0
    for out_idx, f in enumerate(range(args.start, end, args.step)):
        paths = frames[f]['ims']
        for v in views:
            src = os.path.abspath(os.path.join(args.seq, paths[v - 1]))
            ext = os.path.splitext(src)[1]
            dst_dir = os.path.join(args.out, 'images', '{:02d}'.format(v))
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, '{:06d}{}'.format(out_idx, ext))
            if os.path.lexists(dst):
                os.remove(dst)
            link(src, dst)
        count += 1
    print('wrote {} cameras x {} frames to {}'.format(len(views), count, args.out))


if __name__ == '__main__':
    main()
