"""Reproduce a small real-video EasyMocap run on Windows/CPU.

Run with the research environment's python; see SETUP_WINDOWS.md.
The fit stage requires a separately licensed SMPL model and never downloads it.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--frames', type=int, default=100, help='Source frame end, exclusive')
    p.add_argument('--step', type=int, default=2)
    p.add_argument('--views', nargs='+', default=['1', '7', '13', '19'])
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--fit', action='store_true')
    p.add_argument('--smpl', type=Path)
    p.add_argument('--ffmpeg', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    easy = root / 'external/EasyMocap'
    data = root / 'data/zju-ls-feng'
    out = root / 'local-runs/zju-ls-feng-cpu'
    out.mkdir(parents=True, exist_ok=True)
    if args.frames < 1 or args.step < 1:
        p.error('--frames and --step must be positive')
    if args.fit and (args.smpl is None or not args.smpl.is_file()):
        p.error('--fit requires --smpl pointing to your licensed SMPL_NEUTRAL.pkl')
    os.environ['YOLOv5_AUTOINSTALL'] = 'false'
    os.environ['TORCH_HOME'] = str(root / '.cache/torch')
    sys.path.insert(0, str(easy))
    os.chdir(easy)
    import cv2
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    from easymocap.config import Config, load_object
    from easymocap.config.yacs import CfgNode
    from easymocap.mytools.file_utils import write_keypoints3d

    # Extract consecutive source frames so EasyMocap's list-index semantics match
    # the original frame IDs. The run samples these at --step intervals.
    source = []
    for view in args.views:
        video = data / 'videos' / (view + '.mp4')
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError('Cannot open ' + str(video))
        info = dict(view=view, file=str(video), fps=cap.get(cv2.CAP_PROP_FPS),
                    frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if args.frames > info['frames']:
            raise ValueError('Requested frames exceed ' + str(video))
        source.append(info)
        dest = data / 'images' / view
        dest.mkdir(parents=True, exist_ok=True)
        for frame in range(args.frames):
            ok, img = cap.read()
            if not ok:
                raise RuntimeError('Decode failed at %s:%s' % (video, frame))
            path = dest / ('%06d.jpg' % frame)
            if not path.exists() and not cv2.imwrite(str(path), img):
                raise RuntimeError('Cannot write ' + str(path))
        cap.release()
    if len({round(s['fps'], 5) for s in source}) != 1:
        raise ValueError('Camera frame rates differ')
    (out / 'source_videos.json').write_text(json.dumps(source, indent=2))
    cfg_data = Config.load(str(easy / 'config/datasets/mvimage.yml'))
    cfg_data.args.root = str(data)
    cfg_data.args.subs = args.views
    cfg_data.args.subs_vis = args.views
    cfg_data.args.ranges = [0, args.frames, args.step]
    cfg = Config.load(str(easy / 'config/mv1p/detect_triangulate_fitSMPL.yml'))
    cfg.args.output = str(out)
    cfg.args.at_step.detect.module = 'local_cpu.PinnedYOLO'
    cfg.args.at_step.detect.args = CfgNode(dict(repo=str(root / 'external/yolov5'),
        ckpt=str(root / 'downloads/yolov5m.pt'), name='person'))
    cfg.args.at_step.keypoints2d.args.ckpt = str(root / 'downloads/pose_hrnet_w48_384x288.pth')
    if not Path(cfg.args.at_step.keypoints2d.args.ckpt).is_file():
        raise FileNotFoundError('Download the official HRNet checkpoint first')
    # Show all cameras without the upstream center crop.
    cfg.args.at_step.visualize.args.mode = 'none'
    if args.fit:
        cfg.args.at_final.load_body_model.args.model_path = str(args.smpl.resolve())
        cfg.args.at_final.load_body_model.args.regressor_path = str(easy / 'data/smplx/J_regressor_body25.npy')
        render = cfg.args.at_final.render_ground
        render.args = CfgNode(dict(backend='pyrender', view_list=[0],
                                  render_mode='image', name='vis_render', scale=0.5))
        del cfg.args.at_final['make_video']
    else:
        cfg.args.at_final = CfgNode({})
    (out / 'cfg_data.yml').write_text(str(cfg_data), encoding='utf-8')
    (out / 'cfg_exp.yml').write_text(str(cfg), encoding='utf-8')
    dataset = load_object(cfg_data.module, cfg_data.args)
    model = load_object(cfg.module, cfg.args)
    records, errors, valid_counts, timings = [], [], [], []
    for index in range(len(dataset)):
        started = time.perf_counter()
        sample = dataset[index]
        record = model.at_step(sample, index)
        records.append(record)
        k3d, k2d = record['keypoints3d'], record['keypoints']
        if not np.isfinite(k3d).all() or not np.isfinite(k2d).all():
            raise ValueError('Non-finite detected or reconstructed keypoints')
        frame = sample['meta']['frame']
        write_keypoints3d(str(out / 'triangulated' / ('%06d.json' % frame)),
                          [dict(id=0, keypoints3d=k3d)])
        raw = out / 'detections'
        raw.mkdir(exist_ok=True)
        np.savez_compressed(raw / ('%06d.npz' % frame), keypoints2d=k2d,
                            bbox=np.array(record['bbox']), keypoints3d=k3d)
        valid_counts.append(int((k3d[:19, 3] > 0.3).sum()))
        for view_index in range(len(args.views)):
            cams = sample['cameras']
            projected, _ = cv2.projectPoints(np.ascontiguousarray(k3d[:, :3]), cams['R'][view_index],
                cams['T'][view_index], cams['K'][view_index], cams['dist'][view_index])
            valid = (k3d[:, 3] > 0.3) & (k2d[view_index, :, 2] > 0.3)
            errors.extend(np.linalg.norm(projected[:, 0] - k2d[view_index, :, :2], axis=-1)[valid].tolist())
        timings.append(time.perf_counter() - started)
        print('Completed source frame', frame, 'in', round(timings[-1], 2), 'seconds', flush=True)
    report = dict(stage='detection_and_triangulation', synthetic=False,
        source='official EasyMocap zju-ls-feng sample', views=args.views,
        source_frames=list(range(0, args.frames, args.step)),
        frame_count=len(records), device='cpu', torch=torch.__version__,
        valid_joints_out_of_19_mean=float(np.mean(valid_counts)),
        reprojection_comparison_count=len(errors),
        reprojection_median_px=float(np.median(errors)) if errors else None,
        reprojection_p95_px=float(np.percentile(errors, 95)) if errors else None,
        seconds_per_frame_mean=float(np.mean(timings)),
        notes=['Reprojection is consistency with detections, not ground truth accuracy.',
               'COCO17 mapped to BODY25; neck and pelvis synthesized, six foot joints unavailable.'],
        smpl_completed=False)
    (out / 'validation.json').write_text(json.dumps(report, indent=2))
    if args.fit:
        model.at_final(records)
    for folder in ['vis_keypoints2d', 'vis_repro'] + (['vis_render'] if args.fit else []):
        subprocess.run([str(args.ffmpeg), '-hide_banner', '-loglevel', 'error', '-y',
            '-framerate', str(source[0]['fps']/args.step), '-i', str(out / folder / '%06d.jpg'),
            '-frames:v', str(len(records)), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            str(out / (folder + '.mp4'))], check=True)
    if args.fit:
        report['smpl_completed'] = True
        report['stage'] = 'smpl_fit_and_render'
    report['videos_encoded'] = True
    (out / 'validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


# The upstream adapter otherwise pulls a moving GitHub HEAD via torch.hub.
# Keep its detector behavior, changing only the explicitly pinned source/device.
class PinnedYOLO:
    def __new__(cls, repo, ckpt, name='person'):
        import torch
        from myeasymocap.backbone.yolo.yolo import BaseYOLOv5
        detector = BaseYOLOv5.__new__(BaseYOLOv5)
        detector.model = torch.hub.load(repo, 'custom', path=ckpt, source='local', device='cpu')
        detector.multiview = True
        detector.name = name
        return detector


if __name__ == '__main__':
    main()
