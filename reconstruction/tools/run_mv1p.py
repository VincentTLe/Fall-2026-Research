"""Run EasyMocap's mv1p pipeline (detect -> triangulate -> SMPL -> overlay) on a calibrated dataset.

General version of local_cpu.py: same upstream config (config/mv1p/detect_triangulate_fitSMPL.yml),
same pinned YOLOv5 loader and HRNet, but with the data/output paths, cameras, frame range, device,
SMPL model and rendered cameras as arguments.

DATA is an EasyMocap folder: intri.yml, extri.yml and either videos/<cam>.mp4 (frames are extracted
to images/<cam>/%06d.jpg) or images/<cam>/%06d.jpg already numbered from 000000.

2D input, one of:
  default        fresh YOLOv5 + HRNet (cached per image under OUT/)
  --write-annots also save each frame's 2D detections to DATA/annots/<cam>/<frame>.json
  --use-annots   skip detection and read DATA/annots/ (e.g. a make_variant.py copy)

Receipts in OUT/: validation.json (reprojection stats overall, per camera and per joint, valid-joint
counts, timing, smpl_completed, videos_encoded), reprojection.npz (every compared joint/view),
triangulated/, detections/, smpl/ and videos. The fit stage never downloads a model.

Example:
  python run_mv1p.py --data ~/fall2026/data/zju-ls-feng --out ~/fall2026/runs/zju-4cam \\
      --views 1 7 13 19 --end 100 --step 2 --device cuda \\
      --smpl ~/fall2026/models/smpl/SMPL_NEUTRAL.pkl --render-views 1
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TOOLS = Path(__file__).resolve().parent
HOME = Path.home() / 'fall2026'
BODY25 = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist',
          'MidHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye',
          'REar', 'LEar', 'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--views', nargs='+', default=None, help='camera names (default: all in intri.yml)')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=None, help='source frame end, exclusive (default: all)')
    p.add_argument('--step', type=int, default=1)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--threads', type=int, default=4, help='torch CPU threads')
    p.add_argument('--smpl', type=Path, help='licensed SMPL_NEUTRAL.pkl; enables the SMPL fit')
    p.add_argument('--render-views', nargs='*', default=None,
                   help='cameras to render the SMPL overlay on (default: first camera)')
    p.add_argument('--render-scale', type=float, default=0.5)
    p.add_argument('--vis-views', nargs='*', default=None,
                   help='cameras shown in vis_keypoints2d / vis_repro (default: render views)')
    p.add_argument('--no-vis', action='store_true', help='skip 2D and reprojection visualizations')
    p.add_argument('--use-annots', action='store_true', help='read 2D from DATA/annots instead of detecting')
    p.add_argument('--write-annots', action='store_true', help='save fresh 2D to DATA/annots')
    p.add_argument('--fps', type=float, default=None, help='source fps when DATA has no videos/')
    p.add_argument('--easymocap', type=Path, default=HOME / 'external/EasyMocap')
    p.add_argument('--yolo-repo', type=Path, default=HOME / 'external/yolov5')
    p.add_argument('--yolo-ckpt', type=Path, default=HOME / 'models/detectors/yolov5m.pt')
    p.add_argument('--hrnet-ckpt', type=Path, default=HOME / 'models/detectors/pose_hrnet_w48_384x288.pth')
    p.add_argument('--keypoints2d', choices=['hrnet', 'vitpose'], default='hrnet',
                   help='hrnet: COCO-17, no feet; vitpose: ViTPose+-B COCO-17 + 6 COCO-WholeBody foot points')
    p.add_argument('--vitpose-ckpt', type=Path, default=HOME / 'models/detectors/vitpose+_base.pth')
    p.add_argument('--ffmpeg', default='ffmpeg')
    p.add_argument('--label', default='', help='free text stored in validation.json')
    args = p.parse_args()
    if args.start < 0 or args.step < 1 or (args.end is not None and args.end <= args.start):
        p.error('need 0 <= --start < --end and --step >= 1')
    if args.smpl is not None and not args.smpl.is_file():
        p.error('--smpl must point to your licensed SMPL_NEUTRAL.pkl')
    if args.use_annots and args.write_annots:
        p.error('--use-annots and --write-annots exclude each other')
    # Resolve now: main() changes into the EasyMocap folder, and EasyMocap's SMPLLoader
    # downloads a third-party model archive when model_path does not exist.
    for key in ['data', 'out', 'smpl', 'easymocap', 'yolo_repo', 'yolo_ckpt', 'hrnet_ckpt', 'vitpose_ckpt']:
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).expanduser().resolve())
    return args


def extract_frames(data, views, end):
    """Decode videos/<cam>.mp4 into images/<cam>/%06d.jpg for frames [0, end)."""
    import cv2
    source = []
    for view in views:
        video = data / 'videos' / (view + '.mp4')
        dest = data / 'images' / view
        if not video.is_file():
            if not dest.is_dir():
                raise FileNotFoundError('Need %s or %s' % (video, dest))
            count = len([f for f in os.listdir(dest) if f.endswith('.jpg')])
            source.append(dict(view=view, images=str(dest), frames=count))
            continue
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError('Cannot open ' + str(video))
        info = dict(view=view, file=str(video), fps=cap.get(cv2.CAP_PROP_FPS),
                    frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        source.append(info)
        stop = info['frames'] if end is None else end
        if stop > info['frames']:
            raise ValueError('Requested frames exceed ' + str(video))
        dest.mkdir(parents=True, exist_ok=True)
        if all((dest / ('%06d.jpg' % f)).exists() for f in range(stop)):
            cap.release()
            continue
        for frame in range(stop):
            ok, img = cap.read()
            if not ok:
                raise RuntimeError('Decode failed at %s:%s' % (video, frame))
            path = dest / ('%06d.jpg' % frame)
            if not path.exists() and not cv2.imwrite(str(path), img):
                raise RuntimeError('Cannot write ' + str(path))
        cap.release()
    return source


def summarize(values):
    import numpy as np
    values = np.asarray(values)
    if values.size == 0:
        return dict(count=0, median_px=None, p95_px=None, mean_px=None)
    return dict(count=int(values.size), median_px=float(np.median(values)),
                p95_px=float(np.percentile(values, 95)), mean_px=float(np.mean(values)))


def main():
    args = parse_args()
    if args.device == 'cpu':
        # EasyMocap's HRNet, SMPL loader and optimizer all pick cuda when they can see a GPU.
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ['YOLOv5_AUTOINSTALL'] = 'false'
    data, out, easy = args.data.resolve(), args.out.resolve(), args.easymocap.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(TOOLS))
    sys.path.insert(0, str(easy))
    os.chdir(easy)
    import cv2
    import numpy as np
    import torch
    from emc_yml import read_cameras
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('--device cuda but torch sees no GPU')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    from easymocap.config import Config, load_object
    from easymocap.config.yacs import CfgNode
    from easymocap.mytools.file_utils import write_keypoints3d

    names = list(read_cameras(str(data)).keys())
    views = args.views or (sorted(names, key=int) if all(n.isdigit() for n in names) else names)
    missing = [v for v in views if v not in names]
    if missing:
        raise SystemExit('cameras %s not in %s/intri.yml' % (missing, data))
    render_views = views[:1] if args.render_views is None else args.render_views
    vis_views = [] if args.no_vis else (render_views if args.vis_views is None else args.vis_views)
    for v in render_views + vis_views:
        if v not in views:
            raise SystemExit('camera %s is not in --views' % v)

    source = extract_frames(data, views, args.end)
    fps_set = {round(s['fps'], 5) for s in source if 'fps' in s}
    if len(fps_set) > 1:
        raise ValueError('Camera frame rates differ: %s' % fps_set)
    fps = args.fps or (fps_set.pop() if fps_set else 25.)
    end = args.end if args.end is not None else min(s['frames'] for s in source)
    (out / 'source_videos.json').write_text(json.dumps(source, indent=2))

    cfg_data = Config.load(str(easy / 'config/datasets/mvimage.yml'))
    cfg_data.args.root = str(data)
    cfg_data.args.subs = views
    cfg_data.args.subs_vis = vis_views
    cfg_data.args.ranges = [args.start, end, args.step]
    cfg = Config.load(str(easy / 'config/mv1p/detect_triangulate_fitSMPL.yml'))
    cfg.args.output = str(out)
    steps = cfg.args.at_step
    if args.use_annots:
        cfg_data.args.reader.annots = CfgNode(dict(root='annots', ext='.json'))
        del steps['detect'], steps['keypoints2d']
        steps = CfgNode(dict([('from_annots', CfgNode(dict(module='run_mv1p.FromAnnots',
                         key_from_data=['bbox', 'keypoints'], args=CfgNode({}))))] + list(steps.items())))
        cfg.args.at_step = steps
    else:
        kp_ckpt = args.hrnet_ckpt if args.keypoints2d == 'hrnet' else args.vitpose_ckpt
        for ckpt in [args.yolo_ckpt, kp_ckpt]:
            if not ckpt.is_file():
                raise FileNotFoundError('Missing detector weights ' + str(ckpt))
        steps.detect.module = 'run_mv1p.PinnedYOLO'
        steps.detect.args = CfgNode(dict(repo=str(args.yolo_repo), ckpt=str(args.yolo_ckpt), name='person',
                                         device='0' if args.device == 'cuda' else 'cpu'))
        if args.keypoints2d == 'vitpose':
            steps.keypoints2d.module = 'run_mv1p.SafeViT'
            steps.keypoints2d.args = CfgNode(dict(ckpt=str(kp_ckpt)))
        else:
            steps.keypoints2d.args.ckpt = str(kp_ckpt)
    # Show the selected cameras without the upstream center crop.
    steps.visualize.args.mode = 'none'
    if args.no_vis:
        steps.vis2d.skip = True
        steps.visualize.skip = True
    final = cfg.args.at_final
    if args.smpl:
        regressor = easy / 'data/smplx/J_regressor_body25.npy'
        if not args.smpl.is_file() or not regressor.is_file():
            raise SystemExit('SMPL model or regressor missing; refusing to let EasyMocap download one')
        final.load_body_model.args.model_path = str(args.smpl)
        final.load_body_model.args.regressor_path = str(regressor)
        render = final.render_ground
        del final['render_ground'], final['make_video']
        for cam in render_views:
            final['render_' + cam] = CfgNode(dict(module=render.module, key_from_data=render.key_from_data,
                key_from_previous=render.key_from_previous,
                args=CfgNode(dict(backend='pyrender', view_list=[views.index(cam)], render_mode='image',
                                  name='vis_render_' + cam, scale=args.render_scale))))
    else:
        cfg.args.at_final = CfgNode({})
    (out / 'cfg_data.yml').write_text(str(cfg_data), encoding='utf-8')
    (out / 'cfg_exp.yml').write_text(str(cfg), encoding='utf-8')

    dataset = load_object(cfg_data.module, cfg_data.args)
    model = load_object(cfg.module, cfg.args)
    records, valid19, valid25, timings, obs = [], [], [], [], []
    for index in range(len(dataset)):
        started = time.perf_counter()
        sample = dataset[index]
        record = model.at_step(sample, index)
        records.append(record)
        k3d, k2d = record['keypoints3d'], np.asarray(record['keypoints'])
        if not np.isfinite(k3d).all() or not np.isfinite(k2d).all():
            raise ValueError('Non-finite detected or reconstructed keypoints')
        frame = sample['meta']['frame']
        write_keypoints3d(str(out / 'triangulated' / ('%06d.json' % frame)), [dict(id=0, keypoints3d=k3d)])
        bbox = np.zeros((len(views), 5))
        for nv, b in enumerate(record['bbox']):
            b = np.asarray(b).reshape(-1, 5)
            if len(b):
                bbox[nv] = b[0]
        (out / 'detections').mkdir(exist_ok=True)
        np.savez_compressed(out / 'detections' / ('%06d.npz' % frame), keypoints2d=k2d, bbox=bbox, keypoints3d=k3d)
        if args.write_annots:
            for nv, view in enumerate(views):
                imgname = sample['meta']['imgnames'][nv]
                annot = dict(filename=os.path.join(view, os.path.basename(imgname)),
                             annots=[dict(personID=0, bbox=bbox[nv].tolist(), keypoints=k2d[nv].tolist())])
                path = data / 'annots' / view / os.path.basename(imgname).replace('.jpg', '.json')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(annot))
        valid19.append(int((k3d[:19, 3] > 0.3).sum()))
        valid25.append(int((k3d[:, 3] > 0.3).sum()))
        cams = sample['cameras']
        for nv in range(len(views)):
            projected, _ = cv2.projectPoints(np.ascontiguousarray(k3d[:, :3]), cams['R'][nv],
                cams['T'][nv], cams['K'][nv], cams['dist'][nv])
            valid = (k3d[:, 3] > 0.3) & (k2d[nv, :, 2] > 0.3)
            err = np.linalg.norm(projected[:, 0] - k2d[nv, :, :2], axis=-1)
            for j in np.where(valid)[0]:
                obs.append((frame, nv, j, err[j]))
        timings.append(time.perf_counter() - started)
        print('Completed source frame', frame, 'in', round(timings[-1], 2), 'seconds', flush=True)

    obs = np.array(obs, dtype=np.float64).reshape(-1, 4)
    np.savez_compressed(out / 'reprojection.npz', frame=obs[:, 0].astype(int), view=obs[:, 1].astype(int),
                        joint=obs[:, 2].astype(int), error_px=obs[:, 3], views=np.array(views))
    overall = summarize(obs[:, 3])
    report = dict(stage='detection_and_triangulation', synthetic=False, label=args.label,
        data=str(data), views=views, source_frames=list(range(args.start, end, args.step)),
        frame_count=len(records), device=args.device, torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
        input_2d='annots (reused)' if args.use_annots else 'fresh YOLOv5 + ' + args.keypoints2d,
        valid_joints_out_of_19_mean=float(np.mean(valid19)),
        valid_joints_out_of_25_mean=float(np.mean(valid25)),
        reprojection_comparison_count=overall['count'],
        reprojection_median_px=overall['median_px'], reprojection_p95_px=overall['p95_px'],
        reprojection_per_camera={v: summarize(obs[obs[:, 1] == nv, 3]) for nv, v in enumerate(views)},
        reprojection_per_joint={BODY25[j]: summarize(obs[obs[:, 2] == j, 3]) for j in range(25)
                                if (obs[:, 2] == j).any()},
        seconds_per_frame_mean=float(np.mean(timings)),
        seconds_per_frame_median=float(np.median(timings)),
        notes=['Reprojection is consistency with detections, not ground truth accuracy.',
               'Compared joint/views: 3D and 2D confidence both > 0.3.',
               'COCO17 mapped to BODY25; neck and pelvis synthesized, six foot joints unavailable.'
               if args.keypoints2d == 'hrnet' else
               'ViTPose+ COCO17 + 6 WholeBody foot points mapped to BODY25; neck and pelvis synthesized.'],
        smpl_completed=False, videos_encoded=False)
    (out / 'validation.json').write_text(json.dumps(report, indent=2))
    if args.smpl:
        started = time.perf_counter()
        model.at_final(records)
        report['seconds_smpl_fit_and_render'] = time.perf_counter() - started
        report['smpl_files'] = len(list((out / 'smpl').glob('*.json')))
    folders = [] if args.no_vis else ['vis_keypoints2d', 'vis_repro']
    folders += ['vis_render_' + c for c in render_views] if args.smpl else []
    for folder in folders:
        subprocess.run([args.ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
            '-framerate', str(fps / args.step), '-i', str(out / folder / '%06d.jpg'),
            '-frames:v', str(len(records)), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            str(out / (folder + '.mp4'))], check=True)
    if args.smpl:
        report['smpl_completed'] = True
        report['stage'] = 'smpl_fit_and_render'
    report['videos_encoded'] = bool(folders)
    report['videos'] = [f + '.mp4' for f in folders]
    (out / 'validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if not k.startswith('reprojection_per')}, indent=2))


# The upstream adapter otherwise pulls a moving GitHub HEAD via torch.hub.
# Keep its detector behavior, changing only the explicitly pinned source/device.
class PinnedYOLO:
    def __new__(cls, repo, ckpt, name='person', device='cpu'):
        import torch
        from myeasymocap.backbone.yolo.yolo import BaseYOLOv5
        detector = BaseYOLOv5.__new__(BaseYOLOv5)
        detector.model = torch.hub.load(repo, 'custom', path=ckpt, source='local', device=device)
        detector.multiview = True
        detector.name = name
        return detector


class SafeViT:
    """EasyMocap's ViTPose+ wrapper (COCO-17 + feet), loading the checkpoint with weights_only=True.

    Upstream MyViT calls torch.load without weights_only (a full unpickle) and never sets num_joints,
    which its no-person branch needs.
    """
    def __new__(cls, ckpt):
        import torch
        from myeasymocap.backbone.vitpose.vit_moe import MyViT
        load = torch.load
        torch.load = lambda *a, **k: load(*a, **dict(k, weights_only=True))
        try:
            model = MyViT(ckpt=ckpt, single_person=True)
        finally:
            torch.load = load
        model.num_joints = 23
        return model


class FromAnnots:
    """Stage that passes the dataset's annots/ 2D (bbox, keypoints) on as if freshly detected."""
    def __call__(self, bbox, keypoints):
        return dict(bbox=[b[None] for b in bbox], keypoints=keypoints)


if __name__ == '__main__':
    main()
