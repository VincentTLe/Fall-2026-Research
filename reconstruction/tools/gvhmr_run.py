"""(Copied unchanged from P2, branch server/t2-20261027, for T5.)
Run GVHMR's demo pipeline (tools/demo/demo.py at ee960bb) up to hmr4d_results.pt, with
rendering optional. Same preprocessing, model and checkpoint as the demo; only the incam/global
rendering and the merged video are skipped unless --render is given.

  cd ~/fall2026/external/GVHMR
  python <this file> --video clip.mp4 --output_root OUT [--render]
"""
import importlib.util
import os
import sys
from pathlib import Path

render = "--render" in sys.argv
# --f_px F: focal length in pixels (square pixels, principal point at the image centre, as in GVHMR).
# Converted to GVHMR's full-frame f_mm as in hmr4d/utils/geo/hmr_cam.py create_camera_sensor:
# f_px = diag_img / diag_fullframe * f_mm, diag_fullframe = sqrt(24^2 + 36^2). Without it GVHMR uses
# estimate_K (f = image diagonal). demo.py's own --f_mm only takes integers.
f_px = None
if "--f_px" in sys.argv:
    i = sys.argv.index("--f_px")
    f_px = float(sys.argv[i + 1])
    del sys.argv[i:i + 2]
sys.argv = [a for a in sys.argv if a != "--render"] + ["-s"]  # REHAB24-6 cameras are static

spec = importlib.util.spec_from_file_location("gvhmr_demo", "tools/demo/demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)  # defines functions only; the demo's __main__ block does not run

cfg = demo.parse_args_to_cfg()
paths = cfg.paths
if f_px is not None:
    _, w, h = demo.get_video_lwh(cfg.video_path)
    cfg.f_mm = f_px * (24 ** 2 + 36 ** 2) ** 0.5 / (w ** 2 + h ** 2) ** 0.5
    demo.Log.info(f"[K] f_px {f_px} -> f_mm {cfg.f_mm:.4f} for {w}x{h}")
# Start CUDA now, as the demo's __main__ does: ultralytics' select_device() (YOLO tracker in
# run_preprocess) sets CUDA_VISIBLE_DEVICES="0", which would otherwise move this job to
# physical GPU 0 regardless of the CUDA_VISIBLE_DEVICES it was started with.
demo.torch.cuda.init()
demo.Log.info(f"[GPU] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
              f"device 0 = {demo.torch.cuda.get_device_name(0)}")
demo.run_preprocess(cfg)
data = demo.load_data_dict(cfg)
if not Path(paths.hmr4d_results).exists():
    demo.Log.info("[HMR4D] Predicting")
    model = demo.hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model = model.eval().cuda()
    pred = demo.detach_to_cpu(model.predict(data, static_cam=cfg.static_cam))
    demo.torch.save(pred, paths.hmr4d_results)
    demo.Log.info(f"[HMR4D] saved {paths.hmr4d_results}, frames {data['length']}")
if render:
    demo.render_incam(cfg)
