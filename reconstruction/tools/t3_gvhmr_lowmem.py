"""Run gvhmr_run.py with GVHMR's preprocessing batch size lowered from 16 to 4.

The ViTPose and HMR2 feature extractors hard-code batch_size = 16 (about 5 GB of GPU memory). On a GPU
that another user's job almost fills, this wrapper rebuilds those two methods with batch_size = 4 at
runtime, without changing the GVHMR checkout, so the outputs are the same computation in smaller batches.

  cd ~/fall2026/external/GVHMR
  python t3_gvhmr_lowmem.py --video clip.mp4 --output_root OUT [--render]
"""
import inspect
import runpy
import sys
import textwrap
from pathlib import Path

import torch

torch.cuda.init()  # before ultralytics can reset CUDA_VISIBLE_DEVICES (see 2026-10-27-t2-monocular.md)

import hmr4d.utils.preproc.vitfeat_extractor as vf  # noqa: E402
import hmr4d.utils.preproc.vitpose as vp  # noqa: E402

BATCH = 4
for mod, cls, name in ((vp, vp.VitPoseExtractor, "extract"), (vf, vf.Extractor, "extract_video_features")):
    src = textwrap.dedent(inspect.getsource(getattr(cls, name)))
    assert src.count("batch_size = 16") == 1, f"{cls.__name__}.{name}: batch_size line not found"
    ns = {}
    exec(compile(src.replace("batch_size = 16", f"batch_size = {BATCH}"), mod.__file__, "exec"), mod.__dict__, ns)
    setattr(cls, name, ns[name])

runner = Path.home() / "fall2026/Fall-2026-Research/reconstruction/tools/gvhmr_run.py"
sys.argv = [str(runner)] + sys.argv[1:]
runpy.run_path(str(runner), run_name="__main__")
