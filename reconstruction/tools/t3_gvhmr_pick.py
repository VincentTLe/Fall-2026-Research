"""t3_gvhmr_lowmem.py plus a different choice of person.

GVHMR's tracker keeps the track with the largest summed box area, which in internet clips with a
coach picked the coach (T3 rounds 1 and 2). Here the person exercising is taken instead: among tracks
that cover at least half of the frames, the one whose box height varies most relative to its median
(squats and lunges move the body up and down; a coach standing beside does not). The rule was fixed
from the failure mode before this run; the clips it was designed on are not a held-out test.

  cd ~/fall2026/external/GVHMR
  python t3_gvhmr_pick.py --video clip.mp4 --output_root OUT
"""
import inspect
import runpy
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch

torch.cuda.init()  # before ultralytics can reset CUDA_VISIBLE_DEVICES (see 2026-10-27-t2-monocular.md)

import hmr4d.utils.preproc.tracker as tr  # noqa: E402
import hmr4d.utils.preproc.vitfeat_extractor as vf  # noqa: E402
import hmr4d.utils.preproc.vitpose as vp  # noqa: E402

BATCH = 4
for mod, cls, name in ((vp, vp.VitPoseExtractor, "extract"), (vf, vf.Extractor, "extract_video_features")):
    src = textwrap.dedent(inspect.getsource(getattr(cls, name)))
    assert src.count("batch_size = 16") == 1, f"{cls.__name__}.{name}: batch_size line not found"
    ns = {}
    exec(compile(src.replace("batch_size = 16", f"batch_size = {BATCH}"), mod.__file__, "exec"), mod.__dict__, ns)
    setattr(cls, name, ns[name])


def _t3_pick(id_sorted, id_to_frame_ids, id_to_bbx_xyxys, n_frames):
    cands = [i for i in id_sorted if len(id_to_frame_ids[i]) >= 0.5 * n_frames]
    if len(cands) < 2:
        return id_sorted[0]
    score = {}
    for i in cands:
        b = np.asarray(id_to_bbx_xyxys[i], dtype=float)
        h = b[:, 3] - b[:, 1]
        score[i] = float(np.std(h) / max(np.median(h), 1.0))
    pick = max(score, key=score.get)
    print(f"T3PICK default {id_sorted[0]} picked {pick} scores {score}", flush=True)
    return pick


tr._t3_pick = _t3_pick
src = textwrap.dedent(inspect.getsource(tr.Tracker.get_one_track))
old = "track_id = id_sorted[0]"
assert src.count(old) == 1, "get_one_track: track choice line not found"
new = "track_id = _t3_pick(id_sorted, id_to_frame_ids, id_to_bbx_xyxys, get_video_lwh(video_path)[0])"
ns = {}
exec(compile(src.replace(old, new), tr.__file__, "exec"), tr.__dict__, ns)
tr.Tracker.get_one_track = ns["get_one_track"]

runner = Path.home() / "fall2026/Fall-2026-Research/reconstruction/tools/gvhmr_run.py"
sys.argv = [str(runner)] + sys.argv[1:]
runpy.run_path(str(runner), run_name="__main__")
