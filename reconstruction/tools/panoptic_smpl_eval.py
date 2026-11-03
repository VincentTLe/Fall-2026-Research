"""Score an EasyMocap SMPL fit on CMU Panoptic against the dome reference, the T1 way.

SMPL-24 joints (SMPL J_regressor on the fitted vertices, as for GVHMR / T2c) are put into the BODY25 slots that
panoptic_to_easymocap.py eval scores (shoulders, elbows, wrists, hips, knees, ankles; confidence 1), written as
RUN/smpl_body25/%06d.json, and scored with that same eval. A second copy, RUN/smpl_body25_trimask/, keeps a joint
only where RUN/triangulated has it, so SMPL and triangulation can be compared on exactly the same joints.
RUN/smpl_body25reg/ holds EasyMocap's own SMPL keypoints (RUN/keypoints3d: its BODY25 regressor on the same fit, the
joint definition the fit was matched to), confidence 1, as a check of how much is joint definition.

  python panoptic_smpl_eval.py --run runs/t1b/s3 --smpl models/smpl/SMPL_NEUTRAL.pkl \
      --ref data/panoptic/emc_pose1_650/dome/keypoints3d --frames 392
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path.home() / "fall2026/external/EasyMocap"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# BODY25 slot <- SMPL-24 index
SMPL_TO_BODY25 = {2: 17, 3: 19, 4: 21, 5: 16, 6: 18, 7: 20, 9: 2, 10: 5, 11: 8, 12: 1, 13: 4, 14: 7}


def main(a):
    import torch
    from easymocap.bodymodel.smpl import SMPLModel
    from panoptic_to_easymocap import evaluate
    run = Path(a.run)
    model = SMPLModel(model_path=str(a.smpl), model_type="smpl", device="cpu", NUM_SHAPES=10)
    files = sorted((run / "smpl").glob("*.json"))
    b = [json.load(open(f))[0] for f in files]
    p = {k: np.concatenate([np.array(x[k], np.float32).reshape(1, -1) for x in b]) for k in ("Rh", "Th", "poses", "shapes")}
    with torch.no_grad():
        J = torch.einsum("jv,fvc->fjc", model.J_regressor, model.vertices(p, return_tensor=True)).numpy()
    for sub in ("smpl_body25", "smpl_body25_trimask", "smpl_body25reg"):
        (run / sub).mkdir(exist_ok=True)
    for f, j in zip(files, J):
        k = np.zeros((25, 4))
        for dst, src in SMPL_TO_BODY25.items():
            k[dst, :3], k[dst, 3] = j[src], 1.0
        json.dump([{"id": 0, "keypoints3d": k.tolist()}], open(run / "smpl_body25" / f.name, "w"))
        tri = np.array(json.load(open(run / "triangulated" / f.name))[0]["keypoints3d"])
        k[tri[:, 3] <= 0, 3] = 0
        json.dump([{"id": 0, "keypoints3d": k.tolist()}], open(run / "smpl_body25_trimask" / f.name, "w"))
        r = np.array(json.load(open(run / "keypoints3d" / f.name))[0]["keypoints3d"])[:, :3]
        json.dump([{"id": 0, "keypoints3d": np.c_[r, np.ones(len(r))].tolist()}], open(run / "smpl_body25reg" / f.name, "w"))
    out = {}
    for name, test in (("tri", "triangulated"), ("smpl", "smpl_body25"), ("smpl_trimask", "smpl_body25_trimask"),
                       ("smpl_b25reg", "smpl_body25reg")):
        print(name, end=": ", flush=True)
        js = run / f"eval10_{name}.json"
        evaluate(SimpleNamespace(ref=a.ref, test=str(run / test), offset=0, frames=a.frames, no_ankles=True, json=str(js)))
        out[name] = json.load(open(js))
    json.dump(out, open(run / "smpl_eval.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--smpl", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--frames", type=int, default=392)
    main(ap.parse_args())
