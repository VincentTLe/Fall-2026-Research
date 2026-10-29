"""Export SMPL 24-joint positions from a GVHMR demo output folder.

Uses GVHMR's own chain (as in tools/demo/demo.py): SMPL-X "supermotion" model ->
smplx2smpl_sparse vertices -> smpl_neutral_J_regressor. Writes joints.npz next to
hmr4d_results.pt with
  incam  (F, 24, 3)  camera coordinates, meters (x right, y down, z forward)
  glob   (F, 24, 3)  GVHMR gravity-aligned world coordinates, meters (y up)
  vitpose (F, 17, 3) COCO-17 2D keypoints + confidence used by GVHMR
  bbx_xyxy (F, 4)    tracked person box
  K      (3, 3)      intrinsics GVHMR assumed

Run from the GVHMR repo with its environment:
  cd ~/fall2026/external/GVHMR
  python <this file> ~/fall2026/runs/t2-gvhmr/<video_stem> [...]
"""
import sys
from pathlib import Path

import numpy as np
import torch

from hmr4d.utils.smplx_utils import make_smplx


@torch.no_grad()
def smpl24(smplx, smplx2smpl, J_regressor, params):
    out = smplx(**{k: v.cuda() for k, v in params.items()})
    verts = torch.stack([smplx2smpl @ v for v in out.vertices])  # (F, 6890, 3)
    return torch.einsum("jv,fvi->fji", J_regressor, verts).cpu().numpy()


def main(dirs):
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()
    assert J_regressor.shape == (24, 6890), J_regressor.shape
    for d in map(Path, dirs):
        res = torch.load(d / "hmr4d_results.pt", map_location="cpu")
        incam = smpl24(smplx, smplx2smpl, J_regressor, res["smpl_params_incam"])
        glob = smpl24(smplx, smplx2smpl, J_regressor, res["smpl_params_global"])
        np.savez(
            d / "joints.npz",
            incam=incam,
            glob=glob,
            vitpose=torch.load(d / "preprocess/vitpose.pt").numpy(),
            bbx_xyxy=torch.load(d / "preprocess/bbx.pt")["bbx_xyxy"].numpy(),
            K=res["K_fullimg"][0].numpy(),
        )
        print(d.name, "frames", len(incam), "pelvis z (m) median", float(np.median(incam[:, 0, 2])))


if __name__ == "__main__":
    main(sys.argv[1:])
