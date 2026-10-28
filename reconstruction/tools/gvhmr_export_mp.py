"""Export GVHMR results as MediaPipe-33-ordered joints (T5). Based on P2's gvhmr_export_joints.py.

GVHMR chain as in tools/demo/demo.py: SMPL-X "supermotion" -> smplx2smpl_sparse (6890 SMPL vertices)
-> smpl_neutral_J_regressor (24 joints). MediaPipe joints without an SMPL joint use SMPL vertices
(smplx vertex_ids['smplh']): nose 332, heels 3387/6787, big toes 3216/6617.

MediaPipe <- SMPL mapping (other MediaPipe joints are NaN):
  0 nose <- v332        11/12 L/R shoulder <- J16/J17   13/14 elbow <- J18/J19   15/16 wrist <- J20/J21
  23/24 L/R hip <- J1/J2  25/26 knee <- J4/J5   27/28 ankle <- J7/J8   29/30 heel <- v3387/v6787
  31/32 foot index <- v3216/v6617

Writes mp33.npz next to hmr4d_results.pt:
  incam (F,33,3) camera coords, m (x right, y down, z forward); glob (F,33,3) GVHMR world, m (y up);
  smpl24_incam (F,24,3); vitpose (F,17,3); bbx_xyxy (F,4); K (3,3)

  cd ~/fall2026/external/GVHMR && python <this file> <out_dir> [...]
"""
import sys
from pathlib import Path

import numpy as np
import torch

from hmr4d.utils.smplx_utils import make_smplx

MP_FROM_J = {11: 16, 12: 17, 13: 18, 14: 19, 15: 20, 16: 21, 23: 1, 24: 2, 25: 4, 26: 5, 27: 7, 28: 8}
MP_FROM_V = {0: 332, 29: 3387, 30: 6787, 31: 3216, 32: 6617}


@torch.no_grad()
def smpl_verts_joints(smplx, smplx2smpl, J_regressor, params):
    out = smplx(**{k: v.cuda() for k, v in params.items()})
    verts = torch.stack([smplx2smpl @ v for v in out.vertices])  # (F, 6890, 3)
    J = torch.einsum("jv,fvi->fji", J_regressor, verts)
    return verts.cpu().numpy(), J.cpu().numpy()


def to_mp(verts, J):
    mp = np.full((len(J), 33, 3), np.nan, np.float32)
    for m, j in MP_FROM_J.items():
        mp[:, m] = J[:, j]
    for m, v in MP_FROM_V.items():
        mp[:, m] = verts[:, v]
    return mp


def main(dirs):
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()
    assert J_regressor.shape == (24, 6890), J_regressor.shape
    for d in map(Path, dirs):
        res = torch.load(d / "hmr4d_results.pt", map_location="cpu")
        vi, ji = smpl_verts_joints(smplx, smplx2smpl, J_regressor, res["smpl_params_incam"])
        vg, jg = smpl_verts_joints(smplx, smplx2smpl, J_regressor, res["smpl_params_global"])
        np.savez(d / "mp33.npz", incam=to_mp(vi, ji), glob=to_mp(vg, jg), smpl24_incam=ji.astype(np.float32),
                 vitpose=torch.load(d / "preprocess/vitpose.pt").numpy(),
                 bbx_xyxy=torch.load(d / "preprocess/bbx.pt")["bbx_xyxy"].numpy(),
                 K=res["K_fullimg"][0].numpy())
        print(d.name, "frames", len(ji), "pelvis z (m) median", float(np.median(ji[:, 0, 2])))


if __name__ == "__main__":
    main(sys.argv[1:])
