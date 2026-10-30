"""Canonical maps for a GaussianAvatar SMPL dataset, without OpenGL.

GaussianAvatar's scripts/gen_pose_map_cano_smpl.py poses SMPL in its canonical pose (smpl_cpose_param, transl
[0, 0.30, 0]) with the subject's beta, saves the joint matrices `.A` as smpl_cano_joint_mat.pth, and renders the
canonical mesh into the 512x512 UV map with an OpenGL renderer (query_posemap_512_cano_smpl.npz, key 'posmap512').
assets.zip already ships, for each UV pixel, the face id (uv_masks/uv_mask512_with_faceid_smpl.npy, -1 = empty) and
barycentric coordinates (bary_coords_uv_map/bary_coords_smpl_uv512.npy), so the map is computed directly:
    posmap[p] = sum_k bary[p, k] * v_cano[faces[faceid[p], k]]   (0 where faceid = -1)

Run with the gs-avatar env from the GaussianAvatar repo:
    cd ~/fall2026/external/GaussianAvatar
    PYTHONPATH=$PWD python <this file> SPLIT_DIR [SPLIT_DIR ...]          # writes the two files into each dir
    PYTHONPATH=$PWD python <this file> --validate REF_SPLIT_DIR            # recompute from REF's beta, compare
"""
import sys
from os.path import join

import numpy as np
import torch

from arguments import smpl_cpose_param
from submodules import smplx

GA = "."  # GaussianAvatar repo (cwd)


def cano_maps(beta):
    smpl = smplx.SMPL(model_path=join(GA, "assets/smpl_files/smpl"), batch_size=1)
    with torch.no_grad():
        out = smpl.forward(betas=beta, global_orient=smpl_cpose_param[:, :3], transl=torch.tensor([[0, 0.30, 0]]),
                           body_pose=smpl_cpose_param[:, 3:])
    v = out.vertices[0].numpy().astype(np.float64)
    faces = np.load(join(GA, "assets/smpl_faces.npy")).astype(np.int64)
    fid = np.load(join(GA, "assets/uv_masks/uv_mask512_with_faceid_smpl.npy")).reshape(512, 512)
    bary = np.load(join(GA, "assets/bary_coords_uv_map/bary_coords_smpl_uv512.npy")).reshape(512, 512, 3)
    pos = np.zeros((512, 512, 3))
    ok = fid >= 0
    tri = v[faces[fid[ok]]]                              # (P, 3 corners, 3 xyz)
    pos[ok] = np.einsum("pk,pkx->px", bary[ok], tri)
    return pos.astype(np.float32), out.A, ok


def main():
    if sys.argv[1] == "--validate":
        ref = sys.argv[2]
        beta = torch.load(join(ref, "smpl_parms.pth"), map_location="cpu")["beta"].float()
        pos, A, ok = cano_maps(beta)
        ref_pos = np.load(join(ref, "query_posemap_512_cano_smpl.npz"))["posmap512"]
        ref_A = torch.load(join(ref, "smpl_cano_joint_mat.pth"), map_location="cpu")
        d = np.abs(pos - ref_pos)
        ref_ok = np.abs(ref_pos).sum(-1) > 0
        print(f"posmap: shape {pos.shape} vs ref {ref_pos.shape}; covered pixels ours {ok.sum()} ref {ref_ok.sum()}, "
              f"both {(ok & ref_ok).sum()}")
        print(f"posmap |diff| on pixels covered by both: mean {d[ok & ref_ok].mean()*1000:.4f} mm, "
              f"p99 {np.percentile(d[ok & ref_ok], 99)*1000:.4f} mm, max {d[ok & ref_ok].max()*1000:.4f} mm")
        print(f"pixels covered by only one: ours-only {(ok & ~ref_ok).sum()}, ref-only {(~ok & ref_ok).sum()}")
        print(f"joint mat: shape {tuple(A.shape)} vs ref {tuple(ref_A.shape)}, max |diff| {float((A - ref_A).abs().max()):.2e}")
        return
    for d in sys.argv[1:]:
        beta = torch.load(join(d, "smpl_parms.pth"), map_location="cpu")["beta"].float()
        pos, A, ok = cano_maps(beta)
        np.savez(join(d, "query_posemap_512_cano_smpl.npz"), posmap512=pos)
        torch.save(A, join(d, "smpl_cano_joint_mat.pth"))
        print(f"{d}: posmap {pos.shape}, {ok.sum()} covered pixels, joint mat {tuple(A.shape)}")


if __name__ == "__main__":
    main()
