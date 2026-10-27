# Fall 2026 Research: Multi-view GoPro motion capture

Team workstreams: Hardware (Ahsan, Huy, Nghiem), Software (Danny, Katie), Calibration (Seishin), Reconstruction (Tan).

- [`reconstruction/`](reconstruction/): EasyMocap pipeline plan, setup guide, reading list, helper tools

- [Collected research and Claude handoff](reconstruction/research/CLAUDE_HANDOFF.md): verified notes for all 14 supplied links, citation metadata, download requirements and source/code discrepancies (2026-09-29)

## Reconstruction status

| Stage | Status | Evidence |
|---|---|---|
| EasyMocap install, Windows CPU | Verified on Tan's laptop (2026-09-29) | [`SETUP_WINDOWS.md`](reconstruction/SETUP_WINDOWS.md) |
| 2D detection + 3D triangulation on real video (`zju-ls-feng`, 4 cameras, 50 frames) | Verified: reprojection median 5.65 px, p95 19.69 px | [`LOCAL_RUN_STATUS.md`](reconstruction/LOCAL_RUN_STATUS.md) |
| SMPL body fit + overlay video (same subset) | Verified on Tan's laptop (2026-10-27): 50 SMPL frames, mesh overlay on camera 1 | [`LOCAL_RUN_STATUS.md`](reconstruction/LOCAL_RUN_STATUS.md#smpl-fit--2026-10-27) |
| Full run (23 cameras, 800 frames), public datasets, Gaussian Splatting | Planned | [`EXPLORATION.md`](reconstruction/EXPLORATION.md) |
| Reproduction on a second machine | Not done yet | [checklist](reconstruction/SETUP_WINDOWS.md#independent-reproduction-checklist-for-seishin) |

To reproduce on Windows, follow [`SETUP_WINDOWS.md`](reconstruction/SETUP_WINDOWS.md). [`SETUP.md`](reconstruction/SETUP.md) is the Linux guide; its install steps were verified in a cloud container on CPU. The environment, sample data, model weights and run outputs are not in the repo. The guides download them and list their SHA256 hashes.
