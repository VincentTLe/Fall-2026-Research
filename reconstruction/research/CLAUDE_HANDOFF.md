# Claude handoff: collected research sources

Prepared for Tan on 2026-09-29 for `VincentTLe/Fall-2026-Research`.
Starting repository commit: `397ee19f76c6727f87e44679326d54cee23e9f3a`.
The original reconstruction helper scripts were preserved.

## Status update (Claude, 2026-09-29, after this handoff)

- Merged this branch's research into `claude/sweet-wright-k4mkvr`.
- **Verified:** EasyMocap install (commit `e6006fd`, Python 3.9, CPU) with 3 fixes (chumpy build isolation, mediapipe pin, EGL), plus EasyMocap triangulation on synthetic data. See `SETUP.md` §2 and §6b.
- **Added** `tools/synth_rig.py` + `tools/rig_sweep.sh`: a camera-layout/sync simulator with exact ground truth that runs through EasyMocap. Results are in `results/2026-09-29-synthetic-rig-sweep.md`.
- **Still not done:** Tan's machine check, detector weights, SMPL models, sample data (the cloud container can't reach Dropbox/Drive/OneDrive), and anything on real video.

## Read these files first

1. [READING.md](../READING.md): verified identities, summaries, interfaces and limitations for the supplied papers/repos.
2. [SOURCE_NOTES.md](SOURCE_NOTES.md): concrete source/code corrections and command provenance.
3. [DOWNLOADS.md](DOWNLOADS.md): official acquisition routes and what is still missing.
4. [sources.json](sources.json): all 14 unique supplied links and inspected file hashes.
5. [references.bib](references.bib): citation metadata.

The prose notes remain usable if your runtime cannot browse IEEE, PubMed, arXiv or the EasyMocap documentation site. They are summaries, not copies of full papers.

## Findings that change the earlier plan

- IEEE 11303747 is **RehabKD**, a literature-mining/knowledge-graph study. It does not directly score exercise video or skeletons.
- PubMed 40327204 is the CNN-SE/SHAP exercise-assessment paper. Its identity, venue and abstract are verified; implementation details need full-text checking.
- arXiv 2410.23641 studies temporal augmentation for cross-dataset action recognition, not body-format conversion.
- ExerciseLLM's paper describes 3D sensor joints, but the released REHAB24-6 generator/features use a 2D interface. The actual generator is `generate_REHAB246.py`. Choose and validate the intended path before writing a converter.
- The current EasyMocap public quickstart uses `street_dance`; the older repo guide uses `zju-ls-feng`. Both are valid source references with different commands/defaults.
- Some components have a CPU fallback. Do not claim the entire EasyMocap pipeline requires CUDA, or that the entire pipeline works on this laptop, without running it.
- All-view reconstruction is a reference estimate. Agreement with it is not independent ground-truth accuracy.
- Downstream avatar methods also require images/masks/calibration or specialized character assets; skeleton/SMPL output alone is insufficient.

## Continue with a small reproducible run

This is a proposed next task, not work already performed.

1. Inspect the actual execution machine. The pasted earlier local transcript reported an AMD Radeon RX 5500M and no present NVIDIA device; that is historical user-provided context, not a hardware check from this collection session. Do not rely on a stale registry entry as proof of a usable GPU.
2. Choose one EasyMocap interface and pin its commit. Resolve its dependencies in an isolated environment; the installation commands in SETUP.md are still unverified.
3. Acquire one official sample plus the required licensed model/checkpoint assets. Public links are recorded; successful downloads have not been demonstrated.
4. Run a short frame range first. Save exact commands, environment versions, logs, frame counts and reprojection overlays.
5. Expand to the sample sequence after the short run works. Only then evaluate camera subsets/synchronization and the team's five-camera capture.
6. For downstream work, inspect real sample tensors before conversion. Verify semantic joint order, coordinate axes, units, timestamps, root-transform conventions and missing-joint handling.

For calibration/software handoffs, request camera identifiers, unchanged recording settings, resolution, calibration files, measured time offsets and frame/timestamp mappings. Whether a lens configuration is adequate must be established by calibration/reprojection tests; do not turn an untested lens preference into a universal claim about all GoPros.

## What was and was not completed

Completed: primary-source reading across all supplied links, source metadata, static checks of relevant repository paths, corrections to reading/setup documentation and this handoff.

Not completed: installation, sample/model download, dataset-access requests, calibration, reconstruction, timing measurements, export implementation, LLM inference or training. No paper result is evidence that these steps succeeded on our data.

To integrate this collection, use the accompanying GitHub pull request or its branch `codex/research-source-collection`, then continue in the existing reconstruction plan. Do not recreate the repository or replace the helper tools merely because the earlier browser could not access a source.
