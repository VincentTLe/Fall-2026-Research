# Reading session: reconstruction context

These are assistance notes, not a claim that Tan has completed personal reading
or that the professor assigned every supporting paper below. The explicit
supplied ZJU links are EasyMocap, its quickstart and multinb. The existing
`READING.md` retains the wider source list.

## Neural Body (supporting paper)

Read: [CVPR 2021 paper](https://arxiv.org/pdf/2012.15838), especially Sections
3-4 and Figures 2-3. A local copy is in the ignored `downloads/` directory.

The target is novel-view rendering of a moving performer. Calibration, foreground
images and posed SMPL geometry are inputs. Shared learned codes attach to the
6,890 SMPL vertices; pose moves their locations between frames. Sparse 3D
convolutions spread those codes into nearby space. Networks decode sampled
features into density and color, and volume rendering produces image pixels.
Joint training over frames lets observations of one body region inform other
frames rather than solving every image independently.

Figure 3 explains the distinction between surface-attached codes and the
continuous field queried by a camera ray. Section 4 evaluates image synthesis
with held-out camera views using PSNR/SSIM. It explicitly lacks ground-truth
human geometry, so the geometric reconstruction results are qualitative.

For our project, EasyMocap supplies the geometric preparation. Producing an SMPL
overlay does not mean Neural Body training is complete. Preserve images, cameras,
timestamps and masks alongside body parameters for a later renderer experiment.

Self-check: explain why shared surface codes help sparse views, what pose errors
would do to them, and why image-rendering metrics do not establish joint accuracy.

## multinb (explicit supplied link)

The [official project documentation](https://chingswy.github.io/easymocap-public-doc/works/multinb.html)
describes human-interaction rendering and instance masks. It starts from
calibrated video and fitted body parameters, then exports vertices and trains a
neural renderer. This is downstream of the reconstruction task.

The author's current PDF link and the previously indexed PDF URL returned 404
on 2026-09-29; the ACM PDF fetch also failed. Therefore this session has not
verified the complete paper's methods or experiments. Do not replace that
reading requirement with these project-page notes. The publisher record is
[DOI 10.1145/3528233.3530704](https://doi.org/10.1145/3528233.3530704).

## Actual code study

Use `CODE_WALKTHROUGH.md` alongside the local source. It traces the real runner's
frames, person boxes, HRNet heatmaps, coordinate transforms, triangulation,
SMPL objectives and calibrated overlay rendering. Reading status and execution
status are deliberately reported separately in `LOCAL_RUN_STATUS.md`.
