# T4 Step 3 / T4b person masks for REHAB24-6 PM_037 (camera 17 or 18) with SAM 2.1 (hiera_base_plus, commit 2b90b9f), CPU.
# Prompt per frame = GVHMR's person box (preprocess/bbx.pt, bbx_xyxy, same detections in out/ and out-truef/),
# padded by 5% of its size, single-mask output. Writes masks/<frame>.png (0/255) and masks_log.csv (score, area).
# Usage: python rehab_sam2_masks.py RUN_DIR [THREADS]   (RUN_DIR has all/images, all/frame_map.csv)
# T4b change from logs/p3-rehab-sam2-masks.py: the box is clipped to the real image size (camera 18 is 1080x1920),
# not to 1920x1080; thread count is an argument (T4 used 16).
import csv, os, sys, time
import numpy as np, torch, cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

run = sys.argv[1]
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 16)
OUT = "/home/tle/fall2026/runs/t2-chunks/out-truef"
pred = SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_b+.yaml",
                                     "/net/home/tle/fall2026/models/sam2/sam2.1_hiera_base_plus.pt", device="cpu"))
boxes = {}
rows = list(csv.DictReader(open(f"{run}/all/frame_map.csv")))
os.makedirs(f"{run}/all/masks", exist_ok=True)
log = open(f"{run}/all/masks_log.csv", "w"); log.write("global,score,area_px,box_x0,box_y0,box_x1,box_y1\n")
t0 = time.time()
for i, r in enumerate(rows):
    name = f"{int(r['global']):05d}.png"
    if os.path.exists(f"{run}/all/masks/{name}"):
        continue
    if r["clip"] not in boxes:
        boxes[r["clip"]] = torch.load(f"{OUT}/{r['clip']}/preprocess/bbx.pt", map_location="cpu")["bbx_xyxy"].numpy()
    x0, y0, x1, y1 = boxes[r["clip"]][int(r["local"])]
    img = cv2.cvtColor(cv2.imread(f"{run}/all/images/{name}"), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    pw, ph = 0.05 * (x1 - x0), 0.05 * (y1 - y0)
    box = np.array([max(0, x0 - pw), max(0, y0 - ph), min(W - 1, x1 + pw), min(H - 1, y1 + ph)])
    with torch.inference_mode():
        pred.set_image(img)
        m, s, _ = pred.predict(box=box[None], multimask_output=False)
    mask = (m[0] > 0).astype(np.uint8) * 255
    cv2.imwrite(f"{run}/all/masks/{name}", mask)
    log.write(f"{r['global']},{float(s[0]):.4f},{int((mask > 0).sum())},{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}\n"); log.flush()
    if i % 25 == 0:
        print(f"{i+1}/{len(rows)} {name} score {float(s[0]):.3f} area {(mask>0).sum()} {time.time()-t0:.0f}s", flush=True)
print(f"done {len(rows)} frames in {time.time()-t0:.0f}s", flush=True)
