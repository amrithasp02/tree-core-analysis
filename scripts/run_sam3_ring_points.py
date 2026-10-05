"""
Run SAM3 on a tree core image using ring points from a .pos file as positive prompts.
Crops the image width-wise, runs SAM3 per crop with the points that fall in it,
stitches masks back to full resolution, and saves a single overlay.
"""

import re
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageFilter

sys.path.insert(0, '/ofo-share/repos/amritha/sam3')
from sam3 import build_sam3_image_model
from sam3.model.geometry_encoders import Prompt
from sam3.model.sam3_image_processor import Sam3Processor

# ── Config ────────────────────────────────────────────────────────────────────
IMG_PATH   = '/ofo-share/repos/amritha/resin-duct-utils/tree-ring-detection/PILApilot_p01_t06_2400dpi.tif'
POS_PATH   = '/ofo-share/repos/amritha/resin-duct-utils/tree-ring-detection/PILApilot_p01_t06_2400dpi_sda_20260409.pos'
OUTPUT_DIR = '/ofo-share/repos/amritha/resin-duct-utils/tree-ring-detection/output'
BPE_PATH   = '/ofo-share/repos/amritha/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz'

CROP_W               = 1024 #2048   # width of each crop in pixels
CONFIDENCE_THRESHOLD = 0.3
DPI                  = 2400.0
MM_TO_PX             = DPI / 25.4

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Parse .pos file ───────────────────────────────────────────────────────────
def load_pos_points(pos_path, mm_to_px):
    points = []
    with open(pos_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('SCALE'):
                continue
            for x_mm, y_mm in re.findall(r'([\d.]+),([\d.]+)', line):
                points.append((float(x_mm) * mm_to_px, float(y_mm) * mm_to_px))
    return np.array(points)


# ── Cropping helpers ──────────────────────────────────────────────────────────
def crop_x_positions(img_w, crop_w):
    """Non-overlapping left-edge x offsets; last crop snapped to right edge."""
    x = 0
    while x + crop_w < img_w:
        yield x
        x += crop_w
    # Final crop: align to right edge (may overlap slightly with previous)
    if x < img_w:
        yield max(0, img_w - crop_w)


def points_for_crop(points_px, x_start, x_end):
    """
    Return points whose x falls in [x_start, x_end), with x shifted to
    crop-local coordinates.
    """
    inside = (points_px[:, 0] >= x_start) & (points_px[:, 0] < x_end)
    local = points_px[inside].copy()
    local[:, 0] -= x_start
    return local


# ── SAM3 per-crop inference ───────────────────────────────────────────────────
def run_crop(crop_img, local_pts, crop_w, crop_h, model, processor):
    """
    Run SAM3 on one crop with the given local-coordinate positive points.
    Returns a boolean mask of shape (crop_h, crop_w), or None if no points.
    """
    if len(local_pts) == 0:
        return None

    with torch.autocast('cuda', dtype=torch.bfloat16):
        state = processor.set_image(crop_img)

        dummy_text = model.backbone.forward_text(['visual'], device='cuda')
        state['backbone_out'].update(dummy_text)

        coords_norm = local_pts / np.array([crop_w, crop_h], dtype=np.float32)
        pts_tensor  = torch.tensor(coords_norm, device='cuda', dtype=torch.float32).unsqueeze(1)
        lbl_tensor  = torch.ones(len(local_pts), 1, device='cuda', dtype=torch.long)

        state['geometric_prompt'] = Prompt(
            point_embeddings=pts_tensor,
            point_labels=lbl_tensor,
        )

        state = processor._forward_grounding(state)

    n = len(state['scores'])
    if n == 0:
        return None

    # OR all detected masks together into one binary mask
    combined = np.zeros((crop_h, crop_w), dtype=bool)
    for i in range(n):
        combined |= state['masks'][i].squeeze(0).cpu().numpy().astype(bool)
    return combined


# ── Overlay renderer (same style as infer_resin_ducts.py) ─────────────────────
def render_overlay(img_rgb, mask, color=(0, 200, 255), alpha=0.45):
    overlay = img_rgb.copy().astype(np.float32)
    is_on = mask.astype(bool)
    for c, col in enumerate(color):
        overlay[is_on, c] = alpha * col + (1.0 - alpha) * overlay[is_on, c]
    result = np.clip(overlay, 0, 255).astype(np.uint8)
    # 1-px outline
    mask_pil = Image.fromarray((is_on * 255).astype(np.uint8), mode='L')
    dilated  = mask_pil.filter(ImageFilter.MaxFilter(3))
    border   = np.array(dilated, dtype=np.uint8) // 255
    border  &= (~is_on).astype(np.uint8)
    for c, col in enumerate(color):
        result[border == 1, c] = col
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    points_px = load_pos_points(POS_PATH, MM_TO_PX)
    print(f"Loaded {len(points_px)} ring points")

    image = Image.open(IMG_PATH).convert('RGB')
    img_w, img_h = image.size
    img_arr = np.array(image, dtype=np.uint8)
    print(f"Image size: {img_w} x {img_h} (w x h)")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("Loading SAM3 model...")
    model     = build_sam3_image_model(bpe_path=BPE_PATH)
    processor = Sam3Processor(model, confidence_threshold=CONFIDENCE_THRESHOLD)

    # Full-resolution combined mask
    full_mask = np.zeros((img_h, img_w), dtype=bool)

    xs      = list(crop_x_positions(img_w, CROP_W))
    n_crops = len(xs)
    print(f"Splitting into {n_crops} crop(s) of width {CROP_W}px")

    for i, x_start in enumerate(xs):
        x_end  = min(x_start + CROP_W, img_w)
        actual_crop_w = x_end - x_start

        crop_img   = image.crop((x_start, 0, x_end, img_h))
        local_pts  = points_for_crop(points_px, x_start, x_end)

        print(f"  crop {i+1}/{n_crops}  x=[{x_start}:{x_end}]  points={len(local_pts)}", end='  ')

        crop_mask = run_crop(crop_img, local_pts, actual_crop_w, img_h, model, processor)

        if crop_mask is not None:
            full_mask[:, x_start:x_end] |= crop_mask
            print(f"detections: {crop_mask.any()}")
        else:
            print("skipped (no points)")

    # ── Render and save overlay ───────────────────────────────────────────────
    overlay = render_overlay(img_arr, full_mask)

    # Draw ring points on top
    _, ax = plt.subplots(figsize=(30, 4))
    ax.imshow(overlay)
    ax.scatter(points_px[:, 0], points_px[:, 1], c='lime', s=10,
               linewidths=0.5, edgecolors='white', zorder=5)

    stem     = os.path.splitext(os.path.basename(IMG_PATH))[0]
    ax.set_title(f'{stem}  |  {n_crops} crops  |  {len(points_px)} ring points')
    ax.axis('off')
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f'{stem}_sam3_overlay.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved overlay → {out_path}")


if __name__ == '__main__':
    main()
