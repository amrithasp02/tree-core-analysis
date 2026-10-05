"""
prepare_resin_duct_dataset.py

Prepares a semantic segmentation dataset from tree core images and resin duct CSV annotations.
Output format mirrors Cityscapes: image crops in leftImg8bit/ and mask crops in gtFine/.

Cityscapes-style layout:
    dataset_root/
    ├── leftImg8bit/
    │   ├── train/
    │   │   └── <core_id>/
    │   │       └── <core_id>_<crop_idx>_leftImg8bit.png
    │   └── val/
    │       └── ...
    ├── gtFine/
    │   ├── train/
    │   │   └── <core_id>/
    │   │       └── <core_id>_<crop_idx>_gtFine_labelIds.png
    │   └── val/
    │       └── ...
    └── vis/
        └── <core_id>/
            └── <core_id>_<crop_idx>_overlay.png   ← RGB image + semi-transparent duct overlay

Mask convention (single class):
    0   → background (class 0)
    1   → resin duct (class 1)
    255 → ignore (unused pixels, e.g. padding)

CSV conventions handled:
    - Files with "inches" in the name have X, Y, Area columns in INCHES.
      They are converted to pixels using DPI extracted from the filename
      (looks for a token like "2400dpi"; default 2400 if not found).
    - Files without "inches" have X, Y, Area already in PIXELS.
    - Rows where Area == 0 are ring-boundary markers and are skipped.
    - The radius of each duct is derived from its area: r = sqrt(Area / pi).

Usage:
    python prepare_resin_duct_dataset.py \
        --images_dir  /path/to/full_core_images \
        --annots_dir  /path/to/full_core_annotations \
        --output_dir  /path/to/dataset_root \
        --crop_w 1024 \
        --stride_x 512 \
        --val_fraction 0.2 \
        --min_duct_pixels 1 \
        --vis_samples 5 \
        --seed 42
"""

import argparse
import csv
import math
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_dpi(name: str, default: int = 2400) -> int:
    """Pull DPI from a filename token like '2400dpi'. Falls back to default."""
    m = re.search(r'(\d+)dpi', name, re.IGNORECASE)
    return int(m.group(1)) if m else default


def parse_float(val: str):
    """Parse scientific-notation strings like '6.944E-6' as well as plain floats."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def load_annotations(csv_path: Path):
    """
    Returns a list of dicts, each with keys:
        x_px, y_px, radius_px  (all in pixels, floats)

    Handles both pixel and inch CSV files.
    """
    is_inches = 'inches' in csv_path.stem.lower()
    dpi = extract_dpi(csv_path.stem) if is_inches else None

    ducts = []
    with open(csv_path, newline='', encoding='utf-8-sig') as fh:
        # The files start with a leading comma, making the first column unnamed.
        reader = csv.DictReader(fh)
        for row in reader:
            # Clean whitespace from keys
            row = {k.strip(): v.strip() for k, v in row.items()}

            area_raw = parse_float(row.get('Area', '0'))
            x_raw = parse_float(row.get('X'))
            y_raw = parse_float(row.get('Y'))

            if area_raw is None or x_raw is None or y_raw is None:
                continue
            if area_raw == 0:
                # Ring-boundary marker — skip
                continue

            if is_inches:
                # Convert inch measurements to pixels
                area_px2 = area_raw * (dpi ** 2)   # in² → px²
                x_px = x_raw * dpi
                y_px = y_raw * dpi
            else:
                area_px2 = area_raw
                x_px = x_raw
                y_px = y_raw

            radius_px = math.sqrt(area_px2 / math.pi)
            ducts.append({'x_px': x_px, 'y_px': y_px, 'radius_px': radius_px})

    return ducts


def match_csv_to_image(csv_path: Path, image_files: list[Path]) -> Path | None:
    """
    Find the image file whose stem best matches a CSV stem.

    Strategy: extract the core ID (first alpha-numeric token before any
    resolution/date suffixes) and look for a unique image that contains it.
    This is robust to minor filename differences between annotation and image.
    """
    # Extract what looks like a sample ID from the CSV stem
    # e.g. "Results_PCN0903A_bsl_20260212_inches" → "PCN0903A"
    # e.g. "Results_PLS0105A_jec_20260216" → "PLS0105A"
    m = re.search(r'[A-Z]{2,3}\d{4}[A-Z]?', csv_path.stem, re.IGNORECASE)
    if not m:
        return None
    core_id = m.group(0)

    candidates = [p for p in image_files if core_id.lower() in p.stem.lower()]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        print(f"  [WARN] Multiple image candidates for {csv_path.name}: "
              f"{[p.name for p in candidates]} — using first.")
        return candidates[0]
    return None


def render_mask(ducts: list[dict], img_w: int, img_h: int) -> np.ndarray:
    """
    Render a full-resolution uint8 mask for the whole core image.
        0 = background (class 0)
        1 = resin duct (class 1)
      255 = ignore
    """
    mask = Image.new('L', (img_w, img_h), color=0)  # background = 0
    draw = ImageDraw.Draw(mask)
    for d in ducts:
        r = d['radius_px']
        cx, cy = d['x_px'], d['y_px']
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(bbox, fill=1)                   # resin duct = 1
    return np.array(mask, dtype=np.uint8)


def render_overlay(img_crop: np.ndarray,
                   msk_crop: np.ndarray,
                   duct_color: tuple = (255, 80, 0),
                   alpha: float = 0.45) -> np.ndarray:
    """
    Blend a semi-transparent duct highlight onto the RGB image crop.

    Duct pixels (mask == 1) are tinted with `duct_color` at opacity `alpha`.
    A 1-px outline is drawn around each duct region for extra visibility.

    Returns a uint8 RGB numpy array.
    """
    overlay = img_crop.copy().astype(np.float32)
    duct_mask = msk_crop == 1                                    # True where duct

    # Semi-transparent fill
    for c, col in enumerate(duct_color):
        overlay[duct_mask, c] = (
            alpha * col + (1.0 - alpha) * overlay[duct_mask, c]
        )

    result = np.clip(overlay, 0, 255).astype(np.uint8)

    # Crisp 1-px outline: pixels that border background→duct transition
    from PIL import ImageFilter
    mask_pil   = Image.fromarray((duct_mask * 255).astype(np.uint8), mode='L')
    dilated    = mask_pil.filter(ImageFilter.MaxFilter(3))
    border     = np.array(dilated, dtype=np.uint8) // 255
    border    &= (~duct_mask).astype(np.uint8)                   # only on bg side
    for c, col in enumerate(duct_color):
        result[border == 1, c] = col

    return result


def make_crops(image: np.ndarray,
               mask: np.ndarray,
               crop_w: int,
               stride_x: int,
               min_duct_pixels: int):
    """
    Slide a window along the length (X axis) of the image only.
    Each crop spans the full image height so the short dimension is never altered.

    Yields (img_crop_np, mask_crop_np).
    """
    img_h, img_w = image.shape[:2]
    x = 0
    while x + crop_w <= img_w:
        img_crop = image[:, x:x + crop_w]
        msk_crop = mask[:, x:x + crop_w]

        duct_px = int(np.sum(msk_crop == 1))
        if duct_px >= min_duct_pixels:
            yield img_crop, msk_crop

        x += stride_x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Prepare resin duct segmentation dataset.')
    p.add_argument('--images_dir',  required=True, help='Directory with full core .tif images')
    p.add_argument('--annots_dir',  required=True, help='Directory with Results_*.csv annotation files')
    p.add_argument('--output_dir',  required=True, help='Root output directory for the dataset')
    p.add_argument('--crop_w',      type=int, default=1024, help='Crop width in pixels (default 1024); height is always the full image height')
    p.add_argument('--stride_x',    type=int, default=512,  help='Horizontal stride (default 512, i.e. 50% overlap)')
    p.add_argument('--val_fraction',type=float, default=0.2, help='Fraction of crops for validation (default 0.2)')
    p.add_argument('--min_duct_pixels', type=int, default=1,
                   help='Minimum number of duct pixels required to keep a crop (default 1)')
    p.add_argument('--vis_samples',  type=int, default=5,
                   help='Number of overlay visualisation crops to save per core (default 5, 0 = disable)')
    p.add_argument('--seed',        type=int, default=42, help='Random seed for train/val split')
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    images_dir = Path(args.images_dir)
    annots_dir = Path(args.annots_dir)
    output_dir = Path(args.output_dir)

    # Gather files
    image_files = sorted(images_dir.glob('*.tif')) + sorted(images_dir.glob('*.tiff'))
    csv_files   = sorted(annots_dir.glob('Results_*.csv'))

    if not image_files:
        sys.exit(f'No .tif images found in {images_dir}')
    if not csv_files:
        sys.exit(f'No Results_*.csv files found in {annots_dir}')

    print(f'Found {len(image_files)} image(s) and {len(csv_files)} CSV(s).')

    # Create output directory tree
    for split in ('train', 'val'):
        (output_dir / 'leftImg8bit' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'gtFine'      / split).mkdir(parents=True, exist_ok=True)
    if args.vis_samples > 0:
        (output_dir / 'vis').mkdir(parents=True, exist_ok=True)

    total_saved = {'train': 0, 'val': 0}
    cores_processed = 0

    for csv_path in csv_files:
        print(f'\nProcessing {csv_path.name} …')

        img_path = match_csv_to_image(csv_path, image_files)
        if img_path is None:
            print(f'  [SKIP] No matching image found for {csv_path.name}')
            continue

        print(f'  Matched image: {img_path.name}')

        # Load image
        try:
            pil_img = Image.open(img_path)
            # Convert to RGB (handles grayscale, RGBA, palette etc.)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            img_np = np.array(pil_img, dtype=np.uint8)
        except Exception as e:
            print(f'  [SKIP] Could not open image: {e}')
            continue

        img_h, img_w = img_np.shape[:2]
        print(f'  Image size: {img_w} x {img_h}')

        # Load and render annotations
        ducts = load_annotations(csv_path)
        print(f'  Loaded {len(ducts)} resin duct annotations')
        if not ducts:
            print('  [SKIP] No valid duct annotations found.')
            continue

        mask_np = render_mask(ducts, img_w, img_h)
        duct_total = int(np.sum(mask_np == 1))
        print(f'  Duct pixels in full mask: {duct_total}')

        # Core ID for folder naming
        m = re.search(r'[A-Z]{2,3}\d{4}[A-Z]?', csv_path.stem, re.IGNORECASE)
        core_id = m.group(0) if m else csv_path.stem

        # Collect all valid crops
        crops = list(make_crops(
            img_np, mask_np,
            args.crop_w,
            args.stride_x,
            args.min_duct_pixels
        ))
        print(f'  Generated {len(crops)} valid crop(s)')

        if not crops:
            print('  [SKIP] No crops with sufficient duct pixels.')
            continue

        # Shuffle and split train/val
        random.shuffle(crops)
        n_val   = max(1, int(len(crops) * args.val_fraction))
        n_train = len(crops) - n_val
        splits  = ['val'] * n_val + ['train'] * n_train

        # Pick which crop indices get a visualisation
        vis_indices: set[int] = set()
        if args.vis_samples > 0:
            n_vis = min(args.vis_samples, len(crops))
            vis_indices = set(random.sample(range(len(crops)), n_vis))
            vis_dir = output_dir / 'vis' / core_id
            vis_dir.mkdir(parents=True, exist_ok=True)

        for idx, ((img_crop, msk_crop), split) in enumerate(zip(crops, splits)):
            # Ensure per-core subdirectories exist
            img_out_dir = output_dir / 'leftImg8bit' / split / core_id
            msk_out_dir = output_dir / 'gtFine'      / split / core_id
            img_out_dir.mkdir(parents=True, exist_ok=True)
            msk_out_dir.mkdir(parents=True, exist_ok=True)

            base = f'{core_id}_{idx:04d}'

            img_out = img_out_dir / f'{base}_leftImg8bit.png'
            msk_out = msk_out_dir / f'{base}_gtFine_labelIds.png'

            Image.fromarray(img_crop, mode='RGB').save(img_out)
            Image.fromarray(msk_crop, mode='L').save(msk_out)

            # Visualisation overlay
            if idx in vis_indices:
                overlay = render_overlay(img_crop, msk_crop)
                vis_out = vis_dir / f'{base}_overlay.png'
                Image.fromarray(overlay, mode='RGB').save(vis_out)

            total_saved[split] += 1

        cores_processed += 1
        n_vis_saved = len(vis_indices) if args.vis_samples > 0 else 0
        print(f'  Saved {n_train} train / {n_val} val crops for {core_id}'
              + (f' ({n_vis_saved} overlays in vis/)' if n_vis_saved else ''))

    # Summary
    print('\n' + '=' * 60)
    print(f'Done. Processed {cores_processed} core(s).')
    print(f'  Train crops : {total_saved["train"]}')
    print(f'  Val   crops : {total_saved["val"]}')
    print(f'  Output root : {output_dir}')
    print()
    print('Mask convention:')
    print('  pixel value  0 → background (class 0)')
    print('  pixel value  1 → resin duct (class 1)')
    print('  pixel value 255 → ignore index')
    print()
    print('Directory layout (Cityscapes-style):')
    print('  leftImg8bit/{train,val}/<core_id>/*_leftImg8bit.png')
    print('  gtFine/{train,val}/<core_id>/*_gtFine_labelIds.png')
    print('  vis/<core_id>/*_overlay.png   ← RGB + semi-transparent duct highlight')


if __name__ == '__main__':
    main()