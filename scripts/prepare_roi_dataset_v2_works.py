#!/usr/bin/env python3
"""
prepare_roi_dataset_v2.py

Builds mmseg-data/roi-dataset-1 by combining two batches:

  roi-batch-1  — new batch: polygon masks from ImageJ .roi files
  batch-1      — old batch: circle masks from centroid + area in CSV

Output layout (no per-core subdirectory inside train/val):
    mmseg-data/roi-dataset-1/
    ├── leftImg8bit/
    │   ├── train/<core_id>_<idx>_leftImg8bit.png
    │   └── val/  <core_id>_<idx>_leftImg8bit.png
    ├── gtFine/
    │   ├── train/<core_id>_<idx>_gtFine_labelIds.png
    │   └── val/  <core_id>_<idx>_gtFine_labelIds.png
    └── vis/<core_id>_<idx>_overlay.png   (2 per core)

Mask values:
    0   → background
    1   → resin duct
    255 → ignore (unused)

Train/val split is per-core: all crops from one core land in the same split.
"""

import csv
import math
import random
import re
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ---------------------------------------------------------------------------
# Paths and parameters
# ---------------------------------------------------------------------------

ROI_BATCH_DIR  = Path('/ofo-share/repos/amritha/resin-duct-utils/raw-data/roi-batch-1')
OLD_IMAGES_DIR = Path('/ofo-share/repos/amritha/resin-duct-utils/raw-data/batch-1/full_core_images')
OLD_ANNOTS_DIR = Path('/ofo-share/repos/amritha/resin-duct-utils/raw-data/batch-1/full_core_annotations')
OUTPUT_DIR     = Path('/ofo-share/repos/amritha/resin-duct-utils/mmseg-data/roi-dataset-1')

CROP_W          = 1024
STRIDE_X        = 512
VAL_FRACTION    = 0.2
MIN_DUCT_PIXELS = 1
VIS_PER_CORE    = 2
SEED            = 42

# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image_rgb(img_path: Path) -> np.ndarray | None:
    """Load any image as uint8 RGB. Falls back to tifffile for awkward TIFFs."""
    try:
        pil_img = Image.open(img_path)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        return np.array(pil_img, dtype=np.uint8)
    except Exception:
        pass
    try:
        import tifffile
        arr = tifffile.imread(str(img_path))
        if arr.dtype != np.uint8:
            lo, hi = float(arr.min()), float(arr.max())
            arr = (((arr.astype(np.float32) - lo) / max(hi - lo, 1)) * 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[-1] == 1:
            arr = np.concatenate([arr, arr, arr], axis=-1)
        elif arr.shape[-1] > 3:
            arr = arr[..., :3]
        return arr
    except Exception as e:
        print(f'  [ERROR] Cannot load {img_path.name}: {e}')
        return None

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def extract_dpi(name: str, default: int = 2400) -> int:
    m = re.search(r'(\d+)dpi', name, re.IGNORECASE)
    return int(m.group(1)) if m else default


def parse_float(val) -> float | None:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def get_core_id(img_path: Path) -> str:
    """Derive a clean core identifier from an image filename."""
    stem = img_path.stem
    stem = re.sub(r'_\d+dpi.*$', '', stem, flags=re.IGNORECASE)
    stem = re.sub(r'_(mosaic\w*|jstitch\w*).*$', '', stem, flags=re.IGNORECASE)
    return stem

# ---------------------------------------------------------------------------
# File matching
# ---------------------------------------------------------------------------

def _find_by_tokens(stem: str, items: list, name_fn) -> list:
    """Return items whose name contains a key token extracted from stem."""
    # Standard core ID pattern: PLN0201A, PCN0903A, PMN0606A, PLS0105A …
    m = re.search(r'[A-Z]{2,3}\d{4}[A-Z]?', stem, re.IGNORECASE)
    if m:
        token = m.group(0).lower()
        hits = [p for p in items if token in name_fn(p).lower()]
        if hits:
            return hits
    # PILA-style filenames: match on p01_t08 / p01_t11 token
    m = re.search(r'(p\d+_t\d+)', stem, re.IGNORECASE)
    if m:
        token = m.group(1).lower()
        hits = [p for p in items if token in name_fn(p).lower()]
        if hits:
            return hits
    return []


def match_csv_to_image(csv_path: Path, image_files: list[Path]) -> Path | None:
    hits = _find_by_tokens(csv_path.stem, image_files, lambda p: p.stem)
    if len(hits) > 1:
        print(f'  [WARN] Multiple image matches for {csv_path.name} — using first.')
    return hits[0] if hits else None


def match_csv_to_roi_dir(csv_path: Path, roi_dirs: list[Path]) -> Path | None:
    hits = _find_by_tokens(csv_path.stem, roi_dirs, lambda p: p.name)
    if len(hits) > 1:
        print(f'  [WARN] Multiple RoiSet matches for {csv_path.name} — using first.')
    return hits[0] if hits else None

# ---------------------------------------------------------------------------
# ROI parsing (ImageJ binary .roi format)
# ---------------------------------------------------------------------------

def parse_roi(path: Path):
    """
    Parse an ImageJ binary .roi file.
    Returns (cx, cy, coords) where coords is a list of (x, y) float tuples
    in full-image pixel coordinates, or None if no coordinate data is present.
    """
    with open(path, 'rb') as fh:
        data = fh.read()
    if data[:4] != b'Iout':
        raise ValueError(f'Not a valid ImageJ ROI: {path}')
    top    = struct.unpack_from('>h', data,  8)[0]
    left   = struct.unpack_from('>h', data, 10)[0]
    bottom = struct.unpack_from('>h', data, 12)[0]
    right  = struct.unpack_from('>h', data, 14)[0]
    n      = struct.unpack_from('>H', data, 16)[0]
    cx = (left + right) / 2.0
    cy = (top  + bottom) / 2.0
    coords = None
    if n > 0 and len(data) >= 64 + n * 4:
        xs = struct.unpack_from(f'>{n}h', data, 64)
        ys = struct.unpack_from(f'>{n}h', data, 64 + n * 2)
        coords = [(float(x + left), float(y + top)) for x, y in zip(xs, ys)]
    return cx, cy, coords


def load_roi_polygons(roi_dir: Path) -> list[dict]:
    """Parse all .roi files in a directory. Returns list of polygon dicts."""
    polys = []
    for roi_file in sorted(roi_dir.glob('*.roi')):
        try:
            cx, cy, coords = parse_roi(roi_file)
            polys.append({'cx': cx, 'cy': cy, 'coords': coords})
        except Exception as e:
            print(f'  [WARN] Skipping {roi_file.name}: {e}')
    return polys

# ---------------------------------------------------------------------------
# Annotation loading (old batch: circles from CSV)
# ---------------------------------------------------------------------------

def load_annotations_csv(csv_path: Path) -> list[dict]:
    """
    Load duct annotations from a Results_*.csv file.
    Returns list of {x_px, y_px, radius_px}.
    Handles both pixel and inch coordinate files.
    """
    is_inches = 'inches' in csv_path.stem.lower()
    dpi = extract_dpi(csv_path.stem) if is_inches else None
    ducts = []
    with open(csv_path, newline='', encoding='utf-8-sig') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            area_raw = parse_float(row.get('Area', '0'))
            x_raw    = parse_float(row.get('X'))
            y_raw    = parse_float(row.get('Y'))
            if area_raw is None or x_raw is None or y_raw is None:
                continue
            if area_raw == 0:
                continue
            if is_inches:
                area_px2 = area_raw * (dpi ** 2)
                x_px = x_raw * dpi
                y_px = y_raw * dpi
            else:
                area_px2 = area_raw
                x_px = x_raw
                y_px = y_raw
            ducts.append({
                'x_px':      x_px,
                'y_px':      y_px,
                'radius_px': math.sqrt(area_px2 / math.pi),
            })
    return ducts

# ---------------------------------------------------------------------------
# Mask rendering
# ---------------------------------------------------------------------------

def render_mask_circles(ducts: list[dict], img_w: int, img_h: int) -> np.ndarray:
    """Rasterise duct annotations as filled circles (old batch)."""
    mask = Image.new('L', (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)
    for d in ducts:
        r, cx, cy = d['radius_px'], d['x_px'], d['y_px']
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=1)
    return np.array(mask, dtype=np.uint8)


def render_mask_polygons(polys: list[dict], img_w: int, img_h: int) -> np.ndarray:
    """Rasterise duct annotations as filled polygons from .roi files (new batch)."""
    mask = Image.new('L', (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)
    for p in polys:
        coords = p['coords']
        if not coords:
            cx, cy = p['cx'], p['cy']
            draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=1)
        elif len(coords) >= 3:
            draw.polygon(coords, fill=1)
        elif len(coords) == 2:
            draw.line(coords, fill=1, width=3)
        else:
            cx, cy = coords[0]
            draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=1)
    return np.array(mask, dtype=np.uint8)

# ---------------------------------------------------------------------------
# Overlay visualisation
# ---------------------------------------------------------------------------

def render_overlay(img_crop: np.ndarray, msk_crop: np.ndarray,
                   color: tuple = (255, 80, 0), alpha: float = 0.45) -> np.ndarray:
    """Blend a semi-transparent duct highlight onto the image crop."""
    overlay   = img_crop.copy().astype(np.float32)
    duct_mask = msk_crop == 1
    for c, col in enumerate(color):
        overlay[duct_mask, c] = alpha * col + (1 - alpha) * overlay[duct_mask, c]
    result = np.clip(overlay, 0, 255).astype(np.uint8)
    mask_pil = Image.fromarray((duct_mask * 255).astype(np.uint8), 'L')
    border   = np.array(mask_pil.filter(ImageFilter.MaxFilter(3)), np.uint8) // 255
    border  &= (~duct_mask).astype(np.uint8)
    for c, col in enumerate(color):
        result[border == 1, c] = col
    return result

# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------

def make_crops(image: np.ndarray, mask: np.ndarray,
               crop_w: int, stride_x: int, min_duct_pixels: int):
    """Slide a window along the X axis; each crop spans the full image height."""
    img_h, img_w = image.shape[:2]
    x = 0
    while x + crop_w <= img_w:
        img_crop = image[:, x:x + crop_w]
        msk_crop = mask[:, x:x + crop_w]
        if int((msk_crop == 1).sum()) >= min_duct_pixels:
            yield img_crop, msk_crop
        x += stride_x

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)

    for split in ('train', 'val'):
        (OUTPUT_DIR / 'leftImg8bit' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'gtFine'      / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'vis').mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Discover files
    # ------------------------------------------------------------------

    old_images = sorted(OLD_IMAGES_DIR.glob('*.tif')) + sorted(OLD_IMAGES_DIR.glob('*.tiff'))
    old_csvs   = sorted(OLD_ANNOTS_DIR.glob('Results_*.csv'))

    roi_images = sorted(ROI_BATCH_DIR.glob('*.tif')) + sorted(ROI_BATCH_DIR.glob('*.tiff'))
    roi_csvs   = sorted(ROI_BATCH_DIR.glob('Results_*.csv'))
    roi_dirs   = sorted(d for d in ROI_BATCH_DIR.iterdir()
                        if d.is_dir() and d.name.startswith('RoiSet_'))

    print(f'Old batch : {len(old_images)} image(s), {len(old_csvs)} CSV(s)')
    print(f'ROI batch : {len(roi_images)} image(s), {len(roi_csvs)} CSV(s), '
          f'{len(roi_dirs)} RoiSet dir(s)')

    # ------------------------------------------------------------------
    # Build core registry
    # ------------------------------------------------------------------

    cores = []

    for csv_path in old_csvs:
        img_path = match_csv_to_image(csv_path, old_images)
        if img_path is None:
            print(f'[SKIP] No image found for {csv_path.name}')
            continue
        annots = load_annotations_csv(csv_path)
        if not annots:
            print(f'[SKIP] No annotations in {csv_path.name}')
            continue
        cores.append({
            'core_id':  get_core_id(img_path),
            'img_path': img_path,
            'batch':    'old',
            'annots':   annots,
        })

    for csv_path in roi_csvs:
        img_path = match_csv_to_image(csv_path, roi_images)
        if img_path is None:
            print(f'[SKIP] No image found for {csv_path.name}')
            continue
        roi_dir = match_csv_to_roi_dir(csv_path, roi_dirs)
        if roi_dir is None:
            print(f'[SKIP] No RoiSet dir found for {csv_path.name}')
            continue
        polys = load_roi_polygons(roi_dir)
        if not polys:
            print(f'[SKIP] No ROI files in {roi_dir.name}')
            continue
        cores.append({
            'core_id':  get_core_id(img_path),
            'img_path': img_path,
            'batch':    'roi',
            'annots':   polys,
        })

    if not cores:
        sys.exit('No cores found — check input paths.')

    n_old = sum(1 for c in cores if c['batch'] == 'old')
    n_roi = sum(1 for c in cores if c['batch'] == 'roi')
    print(f'\nTotal cores: {len(cores)}  (old={n_old}, roi={n_roi})')

    # ------------------------------------------------------------------
    # Per-core train/val split
    # ------------------------------------------------------------------

    random.shuffle(cores)
    n_val = max(1, round(len(cores) * VAL_FRACTION))
    for i, core in enumerate(cores):
        core['split'] = 'val' if i < n_val else 'train'

    print(f'Val  cores ({n_val}):           {[c["core_id"] for c in cores if c["split"]=="val"]}')
    print(f'Train cores ({len(cores)-n_val}): {[c["core_id"] for c in cores if c["split"]=="train"]}')

    # ------------------------------------------------------------------
    # Process each core
    # ------------------------------------------------------------------

    totals = {'train': 0, 'val': 0}

    for core in sorted(cores, key=lambda c: c['core_id']):
        core_id  = core['core_id']
        img_path = core['img_path']
        batch    = core['batch']
        split    = core['split']

        print(f'\n[{batch}] {core_id} → {split}')

        img_np = load_image_rgb(img_path)
        if img_np is None:
            continue
        img_h, img_w = img_np.shape[:2]
        print(f'  {img_w} × {img_h} px')

        if batch == 'old':
            mask_np = render_mask_circles(core['annots'], img_w, img_h)
        else:
            mask_np = render_mask_polygons(core['annots'], img_w, img_h)

        print(f'  Duct pixels in full mask: {int((mask_np == 1).sum())}')

        crops = list(make_crops(img_np, mask_np, CROP_W, STRIDE_X, MIN_DUCT_PIXELS))
        print(f'  Valid crops: {len(crops)}')
        if not crops:
            print('  [SKIP] No valid crops.')
            continue

        vis_set = set(random.sample(range(len(crops)), min(VIS_PER_CORE, len(crops))))

        img_out = OUTPUT_DIR / 'leftImg8bit' / split
        msk_out = OUTPUT_DIR / 'gtFine'      / split
        vis_out = OUTPUT_DIR / 'vis'

        for idx, (ic, mc) in enumerate(crops):
            base = f'{core_id}_{idx:04d}'
            Image.fromarray(ic, 'RGB').save(img_out / f'{base}_leftImg8bit.png')
            Image.fromarray(mc, 'L')  .save(msk_out / f'{base}_gtFine_labelIds.png')
            if idx in vis_set:
                Image.fromarray(render_overlay(ic, mc), 'RGB').save(
                    vis_out / f'{base}_overlay.png')

        totals[split] += len(crops)
        print(f'  Saved {len(crops)} crops ({len(vis_set)} vis overlays).')

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print('\n' + '=' * 60)
    print('Done.')
    print(f'  Train crops : {totals["train"]}')
    print(f'  Val   crops : {totals["val"]}')
    print(f'  Output      : {OUTPUT_DIR}')
    print()
    print('Mask values:  0=background  1=resin-duct  255=ignore')
    print('Layout:')
    print('  leftImg8bit/{train,val}/<core>_<idx>_leftImg8bit.png')
    print('  gtFine/{train,val}/<core>_<idx>_gtFine_labelIds.png')
    print('  vis/<core>_<idx>_overlay.png')


if __name__ == '__main__':
    main()