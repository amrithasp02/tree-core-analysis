"""
infer_resin_ducts.py

Runs resin duct segmentation inference on full tree core images.

For each image:
  1. Slide a window along the X axis (same logic as data_cropping_script.py),
     keeping the full image height in each crop.
  2. Run every crop through the trained MMSegmentation model in batches.
  3. Accumulate soft duct probabilities from overlapping crops (averaged).
  4. Render a single stitched overlay (semi-transparent duct highlight) and
     write it to the output directory.

Usage:
    python infer_resin_ducts.py \
        --config      /path/to/segformer_config.py \
        --checkpoint  /path/to/iter_10000.pth \
        --input_dir   /path/to/input \
        --output_dir  /path/to/output \
        [--crop_w 1024] \
        [--stride_x 512] \
        [--batch_size 4] \
        [--alpha 0.45] \
        [--duct_color 255 80 0]
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from mmseg.apis import init_model, inference_model


def parse_args():
    p = argparse.ArgumentParser(description='Resin duct inference with crop-and-stitch.')
    p.add_argument('--config',     required=True, type=Path, help='MMSeg config .py file')
    p.add_argument('--checkpoint', required=True, type=Path, help='Model checkpoint .pth file')
    p.add_argument('--input_dir',  required=True, type=Path, help='Directory with input core images')
    p.add_argument('--output_dir', required=True, type=Path, help='Directory to write overlay images')
    p.add_argument('--crop_w',     type=int,   default=1024, help='Crop width in pixels (default 1024)')
    p.add_argument('--stride_x',   type=int,   default=512,  help='Horizontal stride (default 512)')
    p.add_argument('--batch_size', type=int,   default=4,    help='Crops per inference batch (default 4)')
    p.add_argument('--alpha',      type=float, default=0.45, help='Overlay transparency (default 0.45)')
    p.add_argument('--duct_color', type=int,   nargs=3, default=[255, 80, 0],
                   metavar=('R', 'G', 'B'), help='Duct highlight colour (default 255 80 0)')
    p.add_argument('--mask_dir',  type=Path, default=None,
                   help='If given, save binary duct masks as .npy files here')
    return p.parse_args()


def crop_x_positions(img_w: int, crop_w: int, stride_x: int):
    """Yield left-edge x offsets for each sliding-window crop."""
    x = 0
    while x + crop_w <= img_w:
        yield x
        x += stride_x


def stitch_predictions(img_np: np.ndarray, model, crop_w: int, stride_x: int,
                       batch_size: int) -> np.ndarray:
    """
    Slide over the image, run inference on each crop, average overlapping soft
    probabilities, and return a binary duct mask (H x W, uint8, values 0/1).

    Crops are passed to mmseg as BGR numpy arrays (the model preprocessor does
    bgr_to_rgb=True internally). Soft probabilities from the logits are
    accumulated per pixel and averaged across all crops that cover it, giving
    clean boundaries at overlap seams.
    """
    img_h, img_w = img_np.shape[:2]
    prob_accum  = np.zeros((img_h, img_w), dtype=np.float32)
    count_accum = np.zeros((img_h, img_w), dtype=np.float32)

    xs = list(crop_x_positions(img_w, crop_w, stride_x))
    # mmseg expects BGR
    img_bgr = img_np[:, :, ::-1].copy()

    for batch_start in range(0, len(xs), batch_size):
        batch_xs   = xs[batch_start:batch_start + batch_size]
        crops_bgr  = [img_bgr[:, x:x + crop_w] for x in batch_xs]

        n_done = batch_start + len(batch_xs)
        print(f'  crops {n_done}/{len(xs)}', end='\r', flush=True)

        results = inference_model(model, crops_bgr)
        if not isinstance(results, list):
            results = [results]

        for x, result in zip(batch_xs, results):
            # logits: (n_classes, H, crop_w)
            logits = result.seg_logits.data.cpu().numpy()
            # Numerically stable softmax over the class axis
            shifted = logits - logits.max(axis=0, keepdims=True)
            exp     = np.exp(shifted)
            probs   = exp / exp.sum(axis=0, keepdims=True)
            duct_prob = probs[1]  # class 1 = resin duct

            prob_accum[:, x:x + crop_w]  += duct_prob
            count_accum[:, x:x + crop_w] += 1.0

    print(f'  crops {len(xs)}/{len(xs)} — done')

    covered  = count_accum > 0
    avg_prob = np.where(covered, prob_accum / np.maximum(count_accum, 1.0), 0.0)
    return (avg_prob >= 0.5).astype(np.uint8)


def render_overlay(img_rgb: np.ndarray, duct_mask: np.ndarray,
                   duct_color: tuple, alpha: float) -> np.ndarray:
    """
    Blend a semi-transparent duct highlight onto the RGB image, plus a crisp
    1-px outline around each detected duct region.
    """
    overlay = img_rgb.copy().astype(np.float32)
    is_duct = duct_mask == 1

    for c, col in enumerate(duct_color):
        overlay[is_duct, c] = alpha * col + (1.0 - alpha) * overlay[is_duct, c]

    result = np.clip(overlay, 0, 255).astype(np.uint8)

    # 1-px outline on the background side of each duct boundary
    mask_pil = Image.fromarray((is_duct * 255).astype(np.uint8), mode='L')
    dilated  = mask_pil.filter(ImageFilter.MaxFilter(3))
    border   = np.array(dilated, dtype=np.uint8) // 255
    border  &= (~is_duct).astype(np.uint8)
    for c, col in enumerate(duct_color):
        result[border == 1, c] = col

    return result


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    extensions = ('*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png')
    image_files = []
    for ext in extensions:
        image_files.extend(sorted(args.input_dir.glob(ext)))

    if not image_files:
        print(f'No images found in {args.input_dir}')
        return

    if args.mask_dir is not None:
        args.mask_dir.mkdir(parents=True, exist_ok=True)

    print(f'Found {len(image_files)} image(s).')
    print(f'Loading model ...')
    model = init_model(str(args.config), str(args.checkpoint))
    duct_color = tuple(args.duct_color)

    for img_path in image_files:
        print(f'\nProcessing {img_path.name}')

        pil_img = Image.open(img_path)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        img_np = np.array(pil_img, dtype=np.uint8)
        img_h, img_w = img_np.shape[:2]
        print(f'  size: {img_w} x {img_h} px')

        n_crops = len(list(crop_x_positions(img_w, args.crop_w, args.stride_x)))
        print(f'  {n_crops} crop(s) at crop_w={args.crop_w}, stride_x={args.stride_x}')

        mask    = stitch_predictions(img_np, model, args.crop_w, args.stride_x, args.batch_size)

        if args.mask_dir is not None:
            mask_path = args.mask_dir / (img_path.stem + '_duct_mask.npy')
            np.save(mask_path, mask)
            print(f'  mask  → {mask_path}')

        overlay = render_overlay(img_np, mask, duct_color, args.alpha)

        out_path = args.output_dir / (img_path.stem + '_overlay.png')
        Image.fromarray(overlay, mode='RGB').save(out_path)
        print(f'  saved → {out_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
