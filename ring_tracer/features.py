"""Edge map, ring orientation field and core band for a scanned core.

Outputs (saved as features.npz):
  E      edge strength in [0, 1]: sharp latewood -> earlywood jumps across the ring direction
  slope  dx/dy of the ring direction at every pixel (tangent of the boundary)
  top, bot  per-column y of the core's upper and lower edge (excludes the mount rails)
"""
import sys
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, median_filter

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from load import load_image, load_boundaries, load_pith, D

XMAX = None                       # set in main(): annotated part only (+ a margin)
SIG_EDGE, SIG_TENSOR = 2.0, 35.0  # px


def core_band(gray):
    """Find the two dark seams between the core and the mount rails, per column block."""
    H, W = gray.shape
    step = 200
    tops, bots, xs = [], [], []
    for x0 in range(0, W, step):
        prof = gaussian_filter(gray[:, x0:x0 + step].mean(1), 1.5)
        t0, t1 = int(0.08 * H), int(0.33 * H)      # search windows scale with the scan height
        b0, b1 = int(0.68 * H), int(0.96 * H)
        t = t0 + int(np.argmin(prof[t0:t1]))
        b = b0 + int(np.argmin(prof[b0:b1]))
        tops.append(t); bots.append(b); xs.append(x0 + step / 2)
    tops = median_filter(np.array(tops), 7); bots = median_filter(np.array(bots), 7)
    cols = np.arange(W)
    top = np.interp(cols, xs, tops) + 38      # skip the rough, damaged top strip of the core
    bot = np.interp(cols, xs, bots) - 12
    return top, bot


def main():
    xmax = int(max(p[0] for b in load_boundaries() for p in b)) + 1500
    rgb = load_image()[:, :xmax].astype(np.float32)
    gray = rgb.mean(2)
    g = gaussian_filter(gray, SIG_EDGE)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)

    # structure tensor -> dominant gradient direction = normal to the rings
    jxx = gaussian_filter(gx * gx, SIG_TENSOR)
    jyy = gaussian_filter(gy * gy, SIG_TENSOR)
    jxy = gaussian_filter(gx * gy, SIG_TENSOR)
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    nx, ny = np.cos(theta), np.sin(theta)
    # orient every normal toward the pith (from the .pos header); fall back to +x
    pith = load_pith()
    H, W = gray.shape
    if pith is not None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        toward = (pith[0] - xx) * nx + (pith[1] - yy) * ny
        del yy, xx
    else:  # no pith in the header: assume it lies in the direction the points were clicked
        B0 = load_boundaries()
        toward = nx * np.sign(B0[-1][0][0] - B0[0][0][0] or 1.0)
    flip = toward < 0
    nx[flip] *= -1; ny[flip] *= -1
    slope = (-ny / np.clip(nx, 1e-3, None)).astype(np.float32)   # boundary tangent dx/dy

    d = gx * nx + gy * ny                           # derivative across the rings

    top, bot = core_band(gray)
    np.savez_compressed(D + "features_raw.npz", d=d.astype(np.float32), slope=slope,
                        top=top, bot=bot)
    # which sign marks a ring boundary? read it off the annotated points
    vals = []
    for b in load_boundaries():
        for (x, y) in b:
            xi, yi = int(round(x)), int(round(y))
            win = d[max(yi - 2, 0):yi + 3, max(xi - 12, 0):xi + 13]
            if win.size:
                vals.append(win.flat[np.argmax(np.abs(win))])
    vals = np.array(vals)
    sign = -1.0 if (vals < 0).sum() >= (vals > 0).sum() else 1.0
    print(f"boundary edge sign: {'negative' if sign < 0 else 'positive'} "
          f"({(vals < 0).sum()} vs {(vals > 0).sum()} points)")
    E = np.clip(sign * d, 0, None)
    E = np.clip(E / np.percentile(E, 99.5), 0, 1).astype(np.float32)
    np.save(D + "E.npy", E)



if __name__ == "__main__":
    main()
