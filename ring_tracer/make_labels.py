"""Turn cleaned traces into a training label image for the ring head.

label value  meaning
     1       ring boundary (traced line, 5 px wide)
     0       known "no boundary" (inside the core, in the annotated stretch)
   255       ignore: outside the core / damaged top strip, outside the annotated stretch,
             and the unknown continuation of any trace that stops before the core edge
Also writes the traces as JSON (full-resolution pixel coordinates) and a review image.
"""
import sys, json
import numpy as np
import cv2

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from load import load_image, load_boundaries, D

NAME = __import__("os").environ.get("RING_NAME", "core")
LINE_W, IGN_W, EXTEND = 5, 50, 900   # px


def main():
    rgb = load_image()
    H, Wfull = rgb.shape[:2]
    B = load_boundaries()
    f = np.load(D + "features_raw.npz"); top, bot = f["top"], f["bot"]
    W = len(top)
    tr = np.load(D + "traces_clean.npz")
    T = [tr[f"arr_{i}"] for i in range(len(tr.files))]
    status = list(np.load(D + "trace_status.npy"))

    label = np.full((H, Wfull), 255, np.uint8)
    yy = np.arange(H)[:, None]
    inband = (yy >= top[None, :] + 8) & (yy <= bot[None, :] - 8)
    xs_pts = [p[0] for b in B for p in b]
    x_lo, x_hi = int(min(xs_pts)) - 100, int(max(xs_pts)) + 50       # annotated stretch
    known = inband.copy(); known[:, :max(x_lo, 0)] = False; known[:, x_hi:] = False
    label[:, :W][known] = 0

    ignore = np.zeros((H, Wfull), np.uint8)
    for i, t in enumerate(T):
        if status[i] == "missing" or len(t) < 2:
            for x, y in B[i]:                                # nothing usable: ignore around the click
                cv2.circle(ignore, (int(x), int(y)), 120, 1, -1)
            continue
        for end, nb in ((t[0], t[min(30, len(t) - 1)]), (t[-1], t[max(len(t) - 31, 0)])):
            x, y = int(end[0]), int(end[1])
            if y <= top[x] + 14 or y >= bot[x] - 14:
                continue                                     # reached the core edge: complete
            d = end.astype(float) - nb.astype(float)
            d /= (np.linalg.norm(d) + 1e-6)
            far = (int(x + d[0] * EXTEND), int(y + d[1] * EXTEND))
            cv2.line(ignore, (x, y), far, 1, IGN_W)          # unknown continuation of this boundary
    label[ignore > 0] = 255

    bound = np.zeros((H, Wfull), np.uint8)
    for t in T:
        if len(t) > 1:
            cv2.polylines(bound, [t.reshape(-1, 1, 2).astype(np.int32)], False, 1, LINE_W)
    label[bound > 0] = 1

    cv2.imwrite(D + f"{NAME}_ring_label.png", label)
    traces = [{"boundary": i, "status": status[i],
               "cooRecorder_points_px": [[round(x, 1), round(y, 1)] for x, y in B[i]],
               "trace_xy_px": t.astype(int).tolist()} for i, t in enumerate(T)]
    json.dump({"image": __import__("os").path.basename(__import__("load").IMG), "dpi": round(__import__("load").PX_PER_MM * 25.4), "label_values":
               {"boundary": 1, "background": 0, "ignore": 255}, "boundaries": traces},
              open(D + f"{NAME}_ring_traces.json", "w"))

    # review image: boundary green, ignore greyed out, CooRecorder points red
    vis = rgb.copy().astype(np.float32)
    ig = label == 255
    vis[ig] = vis[ig] * 0.45 + np.array([90, 90, 90]) * 0.55
    vis[label == 1] = (20, 160, 60)
    vis = vis.astype(np.uint8)
    for b in B:
        for x, y in b:
            cv2.circle(vis, (int(x), int(y)), 8, (220, 30, 30), -1)
    vis = vis[:, :W]
    q = (W + 3) // 4
    seg = [vis[:, i * q:(i + 1) * q] for i in range(4)]
    seg = [np.pad(s, ((0, 0), (0, q - s.shape[1]), (0, 0)), constant_values=255) for s in seg]
    seg = [cv2.resize(s, (s.shape[1] * 2 // 5, s.shape[0] * 2 // 5), interpolation=cv2.INTER_AREA) for s in seg]
    grid = np.vstack([np.pad(s, ((0, 8), (0, 0), (0, 0)), constant_values=255) for s in seg])
    cv2.imwrite(D + f"{NAME}_ring_label_review.png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    n = {v: int((label == v).sum()) for v in (0, 1, 255)}
    print("pixels  boundary:", n[1], " background:", n[0], " ignore:", n[255])


if __name__ == "__main__":
    main()
