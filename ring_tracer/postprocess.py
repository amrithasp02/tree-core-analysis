"""Clean traces with two rules from ring biology, then score each one.

Walking outward from the stretch between a boundary's annotated points, a trace is cut
  (a) where it touches another boundary (rings never merge or cross), or
  (b) where it runs > GAP px with no visible edge (it has lost the boundary).
Each cleaned trace gets a status: ok / short (covers < 60% of the core height) / missing.
"""
import sys
import numpy as np
import cv2

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from load import load_boundaries, D

GAP, LOW = 25, 0.15


def anchor_span(t, pts):
    d = [int(np.argmin((t[:, 0] - x) ** 2 + (t[:, 1] - y) ** 2)) for x, y in pts]
    return min(d), max(d)


def main():
    B = load_boundaries()
    E = np.load(D + "E.npy")
    f = np.load(D + "features_raw.npz"); top, bot = f["top"], f["bot"]
    tr = np.load(D + "traces.npz")
    T = [tr[f"arr_{i}"] for i in range(len(tr.files))]
    H, W = E.shape

    # index image of all raw traces (2 px wide)
    owner = np.full((H, W), -1, np.int32)
    for i, t in enumerate(T):
        m = np.zeros((H, W), np.uint8)
        cv2.polylines(m, [t.reshape(-1, 1, 2).astype(np.int32)], False, 1, 3)
        owner[(m > 0) & (owner == -1)] = i

    out, status = [], []
    for i, t in enumerate(T):
        if len(t) < 2:
            out.append(t); status.append("missing"); continue
        a0, a1 = anchor_span(t, B[i])
        lo, hi = a0, a1
        for direction in (-1, +1):
            k = a0 if direction < 0 else a1
            run = 0
            while 0 <= k + direction < len(t):
                x, y = t[k + direction]
                o = owner[y, x]
                if o != -1 and o != i:          # touched another boundary
                    break
                run = run + 1 if E[y, x] < LOW else 0
                if run > GAP:                   # lost the edge: cut back to where the gap began
                    k -= direction * (run - 1)
                    break
                k += direction
            if direction < 0: lo = k
            else: hi = k
        c = t[lo:hi + 1]
        out.append(c)
        height = float(np.median(bot[c[:, 0]] - top[c[:, 0]])) if len(c) else 1.0
        span_y = (c[:, 1].max() - c[:, 1].min()) if len(c) > 1 else 0
        span_x = (c[:, 0].max() - c[:, 0].min()) if len(c) > 1 else 0
        # a boundary should cross (most of) the core; near-horizontal ones exit through the side,
        # so measure coverage as the larger of vertical extent and path length / 4
        cover = max(span_y / height, len(c) / (4 * height))
        status.append("ok" if cover >= 0.6 else ("short" if len(c) > 30 else "missing"))
    np.savez_compressed(D + "traces_clean.npz", *out)
    np.save(D + "trace_status.npy", np.array(status))
    print({s: status.count(s) for s in set(status)})
    for i, s in enumerate(status):
        if s != "ok":
            print(f"  boundary {i}: {s} ({len(out[i])} px)")


if __name__ == "__main__":
    main()
