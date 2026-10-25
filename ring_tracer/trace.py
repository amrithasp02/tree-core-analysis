"""Trace every annotated ring boundary across the core with dynamic programming.

For each CooRecorder boundary:
  * the path is forced through every annotated point on that boundary;
  * each step it prefers pixels with a strong edge (E) and moves in the local ring
    direction (slope field), with a small penalty for deviating from it;
  * near-vertical boundaries are traced row by row, tilted / near-horizontal ones
    column by column, so the path never has to move more than a few pixels per step;
  * the result is clipped to the core band.
Output: traces.npz (list of polylines) + summary printed.
"""
import sys
import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from load import load_boundaries, D

LAMBDA = 0.25      # weight of the "follow the ring direction" prior
KMAX = 2           # max deviation from the predicted step, px per step


def dp_segment(cost, prior, s0, s_end=None):
    """Minimum-cost path over steps 0..T-1 through states (columns of `cost`).

    cost[t, s]  : cost of visiting state s at step t (low on edges)
    prior[t, s] : expected state change from step t to t+1 when at state s
    s0          : fixed state at step 0;  s_end: optional fixed state at the last step
    """
    T, S = cost.shape
    acc = np.full(S, np.inf); acc[s0] = 0.0
    back = np.zeros((T, S), np.int32)
    idx = np.arange(S)
    for t in range(1, T):
        best = np.full(S, np.inf); arg = np.zeros(S, np.int32)
        p = prior[t - 1]
        lo, hi = int(np.floor(np.nanmin(p))) - KMAX, int(np.ceil(np.nanmax(p))) + KMAX
        for k in range(lo, hi + 1):                 # candidate move: prev state = s - k
            prev = idx - k
            ok = (prev >= 0) & (prev < S)
            pv = np.where(ok, prev, 0)
            dev = k - p[pv]
            c = np.where(ok & (np.abs(dev) <= KMAX), acc[pv] + LAMBDA * dev * dev, np.inf)
            better = c < best
            best[better] = c[better]; arg[better] = pv[better]
        acc = best + cost[t]
        back[t] = arg
    s = s_end if s_end is not None else int(np.argmin(acc))
    path = np.zeros(T, np.int32); path[-1] = s
    for t in range(T - 1, 0, -1):
        s = back[t, s]; path[t - 1] = s
    return path


def trace_boundary(pts, E, slope, top, bot, inband, fade):
    pts = sorted(pts, key=lambda p: p[1])
    xa, ya = pts[0]
    if len(pts) == 2 and abs(pts[1][1] - pts[0][1]) > 20:
        # two clicks on the same boundary give its direction directly
        s_here = (pts[1][0] - pts[0][0]) / (pts[1][1] - pts[0][1])
    else:
        # otherwise: median ring direction around the point, from the clean middle of the core
        r0 = max(int(ya) - 80, int(top[int(xa)]) + 40); r1 = min(int(ya) + 81, int(bot[int(xa)]) - 40)
        if r1 <= r0: r0, r1 = int(ya) - 5, int(ya) + 6
        s_here = float(np.median(slope[r0:r1, max(int(xa) - 40, 0):int(xa) + 41]))
    H, W = E.shape
    # inside the core: cheap on edges; near the core outline the cost fades to 0 and outside
    # it is free, so a trace exits where its boundary really leaves the core
    EXIT = 0.2   # small cost outside the core: cheaper than walking off-edge, dearer than a real boundary
    cost_full = np.where(inband, (1.0 - E) * fade + EXIT * (1 - fade), EXIT)

    if abs(s_here) <= 1.0:                     # ---- row-by-row: x as a function of y
        ylo = int(top[int(xa)]) - 2; yhi = int(bot[int(xa)]) + 2
        half = int(0.5 * (yhi - ylo) * (abs(s_here) + 0.6)) + 120
        x0 = max(int(xa) - half, 0); x1 = min(int(xa) + half, W - 1)
        C = cost_full[ylo:yhi + 1, x0:x1 + 1]
        # near-vertical boundaries are almost straight: one direction for the whole boundary,
        # taken from the clean middle of the core around the annotated point
        P = np.full(C.shape, np.clip(s_here, -2.5, 2.5), np.float32)   # dx per +1 row
        anchors = [(int(round(y)) - ylo, int(round(x)) - x0) for x, y in sorted(pts, key=lambda p: p[1])]
        xs = np.zeros(C.shape[0], np.int32)
        # top -> first anchor (run upward)
        a_t, a_s = anchors[0]
        up = dp_segment(C[a_t::-1], -P[a_t::-1], a_s)
        xs[:a_t + 1] = up[::-1]
        for (t0, s0), (t1, s1) in zip(anchors, anchors[1:]):  # anchor -> anchor
            xs[t0:t1 + 1] = dp_segment(C[t0:t1 + 1], P[t0:t1 + 1], s0, s1)
        a_t, a_s = anchors[-1]
        xs[a_t:] = dp_segment(C[a_t:], P[a_t:], a_s)
        ys = np.arange(ylo, yhi + 1); xs = xs + x0
    else:                                      # ---- column-by-column: y as a function of x
        ptsx = sorted(pts, key=lambda p: p[0])
        span = int((bot[int(xa)] - top[int(xa)]) * min(abs(s_here), 8) * 1.2) + 300
        x0 = max(int(ptsx[0][0]) - span, 0); x1 = min(int(ptsx[-1][0]) + span, W - 1)
        ylo = int(top[x0:x1 + 1].min()) - 2; yhi = int(bot[x0:x1 + 1].max()) + 2
        C = cost_full[ylo:yhi + 1, x0:x1 + 1].T                    # steps = columns
        P = (1.0 / np.where(np.abs(slope) < 1e-3, 1e-3, slope))[ylo:yhi + 1, x0:x1 + 1].T
        P = np.clip(P, -3, 3)                                      # dy per +1 column
        anchors = [(int(round(x)) - x0, int(round(y)) - ylo) for x, y in ptsx]
        ysel = np.zeros(C.shape[0], np.int32)
        a_t, a_s = anchors[0]
        left = dp_segment(C[a_t::-1], -P[a_t::-1], a_s)
        ysel[:a_t + 1] = left[::-1]
        for (t0, s0), (t1, s1) in zip(anchors, anchors[1:]):
            ysel[t0:t1 + 1] = dp_segment(C[t0:t1 + 1], P[t0:t1 + 1], s0, s1)
        a_t, a_s = anchors[-1]
        ysel[a_t:] = dp_segment(C[a_t:], P[a_t:], a_s)
        xs = np.arange(x0, x1 + 1); ys = ysel + ylo

    # keep the in-band stretch that contains the first anchor
    keep = (ys >= top[xs] + 8) & (ys <= bot[xs] - 8)   # stop once the trace reaches the core outline
    i_anchor = int(np.argmin((xs - xa) ** 2 + (ys - ya) ** 2))
    lo = i_anchor
    while lo > 0 and keep[lo - 1]: lo -= 1
    hi = i_anchor
    while hi < len(xs) - 1 and keep[hi + 1]: hi += 1
    return xs[lo:hi + 1], ys[lo:hi + 1], ("rows" if abs(s_here) <= 1.0 else "cols")


def main():
    f = np.load(D + "features_raw.npz")
    slope, top, bot = f["slope"], f["top"], f["bot"]
    E = np.load(D + "E.npy")
    H, W = E.shape
    yy = np.arange(H)[:, None]
    inband = (yy >= top[None, :]) & (yy <= bot[None, :])
    # fade edges near the core's top/bottom so traces are not pulled along the core outline
    dist_edge = np.minimum(yy - top[None, :], bot[None, :] - yy)
    fade = np.clip(dist_edge / 20.0, 0, 1).astype(np.float32)
    E = E * fade

    B = load_boundaries()
    traces, modes, mean_edge = [], [], []
    check = []                                   # jump pairs: trace from 1st point only, measure miss at 2nd
    for b in B:
        xs, ys, mode = trace_boundary(b, E, slope, top, bot, inband, fade)
        traces.append(np.stack([xs, ys], 1)); modes.append(mode)
        mean_edge.append(float(E[ys, xs].mean()))
        if len(b) == 2:
            xs1, ys1, _ = trace_boundary([b[0]], E, slope, top, bot, inband, fade)
            bx, by = b[1]
            dist = np.sqrt((xs1 - bx) ** 2 + (ys1 - by) ** 2).min()
            check.append(dist)
    np.savez_compressed(D + "traces.npz", *traces)
    np.save(D + "trace_quality.npy", np.array(mean_edge))
    print(f"traced {len(traces)} boundaries  (rows: {modes.count('rows')}, cols: {modes.count('cols')})")
    print(f"mean edge strength along traces: median {np.median(mean_edge):.2f}, min {min(mean_edge):.2f}")
    check = np.array(check)
    print(f"jump-pair check ({len(check)} pairs): trace from point 1 passes point 2 within "
          f"median {np.median(check):.1f}px, max {check.max():.1f}px; "
          f"{(check <= 5).sum()}/{len(check)} within 5 px")
    np.save(D + "pair_check.npy", check)


if __name__ == "__main__":
    main()
