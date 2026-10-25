# Ring-boundary tracer (CooRecorder points → full boundary labels)

Turns the single CooRecorder click on each ring boundary into a full traced line across
the core, and writes a training label for the ring head.

## Run

```
pip install numpy scipy opencv-python-headless pillow
python run_all.py PLN0512A_2400dpi.tif PLN0512A_2400dpi.pos out/ PLN0512A
```

About 1 minute per core on a laptop CPU. Outputs in `out/`:

| file | contents |
|---|---|
| `NAME_ring_label.png` | training label, same size as the scan: **1** = boundary (5 px wide), **0** = no boundary, **255** = ignore (MMSegmentation's default `ignore_index`) |
| `NAME_ring_traces.json` | each boundary's CooRecorder points and traced line (pixel coordinates), with status `ok` / `short` / `missing` |
| `NAME_ring_label_review.png` | overview to check by eye: green = boundary, grey = ignored, red = CooRecorder points |

## How it works

1. **features.py**: edge map of sharp latewood→earlywood jumps measured *across* the local
   ring direction (structure tensor), oriented toward the pith from the `.pos` header; finds
   the core's top/bottom edges and skips the damaged top strip.
2. **trace.py**: dynamic-programming path through every annotated point of each boundary,
   preferring strong edges and the local ring direction. Near-vertical boundaries are traced
   row by row, tilted/arc-shaped ones column by column. Two-point CooRecorder "jumps" fix the
   boundary's direction.
3. **postprocess.py**: rings never merge, so a trace is cut where it touches another boundary
   or runs > 25 px with no visible edge.
4. **make_labels.py**: builds the label. Ignored: outside the core, outside the annotated
   stretch, and the unknown continuation of any trace that stops before the core edge.

## Settings worth knowing (top of each file)

* `features.py`: `SIG_TENSOR` (smoothing of the ring direction), core-edge margins in `core_band`.
* `trace.py`: `LAMBDA` (how strictly traces follow the ring direction), `KMAX`, `EXIT`.
* `postprocess.py`: `GAP` (max no-edge stretch before a trace is cut).
* `make_labels.py`: `LINE_W` (boundary width in the label), `IGN_W`, `EXTEND`.
