"""Shared helpers: load the core image and parse the CooRecorder .pos file."""
import os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
# set by run_all.py; default = the folder this file is in
D = os.environ.get("RING_OUT", os.path.dirname(os.path.abspath(__file__))) + os.sep
IMG = os.environ.get("RING_IMG", D + "core.tif")
POS = os.environ.get("RING_POS", D + "core.pos")


def _dpi():
    for line in open(POS):
        if line.startswith("#DPI"):
            return float(line.split()[1])
    return 2400.0


PX_PER_MM = _dpi() / 25.4


def load_image():
    return np.asarray(Image.open(IMG).convert("RGB"))


def load_boundaries():
    """Return a list of boundaries; each is a list of (x, y) pixel points.

    In a .pos file each data line is one ring boundary. A line with two
    coordinate pairs is a CooRecorder 'jump': both points lie on the same
    boundary, at two different places.
    """
    out = []
    for line in open(POS):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("SCALE"):
            continue
        pts = []
        for tok in line.split():
            x, y = map(float, tok.split(","))
            pts.append((x * PX_PER_MM, y * PX_PER_MM))
        out.append(pts)
    return out


def load_pith():
    """Pith position (px) from the .pos header, e.g. '#C PithCoordinates=194.236,52.948;'."""
    for line in open(POS):
        if "PithCoordinates=" in line:
            v = line.split("PithCoordinates=")[1].split(";")[0]
            x, y = map(float, v.split(","))
            return x * PX_PER_MM, y * PX_PER_MM
    return None
