"""Trace ring boundaries on one scanned core from its CooRecorder .pos file.

usage:  python run_all.py IMAGE.tif POINTS.pos OUTPUT_FOLDER [NAME]

writes to OUTPUT_FOLDER:
  NAME_ring_label.png         training label: 1 = boundary, 0 = no boundary, 255 = ignore
  NAME_ring_traces.json       every traced boundary as pixel coordinates, with a status
  NAME_ring_label_review.png  overview to check by eye (green = boundary, grey = ignored)
"""
import os, sys, subprocess

here = os.path.dirname(os.path.abspath(__file__))
img, pos, out = sys.argv[1], sys.argv[2], sys.argv[3]
name = sys.argv[4] if len(sys.argv) > 4 else os.path.splitext(os.path.basename(img))[0]
os.makedirs(out, exist_ok=True)
env = dict(os.environ, RING_IMG=os.path.abspath(img), RING_POS=os.path.abspath(pos),
           RING_OUT=os.path.abspath(out), RING_NAME=name)
for step in ["features.py", "trace.py", "postprocess.py", "make_labels.py"]:
    print(f"== {step}")
    subprocess.run([sys.executable, os.path.join(here, step)], env=env, check=True)
