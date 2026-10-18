"""Self-check for the box matching (evaluate.py) and the IoU tracker (run_boxes.py).

python YOLO/test_boxes.py
"""
import numpy as np

from evaluate import match
from runner import iou
from run_boxes import Tracker

b = np.array([0, 0, 10, 10.0])
assert abs(iou(b, np.array([[0, 0, 10, 10.0], [5, 0, 15, 10.0]]))[1] - 50 / 150) < 1e-9

# matching: one-to-one, highest confidence first, IoU >= 0.5
gt = np.array([[0, 0, 10, 10.0], [20, 20, 30, 30.0]])
pred = np.array([[1, 1, 10, 10, 0.9], [0, 0, 10, 11, 0.8], [50, 50, 60, 60, 0.7]])
m = match(pred, gt)
assert len(m) == 1 and m[0][0][0] == 1  # the 0.8 box can't take the same ground truth twice

# tracker: overlap with the previous frame keeps the id, a gap longer than max_gap starts a new one
t = Tracker(thr=0.3, max_gap=1)
assert t.update(1, np.array([[0, 0, 10, 10, 0.9]])) == [1]
assert t.update(2, np.array([[1, 1, 11, 11, 0.9], [50, 50, 60, 60, 0.8]])) == [1, 2]
assert t.update(4, np.array([[1, 1, 11, 11, 0.9]])) == [3]
print("boxes OK")
