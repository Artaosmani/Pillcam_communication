"""Run YOLO26 on the frames the classifier flagged and write boxes in the bbox_replay format.

python YOLO/run_boxes.py --clf runs/pipeline/classifier.csv --frames Data/test/frames \
    --model runs/yolo26n/weights/best_int8.onnx --threads 2 --out runs/pipeline/boxes.csv

Reads the classifier CSV (frame_id,t_rx_board_ns,t_infer_start_ns,t_infer_end_ns,t_sent_ns,score,detected[,usb_ns])
and runs YOLO only on rows with detected=1, in frame_id order. Writes:
  boxes.csv   frame_id,track_id,t_capture_ns,t_detect_ns,t_send_ns,x_min,y_min,x_max,y_max,img_w,img_h,confidence
              (same layout as PBS/drift/data/boxes_fake.csv; absolute pixels, origin top-left; one row per box)
  <out>.timing.csv  frame_id,n_boxes,pre_ms,infer_ms,post_ms,yolo_ms  (every detected frame, also those with no box)
Timeline per frame (replay): t_capture = t_rx_board (frame arrives at the board), t_detect = t_sent + usb_ns (the
board -> PC transfer, 0 if not measured) + the measured YOLO time, t_send = t_detect + the time to format the rows.
So t_detect - t_capture is board/stand-in classifier + USB + YOLO, without the gap between the classifier pass and
the YOLO pass of an offline run. track_id: greedy IoU tracker over consecutive detected frames.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Classifier"))
from evaluate import iou  # noqa: E402
from runner import Detector  # noqa: E402

COLUMNS = ["frame_id", "track_id", "t_capture_ns", "t_detect_ns", "t_send_ns",
           "x_min", "y_min", "x_max", "y_max", "img_w", "img_h", "confidence"]


class Tracker:
    """A box continues a track if IoU >= `thr` with that track's box in the previous frame (frame_id - 1 ... - max_gap)."""

    def __init__(self, thr=0.3, max_gap=1):
        self.thr, self.max_gap, self.next_id, self.tracks = thr, max_gap, 1, {}  # id -> (last frame, box)

    def update(self, fid, boxes):
        self.tracks = {k: v for k, v in self.tracks.items() if fid - v[0] <= self.max_gap}
        free, ids = dict(self.tracks), []
        for b in boxes:  # highest confidence first
            best, best_iou = None, self.thr
            for k, (_, tb) in free.items():
                o = float(iou(b[:4], tb[None])[0])
                if o >= best_iou:
                    best, best_iou = k, o
            if best is None:
                best, self.next_id = self.next_id, self.next_id + 1
            else:
                free.pop(best)
            self.tracks[best] = (fid, b[:4])
            ids.append(best)
        return ids


def frame_paths(folder):
    """{frame_id: path}, id = last number in the filename (same rule as Classifier/run_csv.py)."""
    from run_csv import frames
    return dict(frames(folder))


def main():
    a = parse_args()
    conf = a.conf if a.conf is not None else json.loads((a.model.parent / "deploy.json").read_text())["conf"]
    det = Detector(a.model, threads=a.threads, conf=conf)
    paths = frame_paths(a.frames)
    with open(a.clf) as f:
        rows = sorted((r for r in csv.DictReader(f) if int(r["detected"])), key=lambda r: int(r["frame_id"]))
    for i in range(a.warmup):  # first inferences are slow (allocation); keep them out of the numbers
        det(np.zeros((336, 336, 3), np.uint8))

    tracker, n_box = Tracker(a.track_iou, a.track_gap), 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fb, open(a.out.with_suffix(".timing.csv"), "w", newline="") as ft:
        wb, wt = csv.writer(fb), csv.writer(ft)
        wb.writerow(COLUMNS)
        wt.writerow(["frame_id", "n_boxes", "pre_ms", "infer_ms", "post_ms", "yolo_ms"])
        for r in rows:
            fid = int(r["frame_id"])
            img = cv2.imread(str(paths[fid]))  # decode stands in for receiving the frame bytes; not timed
            t0 = time.perf_counter_ns()
            boxes, s = det(img)
            yolo_ns = time.perf_counter_ns() - t0
            ids = tracker.update(fid, boxes)
            t_detect = int(r["t_sent_ns"]) + int(r.get("usb_ns") or 0) + yolo_ns
            t1 = time.perf_counter_ns()
            h, w = img.shape[:2]
            lines = [[fid, k, r["t_rx_board_ns"], None, None, f"{b[0]:.1f}", f"{b[1]:.1f}", f"{b[2]:.1f}",
                      f"{b[3]:.1f}", w, h, f"{b[4]:.4f}"] for k, b in zip(ids, boxes)]
            t_send = t_detect + time.perf_counter_ns() - t1
            for line in lines:
                line[3], line[4] = t_detect, t_send
            wb.writerows(lines)
            wt.writerow([fid, len(boxes), *(f"{s[k] * 1e3:.3f}" for k in ("pre", "infer", "post")),
                         f"{yolo_ns / 1e6:.3f}"])
            n_box += len(boxes)
    print(f"[boxes] {len(rows)} detected frames -> {n_box} boxes (conf {conf:.3f}, {a.threads} threads) -> {a.out}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clf", type=Path, required=True, help="classifier CSV")
    p.add_argument("--frames", type=Path, required=True, help="folder of frames, frame number in each filename")
    p.add_argument("--model", type=Path, required=True, help=".onnx or OpenVINO model (deploy.json read next to it)")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--conf", type=float, help="override the deployment confidence")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--track-iou", type=float, default=0.3)
    p.add_argument("--track-gap", type=int, default=1, help="frames a track may skip")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


if __name__ == "__main__":
    main()
