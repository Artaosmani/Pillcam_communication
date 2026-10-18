"""Test-set metrics for a trained YOLO26 model and its exports, including how tight the boxes are.

python YOLO/evaluate.py --weights runs/yolo26n/weights/best.pt            # .pt + exports found next to it

Per format (pt, onnx = FP32, int8 = ONNX INT8, if the file exists):
  - ultralytics val on `test` (Kvasir-SEG test + every test negative as background) and `test_pos` (polyps only):
    precision, recall, mAP50, mAP50-95 (conf 0.001, as usual for mAP)
  - at the deployment confidence: box tightness on test_pos (greedy IoU >= 0.5 matching per image, highest confidence
    first): mean IoU, mean |width error| and |height error| in pixels (original image and scaled to the 336×336 frame),
    the signed mean (positive = predicted box larger than ground truth), and the false-alarm rate on test background
    images (fraction with at least one box)
The deployment confidence is the F1-optimal confidence of the .pt model on `val`, saved to deploy.json next to the
weights so run_boxes.py uses the same value. Writes runs/<name>/metrics_eval.json.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import Detector  # noqa: E402
from src import DEFAULT_CONFIG, data_yaml_path, resolve  # noqa: E402
from train import resolve_device  # noqa: E402

FRAME = 336  # board frame side; box errors are also reported at this scale


def iou(a, b):
    """IoU of one box a (4,) with boxes b (N, 4)."""
    x0, y0 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x1, y1 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = (x1 - x0).clip(0) * (y1 - y0).clip(0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter)


def gt_boxes(label_path, w, h):
    """YOLO label file (cls cx cy w h, normalised) -> N×4 xyxy pixels."""
    rows = np.loadtxt(label_path, ndmin=2) if label_path.stat().st_size else np.zeros((0, 5))
    cx, cy, bw, bh = rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
    return np.c_[cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]


def match(pred, gt, thr=0.5):
    """Greedy one-to-one matching, highest confidence first. Returns [(pred box, gt box, iou)]."""
    used, out = set(), []
    for p in pred:
        if not len(gt):
            break
        o = iou(p[:4], gt)
        o[list(used)] = 0
        j = int(np.argmax(o))
        if o[j] >= thr:
            used.add(j)
            out.append((p[:4], gt[j], float(o[j])))
    return out


def tightness(det, images_dir):
    """Box tightness on polyp images, plus recall at this confidence."""
    ious, dw, dh, dw336, dh336, n_gt = [], [], [], [], [], 0
    for p in sorted(Path(images_dir).glob("*.jpg")):
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        gt = gt_boxes(Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt"), w, h)
        n_gt += len(gt)
        for pb, gb, o in match(det(img)[0], gt):
            ious.append(o)
            ew, eh = (pb[2] - pb[0]) - (gb[2] - gb[0]), (pb[3] - pb[1]) - (gb[3] - gb[1])
            dw.append(ew), dh.append(eh), dw336.append(ew * FRAME / w), dh336.append(eh * FRAME / h)
    a = lambda v: np.asarray(v, float)
    return {"matched": len(ious), "gt_boxes": n_gt, "recall_iou50": len(ious) / max(n_gt, 1),
            "mean_iou": float(np.mean(ious)), "median_iou": float(np.median(ious)),
            "abs_w_err_px": float(np.mean(np.abs(dw))), "abs_h_err_px": float(np.mean(np.abs(dh))),
            "abs_w_err_px336": float(np.mean(np.abs(dw336))), "abs_h_err_px336": float(np.mean(np.abs(dh336))),
            "signed_w_err_px336": float(a(dw336).mean()), "signed_h_err_px336": float(a(dh336).mean())}


def false_alarms(det, images_dir):
    bg = [p for p in sorted(Path(images_dir).iterdir()) if p.name.startswith("neg_")]
    hits = sum(len(det(cv2.imread(str(p)))[0]) > 0 for p in bg)
    return {"background_images": len(bg), "with_box": int(hits), "false_alarm_rate": hits / max(len(bg), 1)}


def val(model_path, data, split, imgsz, device):
    m = YOLO(str(model_path), task="detect").val(data=data, split=split, imgsz=imgsz, batch=1 if device == "cpu" else 16,
                                                  conf=0.001, iou=0.6, device=device, plots=False, verbose=False)
    return m, {"precision": float(m.box.mp), "recall": float(m.box.mr),
               "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--weights", type=Path, required=True, help="best.pt; exports are looked up next to it")
    a = p.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    data, imgsz = str(data_yaml_path(cfg)), cfg["train"]["imgsz"]
    ds = yaml.safe_load(Path(data).read_text())

    m, _ = val(a.weights, data, "val", imgsz, resolve_device("auto"))
    f1 = m.box.f1_curve.mean(0)
    conf = float(m.box.px[int(np.argmax(f1))])
    (a.weights.parent / "deploy.json").write_text(json.dumps({"conf": conf, "val_f1": float(f1.max())}, indent=2))
    print(f"[eval] deployment confidence {conf:.3f} (val F1 {f1.max():.3f})")

    formats = {"pt": a.weights, "onnx": a.weights.with_suffix(".onnx"),
               "int8": a.weights.with_name(f"{a.weights.stem}_int8.onnx")}
    out = {"weights": str(a.weights), "imgsz": imgsz, "deploy_conf": conf, "formats": {}}
    for name, path in formats.items():
        if not path.exists():
            continue
        dev = resolve_device("auto") if name == "pt" else "cpu"
        r = {"test": val(path, data, "test", imgsz, dev)[1], "test_pos": val(path, data, "test_pos", imgsz, dev)[1]}
        det = Detector(formats["onnx"] if name == "pt" else path, conf=conf)  # pt boxes = FP32 ONNX boxes
        r["tightness"] = tightness(det, ds["test_pos"])
        r["false_alarms"] = false_alarms(det, ds["test"])
        out["formats"][name] = r
        print(f"[eval] {name}: {json.dumps(r)}")
    path = a.weights.parent.parent / "metrics_eval.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[eval] -> {path}")


if __name__ == "__main__":
    main()
