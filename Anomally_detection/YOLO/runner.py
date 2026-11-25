"""CPU inference for an exported YOLO26 model (ONNX via onnxruntime, or OpenVINO), timed per stage.

Shared by bench.py, evaluate.py and run_boxes.py so the benchmarked code is the code that runs in the pipeline.

    det = Detector("runs/yolo26n/weights/best.onnx", threads=2, conf=0.25)
    boxes, t = det(img_bgr)   # boxes: N×5 [x0, y0, x1, y1, conf] in input-image pixels; t: seconds per stage
t has pre / infer / post and their steps (STEPS); with Detector(..., profile=True) on OpenVINO also the inference split
into backbone / neck / head: the measured inference time split in proportion to OpenVINO's per-layer counters
(PERF_COUNT). Proportional because the counters overlap across threads: their sum is often above the wall time.
"""
import re
import time
from pathlib import Path

import cv2
import numpy as np

STEPS = {"pre": ["resize", "pad", "to_tensor"], "infer": ["backbone", "neck", "head"],
         "post": ["conf_filter", "nms", "rescale", "sort"]}


def layer_split(req):
    """Seconds per model part (backbone = layers 0-10, neck = 11-22, head = 23 Detect) from OpenVINO's per-layer
    counters. Fused Subgraph/Reorder nodes can carry another layer's name, so they go to the layer of the node that ran
    just before them (counters are in execution order)."""
    # ponytail: attribution heuristic for fused nodes; an exact per-layer split needs an unfused profile (onnxruntime)
    out, layer = {"backbone": 0.0, "neck": 0.0, "head": 0.0}, 0
    for p in req.profiling_info:
        m = re.search(r"model\.(\d+)", p.node_name)
        if m and p.node_type not in ("Subgraph", "Reorder"):
            layer = int(m.group(1))
        out["backbone" if layer <= 10 else "neck" if layer <= 22 else "head"] += p.real_time.total_seconds()
    return out


class Detector:
    def __init__(self, path, threads=1, conf=0.25, iou=0.6, backend=None, max_det=10, profile=False):
        """path: .onnx file, an OpenVINO .xml / *_openvino_model dir, or the board model best_axon_int8.tflite
        (YOLO/export_axon.py). backend: ort | openvino | tflite (default from path). profile: OpenVINO per-layer counters
        (backbone / neck / head in the timings; adds a little overhead)."""
        path = Path(path)
        if path.is_dir():
            path = next(path.glob("*.xml"))
        self.backend = backend or {".xml": "openvino", ".tflite": "tflite"}.get(path.suffix, "ort")
        self.conf, self.iou, self.threads, self.max_det, self.steps = conf, iou, threads, max_det, {}
        if self.backend == "tflite":
            from ai_edge_litert.interpreter import Interpreter
            self.interp = Interpreter(model_path=str(path), num_threads=threads)
            self.interp.allocate_tensors()
            d = self.interp.get_input_details()[0]
            self.size, self.in_q = int(d["shape"][1]), d["quantization"]
            self._run = self._run_tflite
        elif self.backend == "ort":
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
            self.input = self.sess.get_inputs()[0].name
            self.size = self.sess.get_inputs()[0].shape[2]
            self._run = lambda x: self.sess.run(None, {self.input: x})[0]
        else:
            import openvino as ov
            m = ov.Core().compile_model(str(path), "CPU", {"INFERENCE_NUM_THREADS": threads,
                                                          "PERFORMANCE_HINT": "LATENCY", "PERF_COUNT": profile})
            self.req, self.profile = m.create_infer_request(), profile
            self.size = int(m.inputs[0].shape[2])
            self._run = lambda x: self.req.infer({0: x})[0].copy()

    def preprocess(self, img):
        """BGR uint8 H×W×3 -> 1×3×S×S float32 RGB 0..1, letterboxed like ultralytics (pad 114, centred)."""
        t0 = time.perf_counter()
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        nw, nh = round(w * r), round(h * r)
        px, py = (self.size - nw) / 2, (self.size - nh) / 2
        if (nw, nh) != (w, h):
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        t1 = time.perf_counter()
        top, left = round(py - 0.1), round(px - 0.1)
        img = cv2.copyMakeBorder(img, top, self.size - nh - top, left, self.size - nw - left,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        t2 = time.perf_counter()
        x = np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        self.steps.update(resize=t1 - t0, pad=t2 - t1, to_tensor=time.perf_counter() - t2)
        return x, (r, left, top, w, h)

    def _run_tflite(self, x):
        """INT8 board model: 6 raw head outputs -> 1×N×6 [x0, y0, x1, y1, conf, cls], decoded like the board does it."""
        scale, zp = self.in_q
        q = np.clip(np.round(x[0].transpose(1, 2, 0) / scale + zp), -128, 127).astype(np.int8)[None]
        self.interp.set_tensor(self.interp.get_input_details()[0]["index"], q)
        self.interp.invoke()
        return decode_axon([(self.interp.get_tensor(o["index"]).astype(np.float32) - o["quantization"][1])
                            * o["quantization"][0] for o in self.interp.get_output_details()],
                           self.size, self.conf, self.max_det)

    def postprocess(self, out, meta):
        """Raw output -> N×5 boxes in input-image pixels. End-to-end head (1×300×6: x0,y0,x1,y1,conf,cls) needs no NMS;
        the one-to-many head (1×5×A: cx,cy,w,h,conf) gets OpenCV NMS."""
        t0 = time.perf_counter()
        r, left, top, w, h = meta
        out = out[0]
        if out.shape[-1] == 6:
            b = out[out[:, 4] >= self.conf, :5]
            t1 = t2 = time.perf_counter()
        else:
            p = out.T[out[4] >= self.conf]
            xyxy = np.c_[p[:, :2] - p[:, 2:4] / 2, p[:, :2] + p[:, 2:4] / 2]
            t1 = time.perf_counter()
            keep = cv2.dnn.NMSBoxes(np.c_[xyxy[:, :2], p[:, 2:4]].tolist(), p[:, 4].tolist(), self.conf, self.iou)
            b = np.c_[xyxy, p[:, 4]][np.asarray(keep, int).ravel()] if len(p) else np.zeros((0, 5))
            t2 = time.perf_counter()
        b = b.astype(np.float64)
        b[:, [0, 2]] = ((b[:, [0, 2]] - left) / r).clip(0, w)
        b[:, [1, 3]] = ((b[:, [1, 3]] - top) / r).clip(0, h)
        t3 = time.perf_counter()
        b = b[np.argsort(-b[:, 4])]
        self.steps.update(conf_filter=t1 - t0, nms=t2 - t1, rescale=t3 - t2, sort=time.perf_counter() - t3)
        return b

    def __call__(self, img):
        t0 = time.perf_counter()
        x, meta = self.preprocess(img)
        t1 = time.perf_counter()
        out = self._run(x)
        t2 = time.perf_counter()
        boxes = self.postprocess(out, meta)
        t3 = time.perf_counter()
        t = {**self.steps, "pre": t1 - t0, "infer": t2 - t1, "post": t3 - t2}
        if getattr(self, "profile", False):
            parts = layer_split(self.req)
            total = sum(parts.values()) or 1.0
            t.update({k: t["infer"] * v / total for k, v in parts.items()})
        return boxes, t


def decode_axon(outs, size, conf, max_det):
    """Raw one-to-one head (per level: box 1×g×g×4 = left/top/right/bottom in cells, cls 1×g×g×1 logit; any order)
    -> 1×N×6 [x0, y0, x1, y1, conf, 0] in model-input pixels, best max_det with conf >= conf (no NMS: one-to-one head).
    The board firmware (Board/src/main.c, decode_boxes) does the same."""
    rows = []
    for g in sorted({o.shape[1] for o in outs}, reverse=True):
        box = next(o for o in outs if o.shape[1] == g and o.shape[3] == 4)[0]
        cls = next(o for o in outs if o.shape[1] == g and o.shape[3] == 1)[0, ..., 0]
        s = size / g
        gy, gx = np.mgrid[:g, :g] + 0.5
        p = 1 / (1 + np.exp(-cls))
        rows.append(np.stack([(gx - box[..., 0]) * s, (gy - box[..., 1]) * s, (gx + box[..., 2]) * s,
                              (gy + box[..., 3]) * s, p, np.zeros_like(p)], -1).reshape(-1, 6))
    b = np.concatenate(rows)
    b = b[b[:, 4] >= conf]
    return b[np.argsort(-b[:, 4])[:max_det]][None]


def iou(a, b):
    """IoU of one box a (4,) with boxes b (N, 4)."""
    x0, y0 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x1, y1 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = (x1 - x0).clip(0) * (y1 - y0).clip(0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter)


def frame_paths(folder):
    """{frame_id: path}, id = last number in the filename (same rule as Classifier/run_csv.py)."""
    return {int(re.findall(r"\d+", p.stem)[-1]): p for p in Path(folder).iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
