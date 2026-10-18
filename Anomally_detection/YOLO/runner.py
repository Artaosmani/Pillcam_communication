"""CPU inference for an exported YOLO26 model (ONNX via onnxruntime, or OpenVINO), timed per stage.

Shared by bench.py, evaluate.py and run_boxes.py so the benchmarked code is the code that runs in the pipeline.

    det = Detector("runs/yolo26n/weights/best.onnx", threads=2, conf=0.25)
    boxes, t = det(img_bgr)   # boxes: N×5 [x0, y0, x1, y1, conf] in input-image pixels; t: pre/infer/post seconds
"""
import time
from pathlib import Path

import cv2
import numpy as np


class Detector:
    def __init__(self, path, threads=1, conf=0.25, iou=0.6, backend=None):
        """path: .onnx file, or an OpenVINO .xml / *_openvino_model dir. backend: ort | openvino (default from path)."""
        path = Path(path)
        if path.is_dir():
            path = next(path.glob("*.xml"))
        self.backend = backend or ("openvino" if path.suffix == ".xml" else "ort")
        self.conf, self.iou, self.threads = conf, iou, threads
        if self.backend == "ort":
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
                                                          "PERFORMANCE_HINT": "LATENCY"})
            self.req = m.create_infer_request()
            self.size = int(m.inputs[0].shape[2])
            self._run = lambda x: self.req.infer({0: x})[0].copy()

    def preprocess(self, img):
        """BGR uint8 H×W×3 -> 1×3×S×S float32 RGB 0..1, letterboxed like ultralytics (pad 114, centred)."""
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        nw, nh = round(w * r), round(h * r)
        px, py = (self.size - nw) / 2, (self.size - nh) / 2
        if (nw, nh) != (w, h):
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, left = round(py - 0.1), round(px - 0.1)
        img = cv2.copyMakeBorder(img, top, self.size - nh - top, left, self.size - nw - left,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        x = np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        return x, (r, left, top, w, h)

    def postprocess(self, out, meta):
        """Raw output -> N×5 boxes in input-image pixels. End-to-end head (1×300×6: x0,y0,x1,y1,conf,cls) needs no NMS;
        the one-to-many head (1×5×A: cx,cy,w,h,conf) gets OpenCV NMS."""
        r, left, top, w, h = meta
        out = out[0]
        if out.shape[-1] == 6:
            b = out[out[:, 4] >= self.conf, :5]
        else:
            p = out.T[out[4] >= self.conf]
            xyxy = np.c_[p[:, :2] - p[:, 2:4] / 2, p[:, :2] + p[:, 2:4] / 2]
            keep = cv2.dnn.NMSBoxes(np.c_[xyxy[:, :2], p[:, 2:4]].tolist(), p[:, 4].tolist(), self.conf, self.iou)
            b = np.c_[xyxy, p[:, 4]][np.asarray(keep, int).ravel()] if len(p) else np.zeros((0, 5))
        b = b.astype(np.float64)
        b[:, [0, 2]] = ((b[:, [0, 2]] - left) / r).clip(0, w)
        b[:, [1, 3]] = ((b[:, [1, 3]] - top) / r).clip(0, h)
        return b[np.argsort(-b[:, 4])]

    def __call__(self, img):
        t0 = time.perf_counter()
        x, meta = self.preprocess(img)
        t1 = time.perf_counter()
        out = self._run(x)
        t2 = time.perf_counter()
        boxes = self.postprocess(out, meta)
        t3 = time.perf_counter()
        return boxes, {"pre": t1 - t0, "infer": t2 - t1, "post": t3 - t2}
