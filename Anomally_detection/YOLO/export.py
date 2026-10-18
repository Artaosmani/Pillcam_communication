"""Export trained YOLO26 weights for PC CPU inference: ONNX FP32, ONNX INT8 and OpenVINO FP32.

python YOLO/export.py --weights runs/yolo26n/weights/best.pt

Writes next to the weights: best.onnx, best_int8.onnx, best_openvino_model/.
INT8 is onnxruntime static quantisation (QDQ, per-channel weights), calibrated on `export.calib_images` un-augmented
train images. The same QDQ file also runs INT8 in OpenVINO (bench.py reports it as openvino_int8), so NNCF is not needed.
`export.int8_exclude` keeps nodes whose name contains one of these strings in float; "detect" = the Detect head.
The head must stay float: boxes (0..352 px) and scores (0..1) leave it in one tensor, and one INT8 scale for both
rounds every score to 0 (no boxes at all).
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import onnx
import yaml
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import Detector  # noqa: E402
from src import DEFAULT_CONFIG, resolve  # noqa: E402


class Calib(CalibrationDataReader):
    def __init__(self, det, paths):
        self.items = iter([{det.input: det.preprocess(cv2.imread(str(p)))[0]} for p in paths])

    def get_next(self):
        return next(self.items, None)


def export(cfg, weights):
    e, imgsz = cfg["export"], cfg["train"]["imgsz"]
    fp32 = Path(YOLO(str(weights)).export(format="onnx", imgsz=imgsz, simplify=True))
    YOLO(str(weights)).export(format="openvino", imgsz=imgsz)

    train = sorted((resolve(cfg["paths"]["dataset"]) / "images" / "train").glob("*"))
    paths = random.Random(cfg["train"]["seed"]).sample(train, min(e["calib_images"], len(train)))
    pre, int8 = fp32.with_name(f"{fp32.stem}_pre.onnx"), fp32.with_name(f"{fp32.stem}_int8.onnx")
    quant_pre_process(str(fp32), str(pre))
    names = [n.name for n in onnx.load(str(pre)).graph.node]
    head = f"/model.{YOLO(str(weights)).model.model[-1].i}/"  # "detect" in int8_exclude = the whole Detect head
    keep = [head if s == "detect" else s for s in e.get("int8_exclude") or []]
    exclude = [n for n in names if any(s in n for s in keep)]
    quantize_static(str(pre), str(int8), Calib(Detector(fp32), paths), quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax, nodes_to_exclude=exclude)
    pre.unlink()
    # keep ultralytics metadata (imgsz, names, end2end) so YOLO(int8).val() can read the file
    src, dst = onnx.load(str(fp32)), onnx.load(str(int8))
    del dst.metadata_props[:]
    dst.metadata_props.extend(src.metadata_props)
    onnx.save(dst, str(int8))
    print(f"[export] {fp32}, {int8} ({len(exclude)} nodes kept float), {fp32.with_name(fp32.stem + '_openvino_model')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--weights", type=Path, required=True)
    a = p.parse_args()
    export(yaml.safe_load(a.config.read_text()), a.weights)
